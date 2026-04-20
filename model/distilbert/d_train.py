# ==============================
# 1. Import libraries
# ==============================
import os
import pickle
import json
import random
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from gensim.models import Word2Vec

print("PyTorch version:", torch.__version__)
# ==============================
# 1. Load Dataset
# ==============================
DATA_PATH = "D:\mscis\dataset\consolidated_emails.csv"
df = pd.read_csv(DATA_PATH)

# ==============================
# 2. Load & Clean Dataset (updated)
# ==============================

# Ensure required columns exist, create empty ones if missing
required_cols = ["label", "subject", "body", "sender"]
for col in required_cols:
    if col not in df.columns:
        df[col] = ""

# Keep only rows with label 0 or 1 and non-empty subject/body
df = df[df["label"].isin(["0", "1"])].dropna(subset=["subject", "body"])

# Convert label to int
df["label"] = df["label"].astype(int)

# Combine text columns (exclude url if it's just 0/1)
df["text"] = (
    df["subject"].astype(str)
    + " "
    + df["body"].astype(str)
    + " "
    + df["sender"].astype(str)
)


# Basic text cleaning
def clean_text(text):
    text = str(text).lower()  # Lowercase
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r"<.*?>", "", text)  # Remove HTML tags
    text = re.sub(r"[^a-z0-9\s]", " ", text)  # Keep only letters/numbers
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text


df["text"] = df["text"].apply(clean_text)

# Optional: preview
print("Sample cleaned texts:")
print(df[["label", "text"]].head())
print("Total samples:", len(df))

# ==============================
# 2. DistilBERT Model (PyTorch)
# ==============================
distilbert_path = "/content/drive/MyDrive/distilbert"

dataset = pd.DataFrame({"text": df["text"], "label": df["label"]})
from datasets import Dataset, ClassLabel

dataset = Dataset.from_pandas(dataset)
dataset = dataset.cast_column("label", ClassLabel(names=["0", "1"]))
dataset = dataset.train_test_split(test_size=0.2, stratify_by_column="label", seed=42)
train_ds, val_ds = dataset["train"], dataset["test"]

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")


def tokenize(batch):
    return tokenizer(
        batch["text"], padding="max_length", truncation=True, max_length=256
    )


train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)
train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
val_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)

training_args = TrainingArguments(
    output_dir=distilbert_path,
    eval_strategy="epoch",  # <- correct arg (eval_strategy is deprecated)
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,  # increase
    gradient_accumulation_steps=1,  # remove accumulation if possible
    per_device_eval_batch_size=32,  # bigger eval batch
    num_train_epochs=2,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_dir="./logs",
    logging_steps=500,  # less frequent
    report_to="none",
    fp16=True,  # ✅ use mixed precision on T4
    dataloader_num_workers=2,  # speed up data loading
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": classification_report(labels, preds, output_dict=True)["1"][
            "precision"
        ],
        "recall": classification_report(labels, preds, output_dict=True)["1"]["recall"],
        "f1": classification_report(labels, preds, output_dict=True)["1"]["f1-score"],
    }


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.train()

# Save DistilBERT model
model.save_pretrained(distilbert_path)
tokenizer.save_pretrained(distilbert_path)
print(f"✅ DistilBERT saved at {distilbert_path}")
