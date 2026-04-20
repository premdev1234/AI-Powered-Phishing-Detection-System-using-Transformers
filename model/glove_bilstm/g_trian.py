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

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from gensim.models import Word2Vec

print("PyTorch version:", torch.__version__)
# ==============================
# 1. Load Dataset
# ==============================
DATA_PATH = "D:\mscis\dataset\consolidated_emails.csv"
df = pd.read_csv(DATA_PATH)

# ==============================
# 1. Load & Clean Dataset (updated)
# ==============================

# Ensure required columns exist, create empty ones if missing
required_cols = ['label', 'subject', 'body', 'sender']
for col in required_cols:
    if col not in df.columns:
        df[col] = ""

# Keep only rows with label 0 or 1 and non-empty subject/body
df = df[df['label'].isin(['0','1'])].dropna(subset=['subject','body'])

# Convert label to int
df['label'] = df['label'].astype(int)

# Combine text columns (exclude url if it's just 0/1)
df['text'] = df['subject'].astype(str) + " " + df['body'].astype(str) + " " + df['sender'].astype(str)

# Basic text cleaning
def clean_text(text):
    text = str(text).lower()                          # Lowercase
    text = re.sub(r"http\S+|www\S+", "", text)        # Remove URLs
    text = re.sub(r"<.*?>", "", text)                 # Remove HTML tags
    text = re.sub(r"[^a-z0-9\s]", " ", text)         # Keep only letters/numbers
    text = re.sub(r"\s+", " ", text).strip()         # Remove extra spaces
    return text

df['text'] = df['text'].apply(clean_text)

# Optional: preview
print("Sample cleaned texts:")
print(df[['label','text']].head())
print("Total samples:", len(df))

# ==============================
# 2. Train/Val Split
# ==============================
from sklearn.model_selection import train_test_split

X = df['text'].astype(str)
y = df['label']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("✅ Train/Val shapes:", X_train.shape, X_val.shape)

# ==============================
# 3. GloVe BiLSTM (PyTorch)
# ==============================
golve_path = "/content/drive/MyDrive/Colab_Notebooks/glove_bilstm"
os.makedirs(golve_path, exist_ok=True)





from collections import Counter

# --- Download GloVe 100d embeddings ---
!wget -q http://nlp.stanford.edu/data/glove.6B.zip -O glove.6B.zip
!unzip -o -q glove.6B.zip   # ✅ fixed line
GLOVE_PATH = "glove.6B.100d.txt"

# -------------------------------
# Vocab & Text Processing
# -------------------------------
def build_vocab(texts, max_vocab=100000):
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    vocab = {"<PAD>":0, "<OOV>":1}
    for i, (word, _) in enumerate(counter.most_common(max_vocab-2), 2):
        vocab[word] = i
    return vocab

vocab = build_vocab(X_train)

def text_to_seq(text, vocab, max_len=200):
    seq = [vocab.get(w, 1) for w in text.split()]
    if len(seq) < max_len:
        seq += [0] * (max_len - len(seq))
    else:
        seq = seq[:max_len]
    return seq

Xtr = np.array([text_to_seq(t, vocab) for t in X_train])
Xva = np.array([text_to_seq(t, vocab) for t in X_val])
ytr, yva = y_train.values, y_val.values

with open(os.path.join(golve_path, "vocab.pkl"), "wb") as f:
    pickle.dump(vocab, f)

print("✅ Vocab size:", len(vocab))
print("✅ Train shape:", Xtr.shape, " Val shape:", Xva.shape)
# -------------------------------
# Load GloVe embeddings
# -------------------------------
emb_index = {}
with open(GLOVE_PATH, "r", encoding="utf8") as f:
    for line in f:
        values = line.strip().split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype=np.float32)
        emb_index[word] = coefs

embedding_matrix = np.zeros((len(vocab), 100))
for word, i in vocab.items():
    if word in emb_index:
        embedding_matrix[i] = emb_index[word]


# -------------------------------
# BiLSTM Model
# -------------------------------
class BiLSTMClassifier(nn.Module):
    def __init__(self, embedding_matrix):
        super().__init__()
        vocab_size, embed_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32),
            freeze=True
        )
        self.lstm = nn.LSTM(embed_dim, 128, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(256, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)   # no sigmoid here (use BCEWithLogitsLoss)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x, _ = torch.max(x, 1)               # max pooling
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)                      # raw logits
        return x

# -------------------------------
# Dataset + DataLoader
# -------------------------------
class EmailDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

train_loader = DataLoader(EmailDataset(Xtr, ytr), batch_size=128, shuffle=True)
val_loader   = DataLoader(EmailDataset(Xva, yva), batch_size=128)

# -------------------------------
# Training Setup
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_glove = BiLSTMClassifier(embedding_matrix).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model_glove.parameters(), lr=1e-3)

# -------------------------------
# Helper: Evaluate Accuracy
# -------------------------------
def evaluate_accuracy(model, dataloader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb).squeeze()
            preds = (torch.sigmoid(out) >= 0.5).int()
            y_true.extend(yb.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    return accuracy_score(y_true, y_pred)

# -------------------------------
# Training Loop with Early Stopping
# -------------------------------
num_epochs = 30         # max epochs
patience   = 3          # stop if no improvement for 3 epochs
best_acc   = 0.0
trials     = 0

for epoch in range(num_epochs):
    model_glove.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model_glove(xb).squeeze()
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    val_acc  = evaluate_accuracy(model_glove, val_loader, device)

    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_loss:.4f} - Val Acc: {val_acc:.4f}")

    # Early Stopping Check
    if val_acc > best_acc:
        best_acc = val_acc
        trials = 0
        torch.save(model_glove.state_dict(), os.path.join(golve_path, "glove_bilstm.pt"))
        print(f"✅ New best model saved (Val Acc: {best_acc:.4f})")
    else:
        trials += 1
        if trials >= patience:
            print("⏹️ Early stopping triggered")
            break
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def evaluate_model(model, dataloader, threshold=0.5, device="cpu"):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in dataloader:
            # move data to the same device as model
            xb, yb = xb.to(device), yb.to(device)

            out = model(xb).squeeze()

            # bring back to CPU for sklearn
            y_true.extend(yb.cpu().numpy())
            y_pred.extend((out.cpu().numpy() >= threshold).astype(int))

    acc = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    return y_true, y_pred


# ==============================
# Call with correct device
# ==============================
print("=== GloVe BiLSTM Evaluation ===")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_glove.to(device)  # move model to GPU
val_loader_glove = DataLoader(EmailDataset(Xva, yva), batch_size=128)

y_true_glove, y_pred_glove = evaluate_model(model_glove, val_loader_glove, device=device)


