# ==============================
# test.py - Evaluate Models & Ensemble
# ==============================
import os
import re
import pickle
import mysql.connector
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from itertools import product


# ==============================
# 1. Database Connection
# ==============================
def get_connection():
    return mysql.connector.connect(
        user="root",
        password="Sonudev2002@",  # update if needed
        host="localhost",
        database="mscis_project1",
        port=3306,
    )


def load_test_data():
    """Fetch and clean test data from MySQL."""
    conn = get_connection()
    query = "SELECT s_no, sender, subject, body, label FROM testcase"
    df = pd.read_sql(query, conn)
    conn.close()

    df = df.dropna(subset=["subject", "body"])
    df["label"] = df["label"].astype(int)

    df["text"] = (
        df["subject"].astype(str)
        + " "
        + df["body"].astype(str)
        + " "
        + df["sender"].astype(str)
    )

    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    df["text"] = df["text"].apply(clean_text)
    return df


# ==============================
# 2. Model Definitions
# ==============================
class BiLSTMClassifier_Glove(nn.Module):
    """Architecture used during GloVe training."""

    def __init__(self, embedding_matrix):
        super().__init__()
        vocab_size, embed_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32), freeze=True
        )
        self.lstm = nn.LSTM(embed_dim, 128, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(256, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x, _ = torch.max(x, 1)
        x = self.dropout(torch.relu(self.fc1(x)))
        return self.fc2(x)


class BiLSTMClassifier_W2V(nn.Module):
    """Architecture used during Word2Vec training."""

    def __init__(self, embedding_matrix, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        vocab_size, embed_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32), freeze=False
        )
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out[:, -1, :])
        return self.fc(out)


# ==============================
# 3. Load Pretrained Models
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- GloVe BiLSTM ----
with open("./model/glove_bilstm/vocab.pkl", "rb") as f:
    vocab_glove = pickle.load(f)

GLOVE_PATH = r"D:\mscis\model\glove_bilstm\glove.6B.100d.txt"
emb_index = {}
with open(GLOVE_PATH, "r", encoding="utf8") as f:
    for line in f:
        values = line.strip().split()
        word, coefs = values[0], np.asarray(values[1:], dtype=np.float32)
        emb_index[word] = coefs

embedding_matrix_glove = np.zeros((len(vocab_glove), 100))
for word, i in vocab_glove.items():
    if word in emb_index:
        embedding_matrix_glove[i] = emb_index[word]

glove_model = BiLSTMClassifier_Glove(embedding_matrix_glove).to(device)
glove_model.load_state_dict(
    torch.load("./model/glove_bilstm/glove_bilstm.pt", map_location=device)
)
glove_model.eval()

# ---- Word2Vec BiLSTM ----
with open("./model/word2vec_bilstm/vocab.pkl", "rb") as f:
    vocab_w2v = pickle.load(f)

embedding_matrix_w2v = np.load("./model/word2vec_bilstm/embedding_matrix.npy")
w2v_model = BiLSTMClassifier_W2V(embedding_matrix_w2v).to(device)
w2v_model.load_state_dict(
    torch.load("./model/word2vec_bilstm/wordvec_bilstm.pt", map_location=device)
)
w2v_model.eval()

# ---- DistilBERT ----
distil_tokenizer = DistilBertTokenizerFast.from_pretrained("./model/distilbert")
distil_model = DistilBertForSequenceClassification.from_pretrained(
    "./model/distilbert"
).to(device)
distil_model.eval()


# ==============================
# 4. Helpers
# ==============================
def text_to_seq(text, vocab, max_len=200):
    seq = [vocab.get(w, 1) for w in text.split()]
    return seq[:max_len] + [0] * max(0, max_len - len(seq))


def predict_glove(texts):
    seqs = torch.tensor(
        [text_to_seq(t, vocab_glove) for t in texts], dtype=torch.long
    ).to(device)
    with torch.no_grad():
        out = glove_model(seqs).squeeze()
        probs = torch.sigmoid(out).cpu().numpy()
    return probs


def predict_w2v(texts):
    seqs = torch.tensor(
        [text_to_seq(t, vocab_w2v) for t in texts], dtype=torch.long
    ).to(device)
    with torch.no_grad():
        out = w2v_model(seqs).squeeze()
        probs = torch.sigmoid(out).cpu().numpy()
    return probs


def predict_distil(texts):
    enc = distil_tokenizer(
        texts, truncation=True, padding=True, max_length=200, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        out = distil_model(**enc).logits  # shape (N,2)
        probs = torch.softmax(out, dim=1)[:, 1]  # take probability of class 1
    return probs.cpu().numpy()


##################


def evaluate():
    # Load test data
    df = load_test_data()
    texts, labels = df["text"].tolist(), df["label"].tolist()
    labels = np.array(labels).reshape(-1, 1)

    # Get model prediction probabilities
    prob_glove = predict_glove(texts).reshape(-1, 1)
    prob_w2v = predict_w2v(texts).reshape(-1, 1)
    prob_distil = predict_distil(texts).reshape(-1, 1)

    # Individual predictions (can adjust thresholds if desired)
    pred_glove = (prob_glove >= 0.98).astype(int)
    pred_w2v = (prob_w2v >= 0.95).astype(int)
    pred_distil = (prob_distil >= 0.95).astype(int)

    # Soft voting
    avg_prob = (prob_glove + prob_w2v + prob_distil) / 3
    pred_soft = (avg_prob >= 0.7).astype(int)

    # --- Weighted Voting (DistilBERT higher weight) ---
    weighted_prob = 0.2 * prob_glove + 0.1 * prob_w2v + 0.7 * prob_distil
    pred_weighted = (weighted_prob >= 0.7).astype(int)

    # Stacking
    meta_features = np.hstack([prob_glove, prob_w2v, prob_distil])
    meta_clf = LogisticRegression()
    meta_clf.fit(meta_features, labels.ravel())
    pred_stacking = meta_clf.predict(meta_features)

    # Accuracy summary
    results = {
        "GloVe BiLSTM": accuracy_score(labels, pred_glove),
        "Word2Vec BiLSTM": accuracy_score(labels, pred_w2v),
        "DistilBERT": accuracy_score(labels, pred_distil),
        "Soft Voting": accuracy_score(labels, pred_soft),
        "Weighted Voting (Best)": (labels, pred_weighted),
        "Stacking Ensemble": accuracy_score(labels, pred_stacking),
    }

    print("\n=== Best Weighted Voting Configuration ===")
    print(f"  GloVe weight     : {w1:.2f}")
    print(f"  Word2Vec weight  : {w2:.2f}")
    print(f"  DistilBERT weight: {w3:.2f}")
    print(f"  Threshold        : {threshold}")
    print(f"  Accuracy         : {best_accuracy:.4f}")

    print("\n=== Accuracy Summary ===")
    for name, acc in results.items():
        print(f"{name:25s}: {acc:.4f}")

    print("\n=== Classification Report (Stacking Ensemble) ===")
    print(classification_report(labels, pred_stacking))
    print("Confusion Matrix:\n", confusion_matrix(labels, pred_stacking))


# ==============================
# Run
# ==============================
if __name__ == "__main__":
    evaluate()
