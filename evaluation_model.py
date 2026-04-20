# ===========================================
# evaluation_model.py — Unified Evaluation Dashboard
# ===========================================
import os, re, pickle
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split

# =========================
# CONFIG PATHS
# =========================
DATA_PATH = r"D:\projects\mscis\mscis1\dataset\consolidated_emails.csv"
MODEL_DIR = r"D:\projects\mscis\mscis1\model"

# GloVe BiLSTM
GLOVE_DIR = os.path.join(MODEL_DIR, "glove_bilstm")
GLOVE_PATH = os.path.join(GLOVE_DIR, "glove_bilstm.pt")
GLOVE_VOCAB = os.path.join(GLOVE_DIR, "vocab.pkl")

# Word2Vec BiLSTM
W2V_DIR = os.path.join(MODEL_DIR, "word2vec_bilstm")
W2V_PATH = os.path.join(W2V_DIR, "wordvec_bilstm.pt")
W2V_VOCAB = os.path.join(W2V_DIR, "vocab.pkl")

# DistilBERT
DISTILBERT_PATH = os.path.join(MODEL_DIR, "distilbert")

# Save Directory
SAVE_DIR = os.path.join(MODEL_DIR, "evaluation_outputs")
os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD AND CLEAN DATA
# =========================
df = pd.read_csv(DATA_PATH)
df = df[df["label"].isin(["0", "1"])].dropna(subset=["subject", "body"])
df["label"] = df["label"].astype(int)
df["text"] = (
    df["subject"].astype(str)
    + " "
    + df["body"].astype(str)
    + " "
    + df["sender"].astype(str)
)


def clean_text(t):
    t = t.lower()
    t = re.sub(r"http\S+|www\S+", "", t)
    t = re.sub(r"<.*?>", "", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


df["text"] = df["text"].apply(clean_text)
X_train, X_val, y_train, y_val = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)


# =========================
# DATASET CLASS
# =========================
class EmailDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =========================
# BiLSTM MODEL CLASS
# =========================
class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        embedding_matrix,
        hidden_dim=128,
        num_layers=1,
        dropout=0.3,
        use_two_fc=False,
    ):
        super().__init__()
        vocab_size, embed_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32), freeze=True
        )
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.use_two_fc = use_two_fc
        if use_two_fc:  # for GloVe
            self.fc1 = nn.Linear(hidden_dim * 2, 64)
            self.fc2 = nn.Linear(64, 1)
            self.dropout = nn.Dropout(0.5)
        else:  # for Word2Vec
            self.fc = nn.Linear(hidden_dim * 2, 1)
            self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x, _ = torch.max(x, 1)
        x = self.dropout(x)
        if self.use_two_fc:
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
        else:
            x = self.fc(x)
        return x


# =========================
# HELPER FUNCTIONS
# =========================
def evaluate_model(model, dataloader):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb).squeeze()
            prob = torch.sigmoid(out).cpu().numpy()
            pred = (prob >= 0.5).astype(int)
            y_true.extend(yb.cpu().numpy())
            y_pred.extend(pred)
            y_prob.extend(prob)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return acc, prec, rec, f1, np.array(y_true), np.array(y_pred), np.array(y_prob)


def text_to_seq(text, vocab, max_len=200):
    seq = [vocab.get(w, 1) for w in text.split()]
    return seq[:max_len] + [0] * (max_len - len(seq))


# =========================
# EVALUATE ALL MODELS
# =========================
results = {}
plt.figure(figsize=(6, 5))

# ---- GloVe ----
if os.path.exists(GLOVE_PATH) and os.path.exists(GLOVE_VOCAB):
    with open(GLOVE_VOCAB, "rb") as f:
        vocab = pickle.load(f)
    X_val_seq = np.array([text_to_seq(t, vocab) for t in X_val])
    val_loader = DataLoader(EmailDataset(X_val_seq, y_val), batch_size=128)
    model = BiLSTMClassifier(
        np.zeros((len(vocab), 100)), num_layers=1, use_two_fc=True
    ).to(device)
    model.load_state_dict(torch.load(GLOVE_PATH, map_location=device))
    acc, prec, rec, f1, y_true, y_pred, y_prob = evaluate_model(model, val_loader)
    results["GloVe"] = (acc, prec, rec, f1)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.plot(fpr, tpr, label=f"GloVe (AUC={auc(fpr,tpr):.3f})")
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm).plot(cmap="Blues")
    plt.title("GloVe BiLSTM Confusion Matrix")
    plt.savefig(os.path.join(SAVE_DIR, "glove_confusion.png"))
    plt.close()
else:
    print("⚠️ GloVe model or vocab missing, skipping.")

# ---- Word2Vec ----
if os.path.exists(W2V_PATH) and os.path.exists(W2V_VOCAB):
    with open(W2V_VOCAB, "rb") as f:
        vocab = pickle.load(f)
    X_val_seq = np.array([text_to_seq(t, vocab) for t in X_val])
    val_loader = DataLoader(EmailDataset(X_val_seq, y_val), batch_size=128)
    model = BiLSTMClassifier(
        np.zeros((len(vocab), 100)), num_layers=2, use_two_fc=False
    ).to(device)
    model.load_state_dict(torch.load(W2V_PATH, map_location=device))
    acc, prec, rec, f1, y_true, y_pred, y_prob = evaluate_model(model, val_loader)
    results["Word2Vec"] = (acc, prec, rec, f1)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.plot(fpr, tpr, label=f"Word2Vec (AUC={auc(fpr,tpr):.3f})")
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm).plot(cmap="Oranges")
    plt.title("Word2Vec BiLSTM Confusion Matrix")
    plt.savefig(os.path.join(SAVE_DIR, "word2vec_confusion.png"))
    plt.close()
else:
    print("⚠️ Word2Vec model or vocab missing, skipping.")

# ---- DistilBERT (optimized for low GPU memory) ----
if os.path.exists(DISTILBERT_PATH):
    tokenizer = DistilBertTokenizerFast.from_pretrained(DISTILBERT_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(DISTILBERT_PATH)
    model.to(device)
    model.eval()

    # Encode in batches to prevent OOM
    batch_size = 64  # adjust (try 32 if still OOM)
    texts = list(X_val)
    all_preds, all_probs, all_labels = [], [], []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attn_mask)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
            preds = np.argmax(outputs.logits.cpu().numpy(), axis=1)

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(y_val.values[i : i + batch_size])

        # Free GPU memory after each batch
        del input_ids, attn_mask, outputs
        torch.cuda.empty_cache()

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    results["DistilBERT"] = (acc, prec, rec, f1)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.plot(fpr, tpr, label=f"DistilBERT (AUC={auc(fpr,tpr):.3f})")

    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm).plot(cmap="Greens")
    plt.title("DistilBERT Confusion Matrix")
    plt.savefig(os.path.join(SAVE_DIR, "distilbert_confusion.png"))
    plt.close()

else:
    print("⚠️ DistilBERT model missing, skipping.")


# =========================
# ROC CURVE COMPARISON
# =========================
plt.plot([0, 1], [0, 1], "k--")
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig(os.path.join(SAVE_DIR, "roc_comparison.png"))
plt.close()

# =========================
# PERFORMANCE BAR CHART
# =========================
if results:
    models = list(results.keys())
    metrics = np.array(list(results.values()))
    plt.figure(figsize=(8, 5))
    x = np.arange(len(models))
    plt.bar(x - 0.3, metrics[:, 0], 0.2, label="Accuracy")
    plt.bar(x - 0.1, metrics[:, 1], 0.2, label="Precision")
    plt.bar(x + 0.1, metrics[:, 2], 0.2, label="Recall")
    plt.bar(x + 0.3, metrics[:, 3], 0.2, label="F1")
    plt.xticks(x, models)
    plt.title("Overall Model Performance")
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, "overall_performance.png"))
    plt.close()

summary = pd.DataFrame(
    metrics, columns=["Accuracy", "Precision", "Recall", "F1"], index=models
)
print("\n✅ Evaluation complete! Results saved in:", SAVE_DIR)
print(summary.round(4))
