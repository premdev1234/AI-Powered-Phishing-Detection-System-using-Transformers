# ==============================
# 1. Import libraries
# ==============================
import os
import re
import numpy as np
import pandas as pd
import pickle
from nltk.tokenize import word_tokenize
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from gensim.models import Word2Vec

print("PyTorch version:", torch.__version__)

# ==============================
# 2. Load Dataset
# ==============================
DATA_PATH = r"D:\mscis\dataset\consolidated_emails.csv"
df = pd.read_csv(DATA_PATH)

required_cols = ["label", "subject", "body", "sender"]
for col in required_cols:
    if col not in df.columns:
        df[col] = ""

df = df[df["label"].isin(["0", "1"])].dropna(subset=["subject", "body"])
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

print("Sample cleaned texts:")
print(df[["label", "text"]].head())
print("Total samples:", len(df))

# ==============================
# 3. Train/Val Split
# ==============================
from sklearn.model_selection import train_test_split

X = df["text"].astype(str)
y = df["label"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("✅ Train/Val shapes:", X_train.shape, X_val.shape)

# ==============================
# 4. Word2Vec Training
# ==============================
wordvec_path = r"D:\mscis\model\word2vec_bilstm"
os.makedirs(wordvec_path, exist_ok=True)

print("start word2vec on training data")
sentences = [word_tokenize(text) for text in X_train]
w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=2, workers=4)
w2v_model.save(os.path.join(wordvec_path, "word2vec_email.model"))

# Build vocab
vocab_w2v = {"<PAD>": 0, "<OOV>": 1}
for tokens in sentences:
    for w in tokens:
        if w not in vocab_w2v:
            vocab_w2v[w] = len(vocab_w2v)

print("encode text into padded sequence")
MAX_LEN = 200


def encode_texts(texts, vocab, max_len=MAX_LEN):
    seqs = []
    for text in texts:
        tokens = word_tokenize(text)
        ids = [vocab.get(w, 1) for w in tokens[:max_len]]
        ids = ids + [0] * (max_len - len(ids))
        seqs.append(ids)
    return np.array(seqs)


Xtr_seq = encode_texts(X_train, vocab_w2v, MAX_LEN)
Xva_seq = encode_texts(X_val, vocab_w2v, MAX_LEN)

print("building embedding matrix")
embedding_matrix_w2v = np.zeros((len(vocab_w2v), 100))
for word, i in vocab_w2v.items():
    if word in w2v_model.wv:
        embedding_matrix_w2v[i] = w2v_model.wv[word]

print("✅ Word2Vec embeddings + dataset prepared")


# ==============================
# 5. Dataset Class
# ==============================
class EmailDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==============================
# 6. BiLSTM Model
# ==============================
class BiLSTMClassifier(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim=128, num_layers=2, dropout=0.3):
        super(BiLSTMClassifier, self).__init__()
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
        out = self.dropout(lstm_out[:, -1, :])  # last hidden state
        out = self.fc(out)
        return out


# ==============================
# 7. Training
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_w2v = BiLSTMClassifier(embedding_matrix_w2v).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model_w2v.parameters(), lr=1e-3)

train_loader = DataLoader(EmailDataset(Xtr_seq, y_train), batch_size=128, shuffle=True)
val_loader = DataLoader(EmailDataset(Xva_seq, y_val), batch_size=128)

best_val_loss = float("inf")
patience, patience_counter = 3, 0

for epoch in range(20):
    # Training
    model_w2v.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model_w2v(xb).squeeze()
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)

    # Validation
    model_w2v.eval()
    val_loss, y_true, y_pred = 0, [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model_w2v(xb).squeeze()
            loss = criterion(out, yb)
            val_loss += loss.item()
            y_true.extend(yb.cpu().numpy())
            y_pred.extend((torch.sigmoid(out).cpu().numpy() >= 0.5).astype(int))
    avg_val_loss = val_loss / len(val_loader)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(
        f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
        f"Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}"
    )

    # Early stopping + save artifacts
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(
            model_w2v.state_dict(), os.path.join(wordvec_path, "wordvec_bilstm.pt")
        )
        with open(os.path.join(wordvec_path, "vocab.pkl"), "wb") as f:
            pickle.dump(vocab_w2v, f)
        np.save(
            os.path.join(wordvec_path, "embedding_matrix.npy"), embedding_matrix_w2v
        )
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("⏹️ Early stopping triggered")
            break

print(f"✅ Best Word2Vec BiLSTM + vocab + embedding saved at {wordvec_path}")
