# backend.py
import os
import re
import pickle
from typing import List, Tuple, Optional
from dotenv import load_dotenv
import numpy as np
import torch
import torch.nn as nn
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

load_dotenv()
# ------------------ CONFIG/PATHS ------------------
# Dynamically locate the model folder relative to backend.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# Subdirectories for each model
GLOVE_DIR = os.path.join(MODEL_DIR, "glove_bilstm")
W2V_DIR = os.path.join(MODEL_DIR, "word2vec_bilstm")
DISTIL_DIR = os.path.join(MODEL_DIR, "distilbert")

# File paths for GloVe model
GLOVE_VOCAB = os.path.join(GLOVE_DIR, "vocab.pkl")
GLOVE_PT = os.path.join(GLOVE_DIR, "glove_bilstm.pt")
GLOVE_TEXT = os.path.join(GLOVE_DIR, "glove.6B.100d.txt")

# File paths for Word2Vec model
W2V_VOCAB = os.path.join(W2V_DIR, "vocab.pkl")
W2V_EMB_NPY = os.path.join(W2V_DIR, "embedding_matrix.npy")
W2V_PT = os.path.join(W2V_DIR, "wordvec_bilstm.pt")

# DistilBERT model directory (contains config.json, tokenizer.json, etc.)
DISTIL_DIR = os.path.join(MODEL_DIR, "distilbert")

# default ensemble weights (glove, wordvec, distil)
ENSEMBLE_WEIGHTS = [0.2, 0.1, 0.7]

MAX_LEN = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESH = 0.7  # final threshold for deciding phishing


# ------------------ CLEAN ------------------
def clean_text(text: str) -> str:
    text = str(text).lower()
    # remove obvious full urls so models see content, but keep domain fragments and @
    text = re.sub(r"http\S+|www\S+", "", text)
    # allowable chars: letters, digits, common email/url chars and whitespace
    text = re.sub(r"[^a-z0-9\s@._:-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ------------------ MODEL CLASSES ------------------
class BiLSTM_Glove(nn.Module):
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


class BiLSTM_W2V(nn.Module):
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


# ------------------ LOAD ARTIFACTS (best-effort, non-fatal) ------------------
def load_vocab(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_embedding_matrix_from_glove(vocab, glove_txt_path, dim=100):
    emb_index = {}
    if not os.path.exists(glove_txt_path):
        return None
    with open(glove_txt_path, "r", encoding="utf8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            vec = np.asarray(parts[1:], dtype=np.float32)
            emb_index[word] = vec
    emb = np.zeros((len(vocab), dim), dtype=np.float32)
    for w, i in vocab.items():
        if w in emb_index:
            emb[i] = emb_index[w]
    return emb


glove_available = os.path.exists(GLOVE_PT) and os.path.exists(GLOVE_VOCAB)
glove_model = None
vocab_glove = None
if glove_available:
    try:
        vocab_glove = load_vocab(GLOVE_VOCAB)
        emb_mat = load_embedding_matrix_from_glove(vocab_glove, GLOVE_TEXT, dim=100)
        if emb_mat is None:
            emb_mat = np.random.normal(scale=0.01, size=(len(vocab_glove), 100)).astype(
                np.float32
            )
        glove_model = BiLSTM_Glove(emb_mat).to(DEVICE)
        glove_model.load_state_dict(torch.load(GLOVE_PT, map_location=DEVICE))
        glove_model.eval()
        print("[INFO] Loaded GloVe BiLSTM.")
    except Exception as e:
        print("[WARN] Could not load GloVe model:", e)
        glove_available = False
else:
    print("[WARN] GloVe model or vocab not found; skipping GloVe.")


w2v_available = (
    os.path.exists(W2V_PT) and os.path.exists(W2V_VOCAB) and os.path.exists(W2V_EMB_NPY)
)
w2v_model = None
vocab_w2v = None
if w2v_available:
    try:
        vocab_w2v = load_vocab(W2V_VOCAB)
        emb_w2v = np.load(W2V_EMB_NPY)
        w2v_model = BiLSTM_W2V(emb_w2v).to(DEVICE)
        w2v_model.load_state_dict(torch.load(W2V_PT, map_location=DEVICE))
        w2v_model.eval()
        print("[INFO] Loaded Word2Vec BiLSTM.")
    except Exception as e:
        print("[WARN] Could not load Word2Vec model:", e)
        w2v_available = False
else:
    print("[WARN] Word2Vec model/vocab/embedding not found; skipping Word2Vec.")


distil_available = os.path.isdir(DISTIL_DIR)
distil_tokenizer = None
distil_model = None
if distil_available:
    try:
        distil_tokenizer = DistilBertTokenizerFast.from_pretrained(
            DISTIL_DIR, local_files_only=True
        )
        distil_model = DistilBertForSequenceClassification.from_pretrained(
            DISTIL_DIR, local_files_only=True
        ).to(DEVICE)
        distil_model.eval()
        print("[INFO] Loaded DistilBERT.")
    except Exception as e:
        print("[WARN] Could not load DistilBERT:", e)
        distil_available = False
else:
    print("[WARN] DistilBERT folder not found; skipping DistilBERT.")


# ------------------ UTIL: normalize weights according to availability ----------
def normalized_weights(weights: List[float], avail: List[bool]) -> List[float]:
    w = np.array(weights, dtype=float)
    avail_mask = np.array(avail, dtype=float)
    w = w * avail_mask
    s = w.sum()
    if s == 0:
        avail_count = avail_mask.sum()
        if avail_count == 0:
            return [0.0, 0.0, 0.0]
        # equal weights for available models
        return list((avail_mask / avail_count).tolist())
    return list((w / s).tolist())


# ------------------ PREPROCESSING ------------------
def text_to_seq(text: str, vocab: dict, max_len=MAX_LEN) -> List[int]:
    toks = text.split()
    ids = [vocab.get(t, vocab.get("<OOV>", 1)) for t in toks[:max_len]]
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    return ids


# ------------------ PREDICTION HELPERS (safe shapes) ------------------
def predict_glove(texts: List[str]) -> np.ndarray:
    if not glove_available:
        return np.atleast_1d(np.zeros(len(texts), dtype=float))
    seqs = [text_to_seq(t, vocab_glove, MAX_LEN) for t in texts]
    X = torch.tensor(seqs, dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        out = glove_model(X).squeeze()
        probs = torch.sigmoid(out).cpu().numpy()
    return np.atleast_1d(probs).astype(float)


def predict_w2v(texts: List[str]) -> np.ndarray:
    if not w2v_available:
        return np.atleast_1d(np.zeros(len(texts), dtype=float))
    seqs = [text_to_seq(t, vocab_w2v, MAX_LEN) for t in texts]
    X = torch.tensor(seqs, dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        out = w2v_model(X).squeeze()
        probs = torch.sigmoid(out).cpu().numpy()
    return np.atleast_1d(probs).astype(float)


def predict_distil(texts: List[str]) -> np.ndarray:
    if not distil_available:
        return np.atleast_1d(np.zeros(len(texts), dtype=float))
    enc = distil_tokenizer(
        texts, truncation=True, padding=True, max_length=200, return_tensors="pt"
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        logits = distil_model(**enc).logits
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return np.atleast_1d(probs).astype(float)


# ------------------ CLASSIFICATION ------------------
def classify_email(
    subject: str, body: str, weights: Optional[List[float]] = None
) -> Tuple[str, dict]:
    """
    Returns (final_label_str, probs_dict)
    final_label_str is 'phishing' or 'not_phishing'
    probs_dict contains prob_glove, prob_word2vec, prob_distilbert, weights_used, weighted_prob
    """
    text = clean_text(f"{subject} {body}")
    texts = [text]

    # get probabilities (safe shape)
    try:
        p_glove = float(predict_glove(texts)[0]) if glove_available else 0.0
    except Exception:
        p_glove = 0.0
    try:
        p_w2v = float(predict_w2v(texts)[0]) if w2v_available else 0.0
    except Exception:
        p_w2v = 0.0
    try:
        p_distil = float(predict_distil(texts)[0]) if distil_available else 0.0
    except Exception:
        p_distil = 0.0

    avail = [glove_available, w2v_available, distil_available]
    base_weights = ENSEMBLE_WEIGHTS.copy()
    if weights is not None:
        try:
            w_arr = [float(x) for x in weights]
            if len(w_arr) != 3:
                w_arr = base_weights
        except Exception:
            w_arr = base_weights
    else:
        w_arr = base_weights

    weights_used = normalized_weights(w_arr, avail)
    weighted_prob = float(
        weights_used[0] * p_glove + weights_used[1] * p_w2v + weights_used[2] * p_distil
    )
    final_label = "phishing" if weighted_prob >= THRESH else "not_phishing"

    # small client-side explain: highlight suspicious tokens (this is light-weight)
    suspicious_tokens = []
    for token in text.split():
        t = token.lower()
        if any(
            k in t
            for k in [
                "click",
                "verify",
                "login",
                "password",
                "bank",
                "urgent",
                "free",
                "claim",
                "prize",
                "http",
                "https",
            ]
        ):
            suspicious_tokens.append(token)

    probs = {
        "prob_glove": float(p_glove),
        "prob_word2vec": float(p_w2v),
        "prob_distilbert": float(p_distil),
        "weights_used": [float(w) for w in weights_used],
        "weighted_prob": float(weighted_prob),
        "explain_html": (
            " ".join([f"<b>{t}</b>" for t in suspicious_tokens])
            if suspicious_tokens
            else ""
        ),
    }
    print(
        f"[INFO] probs -> glove: {p_glove:.4f}, w2v: {p_w2v:.4f}, distil: {p_distil:.4f} | weights={weights_used} => {weighted_prob:.4f}"
    )
    return final_label, probs
