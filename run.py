# diagnostics.py
import os, json, traceback
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

BASE = os.path.dirname(os.path.abspath(__file__))
DISTIL_DIR = os.path.join(BASE, "model", "distilbert")

print("DISTIL_DIR exists:", os.path.isdir(DISTIL_DIR))
if os.path.isdir(DISTIL_DIR):
    print("files:", os.listdir(DISTIL_DIR))
    # show config summary
    try:
        with open(os.path.join(DISTIL_DIR, "config.json")) as f:
            cfg = json.load(f)
        print("config keys:", list(cfg.keys()))
        print("num_labels:", cfg.get("num_labels"))
        print("architectures:", cfg.get("architectures"))
    except Exception as e:
        print("config.json read failed:", e)

    # Try loading tokenizer
    try:
        tok = DistilBertTokenizerFast.from_pretrained(DISTIL_DIR, local_files_only=True)
        print("Tokenizer loaded OK")
    except Exception as e:
        print("Tokenizer load failed:", e)
        traceback.print_exc()

    # Try loading model (weights); show helpful error
    try:
        m = DistilBertForSequenceClassification.from_pretrained(
            DISTIL_DIR, local_files_only=True
        )
        print("Model loaded OK")
    except Exception as e:
        print("Model load failed:", e)
        traceback.print_exc()
else:
    print("DISTIL_DIR not found")
