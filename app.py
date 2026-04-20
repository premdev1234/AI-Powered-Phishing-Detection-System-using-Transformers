# app.py
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import backend
import os
import traceback

# Serve frontend.html from current directory
app = Flask(__name__, template_folder=".")
CORS(app)  # allow local JS requests (safe for dev)


@app.route("/")
def home():
    return render_template("frontend.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        sender = data.get("sender", "") or ""
        subject = data.get("subject", "") or ""
        body = data.get("body", "") or ""
        weights = data.get("weights", None)

        if not subject and not body and not sender:
            return jsonify({"error": "Provide subject or body (or sender)"}), 400

        final_label, probs = backend.classify_email(subject, body, weights=weights)

        out = {
            "label": final_label,
            "prob_glove": float(probs.get("prob_glove", 0.0)),
            "prob_word2vec": float(probs.get("prob_word2vec", 0.0)),
            "prob_distilbert": float(probs.get("prob_distilbert", 0.0)),
            "weights_used": [float(x) for x in probs.get("weights_used", [0, 0, 0])],
            "weighted_prob": float(probs.get("weighted_prob", 0.0)),
            "explain_html": probs.get("explain_html", ""),
        }
        return jsonify(out)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[INFO] Starting server on 0.0.0.0:{port}")
    app.run(debug=True, host="0.0.0.0", port=port)
