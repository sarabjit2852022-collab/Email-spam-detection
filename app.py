"""
app.py
------
Flask backend for the Email Spam Detection web application.

Responsibilities:
    1. Serve the frontend (templates/index.html + static assets)
    2. Load the pre-trained TF-IDF vectorizer and Multinomial Naive Bayes
       model (created by train_model.py) from the models/ folder
    3. Expose a POST /predict API endpoint that:
         - accepts JSON: { "email_text": "..." }
         - cleans the text the same way it was cleaned during training
         - vectorizes it using the saved TF-IDF vectorizer
         - predicts Spam / Ham using the saved model
         - returns the prediction label + confidence percentage as JSON
    4. Expose a GET /metrics API endpoint that returns the saved training
       metrics (Accuracy, Precision, Recall, F1-Score) for the dashboard

Run with:
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

import re
import string
import pickle
import os

from flask import Flask, render_template, request, jsonify

# ---------------------------------------------------------------------------
# Flask app initialization
# ---------------------------------------------------------------------------
app = Flask(__name__)

MODEL_PATH = os.path.join("models", "spam_model.pkl")
VECTORIZER_PATH = os.path.join("models", "vectorizer.pkl")
METRICS_PATH = os.path.join("models", "metrics.pkl")

# ---------------------------------------------------------------------------
# Load trained model, vectorizer & metrics at startup
# ---------------------------------------------------------------------------
model = None
vectorizer = None
metrics = {"accuracy": "N/A", "precision": "N/A", "recall": "N/A", "f1_score": "N/A"}

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    print("[INFO] Model and vectorizer loaded successfully.")
except FileNotFoundError:
    print("[WARNING] Model/vectorizer not found. Please run 'python train_model.py' first.")

try:
    with open(METRICS_PATH, "rb") as f:
        metrics = pickle.load(f)
    print("[INFO] Training metrics loaded successfully.")
except FileNotFoundError:
    print("[WARNING] Metrics file not found. Run 'python train_model.py' to generate it.")


# ---------------------------------------------------------------------------
# Text cleaning function (MUST match the preprocessing used in train_model.py)
# ---------------------------------------------------------------------------
def clean_text(text):
    """Lowercase, remove URLs/numbers/punctuation/extra whitespace."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    """Render the main dashboard page."""
    return render_template("index.html", metrics=metrics)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict whether the given email/message text is Spam or Ham.
    Expects JSON body: { "email_text": "some text" }
    Returns JSON: { "prediction": "Spam"/"Ham", "confidence": 97.34, "success": true }
    """
    if model is None or vectorizer is None:
        return jsonify({
            "success": False,
            "error": "Model not loaded. Please run 'python train_model.py' first."
        }), 500

    data = request.get_json(silent=True) or {}
    email_text = data.get("email_text", "").strip()

    # Server-side validation
    if not email_text:
        return jsonify({"success": False, "error": "Email text cannot be empty."}), 400

    if len(email_text) < 3:
        return jsonify({"success": False, "error": "Please enter a longer message to analyze."}), 400

    # Clean & vectorize
    cleaned = clean_text(email_text)
    vectorized = vectorizer.transform([cleaned])

    # Predict class and probability
    prediction = model.predict(vectorized)[0]
    probabilities = model.predict_proba(vectorized)[0]  # [P(ham), P(spam)]

    is_spam = bool(prediction == 1)
    confidence = probabilities[1] if is_spam else probabilities[0]
    confidence_pct = round(float(confidence) * 100, 2)

    result = {
        "success": True,
        "prediction": "Spam" if is_spam else "Ham",
        "is_spam": is_spam,
        "confidence": confidence_pct,
        "spam_probability": round(float(probabilities[1]) * 100, 2),
        "ham_probability": round(float(probabilities[0]) * 100, 2),
    }
    return jsonify(result)


@app.route("/metrics", methods=["GET"])
def get_metrics():
    """Return the saved model training metrics as JSON."""
    return jsonify(metrics)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # debug=True enables auto-reload during development; set to False in production
    app.run(debug=True, host="0.0.0.0", port=5000)
