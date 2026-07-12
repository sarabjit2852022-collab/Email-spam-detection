# 📧 SpamGuard AI — Email Spam Detection Web App

A complete, production-ready **Email/SMS Spam Detection** web application built with **Flask**, **Scikit-learn**, and a modern, animated, responsive frontend.

The app uses a **TF-IDF Vectorizer** + **Multinomial Naive Bayes** classifier to detect whether a given email/message is **Spam** or **Ham (legitimate)**, and displays the prediction with a confidence percentage on a polished dashboard UI.

---

## ✨ Features

- 🧠 Machine Learning powered by TF-IDF + Multinomial Naive Bayes
- 🎯 Displays Accuracy, Precision, Recall & F1-Score on the dashboard
- 🔴 Spam results shown in **red**, 🟢 Ham results shown in **green**
- 📊 Confidence percentage with an animated progress bar
- ⚡ Real-time client-side input validation
- ⏳ Loading spinner animation while the model predicts
- 📱 Fully responsive, mobile-friendly design
- 🎨 Modern UI: gradient background, glassmorphism cards, hover animations
- 🚀 Simple two-step setup: train once, then run the app

---

## 🗂️ Project Structure

```
Email-Spam-Detection/
│
├── dataset/
│     └── spam.csv                 # Training dataset (label, message)
│
├── models/
│     ├── spam_model.pkl           # Trained Naive Bayes model (generated)
│     ├── vectorizer.pkl           # Fitted TF-IDF vectorizer (generated)
│     └── metrics.pkl              # Saved evaluation metrics (generated)
│
├── static/
│     ├── css/
│     │      └── style.css         # Dashboard styling & animations
│     ├── js/
│     │      └── script.js         # Frontend logic / AJAX calls
│     └── images/                  # (optional) image assets
│
├── templates/
│     └── index.html               # Main dashboard page
│
├── app.py                         # Flask application (backend + API)
├── train_model.py                 # ML training script
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── .gitignore
```

---

## ⚙️ Setup Instructions (VS Code / Local Machine)

### 1. Clone / Copy the project
Place the entire `Email-Spam-Detection/` folder anywhere on your machine and open it in VS Code.

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# Activate it:
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Machine Learning model
This reads `dataset/spam.csv`, trains the TF-IDF + Multinomial Naive Bayes model, prints Accuracy/Precision/Recall/F1, and saves the model + vectorizer into `models/`.

```bash
python train_model.py
```

You should see output similar to:
```
Accuracy  : 98.10%
Precision : 97.50%
Recall    : 94.20%
F1 Score  : 95.82%
...
Saved: models/spam_model.pkl
Saved: models/vectorizer.pkl
```

### 5. Run the Flask web application
```bash
python app.py
```

### 6. Open the app in your browser
```
http://127.0.0.1:5000
```

That's it! Paste any email/message text into the textbox, click **Predict**, and see whether it's classified as **Spam** or **Ham**, along with the confidence percentage.

---

## 🔬 How It Works

1. **Text Cleaning** — messages are lowercased, URLs/numbers/punctuation are stripped, and whitespace is normalized (identical logic is used at both training time and prediction time to avoid train/serve skew).
2. **TF-IDF Vectorization** — converts cleaned text into numerical feature vectors (unigrams + bigrams, top 5000 features, English stop-words removed).
3. **Multinomial Naive Bayes** — a probabilistic classifier well-suited for text/word-count-style features; trained on an 80/20 train/test split.
4. **Evaluation** — Accuracy, Precision, Recall, and F1-Score are computed on the held-out test set and persisted to `models/metrics.pkl` so the dashboard can display them.
5. **Flask API** — `POST /predict` accepts `{ "email_text": "..." }`, applies the same cleaning + vectorizer + model pipeline, and returns the predicted label and confidence.

---

## 🧪 Using Your Own Dataset

Replace `dataset/spam.csv` with your own dataset as long as it has two columns:

| Column | Description |
|--------|-------------|
| `v1`   | Label — `ham` or `spam` |
| `v2`   | The raw email/message text |

(This matches the format of the well-known UCI SMS Spam Collection dataset — if you have that file, you can drop it in directly.)

Then simply re-run:
```bash
python train_model.py
```
The model and vectorizer will be regenerated automatically.

---

## 🌐 API Reference

### `POST /predict`
**Request body:**
```json
{ "email_text": "Congratulations! You have won a free prize, click here to claim!" }
```

**Response:**
```json
{
  "success": true,
  "prediction": "Spam",
  "is_spam": true,
  "confidence": 98.42,
  "spam_probability": 98.42,
  "ham_probability": 1.58
}
```

### `GET /metrics`
Returns the saved model evaluation metrics as JSON.

---

## 🛠️ Tech Stack

| Layer      | Technology |
|------------|------------|
| Backend    | Python, Flask |
| ML         | Scikit-learn (TfidfVectorizer, MultinomialNB), Pandas, NumPy, Pickle |
| Frontend   | HTML5, CSS3, Vanilla JavaScript |
| Styling    | Custom CSS with gradients, glassmorphism, animations, responsive grid |

---

## 📌 Notes

- If you see `Model not loaded` errors in the app, make sure you ran `python train_model.py` **before** `python app.py`.
- The app runs in debug mode by default (`app.run(debug=True)`); set `debug=False` before deploying to production.
- For production deployment, use a WSGI server such as **Gunicorn** or **Waitress** instead of Flask's built-in dev server.

---

## 📄 License
Free to use for learning, portfolio, and personal projects.
