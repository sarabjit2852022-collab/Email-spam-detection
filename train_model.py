"""
train_model.py
---------------
Trains an Email/SMS Spam Detection model using:
    - TF-IDF Vectorizer  -> converts raw text into numeric feature vectors
    - Multinomial Naive Bayes -> classic, fast, and highly effective
      classifier for text classification problems such as spam detection.

Steps performed:
    1. Load dataset/spam.csv
    2. Clean & preprocess the text (lowercase, remove punctuation/numbers,
       remove extra whitespace)
    3. Split into train/test sets
    4. Vectorize text using TF-IDF
    5. Train MultinomialNB classifier
    6. Evaluate using Accuracy, Precision, Recall and F1-Score
    7. Save the trained model  -> models/spam_model.pkl
    8. Save the fitted vectorizer -> models/vectorizer.pkl

Run this file once before starting app.py:
    python train_model.py
"""

import re
import string
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ---------------------------------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------------------------------
DATASET_PATH = "dataset/spam.csv"

print("=" * 60)
print("EMAIL SPAM DETECTION - MODEL TRAINING")
print("=" * 60)
print(f"\n[1/7] Loading dataset from '{DATASET_PATH}' ...")

# The classic spam dataset (UCI SMS Spam Collection) is usually encoded in
# latin-1 and has columns v1 (label) and v2 (message text), sometimes with
# extra unnamed columns which we drop.
df = pd.read_csv(DATASET_PATH, encoding="latin-1")

# Keep only the first two relevant columns and rename them
df = df.iloc[:, :2]
df.columns = ["label", "message"]

# Drop any rows with missing values
df.dropna(inplace=True)

print(f"      Dataset loaded successfully with {len(df)} rows.")
print(f"      Label distribution:\n{df['label'].value_counts().to_string()}")

# ---------------------------------------------------------------------------
# 2. Text Cleaning / Preprocessing
# ---------------------------------------------------------------------------
print("\n[2/7] Cleaning and preprocessing text ...")


def clean_text(text):
    """Lowercase, remove punctuation/numbers/extra whitespace from text."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)   # remove URLs
    text = re.sub(r"\d+", " ", text)                        # remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()                # remove extra whitespace
    return text


df["clean_message"] = df["message"].apply(clean_text)

# Encode labels: ham -> 0, spam -> 1
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

print("      Text cleaning completed.")

# ---------------------------------------------------------------------------
# 3. Train / Test Split
# ---------------------------------------------------------------------------
print("\n[3/7] Splitting dataset into train and test sets (80/20) ...")

X = df["clean_message"]
y = df["label_num"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"      Train size: {len(X_train)}  |  Test size: {len(X_test)}")

# ---------------------------------------------------------------------------
# 4. TF-IDF Vectorization
# ---------------------------------------------------------------------------
print("\n[4/7] Vectorizing text using TF-IDF ...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000,
    ngram_range=(1, 2),  # unigrams + bigrams for better context capture
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"      TF-IDF vocabulary size: {len(vectorizer.vocabulary_)}")

# ---------------------------------------------------------------------------
# 5. Train Multinomial Naive Bayes Classifier
# ---------------------------------------------------------------------------
print("\n[5/7] Training Multinomial Naive Bayes classifier ...")

model = MultinomialNB(alpha=0.3)
model.fit(X_train_tfidf, y_train)

print("      Model training completed.")

# ---------------------------------------------------------------------------
# 6. Evaluate the Model
# ---------------------------------------------------------------------------
print("\n[6/7] Evaluating model performance on the test set ...")

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n" + "-" * 60)
print("MODEL PERFORMANCE METRICS")
print("-" * 60)
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")
print("-" * 60)
print("Confusion Matrix:")
print(f"                 Predicted Ham   Predicted Spam")
print(f"Actual Ham        {cm[0][0]:<15} {cm[0][1]}")
print(f"Actual Spam       {cm[1][0]:<15} {cm[1][1]}")
print("-" * 60)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

# ---------------------------------------------------------------------------
# 7. Save Model & Vectorizer as Pickle files
# ---------------------------------------------------------------------------
print("[7/7] Saving trained model and vectorizer to 'models/' folder ...")

with open("models/spam_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("models/vectorizer.pkl", "wb") as vec_file:
    pickle.dump(vectorizer, vec_file)

# Also persist the metrics so the Flask app can display them on the dashboard
metrics = {
    "accuracy": round(accuracy * 100, 2),
    "precision": round(precision * 100, 2),
    "recall": round(recall * 100, 2),
    "f1_score": round(f1 * 100, 2),
    "train_size": len(X_train),
    "test_size": len(X_test),
}
with open("models/metrics.pkl", "wb") as metrics_file:
    pickle.dump(metrics, metrics_file)

print("      Saved: models/spam_model.pkl")
print("      Saved: models/vectorizer.pkl")
print("      Saved: models/metrics.pkl")
print("\n" + "=" * 60)
print("TRAINING COMPLETE! You can now run the Flask app:  python app.py")
print("=" * 60)
