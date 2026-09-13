import pandas as pd
import pickle
import re
import json
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# ============================================================
# 1. FIND THE ASSIGNMENT-02 FOLDER AUTOMATICALLY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

print("Working folder:", BASE_DIR)

# ============================================================
# 2. LOAD DATASET
# ============================================================

csv_path = BASE_DIR / "spam.csv"

if not csv_path.exists():
    raise FileNotFoundError(
        f"spam.csv was not found here:\n{csv_path}\n\n"
        "Make sure spam.csv is inside the Assignment-02 folder."
    )

print("Loading dataset:", csv_path)

df = pd.read_csv(
    csv_path,
    encoding="latin-1"
)[["v1", "v2"]]

df.columns = ["label", "message"]

df["message"] = df["message"].fillna("").astype(str)

print("\nOriginal dataset:")
print(df["label"].value_counts())


# ============================================================
# 3. ADD PHISHING EXAMPLES
# ============================================================

phishing_examples = [
    (
        "phishing",
        "Your bank account will be suspended today. Verify your login immediately."
    ),
    (
        "phishing",
        "Your UPI account is locked. Click the verification link to restore access."
    ),
    (
        "phishing",
        "Your KYC has expired. Enter your PAN and OTP on the verification page."
    ),
    (
        "phishing",
        "Your debit card is blocked. Confirm your card number and CVV to reactivate it."
    ),
    (
        "phishing",
        "Security alert: verify your password and one-time code to keep your account active."
    ),
    (
        "phishing",
        "Your email storage is full. Login to prevent your mailbox from being disabled."
    ),
    (
        "phishing",
        "Your account will be deleted in 24 hours. Confirm your password now."
    ),
    (
        "phishing",
        "A new device signed into your account. Verify your password now."
    ),
    (
        "phishing",
        "Your mobile SIM will be deactivated. Verify your identity using the link."
    ),
    (
        "phishing",
        "Account verification required: enter your username, password and OTP."
    ),
    (
        "phishing",
        "Your parcel is on hold. Click the verification link and pay the delivery fee."
    ),
    (
        "phishing",
        "Your payment failed. Sign in and update your card details."
    ),
    (
        "phishing",
        "Your tax refund is ready. Enter your bank information to receive the refund."
    ),
    (
        "phishing",
        "Your credit card rewards are expiring. Confirm your card details."
    ),
    (
        "phishing",
        "Your shopping account has suspicious activity. Sign in to verify your identity."
    ),
    (
        "phishing",
        "Your courier delivery failed. Confirm your address and pay the rescheduling fee."
    ),
    (
        "phishing",
        "Your bank needs identity verification. Reply with your OTP to complete the process."
    ),
    (
        "phishing",
        "Urgent: confirm your banking information or your access will be disabled."
    ),
    (
        "phishing",
        "Your electricity bill is overdue. Pay immediately through this verification link."
    ),
    (
        "phishing",
        "Your insurance policy is suspended. Confirm personal information to reactivate it."
    )
]

phishing_df = pd.DataFrame(
    phishing_examples,
    columns=["label", "message"]
)

df = pd.concat(
    [df, phishing_df],
    ignore_index=True
)

print("\nDataset after adding phishing examples:")
print(df["label"].value_counts())


# ============================================================
# 4. TEXT CLEANING
# ============================================================

def clean_text(text):

    text = text.lower()

    # Replace URLs
    text = re.sub(
        r"http\S+|www\S+|bit\.ly\S+",
        " link ",
        text
    )

    # Replace numbers
    text = re.sub(
        r"\d+",
        " number ",
        text
    )

    # Remove special characters
    text = re.sub(
        r"[^a-zA-Z ]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


df["message"] = df["message"].apply(clean_text)


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

X = df["message"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 6. TF-IDF
# ============================================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=10000,
    sublinear_tf=True
)

# IMPORTANT:
# Fit ONLY on training data.
X_train_vec = vectorizer.fit_transform(X_train)

# Only transform test data.
X_test_vec = vectorizer.transform(X_test)

print("\nTF-IDF training shape:", X_train_vec.shape)
print("TF-IDF testing shape :", X_test_vec.shape)


# ============================================================
# 7. TRAIN MODEL
# ============================================================

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    C=2.0,
    random_state=42
)

model.fit(
    X_train_vec,
    y_train
)

print("\nModel training completed!")


# ============================================================
# 8. PREDICTION
# ============================================================

y_pred = model.predict(X_test_vec)


# ============================================================
# 9. EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision, recall, f1, _ = precision_recall_fscore_support(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

print("\n======================================")
print("MODEL PERFORMANCE")
print("======================================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


print("\n======================================")
print("CLASSIFICATION REPORT")
print("======================================")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

labels = [
    "ham",
    "spam",
    "phishing"
]

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

print("\n======================================")
print("CONFUSION MATRIX")
print("======================================")

print("Labels:", labels)
print(cm)


# ============================================================
# 11. SAVE METRICS
# ============================================================
# Saved automatically from THIS run so metrics.json can never drift
# out of sync with the model actually being shipped.

metrics_path = BASE_DIR / "metrics.json"

metrics = {
    "accuracy": round(float(accuracy), 4),
    "weighted_precision": round(float(precision), 4),
    "weighted_recall": round(float(recall), 4),
    "weighted_f1": round(float(f1), 4),
    "test_size": 0.2,
    "random_state": 42,
    "classes": sorted(y.unique().tolist()),
    "confusion_matrix_labels": labels,
    "confusion_matrix": cm.tolist()
}

with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

print("Metrics saved:", metrics_path)


# ============================================================
# 12. SAVE MODEL
# ============================================================

model_path = BASE_DIR / "model.pkl"
vectorizer_path = BASE_DIR / "vectorizer.pkl"

with open(model_path, "wb") as file:
    pickle.dump(
        model,
        file
    )

with open(vectorizer_path, "wb") as file:
    pickle.dump(
        vectorizer,
        file
    )

print("\n======================================")
print("FILES SAVED")
print("======================================")

print("Model     :", model_path)
print("Vectorizer:", vectorizer_path)

print("\nTraining completed successfully!")
