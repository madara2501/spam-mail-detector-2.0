# Spam Detector 2.0 — Assignment 02

## Novum Labs AI – Teaching & Technical Interview Assignment

A lightweight NLP-based spam detection system extended from a basic
Spam/Ham classifier to a **3-class text classifier** that detects:

- Ham
- Spam
- Phishing

The project uses TF-IDF features with Logistic Regression and exposes the
trained model through a minimal Flask REST API.

---

## 1. Objective

The objective of this assignment was to extend an existing spam/ham text
classifier by adding one meaningful capability.

The implemented system:

- Detects `ham`, `spam`, and `phishing` messages.
- Returns a confidence score for each prediction.
- Provides a short human-readable explanation based on important words.
- Exposes the model through a Flask `/predict` API.
- Provides a `/health` endpoint.
- Performs basic JSON and input validation.

---

## 2. Technology Stack

- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- TF-IDF
- Logistic Regression
- REST API
- JSON

---

## 3. Project Structure

```text
Assignment-02/
│
├── app.py
├── train_model.py
├── model.pkl
├── vectorizer.pkl
├── spam.csv
├── requirements.txt
├── examples.json
├── metrics.json
├── API_TEST_RESULTS.md
├── AI_Usage_Log.md
└── README.md

#Training Process

The model uses an 80/20 stratified train-test split.

The TF-IDF vectorizer is fitted only on the training data to avoid
data leakage.

The general training workflow is:

Dataset
   ↓
Text Cleaning
   ↓
Train/Test Split
   ↓
TF-IDF Vectorization
   ↓
Logistic Regression
   ↓
Model Evaluation
   ↓
Save Model + Vectorizer

#Model Performance

The model was evaluated on a held-out test set.

Results
Metric	Score
Accuracy	0.9778
Weighted Precision	0.9796
Weighted Recall	0.9778
Weighted F1	0.9783

#Design and Scope Decisions

The assignment had a limited time budget, so the implementation focused
on the required functionality rather than unnecessary complexity.

Implemented
3-class classification
TF-IDF features
Logistic Regression
Confidence score
Prediction explanation
Flask /predict
Flask /health
Input validation
Model evaluation
Example API requests/responses
Deliberately Not Added
Deep-learning model
Database
Authentication system
Large explainability frameworks
Complex frontend framework

These were not required for the assignment and would increase complexity
without directly improving the requested deliverables.


# Limitations
The phishing class is based on 30 project-created examples because the
original dataset did not contain phishing labels.
The model is intended as an assignment-level text classification system,
not a production-grade email security solution.
Confidence values represent the model's predicted class probability and
should not be interpreted as a guarantee that a message is malicious.
Real-world phishing detection would require a larger and more diverse
phishing dataset.

#AI Usage

ChatGPT was used as an assistance tool during development for:

Planning the 3-class classifier
Designing the Flask API
Input validation
Confidence-score implementation
Prediction explanation approach
Debugging and documentation

AI-generated suggestions were reviewed and modified where necessary.

The implementation was tested by running the training pipeline, checking
held-out evaluation metrics, loading the saved artifacts, and testing
representative API inputs.

Detailed information is available in:

AI_Usage_Log.md
