# Spam Detector 2.0 — Assignment 02

## Objective
Extend a basic spam/ham text classifier to detect **Phishing**, return a confidence score, and provide a short human-readable explanation through a Flask `/predict` endpoint.

## Implementation
- 3 classes: `ham`, `spam`, `phishing`
- TF-IDF unigrams + bigrams
- Logistic Regression
- Stratified 80/20 train-test split
- Training-only TF-IDF fitting to avoid leakage
- Confidence via `predict_proba`
- Top contributing TF-IDF terms for explanation
- `POST /predict`
- `GET /health`
- Basic JSON/input validation

## Dataset
Original SMS Spam Collection data was retained. Because the supplied dataset contains only ham/spam labels, 30 project-created phishing examples were added for the new class. This is explicitly documented rather than presented as original labeled data.

Class counts:
label
ham         4830
spam         758
phishing      20

## Held-out test metrics
Accuracy: **0.9750**
Weighted Precision: **0.9769**
Weighted Recall: **0.9750**
Weighted F1: **0.9755**

Confusion matrix labels: `[ham, spam, phishing]`

```text
[[947  19   0]
 [  3 147   2]
 [  0   1   5]]
```

## Run locally
```bash
pip install -r requirements.txt
python train_model.py
python app.py
```

The API runs on port 5000.

### Health
`GET /health`

Expected:
{
  "service": "spam-detector-2.0",
  "status": "healthy"
}

### Predict
`POST /predict`

Request:
```json
{"message":"Your account is locked. Verify your password and OTP immediately."}
```

Response shape:
```json
{
  "class":"phishing",
  "confidence":0.80,
  "explanation":"The prediction was influenced mainly by: verify, password, account."
}
```

See `examples.json` for actual outputs from the trained model.

## Scope decision
The requested 3-class version was implemented while keeping the API intentionally small. No unnecessary authentication, database, frontend framework, or deep-learning model was added because the assignment prioritizes a working, clearly scoped feature within the time limit.
