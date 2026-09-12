# AI Usage Log — Assignment 02

## Tool used
ChatGPT

## Purpose
Used AI assistance to plan the 3-class classifier, Flask endpoint structure, input validation, and a lightweight feature-based explanation.

## Key prompts used
1. How can I extend a TF-IDF spam classifier from two classes to spam, ham and phishing?
2. How can Logistic Regression provide a confidence score and a simple explanation from TF-IDF features?
3. How can I create a minimal Flask POST /predict endpoint with JSON validation?

## What I accepted
- A 3-class label set: ham, spam, phishing.
- TF-IDF + Logistic Regression as a lightweight approach suitable for the time limit.
- predict_proba for confidence.
- Model coefficients combined with TF-IDF values for top contributing terms.

## What I changed / verified
- Fit the vectorizer only on the training split.
- Used stratified train/test splitting.
- Added input validation for missing, empty, and non-string messages.
- Added project-created phishing examples because the original dataset only contains ham/spam.
- Ran the training pipeline and checked held-out metrics.
- Tested representative prediction inputs using the saved model and vectorizer.

## What I rejected
I avoided a large deep-learning model and unnecessary explanation frameworks because they would add complexity without being required by the assignment.

## Verification
The trained model was evaluated on a held-out test set. The saved artifacts were also loaded and used for prediction examples. The Flask source was syntax-checked; Flask itself must be installed from requirements.txt before running the API.
