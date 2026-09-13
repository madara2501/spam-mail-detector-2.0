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

## Follow-up debugging pass (AI-assisted)
Used Claude to actually install Flask and run the full app end-to-end, rather than relying on a syntax check.

### What this caught
- `GET /` returned a 500 error: `render_template("index.html")` needs Flask's
  default `templates/` folder, but `index.html` was at the project root. Reproduced
  with the server log showing `TemplateNotFound: index.html`.
- `examples.json` and `metrics.json` did not match the actual output of the shipped
  `model.pkl`/`vectorizer.pkl` — predicted classes were right but confidence values
  and explanation words were wrong, meaning they were not generated from a real run
  of this exact model.

### What I accepted
- Moving `index.html` into `templates/index.html` as the fix.
- Adding automatic `metrics.json` export inside `train_model.py` so metrics can't
  drift from the shipped model again.
- Adding `generate_examples.py`, which calls the real running `/predict` endpoint
  to produce `examples.json`, instead of hand-writing example outputs.

### What I verified myself
- Re-ran `train_model.py` and confirmed the printed classification report matches
  the new `metrics.json` exactly.
- Started the Flask app and hit `/`, `/health`, and `/predict` directly (valid
  inputs, empty/missing/non-string message, malformed JSON, no body, and
  symbols/emoji-only edge cases) to confirm the fix and that nothing else was
  broken by it.
