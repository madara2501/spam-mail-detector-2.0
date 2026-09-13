Runtime verification completed (Flask installed and actually run this time, not just syntax-checked).

- Training pipeline executed successfully; metrics.json regenerated automatically from this run.
- Saved model and vectorizer loaded successfully by app.py at startup.
- `GET /health` → 200, correct JSON body.
- `GET /` → was returning 500 (TemplateNotFound: index.html) because index.html sat in
  the project root instead of Flask's default `templates/` folder. Fixed by moving it
  to `templates/index.html`. Re-tested: now returns 200 with the full page.
- `POST /predict` tested with:
  - valid spam / phishing / ham messages → correct class, confidence, explanation
  - empty string message → 400, "'message' must be a non-empty string."
  - missing "message" key → 400, same message
  - non-string message (int) → 400, same message
  - malformed JSON body → 400, "Request body must be JSON."
  - no body at all → 400, "Request body must be JSON."
  - symbols-only / emoji-only / single-character messages (edge cases where TF-IDF
    features are sparse or empty) → handled gracefully, no crash, falls back to
    "The model did not find a small set of strongly contributing words."
- examples.json regenerated from these real /predict responses (previous version had
  correct predicted classes but incorrect confidence values and explanation words —
  it did not match the actual shipped model).
