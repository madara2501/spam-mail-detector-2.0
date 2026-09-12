from flask import Flask, request, jsonify
import pickle
import numpy as np
import re
import os
import time
import logging
from pathlib import Path
from datetime import datetime, timezone


# ============================================================
# APP CONFIGURATION
# ============================================================

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = BASE_DIR / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"


with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)


logger.info("Model and vectorizer loaded successfully.")


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"http\S+|www\S+|bit\.ly\S+",
        " link ",
        text
    )

    text = re.sub(
        r"\d+",
        " number ",
        text
    )

    text = re.sub(
        r"[^a-zA-Z ]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# EXPLANATION
# ============================================================

def explain_prediction(
    transformed_message,
    predicted_class
):

    feature_names = np.array(
        vectorizer.get_feature_names_out()
    )

    row = transformed_message.toarray()[0]

    class_index = list(
        model.classes_
    ).index(predicted_class)

    contributions = (
        row * model.coef_[class_index]
    )

    top_indices = np.argsort(
        contributions
    )[::-1]

    words = [
        feature_names[i]
        for i in top_indices
        if contributions[i] > 0
        and row[i] > 0
    ][:3]

    if not words:

        return (
            "The model did not find a small set "
            "of strongly contributing words."
        )

    return (
        "The prediction was influenced mainly by: "
        + ", ".join(words)
        + "."
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return jsonify({
        "status": "healthy",
        "service": "spam-detector-2.0"
    })


# ============================================================
# PREDICTION API
# ============================================================

@app.post("/predict")
def predict():

    start_time = time.perf_counter()

    timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    try:

        data = request.get_json(
            silent=True
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if not isinstance(data, dict):

            logger.warning(
                "%s | Invalid JSON request",
                timestamp
            )

            return jsonify({
                "error": "Request body must be JSON."
            }), 400


        message = data.get("message")


        if (
            not isinstance(message, str)
            or not message.strip()
        ):

            logger.warning(
                "%s | Empty or invalid message",
                timestamp
            )

            return jsonify({
                "error": "'message' must be a non-empty string."
            }), 400


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        transformed = vectorizer.transform(
            [clean_text(message)]
        )

        prediction = model.predict(
            transformed
        )[0]

        probabilities = model.predict_proba(
            transformed
        )[0]

        confidence = float(
            np.max(probabilities)
        )

        explanation = explain_prediction(
            transformed,
            prediction
        )


        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        response = jsonify({

            "class": str(prediction),

            "confidence": round(
                confidence,
                4
            ),

            "explanation": explanation

        })


        # ----------------------------------------------------
        # REQUEST LOGGING
        # ----------------------------------------------------

        elapsed_ms = (
            time.perf_counter()
            - start_time
        ) * 1000

        logger.info(
            "%s | POST /predict | "
            "class=%s | response_time=%.2f ms",
            timestamp,
            prediction,
            elapsed_ms
        )


        return response


    except Exception as e:

        elapsed_ms = (
            time.perf_counter()
            - start_time
        ) * 1000

        logger.exception(
            "%s | POST /predict | "
            "error=%s | response_time=%.2f ms",
            timestamp,
            str(e),
            elapsed_ms
        )

        return jsonify({
            "error": "Internal server error."
        }), 500


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def home():

    return jsonify({

        "service": "Spam Detector 2.0",

        "status": "running",

        "endpoints": {
            "health": "GET /health",
            "predict": "POST /predict"
        }

    })


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )