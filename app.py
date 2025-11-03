# app.py
import os
import pickle
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, render_template, request
import logging

from src.logging.logger import logger
from src.utility.preprocess import clean_text
from src.exceptions.exception import CustomException
from src.model.predict import predict_sarcasm

app_logger = logging.getLogger(__name__)
app = Flask(__name__)


def load_resources():
    app_logger.info("Loading model, tokenizer, and config from artifacts...")
    base_path = "artifacts"

    model = tf.keras.models.load_model(os.path.join(base_path, "sarcasm_detector.h5"))
    with open(os.path.join(base_path, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)
    with open(os.path.join(base_path, "preprocessing.json"), "r") as f:
        config = json.load(f)

    app_logger.info("All resources loaded successfully.")
    return model, tokenizer, config

model, tokenizer, config = load_resources()
max_length = config["max_length"]


@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["GET", "POST"])
def predict_page():
    if request.method == "POST":
        headline = request.form.get("headline", "").strip()
        if not headline:
            return render_template("predict.html", error="Please enter a headline.")
        if len(headline) > 500:
            return render_template("predict.html", error="Headline too long (max 500 characters).")

        try:
            result = predict_sarcasm(headline, model, tokenizer, config)
            emoji = "😏" if result["label"] == "Sarcastic" else "🙂"
            return render_template(
                "predict.html",
                headline=headline,
                result=result["label"],
                emoji=emoji,
                confidence=f"{result['confidence'] * 100:.2f}%"
            )
        except Exception as e:
            app_logger.error(f"Prediction failed: {str(e)}", exc_info=True)
            return render_template("predict.html", error="Oops! Something went wrong.")

    return render_template("predict.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
