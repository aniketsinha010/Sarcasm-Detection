import pickle
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.utility.preprocess import clean_text
import logging
from src.exceptions.exception import CustomException
import sys
import os

logger = logging.getLogger(__name__)

def load_resources():
    try:
        logger.info("Loading model, tokenizer, and config from artifacts...")

        base_path = os.path.join(os.path.dirname(__file__), "..", "..", "artifacts")
        model_path = os.path.join(base_path, "sarcasm_detector.h5")
        tokenizer_path = os.path.join(base_path, "tokenizer.pkl")
        config_path = os.path.join(base_path, "preprocessing.json")

        model = tf.keras.models.load_model(model_path)
        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)
        with open(config_path, "r") as f:
            config = json.load(f)

        logger.info("All resources loaded successfully.")
        return model, tokenizer, config

    except Exception as e:
        logger.info(f"Failed to load resources: {str(e)}")
        raise CustomException(e, sys)


def predict_sarcasm(text: str) -> dict:
    try:
        logger.info(f"Received prediction request for text: {text[:100]}")

        model, tokenizer, config = load_resources()
        cleaned = [clean_text(text)]
        seq = tokenizer.texts_to_sequences(cleaned)
        padded = pad_sequences(seq, maxlen=config["max_length"], padding='post')
        pred = model.predict(padded, verbose=0)

        confidence_score = float(pred[0][0])
        is_sarcastic = confidence_score > 0.5
        label = "Sarcastic" if is_sarcastic else "Not Sarcastic"
        confidence = confidence_score if is_sarcastic else 1 - confidence_score

        result = {"label": label, "confidence": round(confidence, 4)}
        logger.info(f"Prediction result: {result}")
        return result

    except Exception as e:
        logger.info(f"Prediction failed for text: {text[:50]}... Error: {str(e)}")
        raise CustomException(e, sys)