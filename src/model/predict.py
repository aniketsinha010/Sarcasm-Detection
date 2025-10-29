# src/model/predict.py
import logging
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.utility.preprocess import clean_text

logger = logging.getLogger(__name__)

def predict_sarcasm(text: str, model, tokenizer, config) -> dict:
    logger.info(f"Received prediction request for text: {text[:100]}")

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