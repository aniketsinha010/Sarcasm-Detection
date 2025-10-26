from flask import Flask, render_template, request, jsonify
import pickle, json, tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.utility.preprocess import clean_text
from src.exceptions.exception import CustomException
import sys

# Initialize Flask app
app = Flask(__name__)

# Load model, tokenizer, and config once
def load_resources():
    try:
        model = tf.keras.models.load_model("artifacts/sarcasm_detector.h5")
        with open("artifacts/tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        with open("artifacts/preprocessing.json", "r") as f:
            config = json.load(f)
        return model, tokenizer, config
    except Exception as e:
        raise CustomException(e, sys)

model, tokenizer, config = load_resources()
max_length = config["max_length"]

# Home route (renders web UI)
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        headline = request.form["headline"]

        if not headline.strip():
            return render_template("index.html", error="Please enter a headline first.")

        try:
            cleaned = [clean_text(headline)]
            seq = tokenizer.texts_to_sequences(cleaned)
            padded = pad_sequences(seq, maxlen=max_length, padding='post')
            pred = model.predict(padded)
            label = "Sarcastic" if pred[0][0] > 0.5 else "Not Sarcastic"
            emoji = "😏" if label == "Sarcastic" else "🙂"
            confidence = float(pred[0][0]) if label == "Sarcastic" else 1 - float(pred[0][0])

            return render_template(
                "index.html",
                headline=headline,
                result=label,
                emoji=emoji,
                confidence=f"{confidence * 100:.2f}%"
            )
        except Exception as e:
            raise CustomException(e, sys)
    return render_template("index.html")

# Optional: API route for programmatic access
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    headline = data.get("headline", "")
    cleaned = [clean_text(headline)]
    seq = tokenizer.texts_to_sequences(cleaned)
    padded = pad_sequences(seq, maxlen=max_length, padding='post')
    pred = model.predict(padded)
    label = "Sarcastic" if pred[0][0] > 0.5 else "Not Sarcastic"
    confidence = float(pred[0][0]) if pred[0][0] > 0.5 else 1 - float(pred[0][0])
    return jsonify({"label": label, "confidence": confidence})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


## http://127.0.0.1:5000