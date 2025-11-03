# app.py
import os
import pickle
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import logging
import gradio as gr

# 🚀 Initialize logging FIRST (like your training pipeline)
from src.logging.logger import logger  # Triggers basicConfig

# Use module logger
app_logger = logging.getLogger(__name__)

# Load model, tokenizer, config ONCE at startup
def load_resources():
    app_logger.info("Loading model, tokenizer, and config from artifacts")
    base_path = "artifacts"
    model = tf.keras.models.load_model(os.path.join(base_path, "sarcasm_detector.h5"))
    with open(os.path.join(base_path, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)
    with open(os.path.join(base_path, "preprocessing.json"), "r") as f:
        config = json.load(f)
    app_logger.info("All resources loaded successfully.")
    return model, tokenizer, config

# Global resources — loaded once
model, tokenizer, config = load_resources()
max_length = config["max_length"]

# Import after logging is ready (exactly as in Flask app)
from src.utility.preprocess import clean_text
from src.exceptions.exception import CustomException
from src.model.predict import predict_sarcasm


# =============== GRADIO PREDICTION FUNCTION ===============
def gradio_predict(headline: str):
    """
    Mirrors the logic of Flask's /predict POST route exactly.
    Returns (result_label_with_emoji, confidence_percent_str)
    """
    headline = headline.strip()
    if not headline:
        raise gr.Error("Please enter a headline.")
    if len(headline) > 500:
        raise gr.Error("Headline too long (max 500 characters).")

    try:
        result = predict_sarcasm(headline, model, tokenizer, config)
        emoji = "😏" if result["label"] == "Sarcastic" else "🙂"
        result_text = f"{emoji} {result['label']}"
        confidence_text = f"{result['confidence'] * 100:.2f}%"
        return result_text, confidence_text
    except Exception as e:
        app_logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise gr.Error("Oops! Something went wrong.")


# =============== GRADIO INTERFACE ===============
with gr.Blocks(title="Sarcasm Detector") as demo:
    gr.Markdown("# 🕵️ Sarcasm Detector")
    gr.Markdown("Enter a headline to detect sarcasm — just like the original web app.")

    headline_input = gr.Textbox(
        label="Headline",
        placeholder="Type your headline here...",
        max_lines=3
    )
    submit_btn = gr.Button("Detect Sarcasm")

    with gr.Row():
        result_output = gr.Textbox(label="Prediction", interactive=False)
        confidence_output = gr.Textbox(label="Confidence", interactive=False)

    submit_btn.click(
        fn=gradio_predict,
        inputs=headline_input,
        outputs=[result_output, confidence_output]
    )

    gr.Examples(
        examples=[
            ["I just love getting stuck in traffic!"],
            ["Wow, another Monday — my favorite day of the week."],
            ["Local man wins million-dollar lottery, still can't find his keys."]
        ],
        inputs=headline_input
    )

    gr.Markdown("Powered by TensorFlow • Deployed on Hugging Face Spaces")


# =============== LAUNCH ===============
if __name__ == "__main__":
    demo.launch()