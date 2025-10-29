# src/utility/preprocess.py
import re
import contractions
import logging
from src.exceptions.exception import CustomException

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    try:
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        logger.info(f"Original text: {text[:50]}")
        text = text.lower()
        text = contractions.fix(text)
        text = re.sub(r'[^a-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        logger.info(f"Cleaned text: {text[:50]}")
        return text

    except Exception as e:
        logger.error(f"Text cleaning failed: {str(e)}")
        raise CustomException(str(e))