# src/logging/logger.py
import logging
import os
from datetime import datetime

# Create logs directory
logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)

# One log file per run — just like your training pipeline
LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

# Configure root logger ONCE
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] Line: %(lineno)d | Logger: %(name)s - [Module: %(module)s] - %(levelname)s - %(message)s",
    level=logging.INFO,
    force=True  # Python >=3.8: override any prior config
)

# Optional: export for convenience (but modules should use getLogger(__name__))
logger = logging.getLogger("sarcasm_detector")