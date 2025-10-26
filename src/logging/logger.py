# src/logging/logger.py
import logging
import os
import sys
from datetime import datetime

# Create logs directory
logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Generate unique log filename
LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="[ %(asctime)s ] Line: %(lineno)d | Logger: %(name)s - [Module: %(module)s] - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler(sys.stdout)  # Optional: also print to terminal
    ],
    force=True  # Ensures config is applied even if logging was used before
)