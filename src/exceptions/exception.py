# src/exceptions/exception.py
import sys
import logging

logger = logging.getLogger(__name__)

def error_message_detail(error):
    _, _, exc_tb = sys.exc_info()
    if exc_tb is None:
        return str(error)
    file_name = exc_tb.tb_frame.f_code.co_filename
    return f"Error occurred in script [{file_name}] line [{exc_tb.tb_lineno}]: {str(error)}"

class CustomException(Exception):
    def __init__(self, error_message):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message)
        logger.error(self.error_message)

    def __str__(self):
        return self.error_message