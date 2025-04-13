import logging
import sys
from datetime import datetime

class CustomFormatter(logging.Formatter):
    def __init__(self):
        super().__init__()
        self.template = "[%(asctime)s] %(levelname)s - %(name)s - %(module)s:%(lineno)s - %(message)s"

    def format(self, record):
        return self.template % record.__dict__

class Logger:
    def __init__(self, name: str, level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self._setup_handlers()

    def _setup_handlers(self):
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_formatter = CustomFormatter()
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        file_handler = logging.FileHandler(f"{self.logger.name}.log")
        file_handler.setFormatter(console_formatter)
        self.logger.addHandler(file_handler)

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg, exc_info=False):
        self.logger.error(msg, exc_info=exc_info)

    def critical(self, msg):
        self.logger.critical(msg)


if __name__ == "__main__":
    logger = Logger(__name__)
    logger.info("Logger initialized.")
    logger.warning("Warning: Potential issue detected.")
    logger.error("Error occurred during execution.", exc_info=True)
