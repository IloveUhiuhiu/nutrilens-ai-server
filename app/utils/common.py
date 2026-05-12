from __future__ import annotations

import logging

def get_logger(name: str) -> logging.Logger:
    if not logging.getLogger().handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(filename)s - %(funcName)s - %(message)s")
        handler.setFormatter(formatter)
        logging.getLogger().addHandler(handler)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger
