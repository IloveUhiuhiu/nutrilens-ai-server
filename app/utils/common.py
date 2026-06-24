from __future__ import annotations

import logging

def get_logger(name: str) -> logging.Logger:
    # Không tự gắn handler/level riêng ở đây: để root logger (cấu hình bởi
    # app.core.logging.configure_logging) là nguồn format/level duy nhất,
    # tránh format không đồng nhất và LOG_LEVEL bị ghi đè.
    return logging.getLogger(name)
