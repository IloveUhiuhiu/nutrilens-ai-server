from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> None:
    # force=True: một số module (qua get_logger) có thể đã gắn handler vào
    # root logger trước khi configure_logging chạy (do import side-effect),
    # khiến basicConfig thường bị no-op và format/level cấu hình bị bỏ qua.
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        force=True,
    )
