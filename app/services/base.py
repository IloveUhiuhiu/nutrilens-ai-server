from __future__ import annotations

from app.utils.common import get_logger

class ServiceBase:
    logger = get_logger(__name__)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.logger = get_logger(cls.__module__)

    @classmethod
    def _log_info(cls, message: str) -> None:
        cls.logger.info(message)

    @classmethod
    def _log_error(cls, message: str) -> None:
        cls.logger.error(message)
