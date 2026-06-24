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
    def _log_warning(cls, message: str) -> None:
        cls.logger.warning(message)

    @classmethod
    def _log_error(cls, message: str, exc_info: bool = True) -> None:
        """exc_info=True theo mặc định vì _log_error luôn được gọi trong except block;
        nếu không bật, traceback gốc sẽ không bao giờ được ghi lại ở server."""
        cls.logger.error(message, exc_info=exc_info)
