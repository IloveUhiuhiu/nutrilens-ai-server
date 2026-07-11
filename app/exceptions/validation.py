from __future__ import annotations

from .base import AppError


class ValidationError(AppError):
    """Chức năng: lỗi do input/request không hợp lệ (request validation, depth map client, camera metadata...). HTTP 400."""

    status_code = 400

    def __init__(self, message: str, detail: dict | None = None) -> None:
        super().__init__("validation_error", message, detail)
