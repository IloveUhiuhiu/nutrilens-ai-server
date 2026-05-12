from __future__ import annotations

from .base import AppError

class ValidationError(AppError):
    def __init__(self, message: str, detail: dict | None = None) -> None:
        super().__init__("validation_error", message, detail)
