from __future__ import annotations

from .base import AppError

class InferenceError(AppError):
    def __init__(self, service: str, message: str, detail: dict | None = None) -> None:
        super().__init__("inference_error", message, {"service": service, **(detail or {})})
