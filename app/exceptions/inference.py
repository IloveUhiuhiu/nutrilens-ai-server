from __future__ import annotations

from .base import AppError


class InferenceError(AppError):
    """Chức năng: lỗi không lường trước khi chạy model/inference (YOLO, Qwen3-VL, SAM3, DepthAnythingV2...). HTTP 500."""

    status_code = 500

    def __init__(self, service: str, message: str, detail: dict | None = None) -> None:
        super().__init__("inference_error", message, {"service": service, **(detail or {})})
