from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from app.exceptions import ValidationError

logger = logging.getLogger(__name__)


@dataclass
class AnalyzeNutritionInput:
    """Chức năng: dữ liệu request đã parse cho pipeline. Đầu vào: multipart. Đầu ra: object nội bộ."""

    image_bytes: bytes
    image_filename: str
    job_id: str
    camera_metadata: dict
    depth_metadata: dict
    depth_bytes: bytes | None

    @property
    def dish_id(self) -> str:
        """Chức năng: lấy id debug/storage. Đầu vào: job_id hoặc filename. Đầu ra: id."""
        from pathlib import Path

        return self.job_id or Path(self.image_filename or "inference").stem


def parse_json_form(value: str | dict | None) -> dict:
    """Chức năng: parse JSON từ form multipart. Đầu vào: string/dict/None. Đầu ra: dict."""
    if value in (None, ""):
        return {}
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        logger.warning(f"step=request_validation failed: invalid JSON metadata - {exc}")
        raise ValidationError("Invalid JSON metadata", {"value": value}) from exc
    if not isinstance(parsed, dict):
        logger.warning("step=request_validation failed: metadata JSON is not an object")
        raise ValidationError("Metadata must be a JSON object", {"value": value})
    return parsed
