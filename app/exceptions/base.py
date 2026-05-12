from __future__ import annotations

from typing import Any

class AppError(Exception):
    def __init__(self, code: str, message: str, detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.detail = detail or {}

    def to_detail(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": str(self),
            "detail": self.detail,
        }
