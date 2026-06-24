from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Chức năng: lỗi nghiệp vụ chuẩn hóa cho toàn pipeline. Đầu vào: code/message/detail/status_code. Đầu ra: exception mang error_code rõ ràng cho client."""

    status_code: int = 500

    def __init__(
        self,
        code: str,
        message: str,
        detail: dict[str, Any] | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.detail = detail or {}
        if status_code is not None:
            self.status_code = status_code

    def to_detail(self) -> dict[str, Any]:
        """Chức năng: chuẩn hóa lỗi cho response API. Đầu vào: self. Đầu ra: dict an toàn cho client (không traceback nội bộ)."""
        return {
            "error_code": self.code,
            "message": str(self),
            "context": self.detail,
        }
