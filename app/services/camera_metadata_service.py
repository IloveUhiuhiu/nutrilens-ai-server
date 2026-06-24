from __future__ import annotations

import logging

from app.exceptions import ValidationError

logger = logging.getLogger(__name__)

DEFAULT_CAMERA_HEIGHT_CM = 40.0


def get_nested(data: dict, *keys, default=None):
    """Chức năng: lấy giá trị nested dict. Đầu vào: dict và keys. Đầu ra: value hoặc default."""
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


class CameraMetadataService:
    """Chức năng: chuẩn hóa thông số camera. Đầu vào: metadata client. Đầu ra: chiều cao fallback và intrinsics."""

    def derive_absolute_distance(self, camera_metadata: dict) -> tuple[bool, float | None]:
        """Chức năng: lấy khoảng cách tuyệt đối từ AR (CASE A). Đầu vào: metadata. Đầu ra: (has_absolute, distance_cm)."""
        has_absolute = bool(camera_metadata.get("has_absolute_depth"))
        raw = (
            camera_metadata.get("camera_to_object_distance")
            or camera_metadata.get("distance_cm")
            or get_nested(camera_metadata, "depth", "camera_to_object_distance")
        )
        try:
            distance = float(raw) if raw is not None else None
        except (TypeError, ValueError):
            distance = None
        if not has_absolute or distance is None or distance <= 0:
            return False, None
        return True, distance

    def derive_camera_height_cm(self, camera_metadata: dict, fallback: float | None = None) -> float:
        """Chức năng: lấy chiều cao camera cm. Đầu vào: metadata. Đầu ra: số cm."""
        raw_value = (
            camera_metadata.get("camera_height_cm")
            or camera_metadata.get("camera_height_ref")
            or get_nested(camera_metadata, "camera", "camera_height_cm")
        )
        if raw_value is not None:
            return float(raw_value)

        raw_mm = camera_metadata.get("camera_height_mm") or get_nested(camera_metadata, "camera", "camera_height_mm")
        if raw_mm is not None:
            return float(raw_mm) / 10.0

        if fallback is not None:
            return float(fallback)
        return DEFAULT_CAMERA_HEIGHT_CM

    def derive_intrinsics(self, camera_metadata: dict) -> dict:
        """Chức năng: lấy intrinsics camera từ metadata. Đầu vào: metadata. Đầu ra: fx/fy/cx/cy."""
        intrinsics = camera_metadata.get("intrinsics") or get_nested(camera_metadata, "camera", "intrinsics") or {}
        fx = intrinsics.get("fx") or camera_metadata.get("fx")
        fy = intrinsics.get("fy") or camera_metadata.get("fy")
        cx = intrinsics.get("cx") or camera_metadata.get("cx")
        cy = intrinsics.get("cy") or camera_metadata.get("cy")
        if not fx or not fy:
            logger.warning("step=request_validation failed: camera_metadata is missing intrinsics fx/fy")
            raise ValidationError("camera intrinsics fx/fy is required", {"field": "camera_metadata"})
        return {
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx) if cx is not None else None,
            "cy": float(cy) if cy is not None else None,
        }
