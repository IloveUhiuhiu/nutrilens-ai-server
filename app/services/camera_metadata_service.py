from __future__ import annotations

from app.exceptions import ValidationError


def get_nested(data: dict, *keys, default=None):
    """Chức năng: lấy giá trị nested dict. Đầu vào: dict và keys. Đầu ra: value hoặc default."""
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


class CameraMetadataService:
    """Chức năng: chuẩn hóa thông số camera. Đầu vào: metadata client. Đầu ra: camera_height_ref và pixel_area_ref."""

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
        raise ValidationError("camera_height_cm or camera_height_mm is required", {"field": "camera_metadata"})

    def derive_pixel_area_cm2(self, camera_metadata: dict, camera_height_cm: float, fallback: float | None = None) -> float:
        """Chức năng: tính/lấy diện tích pixel cm2. Đầu vào: metadata và chiều cao camera. Đầu ra: cm2."""
        pixel_area_cm2 = (
            camera_metadata.get("pixel_area_cm2")
            or get_nested(camera_metadata, "pixel_size", "pixel_area_cm2")
            or get_nested(camera_metadata, "camera", "pixel_size", "pixel_area_cm2")
        )
        if pixel_area_cm2 is not None:
            return float(pixel_area_cm2)

        pixel_area_mm2 = (
            camera_metadata.get("pixel_area_mm2")
            or get_nested(camera_metadata, "pixel_size", "pixel_area_mm2")
            or get_nested(camera_metadata, "camera", "pixel_size", "pixel_area_mm2")
        )
        if pixel_area_mm2 is not None:
            return float(pixel_area_mm2) / 100.0

        intrinsics = camera_metadata.get("intrinsics") or get_nested(camera_metadata, "camera", "intrinsics") or {}
        fx = intrinsics.get("fx")
        fy = intrinsics.get("fy")
        if fx and fy:
            return (camera_height_cm / float(fx)) * (camera_height_cm / float(fy))

        if fallback is not None:
            return float(fallback)
        raise ValidationError("pixel_area_cm2 or camera intrinsics fx/fy is required", {"field": "camera_metadata"})
