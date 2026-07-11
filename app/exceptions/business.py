from __future__ import annotations

from .base import AppError


class NoFoodDetectedError(AppError):
    """Chức năng: ảnh không chứa món ăn nào được YOLO food detect ra. HTTP 422 - dừng pipeline ngay tại bước detection."""

    status_code = 422

    def __init__(
        self,
        message: str = "No food detected in image.",
        detail: dict | None = None,
    ) -> None:
        super().__init__("no_food_detected", message, detail)


class NoPlateDetectedError(AppError):
    """Chức năng: ảnh không chứa đĩa/vật chứa nào được YOLO plate detect ra. HTTP 422 - dừng pipeline ngay tại bước detection."""

    status_code = 422

    def __init__(
        self,
        message: str = "No plate or container detected in image.",
        detail: dict | None = None,
    ) -> None:
        super().__init__("no_plate_detected", message, detail)


class ImageDecodeError(AppError):
    """Chức năng: ảnh upload không đọc được hoặc file bị hỏng. HTTP 400."""

    status_code = 400

    def __init__(
        self,
        message: str = "Unable to read the uploaded image. The file may be corrupted or in an unsupported format.",
        detail: dict | None = None,
    ) -> None:
        super().__init__("image_decode_error", message, detail)


class NoIngredientsIdentifiedError(AppError):
    """Chức năng: Qwen3-VL không suy luận được nguyên liệu nào cho bất kỳ food box nào. HTTP 422 - dừng pipeline tại bước extraction."""

    status_code = 422

    def __init__(
        self,
        message: str = "Could not identify any ingredients in the detected food.",
        detail: dict | None = None,
    ) -> None:
        super().__init__("no_ingredients_identified", message, detail)


class NoSegmentsProducedError(AppError):
    """Chức năng: SAM3 không phân đoạn được mask nào cho bất kỳ nguyên liệu nào. HTTP 422 - dừng pipeline tại bước segmentation."""

    status_code = 422

    def __init__(
        self,
        message: str = "Could not segment any ingredient region in the image.",
        detail: dict | None = None,
    ) -> None:
        super().__init__("no_segments_produced", message, detail)


class ImageCropError(AppError):
    """Chức năng: crop vùng ảnh theo bbox thất bại (bbox không hợp lệ hoặc nằm ngoài ảnh). HTTP 422."""

    status_code = 422

    def __init__(
        self,
        message: str = "Failed to crop the detected region from the image.",
        detail: dict | None = None,
    ) -> None:
        super().__init__("image_crop_error", message, detail)


class ModelLoadError(AppError):
    """Chức năng: thiếu file model hoặc load model lỗi khi khởi động server. HTTP 500."""

    status_code = 500

    def __init__(self, message: str, detail: dict | None = None) -> None:
        super().__init__("model_load_error", message, detail)
