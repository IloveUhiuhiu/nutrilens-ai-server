from __future__ import annotations

import numpy as np
from app.utils.cv.image import decode_image_bytes
from app.exceptions import AppError, InferenceError

from app.services.detection_service import DetectionService
from app.services.extraction_service import ExtractionService
from app.services.segmentation_service import SegmentationService
from app.services.depth_service import DepthService
from app.services.geometry_service import GeometryService

class NutritionPipeline:
    def __init__(
        self,
        detection: DetectionService,
        extraction: ExtractionService,
        segmentation: SegmentationService,
        depth: DepthService,
        geometry: GeometryService,
    ) -> None:
        self.detection = detection
        self.extraction = extraction
        self.segmentation = segmentation
        self.depth = depth
        self.geometry = geometry

    def _call(self, service_name: str, func, *args, **kwargs):
        """Chức năng: chạy 1 bước pipeline. Đầu vào: tên service + callable. Đầu ra: kết quả bước.
        AppError đã được service chuẩn hóa (kèm log gốc tại đúng nơi phát sinh) nên truyền thẳng ra,
        không bọc lại - tránh mất error_code/status_code cụ thể (vd. no_food_detected -> inference_error).
        Exception lạ (chưa được service nào đoán trước) mới bị bọc thành InferenceError với message
        an toàn cho client, traceback gốc đã được service log ở nơi phát sinh và giữ qua `from exc`."""
        try:
            return func(*args, **kwargs)
        except AppError:
            raise
        except Exception as exc:
            raise InferenceError(
                service_name,
                f"{service_name.capitalize()} step failed due to an internal processing error.",
                {"step": service_name},
            ) from exc

    def run_pipeline(
        self,
        image_bytes: bytes,
        models: object,
        camera_height_ref: float,
        camera_intrinsics: dict,
        templates_dir: str,
        depth_bytes: bytes | None = None,
        depth_metadata: dict | None = None,
        has_absolute_depth: bool = False,
        anchor_distance_cm: float | None = None,
        anchor_pixel: tuple[float, float] | None = None,
        anchor_candidates: list[tuple[float, float, float]] | None = None,
    ) -> dict:
        image_rgb = decode_image_bytes(image_bytes)
        detections = self._call(
            "detection",
            self.detection.detect_food_and_plate,
            image_bytes,
            models.yolo_food,
            models.yolo_plate,
        )
        ingredients_map = self._call(
            "extraction",
            self.extraction.extract_ingredients,
            image_bytes,
            detections["food_boxes"],
            models.qwen3_vl,
        )
        segments = self._call(
            "segmentation",
            self.segmentation.segment_ingredients,
            image_bytes,
            ingredients_map,
            models.sam3,
            detections["food_boxes"],
        )

        orig_h, orig_w = image_rgb.shape[:2]
        food_mask_combined = np.zeros((orig_h, orig_w), dtype=np.uint8)
        for mask in segments["global_masks"].values():
            food_mask_combined = np.maximum(food_mask_combined, mask)

        # Chốt cờ trước khi (có thể) ghi đè mask ở dưới — derive_table_height
        # cần biết plate_mask có phản ánh đĩa thật hay chỉ là toàn khung giả
        # lập, để không lấy giao với nó (sẽ rỗng) khi không có đĩa thật.
        plate_detected = detections["plate_mask"].get("mask") is not None
        if not plate_detected:
            # YOLO Plate không detect được đĩa/vật chứa nào (miss, confidence
            # thấp, hoặc ảnh thực sự không có đĩa) -> không chặn pipeline, coi
            # món ăn đặt trực tiếp trên mặt bàn: dùng toàn khung ảnh làm vùng
            # tham chiếu mặt sàn. plate_type vẫn None nên inpaint_plate_depth
            # tự rơi vào nhánh "flat plate" sẵn có; chỉ cần đảm bảo plate_mask
            # không còn là None để get_clean_plate_samples/DepthScaleResolver
            # không crash AttributeError khi gọi .astype()/.astype(bool).
            detections["plate_mask"]["mask"] = np.ones((orig_h, orig_w), dtype=np.uint8)

        plate_type = detections["plate_mask"].get("class")
        if depth_bytes:
            depth_data = self._call(
                "depth",
                self.depth.prepare_client_depth,
                depth_bytes=depth_bytes,
                depth_metadata=depth_metadata or {},
                image_bytes=image_bytes,
                plate_mask=detections["plate_mask"]["mask"],
                food_mask=food_mask_combined,
                plate_type=plate_type,
                camera_h_ref=camera_height_ref,
                templates_dir=templates_dir,
                plate_detected=plate_detected,
            )
        else:
            # The AR raycast measures distance at one or more candidate
            # pixels — wherever the native ring-search landed on the
            # detected table/plate plane that frame (see
            # ArKitPlatformView.swift / ArPlatformView.kt). Native can't know
            # on-device which pixel will end up overlapping food once
            # segmentation runs here, so it reports every candidate it found;
            # picking the best one is deferred to depth_scale_service, after
            # food_mask is known. Only fall back to a single pixel (and, for
            # very old builds without even that, the principal point) when
            # the client didn't send the candidate list at all.
            resolved_anchor_pixel = anchor_pixel
            if has_absolute_depth and not anchor_candidates and resolved_anchor_pixel is None:
                cx = camera_intrinsics.get("cx")
                cy = camera_intrinsics.get("cy")
                resolved_anchor_pixel = (
                    cx if cx is not None else orig_w / 2.0,
                    cy if cy is not None else orig_h / 2.0,
                )

            depth_data = self._call(
                "depth",
                self.depth.estimate_depth,
                image_bytes=image_bytes,
                plate_mask=detections["plate_mask"]["mask"],
                food_mask=food_mask_combined,
                plate_type=plate_type,
                camera_h_ref=camera_height_ref,
                depth_bundle=models.depth_anything,
                templates_dir=templates_dir,
                has_absolute_depth=has_absolute_depth,
                anchor_distance_cm=anchor_distance_cm,
                anchor_pixel=resolved_anchor_pixel,
                anchor_candidates=anchor_candidates,
                plate_detected=plate_detected,
            )

        geometry_data = self._call(
            "geometry",
            self.geometry.compute_geometry,
            segments,
            depth_map=depth_data["depth_map"],
            depth_plate=depth_data["plate_depth"],
            camera_intrinsics=camera_intrinsics,
        )

        return {
            "image_rgb": image_rgb,
            "detections": detections,
            "ingredients_map": ingredients_map,
            "segments": segments,
            "food_mask_combined": food_mask_combined,
            "depth_data": depth_data,
            "geometry_data": geometry_data,
        }
