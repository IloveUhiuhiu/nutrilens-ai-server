from __future__ import annotations

import numpy as np
from app.utils.cv.image import decode_image_bytes
from app.utils.visualization.debug import DebugVisualizer
from app.exceptions import InferenceError

from app.services.detection_service import DetectionService
from app.services.extraction_service import ExtractionService
from app.services.segmentation_service import SegmentationService
from app.services.depth_service import DepthService
from app.services.geometry_service import GeometryService
from app.services.nutrition_service import NutritionService

class NutritionPipeline:
    def __init__(
        self,
        detection: DetectionService,
        extraction: ExtractionService,
        segmentation: SegmentationService,
        depth: DepthService,
        geometry: GeometryService,
        nutrition: NutritionService,
    ) -> None:
        self.detection = detection
        self.extraction = extraction
        self.segmentation = segmentation
        self.depth = depth
        self.geometry = geometry
        self.nutrition = nutrition

    def _call(self, service_name: str, func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            import traceback
            error_msg = f"{service_name}: {str(exc)}\n{traceback.format_exc()}"
            raise InferenceError(service_name, error_msg) from exc

    def run_pipeline(
        self,
        image_bytes: bytes,
        models: object,
        nutrition_db: dict,
        camera_height_ref: float,
        pixel_area_ref: float,
        templates_dir: str,
        debugger: DebugVisualizer | None = None,
    ) -> dict:
        image_rgb = decode_image_bytes(image_bytes)
        if debugger:
            debugger.save_image_rgb("00_original.png", image_rgb)

        detections = self._call(
            "detection",
            self.detection.detect_food_and_plate,
            image_bytes,
            models.yolo_food,
            models.yolo_plate,
        )
        if debugger:
            debugger.save_detection_boxes("01_food_detection.png", image_rgb, detections["food_boxes"])
            if detections["plate_mask"].get("mask") is not None:
                debugger.save_mask("02_plate_mask.png", detections["plate_mask"]["mask"])

        ingredients_map = self._call(
            "extraction",
            self.extraction.extract_ingredients,
            image_bytes,
            detections["food_boxes"],
            models.qwen3_vl,
        )
        if debugger:
            debugger.save_json("03_extracted_ingredients.json", ingredients_map)

        segments = self._call(
            "segmentation",
            self.segmentation.segment_ingredients,
            image_bytes,
            ingredients_map,
            models.sam3,
            detections["food_boxes"],
        )
        if debugger:
            debugger.save_mask("05_instance_masks.png", np.zeros_like(image_rgb[:,:,0]))

        orig_h, orig_w = image_rgb.shape[:2]
        food_mask_combined = np.zeros((orig_h, orig_w), dtype=np.uint8)
        for mask in segments["global_masks"].values():
            food_mask_combined = np.maximum(food_mask_combined, mask)
        if debugger:
            debugger.save_mask("06_food_mask_combined.png", food_mask_combined)

        plate_type = detections["plate_mask"].get("class")
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
        )
        try:
            if debugger:
                debugger.save_depth("07_depth_map_raw.png", depth_data["depth_map"])
                debugger.save_depth("08_plate_depth_inpainted.png", depth_data["plate_depth"])

            geometry_data = self._call(
                "geometry",
                self.geometry.compute_geometry,
                segments,
                depth_map=depth_data["depth_map"],
                depth_plate=depth_data["plate_depth"],
                camera_height_ref=camera_height_ref,
                pixel_area_ref=pixel_area_ref,
            )
            if debugger:
                debugger.save_json("09_geometry_results.json", geometry_data.get("geometry", []))
                # Save topological order overlay
                if geometry_data.get("instance_masks") and geometry_data.get("instance_labels"):
                    try:
                        debugger.save_topological_order_overlay(
                            "10_topological_order.png",
                            image_rgb,
                            geometry_data["instance_masks"],
                            geometry_data["instance_labels"],
                            geometry_data.get("topological_order", [])
                        )
                    except Exception as e:
                        print(f"Failed to save topological overlay: {e}")

            nutrition_results = self._call(
                "nutrition",
                self.nutrition.estimate_nutrition,
                geometry_data["geometry"],
                nutrition_db.get("foods", {}),
            )
            if debugger:
                debugger.save_json("11_nutrition_results.json", nutrition_results)
        except Exception as e:
            print(e)
        return {
            "image_rgb": image_rgb,
            "detections": detections,
            "ingredients_map": ingredients_map,
            "segments": segments,
            "food_mask_combined": food_mask_combined,
            "depth_data": depth_data,
            "geometry_data": geometry_data,
            "nutrition_results": nutrition_results,
        }
