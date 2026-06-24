from __future__ import annotations

import logging
import time
import cv2
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from ultralytics import YOLO
from app.exceptions import AppError, InferenceError, ModelLoadError, NoFoodDetectedError
from app.utils.cv.image import decode_image_bytes
from app.services.base import ServiceBase

class DetectionService(ServiceBase):
    service_name = "detection"

    def __init__(self) -> None:
        pass

    def load_yolo_food(self, weights_path: str, device: str, conf: float = 0.5) -> dict:
        self._log_info(f"loading YOLO Food model on {device} with conf={conf}")
        try:
            model = YOLO(weights_path)
        except Exception as exc:
            self._log_error(f"model_load failed: YOLO Food weights_path={weights_path} device={device}")
            raise ModelLoadError(
                "Failed to load the food detection model.",
                {"model": "yolo_food", "weights_path": weights_path},
            ) from exc
        return {"model": model, "device": device, "conf": conf, "task": "food"}

    def load_yolo_plate(self, weights_path: str, device: str, conf: float = 0.9) -> dict:
        self._log_info(f"loading YOLO Plate model on {device} with conf={conf}")
        try:
            model = YOLO(weights_path)
        except Exception as exc:
            self._log_error(f"model_load failed: YOLO Plate weights_path={weights_path} device={device}")
            raise ModelLoadError(
                "Failed to load the plate detection model.",
                {"model": "yolo_plate", "weights_path": weights_path},
            ) from exc
        return {"model": model, "device": device, "conf": conf, "task": "plate"}

    def _run_yolo(self, model_dict: dict, image: np.ndarray) -> Any:
        self._log_info("run yolo")
        """Thực hiện inference trên thiết bị %s."""
        model = model_dict["model"]
        # Chuyển sang BGR vì Ultralytics tối ưu trên định dạng của OpenCV
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        results = model.predict(
            img_bgr,
            device=model_dict["device"],
            conf=model_dict["conf"],
            verbose=False
        )
        return results

    def _parse_food_boxes(self, results: Any) -> list[dict]:
        self._log_info("parse food boxes")
        boxes = []
        if not results or len(results) == 0:
            self._log_info("no food detection results")
            return boxes
        
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            self._log_info("empty food boxes branch")
            return boxes

        xyxy = result.boxes.xyxy.cpu().numpy()
        conf = result.boxes.conf.cpu().numpy()
        
        for idx, bbox in enumerate(xyxy):
            box_id = f"food_{idx}"
            
            boxes.append({
                "id": box_id,
                "label": "food",
                "bbox": bbox.astype(int).tolist(),
                "score": float(conf[idx])
            })
        return boxes

    def _process_plate_results(self, results: Any) -> dict:
        self._log_info("process plate results")
        """
        Xử lý kết quả Segmentation và chọn vật chứa có diện tích lớn nhất.
        """
        output = {"mask": None, "class": None, "bbox": None, "score": 0.0}
        
        if not results or len(results) == 0 or results[0].masks is None:
            self._log_info("no plate segmentation results")
            return output

        result = results[0]
        img_h, img_w = result.orig_shape
        masks_data = result.masks.data 
        
        # 1. Tính diện tích của tất cả các mặt nạ để tìm cái lớn nhất
        # Sum trên pixel (True=1, False=0)
        areas = masks_data.sum(dim=(1, 2)) 
        best_idx = int(areas.argmax())

        # 2. Trích xuất thông tin theo best_idx
        mask_tensor = masks_data[best_idx]
        cls_id = int(result.boxes.cls[best_idx])
        cls_name = result.names[cls_id]
        score = float(result.boxes.conf[best_idx])
        bbox = result.boxes.xyxy[best_idx].cpu().numpy().astype(int).tolist()

        # 3. Hậu xử lý mặt nạ
        mask_np = mask_tensor.cpu().numpy()
        if mask_np.shape[:2] != (img_h, img_w):
            mask_np = cv2.resize(mask_np, (img_w, img_h), interpolation=cv2.INTER_LINEAR)

        # Nhị phân hóa
        binary_mask = (mask_np > 0.5).astype(np.uint8)

        output.update({
            "mask": binary_mask,
            "class": cls_name,
            "bbox": bbox,
            "score": score
        })
        return output

    def _run_yolo_food(self, yolo_food_bundle: dict, image_rgb: np.ndarray) -> Any:
        """Chức năng: chạy YOLO food, log rõ nguyên nhân nếu model thất bại. Đầu ra: raw results hoặc raise."""
        try:
            return self._run_yolo(yolo_food_bundle, image_rgb)
        except torch.cuda.OutOfMemoryError as exc:
            self._log_error("step=detection(food) failed: GPU out of memory while running YOLO food inference")
            raise InferenceError(
                "detection", "GPU ran out of memory while detecting food.", {"step": "detection_food", "reason": "cuda_oom"}
            ) from exc
        except Exception as exc:
            self._log_error(f"step=detection(food) failed: YOLO food inference raised {type(exc).__name__}")
            raise InferenceError(
                "detection", "Food detection model failed to run.", {"step": "detection_food"}
            ) from exc

    def _run_yolo_plate(self, yolo_plate_bundle: dict, image_rgb: np.ndarray) -> Any:
        """Chức năng: chạy YOLO plate, log rõ nguyên nhân nếu model thất bại. Đầu ra: raw results hoặc raise."""
        try:
            return self._run_yolo(yolo_plate_bundle, image_rgb)
        except torch.cuda.OutOfMemoryError as exc:
            self._log_error("step=detection(plate) failed: GPU out of memory while running YOLO plate inference")
            raise InferenceError(
                "detection", "GPU ran out of memory while detecting the plate.", {"step": "detection_plate", "reason": "cuda_oom"}
            ) from exc
        except Exception as exc:
            self._log_error(f"step=detection(plate) failed: YOLO plate inference raised {type(exc).__name__}")
            raise InferenceError(
                "detection", "Plate detection model failed to run.", {"step": "detection_plate"}
            ) from exc

    def detect_food_and_plate(
        self,
        image_bytes: bytes,
        yolo_food_bundle: dict,
        yolo_plate_bundle: dict,
        parallel: bool = True,
    ) -> dict:
        self._log_info("step=detection: start detection_service")
        image_rgb = decode_image_bytes(image_bytes)

        try:
            if parallel:
                self._log_info("parallel yolo inference branch")
                with ThreadPoolExecutor(max_workers=2) as executor:
                    food_future = executor.submit(self._run_yolo_food, yolo_food_bundle, image_rgb)
                    plate_future = executor.submit(self._run_yolo_plate, yolo_plate_bundle, image_rgb)
                    food_raw = food_future.result()
                    plate_raw = plate_future.result()
            else:
                self._log_info("sequential yolo inference branch")
                food_raw = self._run_yolo_food(yolo_food_bundle, image_rgb)
                plate_raw = self._run_yolo_plate(yolo_plate_bundle, image_rgb)

            food_boxes = self._parse_food_boxes(food_raw)
            plate_data = self._process_plate_results(plate_raw)

            self._log_info(
                f"step=detection completed: food_boxes={len(food_boxes)}, "
                f"plate_detected={plate_data['mask'] is not None}"
            )

            if not food_boxes:
                self._log_warning("step=detection completed: zero food objects detected in image")
                raise NoFoodDetectedError(detail={"step": "detection"})

            if plate_data["mask"] is None:
                # Không coi đây là lỗi cứng: pipeline (nutrition_pipeline.run_pipeline)
                # sẽ fallback sang giả định món ăn đặt trực tiếp trên mặt bàn
                # (dùng toàn khung ảnh làm vùng tham chiếu mặt sàn) thay vì chặn request.
                self._log_warning(
                    "step=detection completed: zero plate/container objects detected in image, "
                    "pipeline will fall back to a flat table-surface assumption"
                )

            return {
                "food_boxes": food_boxes,
                "plate_mask": plate_data
            }

        except AppError:
            raise
        except Exception as exc:
            self._log_error(f"step=detection failed: unexpected error {type(exc).__name__} while parsing detection results")
            raise InferenceError("detection", "Food/plate detection failed due to an internal error.", {"step": "detection"}) from exc
        finally:
            self._log_info("step=detection: detection_service finished")

__all__ = ["DetectionService"]