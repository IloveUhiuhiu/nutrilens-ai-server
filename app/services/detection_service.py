from __future__ import annotations

import logging
import time
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from ultralytics import YOLO
from app.utils.image import decode_image_bytes
from app.services.base import ServiceBase

class DetectionService(ServiceBase):
    service_name = "detection"

    def __init__(self) -> None:
        pass

    def load_yolo_food(self, weights_path: str, device: str, conf: float = 0.5) -> dict:
        self._log_info(f"loading YOLO Food model on {device} with conf={conf}")
        model = YOLO(weights_path)
        return {"model": model, "device": device, "conf": conf, "task": "food"}

    def load_yolo_plate(self, weights_path: str, device: str, conf: float = 0.9) -> dict:
        self._log_info(f"loading YOLO Plate model on {device} with conf={conf}")
        model = YOLO(weights_path)
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

    def detect_food_and_plate(
        self,
        image_bytes: bytes,
        yolo_food_bundle: dict,
        yolo_plate_bundle: dict,
        parallel: bool = True,
    ) -> dict:
        self._log_info("start detection_service")
        try:
            image_rgb = decode_image_bytes(image_bytes)
            
            if parallel:
                self._log_info("parallel yolo inference branch")
                with ThreadPoolExecutor(max_workers=2) as executor:
                    food_future = executor.submit(self._run_yolo, yolo_food_bundle, image_rgb)
                    plate_future = executor.submit(self._run_yolo, yolo_plate_bundle, image_rgb)
                    food_raw = food_future.result()
                    plate_raw = plate_future.result()
            else:
                self._log_info("sequential yolo inference branch")
                food_raw = self._run_yolo(yolo_food_bundle, image_rgb)
                plate_raw = self._run_yolo(yolo_plate_bundle, image_rgb)

            food_boxes = self._parse_food_boxes(food_raw)
            plate_data = self._process_plate_results(plate_raw)

            return {
                "food_boxes": food_boxes,
                "plate_mask": plate_data 
            }

        except Exception:
            self._log_error("detection_service failed")
            raise
        finally:
            self._log_info("detection_service finished")

__all__ = ["DetectionService"]