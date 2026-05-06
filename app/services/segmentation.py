from __future__ import annotations

import sys
from pathlib import Path
import logging
import time
import os
import torch
import gc
import numpy as np
from PIL import Image
from typing import Any
from app.utils.image_processing import decode_image_bytes, crop_image, merge_masks_and_instances

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
SAM3_DIR = MODELS_DIR / "SAM3_LoRA"
if str(SAM3_DIR) not in sys.path:
    sys.path.append(str(SAM3_DIR))
os.environ.setdefault("SAM3_ASSETS_DIR", str(SAM3_DIR / "sam3" / "assets"))
from infer_sam import SAM3LoRAInference

logger = logging.getLogger(__name__)

def load_sam3(config_path: str, weights_path: str, device: str, conf: float = 0.75) -> dict:
    """
    Khởi tạo SAM3 LoRA và đóng gói vào bundle kèm ngưỡng tin cậy conf.
    """
    logger.info("[DEBUG] Loading SAM3 LoRA model on %s with conf=%s", device, conf)
    from infer_sam import SAM3LoRAInference
    
    # detection_threshold trong SAM3 tương đương với conf
    inferencer = SAM3LoRAInference(
        config_path=config_path,
        weights_path=weights_path,
        device=device,
        detection_threshold=conf, 
        nms_iou_threshold=0.5
    )
    
    return {
        "model": inferencer,
        "device": device,
        "conf": conf,  # Lưu lại ngưỡng mặc định
    }

# ==============================================================================
# 2. MAIN SERVICE FUNCTION
# ==============================================================================

def segment_ingredients(
    image_bytes: bytes,
    ingredients_map: dict[str, list[str]],
    sam3_bundle: dict,
    food_boxes: list[dict]
) -> dict:
    """
    Phân đoạn nguyên liệu sử dụng ngưỡng conf từ bundle hoặc tham số ghi đè.
    """
    start = time.perf_counter()
    logger.info("[DEBUG] Starting segmentation_service...")
    
    
    threshold = sam3_bundle["conf"]
    temp_path = "/tmp/temp_crop.png"
    segmentation_results = {}
    
    try:
        image_rgb = decode_image_bytes(image_bytes)
        full_image_shape = image_rgb.shape
        
        inferencer = sam3_bundle["model"]
        inferencer.model.eval()
        
        with torch.no_grad():
            for box in food_boxes:
                box_id = box["id"]
                bbox = box["bbox"]
                target_ingredients = ingredients_map.get(box_id, [])
                
                if not target_ingredients:
                    segmentation_results[box_id] = []
                    continue
                
                # Cắt ảnh theo vùng thực phẩm
                crop_np = crop_image(image_rgb, bbox)
                Image.fromarray(crop_np).save(temp_path)
                
                prompts = [ing.strip().lower() for ing in target_ingredients if ing.strip()]
                
                # Chạy inference trên file tạm
                predictions = inferencer.predict(temp_path, text_prompts=prompts)
                
                masks_for_this_crop = []
                for p_idx, prompt_text in enumerate(prompts):
                    res = predictions.get(p_idx)
                    
                    if res and res['num_detections'] > 0:
                        masks = res.get('masks')
                        scores = res.get('scores')
                        
                        if masks is not None and scores is not None:
                            for m, score in zip(masks, scores):
                                # Kiểm tra ngưỡng tin cậy cho từng instance
                                if float(score) >= threshold:
                                    masks_for_this_crop.append((prompt_text, m))
                
                segmentation_results[box_id] = masks_for_this_crop

        # Hợp nhất các mask về hệ tọa độ ảnh gốc
        food_bboxes_map = {box["id"]: box["bbox"] for box in food_boxes}
        global_masks, instance_masks = merge_masks_and_instances(
            segmentation_results, 
            food_bboxes_map, 
            full_image_shape
        )

        return {
            "global_masks": global_masks,
            "instance_masks": instance_masks
        }

    except Exception:
        logger.exception("[ERROR] segmentation_service failed")
        raise
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        gc.collect()
        logger.info("[DEBUG] segmentation_service finished in %.2fs", time.perf_counter() - start)