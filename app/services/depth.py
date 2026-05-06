from __future__ import annotations

import sys
from pathlib import Path
import logging
import time
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Any
from app.utils.image_processing import decode_image_bytes, inpaint_plate_depth

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
DA2_DIR = MODELS_DIR / "Depth-Anything-V2" / "metric_depth"
if str(DA2_DIR) not in sys.path:
    sys.path.append(str(DA2_DIR))

from depth_anything_v2.dpt import DepthAnythingV2
from dataset.transform import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose

logger = logging.getLogger(__name__)


def get_inference_transform() -> Compose:
    return Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

def load_depth_anything(weights_path: str, device: str, encoder: str = 'vits') -> dict:
    """
    Nạp model và bộ transform vào một Bundle duy nhất.
    """
    logger.info("[DEBUG] Loading DepthAnythingV2 (%s) on %s", encoder, device)
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    }
    
    from depth_anything_v2.dpt import DepthAnythingV2
    
    model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': 0.4})
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.to(device).eval()
    
    return {
        "model": model,
        "transform": get_inference_transform(),
        "device": device,
        "encoder": encoder
    }

def _run_depth_inference(model_bundle: dict, image_rgb: np.ndarray) -> np.ndarray:
    model = model_bundle["model"]
    transform = model_bundle["transform"]
    device = model_bundle["device"]
    orig_h, orig_w = image_rgb.shape[:2]
    
    # Chuẩn hóa về [0, 1] trước khi đưa vào Transform
    image_float = image_rgb.astype(np.float32) / 255.0
    
    # Áp dụng Compose Transform
    sample = transform({'image': image_float})
    image_tensor = torch.from_numpy(sample['image']).unsqueeze(0).to(device)
    
    with torch.inference_mode():
        pred = model(image_tensor)
        
        pred = F.interpolate(pred[:, None], (orig_h, orig_w), mode='bilinear', align_corners=True)[0, 0]
        
        depth_cm = pred.cpu().numpy() * 100.0
        
    return depth_cm

def estimate_depth(
    image_bytes: bytes,
    plate_mask: np.ndarray,      # Mask đĩa (numpy)
    food_mask: np.ndarray,       # Mask tổng hợp thức ăn (numpy)
    plate_type: str | None,      # Loại đĩa để chọn Template
    camera_h_ref: float,         # Chiều cao camera (cm)
    depth_bundle: dict,          # Bundle chứa model và device
    templates_dir: str,          # Đường dẫn folder templates
) -> dict:
    start = time.perf_counter()
    logger.info("[DEBUG] Starting depth_service with SOTA Inpainting...")
    
    try:
        image_rgb = decode_image_bytes(image_bytes)
        
        # 1. Chạy dự đoán độ sâu (đơn vị CM)
        depth_map = _run_depth_inference(depth_bundle, image_rgb)
        
        # 2. Inpainting chuyên sâu (Affine + Z-Offset)
        # Sử dụng mặt sàn thực tế từ Template thay vì median đơn thuần
        plate_depth = inpaint_plate_depth(
            depth_map=depth_map,
            plate_mask=plate_mask,
            food_mask=food_mask,
            plate_type=plate_type,
            camera_h_ref=camera_h_ref,
            template_dir=templates_dir
        )
            

        # --- PHẦN LOG THÊM VÀO ---
        # 1. Tính mean depth của vùng thực phẩm (mặt trên)
        if np.any(food_mask > 0):
            food_mean = np.mean(depth_map[food_mask > 0])
            print(f"[DEBUG] Food surface depth mean: {food_mean:.2f} cm")
        else:
            print("[DEBUG] Food mask is empty, cannot calculate food mean depth.")

        # 2. Tính mean depth của mặt đĩa đã phục hồi (mặt sàn)
        if np.any(plate_mask > 0):
            plate_mean = np.mean(plate_depth[plate_mask > 0])
            print(f"[DEBUG] Plate surface (inpainted) depth mean: {plate_mean:.2f} cm")
            
            # 3. Tính chiều cao trung bình thực tế
            if np.any(food_mask > 0):
                # Dùng np.maximum(..., 0) để đảm bảo không bị chiều cao âm do nhiễu sensor
                height_food = np.maximum(plate_depth - depth_map, 0)
                
                # Chỉ tính trung bình trên vùng có thực phẩm
                avg_h = np.mean(height_food[food_mask > 0])
                print(f"[DEBUG] Estimated average food height: {avg_h:.2f} cm")
                
                # Log thêm giá trị max để check xem có điểm nào vọt lên bất thường không
                max_h = np.max(height_food[food_mask > 0])
                print(f"[DEBUG] Max food height detected: {max_h:.2f} cm")
        else:
            print(f"[DEBUG] Plate mask is empty, fallback camera_h_ref: {camera_h_ref:.2f} cm")
            
        return {
            "depth_map": depth_map,     # Độ sâu mặt trên (food + plate)
            "plate_depth": plate_depth  # Độ sâu mặt sàn (đã khôi phục)
        }
        
    except Exception:
        logger.exception("[ERROR] depth_service failed")
        raise
    finally:
        logger.info("[DEBUG] depth_service finished in %.2fs", time.perf_counter() - start)
