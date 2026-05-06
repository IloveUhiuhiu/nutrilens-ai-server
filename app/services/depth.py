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

def _ensure_logging() -> None:
    if not logging.getLogger().handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(filename)s - %(funcName)s - %(message)s")
        handler.setFormatter(formatter)
        logging.getLogger().addHandler(handler)
    logger.setLevel(logging.INFO)

def _log_info(message: str) -> None:
    _ensure_logging()
    logger.info(message)

def _log_error(message: str) -> None:
    _ensure_logging()
    logger.error(message)

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
    _log_info(f"loading DepthAnythingV2 {encoder} on {device}")
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    }
    
    
    
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
    _log_info("enter _run_depth_inference")
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
    plate_mask: np.ndarray,
    food_mask: np.ndarray,
    plate_type: str | None,
    camera_h_ref: float,
    depth_bundle: dict,
    templates_dir: str,
) -> dict:
    _log_info("enter estimate_depth")
    start = time.perf_counter()
    
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
                
        return {
            "depth_map": depth_map,     # Độ sâu mặt trên (food + plate)
            "plate_depth": plate_depth  # Độ sâu mặt sàn (đã khôi phục)
        }
        
    except Exception:
        _log_error("depth_service failed")
        raise
    finally:
        _log_info("depth_service finished")
