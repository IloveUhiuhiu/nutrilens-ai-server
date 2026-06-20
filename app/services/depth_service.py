from __future__ import annotations

import sys
from pathlib import Path
import time
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from app.utils.cv.image import decode_image_bytes
from app.utils.processing import inpaint_plate_depth
from app.services.base import ServiceBase
from app.services.depth_scale_service import DepthScaleResolver

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
DA2_DIR = MODELS_DIR / "Depth-Anything-V2" / "metric_depth"
if str(DA2_DIR) not in sys.path:
    sys.path.append(str(DA2_DIR))

from depth_anything_v2.dpt import DepthAnythingV2
from dataset.transform import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose

class DepthService(ServiceBase):
    service_name = "depth"

    def __init__(self) -> None:
        self._scale_resolver = DepthScaleResolver()

    def get_inference_transform(self) -> Compose:
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

    def load_depth_anything(self, weights_path: str, device: str, encoder: str = "vits") -> dict:
        self._log_info(f"loading DepthAnythingV2 {encoder} on {device}")
        
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        }
        
        
        
        model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': 0.4})
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        model.to(device).eval()
        
        return {
            "model": model,
            "transform": self.get_inference_transform(),
            "device": device,
            "encoder": encoder
        }

    def _run_depth_inference(self, model_bundle: dict, image_rgb: np.ndarray) -> np.ndarray:
        self._log_info("enter _run_depth_inference")
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
        self,
        image_bytes: bytes,
        plate_mask: np.ndarray,
        food_mask: np.ndarray,
        plate_type: str | None,
        camera_h_ref: float,
        depth_bundle: dict,
        templates_dir: str,
        has_absolute_depth: bool = False,
        anchor_distance_cm: float | None = None,
        anchor_pixel: tuple[float, float] | None = None,
    ) -> dict:
        self._log_info("enter estimate_depth")
        start = time.perf_counter()

        try:
            image_rgb = decode_image_bytes(image_bytes)

            # 1. Chạy dự đoán độ sâu (đơn vị CM)
            depth_map = self._run_depth_inference(depth_bundle, image_rgb)

            # 1b. CASE A — anchor depth scale to the measured camera-to-object
            # distance so the metric depth (and volume) is absolute instead of
            # relying on the model's fixed max_depth assumption. anchor_pixel
            # is the exact image pixel the AR raycast measured (camera
            # principal point) — the depth value must be read at that same
            # pixel, not from an unrelated whole-plate aggregate.
            scale = 1.0
            scale_source = "da2_metric"
            if has_absolute_depth:
                depth_map, scale, scale_source = self._scale_resolver.anchor_with_distance(
                    depth_map=depth_map,
                    plate_mask=plate_mask,
                    food_mask=food_mask,
                    anchor_distance_cm=anchor_distance_cm,
                    anchor_pixel=anchor_pixel,
                )

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
                "plate_depth": plate_depth, # Độ sâu mặt sàn (đã khôi phục)
                "scale": scale,
                "scale_source": scale_source,
            }

        except Exception:
            self._log_error("depth_service failed")
            raise
        finally:
            self._log_info("depth_service finished")

    def load_client_depth_map(
        self,
        depth_bytes: bytes,
        depth_metadata: dict,
        target_shape: tuple[int, int],
    ) -> np.ndarray:
        """Chức năng: đọc depth map client gửi. Đầu vào: bytes, metadata, shape ảnh. Đầu ra: depth cm."""
        self._log_info("enter load_client_depth_map")
        suffix = (depth_metadata or {}).get("file_extension", "").lower()
        depth_unit = (depth_metadata or {}).get("depth_unit", "cm").lower()
        try:
            depth_map = self._decode_depth_bytes(depth_bytes, suffix)
            depth_map = self._convert_depth_to_cm(depth_map, depth_unit)
            target_h, target_w = target_shape
            if depth_map.shape[:2] != (target_h, target_w):
                depth_map = cv2.resize(depth_map, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            return depth_map.astype(np.float32)
        except Exception:
            self._log_error("load_client_depth_map failed")
            raise

    def prepare_client_depth(
        self,
        depth_bytes: bytes,
        depth_metadata: dict,
        image_bytes: bytes,
        plate_mask: np.ndarray,
        food_mask: np.ndarray,
        plate_type: str | None,
        camera_h_ref: float,
        templates_dir: str,
    ) -> dict:
        """Chức năng: dùng depth map client và tính plate depth. Đầu vào: depth bytes + mask. Đầu ra: depth_data."""
        self._log_info("enter prepare_client_depth")
        image_rgb = decode_image_bytes(image_bytes)
        depth_map = self.load_client_depth_map(
            depth_bytes,
            depth_metadata,
            target_shape=image_rgb.shape[:2],
        )
        plate_depth = inpaint_plate_depth(
            depth_map=depth_map,
            plate_mask=plate_mask,
            food_mask=food_mask,
            plate_type=plate_type,
            camera_h_ref=camera_h_ref,
            template_dir=templates_dir,
        )
        return {
            "depth_map": depth_map,
            "plate_depth": plate_depth,
            "source": "client_depth_map",
        }

    def _decode_depth_bytes(self, depth_bytes: bytes, suffix: str) -> np.ndarray:
        """Chức năng: decode depth map từ npy hoặc ảnh. Đầu vào: bytes/suffix. Đầu ra: ndarray."""
        if suffix == ".npy":
            from io import BytesIO

            return np.load(BytesIO(depth_bytes)).astype(np.float32)
        buffer = np.frombuffer(depth_bytes, dtype=np.uint8)
        depth_map = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
        if depth_map is None:
            raise ValueError("Unsupported depth map format.")
        if depth_map.ndim == 3:
            depth_map = depth_map[:, :, 0]
        return depth_map.astype(np.float32)

    def _convert_depth_to_cm(self, depth_map: np.ndarray, depth_unit: str) -> np.ndarray:
        """Chức năng: đổi depth map về cm. Đầu vào: depth map và unit. Đầu ra: depth cm."""
        if depth_unit in {"m", "meter", "meters"}:
            return depth_map * 100.0
        if depth_unit in {"mm", "millimeter", "millimeters"}:
            return depth_map / 10.0
        return depth_map
