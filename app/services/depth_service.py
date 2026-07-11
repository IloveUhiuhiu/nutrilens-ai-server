from __future__ import annotations

import sys
from pathlib import Path
import time
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from app.exceptions import AppError, InferenceError, ModelLoadError, ValidationError
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

        try:
            model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': 0.6})
            model.load_state_dict(torch.load(weights_path, map_location='cpu'))
            model.to(device).eval()
        except Exception as exc:
            self._log_error(
                f"model_load failed: DepthAnythingV2 encoder={encoder} weights_path={weights_path} device={device}"
            )
            raise ModelLoadError(
                "Failed to load the depth estimation model (DepthAnythingV2).",
                {"model": "depth_anything", "encoder": encoder, "weights_path": weights_path},
            ) from exc

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
        anchor_candidates: list[tuple[float, float, float]] | None = None,
        plate_detected: bool = True,
    ) -> dict:
        self._log_info("step=depth: enter estimate_depth")
        start = time.perf_counter()

        try:
            image_rgb = decode_image_bytes(image_bytes)

            # 1. Chạy dự đoán độ sâu (đơn vị CM)
            try:
                depth_map = self._run_depth_inference(depth_bundle, image_rgb)
            except torch.cuda.OutOfMemoryError as exc:
                self._log_error("step=depth failed: GPU out of memory while running DepthAnythingV2 inference")
                raise InferenceError(
                    "depth", "GPU ran out of memory while estimating depth.", {"step": "depth", "reason": "cuda_oom"}
                ) from exc

            if not np.isfinite(depth_map).any():
                self._log_error(
                    f"step=depth failed: DepthAnythingV2 produced no finite depth values (shape={depth_map.shape})"
                )
                raise InferenceError(
                    "depth", "Depth estimation produced no valid depth values for this image.", {"step": "depth"}
                )

            finite_raw = depth_map[np.isfinite(depth_map)]
            self._log_info(
                f"step=depth: raw depth_map (pre-scale) min={finite_raw.min():.2f}cm "
                f"max={finite_raw.max():.2f}cm mean={finite_raw.mean():.2f}cm"
            )

            # 1b. CASE A — anchor depth scale to the measured camera-to-object
            # distance so the metric depth (and volume) is absolute instead of
            # relying on the model's fixed max_depth assumption. anchor_pixel
            # is the exact image pixel the AR raycast measured (camera
            # principal point) — the depth value must be read at that same
            # pixel, not from an unrelated whole-plate aggregate.
            self._log_info(
                f"step=depth: has_absolute_depth={has_absolute_depth} "
                f"anchor_distance_cm={anchor_distance_cm} anchor_pixel={anchor_pixel} "
                f"anchor_candidates={len(anchor_candidates) if anchor_candidates else 0}"
            )
            scale = 1.0
            scale_source = "da2_metric"
            if has_absolute_depth:
                depth_map, scale, scale_source = self._scale_resolver.anchor_with_distance(
                    depth_map=depth_map,
                    plate_mask=plate_mask,
                    food_mask=food_mask,
                    anchor_distance_cm=anchor_distance_cm,
                    anchor_pixel=anchor_pixel,
                    anchor_candidates=anchor_candidates,
                )
                finite_scaled = depth_map[np.isfinite(depth_map)]
                self._log_info(
                    f"step=depth: depth_map (post-scale) min={finite_scaled.min():.2f}cm "
                    f"max={finite_scaled.max():.2f}cm mean={finite_scaled.mean():.2f}cm scale={scale:.4f}"
                )

            # 1c. Tách biệt không gian với bước anchor ở trên: anchor đọc 1
            # patch nhỏ gần đúng điểm AR đo để tính scale; ở đây suy ra
            # camera_h_ref thực tế từ median cả vùng mặt bàn quan sát được
            # trên depth_map ĐÃ scale — ổn định hơn giá trị tĩnh client khai
            # báo/mặc định, dùng cho việc tái tạo mặt đĩa bên dưới.
            table_height = self._scale_resolver.derive_table_height(
                depth_map, plate_mask, food_mask, plate_detected=plate_detected,
            )
            resolved_camera_h_ref = table_height if table_height is not None else camera_h_ref
            self._log_info(
                f"camera_h_ref: table_derived={table_height}, "
                f"client_fallback={camera_h_ref} -> using={resolved_camera_h_ref}"
            )

            valid_dm = np.isfinite(depth_map) & (depth_map > 0)
            food_b = food_mask.astype(bool)
            plate_b = plate_mask.astype(bool)
            regions = {
                "food": valid_dm & food_b,
                "plate_clean": valid_dm & plate_b & ~food_b,
                "table_remaining": valid_dm & ~plate_b & ~food_b,
            }
            for region_name, region_mask in regions.items():
                vals = depth_map[region_mask]
                if vals.size > 0:
                    self._log_info(
                        f"step=depth region {region_name}: min={vals.min():.2f}cm "
                        f"max={vals.max():.2f}cm mean={vals.mean():.2f}cm pixels={vals.size}"
                    )
                else:
                    self._log_info(f"step=depth region {region_name}: no valid pixels")

            # 2. Inpainting chuyên sâu (Affine + Z-Offset)
            # Sử dụng mặt sàn thực tế từ Template thay vì median đơn thuần
            plate_depth = inpaint_plate_depth(
                depth_map=depth_map,
                plate_mask=plate_mask,
                food_mask=food_mask,
                plate_type=plate_type,
                camera_h_ref=resolved_camera_h_ref,
                template_dir=templates_dir
            )

            return {
                "depth_map": depth_map,     # Độ sâu mặt trên (food + plate)
                "plate_depth": plate_depth, # Độ sâu mặt sàn (đã khôi phục)
                "scale": scale,
                "scale_source": scale_source,
                "camera_h_ref": resolved_camera_h_ref,
            }

        except AppError:
            raise
        except Exception as exc:
            self._log_error(f"step=depth failed: unexpected error {type(exc).__name__} during depth estimation")
            raise InferenceError("depth", "Depth estimation failed due to an internal error.", {"step": "depth"}) from exc
        finally:
            self._log_info("step=depth: depth_service finished")

    def load_client_depth_map(
        self,
        depth_bytes: bytes,
        depth_metadata: dict,
        target_shape: tuple[int, int],
    ) -> np.ndarray:
        """Chức năng: đọc depth map client gửi. Đầu vào: bytes, metadata, shape ảnh. Đầu ra: depth cm."""
        self._log_info("step=depth(client): enter load_client_depth_map")
        suffix = (depth_metadata or {}).get("file_extension", "").lower()
        depth_unit = (depth_metadata or {}).get("depth_unit", "cm").lower()
        try:
            depth_map = self._decode_depth_bytes(depth_bytes, suffix)
            depth_map = self._convert_depth_to_cm(depth_map, depth_unit)
            target_h, target_w = target_shape
            if depth_map.shape[:2] != (target_h, target_w):
                depth_map = cv2.resize(depth_map, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            return depth_map.astype(np.float32)
        except AppError:
            raise
        except Exception as exc:
            self._log_error(
                f"step=depth(client) failed: unexpected error {type(exc).__name__} while processing client "
                f"depth map (suffix={suffix}, depth_unit={depth_unit})"
            )
            raise ValidationError(
                "The provided depth map could not be processed.",
                {"step": "depth_client", "file_extension": suffix, "depth_unit": depth_unit},
            ) from exc

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
        plate_detected: bool = True,
    ) -> dict:
        """Chức năng: dùng depth map client và tính plate depth. Đầu vào: depth bytes + mask. Đầu ra: depth_data."""
        self._log_info("step=depth(client): enter prepare_client_depth")
        try:
            image_rgb = decode_image_bytes(image_bytes)
            depth_map = self.load_client_depth_map(
                depth_bytes,
                depth_metadata,
                target_shape=image_rgb.shape[:2],
            )

            if not np.isfinite(depth_map).any():
                self._log_error(
                    f"step=depth(client) failed: client depth map has no finite values (shape={depth_map.shape})"
                )
                raise ValidationError(
                    "The provided depth map contains no valid depth values.",
                    {"step": "depth_client"},
                )

            table_height = self._scale_resolver.derive_table_height(
                depth_map, plate_mask, food_mask, plate_detected=plate_detected,
            )
            resolved_camera_h_ref = table_height if table_height is not None else camera_h_ref
            self._log_info(
                f"camera_h_ref(client): table_derived={table_height}, "
                f"client_fallback={camera_h_ref} -> using={resolved_camera_h_ref}"
            )

            plate_depth = inpaint_plate_depth(
                depth_map=depth_map,
                plate_mask=plate_mask,
                food_mask=food_mask,
                plate_type=plate_type,
                camera_h_ref=resolved_camera_h_ref,
                template_dir=templates_dir,
            )
            return {
                "depth_map": depth_map,
                "plate_depth": plate_depth,
                "source": "client_depth_map",
                "camera_h_ref": resolved_camera_h_ref,
            }
        except AppError:
            raise
        except Exception as exc:
            self._log_error(f"step=depth(client) failed: unexpected error {type(exc).__name__}")
            raise InferenceError(
                "depth", "Processing the client-provided depth map failed due to an internal error.", {"step": "depth_client"}
            ) from exc
        finally:
            self._log_info("step=depth(client): prepare_client_depth finished")

    def _decode_depth_bytes(self, depth_bytes: bytes, suffix: str) -> np.ndarray:
        """Chức năng: decode depth map từ npy hoặc ảnh. Đầu vào: bytes/suffix. Đầu ra: ndarray.
        Raise ValidationError nếu file depth map client gửi bị hỏng hoặc không đúng định dạng hỗ trợ."""
        if suffix == ".npy":
            from io import BytesIO

            try:
                return np.load(BytesIO(depth_bytes)).astype(np.float32)
            except Exception as exc:
                self._log_error(f"step=depth(client) failed: corrupted or unsupported .npy depth map ({len(depth_bytes)} bytes)")
                raise ValidationError(
                    "The provided depth map (.npy) is corrupted or unsupported.",
                    {"step": "depth_client", "file_extension": suffix},
                ) from exc

        buffer = np.frombuffer(depth_bytes, dtype=np.uint8)
        depth_map = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
        if depth_map is None:
            self._log_error(
                f"step=depth(client) failed: cv2.imdecode could not parse depth map "
                f"({len(depth_bytes)} bytes, suffix={suffix})"
            )
            raise ValidationError(
                "Unsupported depth map format.",
                {"step": "depth_client", "file_extension": suffix},
            )
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
