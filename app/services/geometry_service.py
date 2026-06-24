from __future__ import annotations

import time
import numpy as np
from collections import defaultdict
from app.exceptions import InferenceError, ValidationError
from app.utils.processing import compute_instance_heights
from app.utils.math.fitting import complete_depth_instance
from app.utils.math.topology import infer_instance_order
from app.services.base import ServiceBase

class GeometryService(ServiceBase):
    service_name = "geometry"

    def __init__(self) -> None:
        pass

    def compute_geometry(
        self,
        segments_dict: dict,
        depth_map: np.ndarray,       # Bản đồ độ sâu mặt trên thực phẩm (cm)
        depth_plate: np.ndarray,     # Bản đồ độ sâu mặt đĩa đã inpaint (cm)
        camera_intrinsics: dict,     # Thông số nội tại camera, fx/fy theo pixel
    ) -> dict:
        self._log_info("step=geometry: enter compute_geometry")
        start = time.perf_counter()

        try:
            # 1. Giải nén dữ liệu instance từ segmentation_service
            instance_masks = []
            instance_labels = []
            # Duyệt qua dict { 'nguyên_liệu': [mask1, mask2, ...] }
            for name, masks in segments_dict.get("instance_masks", {}).items():
                for m in masks:
                    instance_masks.append(m.astype(bool))
                    instance_labels.append(name)

            n_instances = len(instance_masks)
            if n_instances == 0:
                self._log_warning("step=geometry completed: zero ingredient instances available for geometry calculation")
                return {
                    "geometry": [],
                    "topological_order": [],
                    "cycle_detected": False,
                    "instance_masks": [],
                    "instance_labels": [],
                    "instance_geometry": [],
                }

            instance_results = []

            # 2. Xác định thứ tự xếp chồng (Top -> Bottom)
            # sorted_idx: List chỉ số instance từ Trên xuống Dưới
            sorted_idx, is_cycle = infer_instance_order(instance_masks, depth_map)
            
            results_agg = defaultdict(lambda: {"volume": 0.0, "sum_height": 0.0, "pixels": 0})
            
            # ======================================================================
            # TRƯỜNG HỢP 1: CÓ CHU TRÌNH (CYCLE) - Dùng Fallback (Chia đều chiều cao)
            # ======================================================================
            if is_cycle:
                self._log_info("stacking cycle detected: fallback mode")
                # Tổng chiều cao từ đĩa đến đỉnh thực phẩm
                height_global = np.clip(depth_plate - depth_map, 0, None)
                
                # Đếm số lớp thực phẩm chồng lên nhau tại mỗi pixel
                overlap_count = np.zeros_like(depth_map, dtype=np.int32)
                for m in instance_masks:
                    overlap_count += m.astype(np.int32)
                overlap_count = np.maximum(overlap_count, 1) # Tránh chia cho 0
                
                # Chia đều chiều cao cho các lớp tại pixel đó
                shared_height = height_global / overlap_count
                
                for i, mask in enumerate(instance_masks):
                    # Diện tích pixel metric từ depth thật: dA ~= Z^2 / (fx * fy)
                    area_map_i = self._pixel_area_from_metric_depth(depth_map, camera_intrinsics)
                    
                    # Thể tích = diện tích pixel * chiều cao chia sẻ
                    vol = np.sum(shared_height[mask] * area_map_i[mask])
                    
                    name = instance_labels[i]
                    results_agg[name]["volume"] += vol
                    results_agg[name]["sum_height"] += np.sum(shared_height[mask])
                    results_agg[name]["pixels"] += np.sum(mask)
                    instance_results.append({
                        "instance_index": i,
                        "ingredient": name,
                        "volume_cm3": round(float(vol), 2),
                        "avg_height_cm": round(float(np.mean(shared_height[mask])) if np.sum(mask) > 0 else 0.0, 2),
                    })

            # ======================================================================
            # TRƯỜNG HỢP 2: NORMAL MODE - Pipeline chuẩn (Stacking + Completion)
            # ======================================================================
            else:
                self._log_info("normal geometry pipeline")
                depth_completed_ref = depth_map.copy()
                instance_depth_maps = {} # Lưu bề mặt (top surface) sau phục hồi của mỗi instance
                occlusion_mask = np.zeros_like(depth_map, dtype=bool)

                # Duyệt từ TRÊN xuống DƯỚI để xác định vùng bị che và phục hồi bề mặt
                for idx in sorted_idx:
                    mask_i = instance_masks[idx]
                    
                    # Vùng bị che là vùng thuộc mask_i nhưng đã bị các lớp trên (trong occlusion_mask) chiếm chỗ
                    missing_mask = mask_i & occlusion_mask
                    
                    # Phục hồi bề mặt lý tưởng bằng Polynomial Fitting bậc 2
                    d_i = complete_depth_instance(
                        mask=mask_i,
                        depth_food=depth_completed_ref,
                        depth_below=depth_plate,
                        missing_mask=missing_mask
                    )
                    instance_depth_maps[idx] = d_i
                    
                    # Cập nhật bề mặt tham chiếu cho các lớp bên dưới
                    depth_completed_ref[mask_i] = d_i[mask_i]
                    occlusion_mask |= mask_i

                # Tính toán chiều cao (độ dày) thực tế theo logic mặt sàn động (Bottom -> Top)
                height_instances = compute_instance_heights(
                    instance_masks, sorted_idx, instance_depth_maps, depth_plate
                )

                # Tổng hợp kết quả thể tích
                for i in range(n_instances):
                    mask_i = instance_masks[i]
                    h_i = height_instances[i]
                    d_i = instance_depth_maps[i] # Bề mặt trên của vật i
                    
                    # Diện tích pixel metric từ depth thật: dA ~= Z^2 / (fx * fy)
                    area_map_i = self._pixel_area_from_metric_depth(d_i, camera_intrinsics)
                    
                    # Volume (cm3)
                    vol = np.sum(h_i[mask_i] * area_map_i[mask_i])
                    
                    name = instance_labels[i]
                    results_agg[name]["volume"] += vol
                    results_agg[name]["sum_height"] += np.sum(h_i[mask_i])
                    results_agg[name]["pixels"] += np.sum(mask_i)
                    instance_results.append({
                        "instance_index": i,
                        "ingredient": name,
                        "volume_cm3": round(float(vol), 2),
                        "avg_height_cm": round(float(np.mean(h_i[mask_i])) if np.sum(mask_i) > 0 else 0.0, 2),
                    })

            # 4. Chuyển đổi sang format response cuối cùng
            final_results = []
            for name, data in results_agg.items():
                avg_h = data["sum_height"] / data["pixels"] if data["pixels"] > 0 else 0.0
                final_results.append({
                    "ingredient": name,
                    "volume_cm3": round(float(data["volume"]), 2),
                    "avg_height_cm": round(float(avg_h), 2)
                })
                
            return {
                "geometry": final_results,
                "topological_order": sorted_idx,
                "cycle_detected": is_cycle,
                "instance_masks": instance_masks,
                "instance_labels": instance_labels,
                "instance_geometry": instance_results,
            }
        except ValueError as exc:
            # fx/fy không hợp lệ là lỗi dữ liệu đầu vào (camera intrinsics), không phải lỗi inference.
            self._log_error(f"step=geometry failed: invalid camera intrinsics - {exc}")
            raise ValidationError(str(exc), {"step": "geometry", "field": "camera_intrinsics"}) from exc
        except Exception as exc:
            self._log_error(f"step=geometry failed: unexpected error {type(exc).__name__}")
            raise InferenceError("geometry", "Geometry/volume computation failed due to an internal error.", {"step": "geometry"}) from exc
        finally:
            self._log_info("step=geometry: geometry_service finished")

    def _pixel_area_from_metric_depth(self, depth_cm: np.ndarray, camera_intrinsics: dict) -> np.ndarray:
        """Chức năng: tính diện tích mỗi pixel từ metric depth và intrinsics. Đầu vào: depth cm, fx/fy pixel. Đầu ra: cm2/pixel."""
        fx = float(camera_intrinsics["fx"])
        fy = float(camera_intrinsics["fy"])
        if fx <= 0 or fy <= 0:
            raise ValueError("camera intrinsics fx/fy must be positive")
        valid_depth = np.maximum(depth_cm.astype(np.float32), 0)
        return (valid_depth * valid_depth) / (fx * fy)

__all__ = ["GeometryService"]
