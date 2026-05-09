from __future__ import annotations

import logging
import time
import numpy as np
from collections import defaultdict
from typing import Any

from app.utils.math_helpers import (
    complete_depth_instance, 
    infer_instance_order, 
    compute_instance_heights
)

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

def compute_geometry(
    segments_dict: dict,
    depth_map: np.ndarray,       # Bản đồ độ sâu mặt trên thực phẩm (cm)
    depth_plate: np.ndarray,     # Bản đồ độ sâu mặt đĩa đã inpaint (cm)
    camera_height_ref: float,    # Chiều cao camera thực tế (cm)
    pixel_area_ref: float,       # Diện tích 1 pixel tại mặt sàn tham chiếu (cm2)
) -> list[dict]:
    _log_info("enter compute_geometry")
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
            _log_info("no instances found for geometry calculation")
            return []

        # 2. Xác định thứ tự xếp chồng (Top -> Bottom)
        # sorted_idx: List chỉ số instance từ Trên xuống Dưới
        sorted_idx, is_cycle = infer_instance_order(instance_masks, depth_map)
        
        results_agg = defaultdict(lambda: {"volume": 0.0, "sum_height": 0.0, "pixels": 0})
        
        # ======================================================================
        # TRƯỜNG HỢP 1: CÓ CHU TRÌNH (CYCLE) - Dùng Fallback (Chia đều chiều cao)
        # ======================================================================
        if is_cycle:
            _log_info("stacking cycle detected: fallback mode")
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
                # Công thức Adaptive Area cho từng pixel dựa trên độ sâu
                area_map_i = pixel_area_ref * (depth_map / camera_height_ref) ** 2
                
                # Thể tích = diện tích pixel * chiều cao chia sẻ
                vol = np.sum(shared_height[mask] * area_map_i[mask])
                
                name = instance_labels[i]
                results_agg[name]["volume"] += vol
                results_agg[name]["sum_height"] += np.sum(shared_height[mask])
                results_agg[name]["pixels"] += np.sum(mask)

        # ======================================================================
        # TRƯỜNG HỢP 2: NORMAL MODE - Pipeline chuẩn (Stacking + Completion)
        # ======================================================================
        else:
            _log_info("normal geometry pipeline")
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
                
                # Công thức Adaptive Area (Perspective Compensation)
                area_map_i = pixel_area_ref * (d_i / camera_height_ref) ** 2
                
                # Volume (cm3)
                vol = np.sum(h_i[mask_i] * area_map_i[mask_i])
                
                name = instance_labels[i]
                results_agg[name]["volume"] += vol
                results_agg[name]["sum_height"] += np.sum(h_i[mask_i])
                results_agg[name]["pixels"] += np.sum(mask_i)

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
            "instance_labels": instance_labels
        }
    except Exception:
        _log_error("geometry_service failed")
        raise
    finally:
        _log_info("geometry_service finished")