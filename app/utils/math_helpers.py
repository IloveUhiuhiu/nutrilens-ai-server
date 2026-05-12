from __future__ import annotations
import numpy as np
import logging
from app.utils.geometry import (
    get_contour as _get_contour,
    fit_shape as _fit_shape,
    build_affine as _build_affine,
    estimate_affine_from_shape as _estimate_affine_from_shape,
)
from app.utils.depth import (
    complete_depth_instance as _complete_depth_instance,
    compute_instance_heights as _compute_instance_heights,
    load_template_data as _load_template_data,
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

def infer_instance_order(instance_masks: list[np.ndarray], depth_food: np.ndarray) -> tuple[list[int], bool]:
    _log_info("enter infer_instance_order")
    """
    Xác định thứ tự Top -> Bottom dựa trên độ đè và độ sâu (Matching Notebook).
    """
    is_cycle = False
    n_masks = len(instance_masks)
    if n_masks == 0:
        return [], False

    graph: dict[int, set[int]] = {i: set() for i in range(n_masks)}

    for i in range(n_masks):
        for j in range(i + 1, n_masks):
            mask_a = instance_masks[i]
            mask_b = instance_masks[j]

            if np.array_equal(mask_a, mask_b):
                graph[i].add(j)
                continue

            overlap = mask_a & mask_b
            if not np.any(overlap):
                continue

            a_only = mask_a & (~mask_b)
            b_only = mask_b & (~mask_a)

            if np.sum(a_only) == 0:
                graph[i].add(j)
                continue
            if np.sum(b_only) == 0:
                graph[j].add(i)
                continue

            min_pixels = 10
            if np.sum(a_only) > min_pixels and np.sum(b_only) > min_pixels:
                d_a = np.max(depth_food[a_only])
                d_b = np.max(depth_food[b_only])
                if d_a < d_b:
                    graph[i].add(j)
                elif d_a > d_b:
                    graph[j].add(i)
                else:
                    graph[i].add(j)

    indeg = {i: 0 for i in range(n_masks)}
    for u in graph:
        for v in graph[u]:
            indeg[v] += 1

    queue = [i for i in range(n_masks) if indeg[i] == 0]
    queue.sort()
    order: list[int] = []
    while queue:
        u = queue.pop(0)
        order.append(u)
        for v in sorted(list(graph[u])):
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)

    if len(order) < n_masks:
        remaining = [i for i in range(n_masks) if i not in order]
        order.extend(remaining)
        is_cycle = True
        
    return order, is_cycle

def complete_depth_instance(mask, depth_food, depth_below, missing_mask):
    _log_info("enter complete_depth_instance")
    return _complete_depth_instance(mask, depth_food, depth_below, missing_mask)

def compute_instance_heights(instance_masks, sorted_idx, instance_depth_maps, depth_plate):
    _log_info("enter compute_instance_heights")
    return _compute_instance_heights(instance_masks, sorted_idx, instance_depth_maps, depth_plate)

def load_template_data(template_dir: str, plate_type: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    _log_info("enter load_template_data")
    return _load_template_data(template_dir, plate_type)

def get_contour(mask: np.ndarray):
    _log_info("enter get_contour")
    return _get_contour(mask)

def fit_shape(mask: np.ndarray, plate_type: str):
    _log_info("enter fit_shape")
    return _fit_shape(mask, plate_type)

def build_affine(c_ref, c_cur, angle_deg, sx=1.0, sy=1.0) -> np.ndarray:
    _log_info("enter build_affine")
    return _build_affine(c_ref, c_cur, angle_deg, sx=sx, sy=sy)

def estimate_affine_from_shape(ref_mask: np.ndarray, cur_mask: np.ndarray, plate_type: str) -> np.ndarray | None:
    _log_info("enter estimate_affine_from_shape")
    return _estimate_affine_from_shape(ref_mask, cur_mask, plate_type)