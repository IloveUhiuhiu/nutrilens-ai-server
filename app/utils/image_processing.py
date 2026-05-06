from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
from scipy.ndimage import binary_erosion
from app.utils.math_helpers import estimate_affine_from_shape, load_template_data
from app.core.constants import MIN_PLATE_DEPTH_CM
import logging


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


def resize_with_padding(image: np.ndarray, size: int) -> np.ndarray:
    """Resize (nearest) and pad to a square size."""
    if image.size == 0:
        return image
    height, width = image.shape[:2]
    scale = min(size / height, size / width)
    new_h = max(1, int(height * scale))
    new_w = max(1, int(width * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((size, size) + image.shape[2:], dtype=image.dtype)
    top = (size - new_h) // 2
    left = (size - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    _log_info("enter decode_image_bytes")
    """
    Giải mã image bytes với tùy chọn định dạng màu đầu ra.
    
    Args:
        image_bytes: Dữ liệu ảnh dạng bytes.
        
    Returns:
        np.ndarray: Mảng numpy của ảnh hoặc mảng rỗng nếu lỗi.
    """
    if not image_bytes:
        _log_info("empty image_bytes branch")
        return np.zeros((0, 0, 3), dtype=np.uint8)

    # 1. Chuyển bytes sang numpy array 1 chiều
    nparr = np.frombuffer(image_bytes, np.uint8)
    
    # 2. Giải mã ảnh (OpenCV mặc định luôn trả về BGR)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        _log_info("image decode failed branch")
        return np.zeros((0, 0, 3), dtype=np.uint8)

    
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    


def crop_image(image: np.ndarray, bbox: list[float] | tuple[float, float, float, float]) -> np.ndarray:
    _log_info("enter crop_image")
    if image.size == 0:
        return image
    x1, y1, x2, y2 = [int(max(0, v)) for v in bbox]
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)
    return image[y1:y2, x1:x2]


def warp_affine(image: np.ndarray, matrix: np.ndarray, output_shape: Tuple[int, int]) -> np.ndarray:
    """Apply a simple affine warp using nearest neighbor sampling."""
    if image.size == 0:
        return image
    out_h, out_w = output_shape
    return cv2.warpAffine(image, matrix[:2], (out_w, out_h), flags=cv2.INTER_NEAREST)


def template_match(depth_map: np.ndarray, template: np.ndarray) -> Tuple[int, int]:
    """Return top-left offset for the best match (simple SSD)."""
    if depth_map.size == 0 or template.size == 0:
        return 0, 0
    h, w = template.shape
    best_score = float("inf")
    best_offset = (0, 0)
    for y in range(0, max(depth_map.shape[0] - h, 1)):
        for x in range(0, max(depth_map.shape[1] - w, 1)):
            patch = depth_map[y : y + h, x : x + w]
            score = np.sum((patch - template) ** 2)
            if score < best_score:
                best_score = score
                best_offset = (y, x)
    return best_offset


def inpaint_plate_depth(
    depth_map: np.ndarray,
    plate_mask: np.ndarray,
    food_mask: np.ndarray,
    plate_type: str | None,
    camera_h_ref: float,
    template_dir: str = "templates" # Đường dẫn template trên server
) -> np.ndarray:
    _log_info("enter inpaint_plate_depth")
    """
    Phiên bản inpaint chuyên sâu từ Notebook: Khớp hình dạng + Hiệu chỉnh Z + Lọc mịn.
    """
    img_h, img_w = depth_map.shape
    
    # 1. Lấy mẫu sạch (Anchors) để làm mốc độ sâu thực tế
    x_s, y_s, z_s = get_clean_plate_samples(depth_map, plate_mask, food_mask, camera_h_ref)

    num_anchors = len(x_s) if x_s is not None else 0
    
    # 2. Xử lý đĩa phẳng (Trường hợp đơn giản nhất)
    if plate_type in ["plate_flat", None]:
        _log_info("flat plate branch")
        depth_plate = np.full((img_h, img_w), camera_h_ref, dtype=np.float32)
        plate_z_level = np.median(z_s) if (x_s is not None and len(z_s) > 0) else camera_h_ref
        depth_plate[plate_mask > 0] = plate_z_level
        return depth_plate

    # 3. Load Template và tính toán Affine Transform (Xoay/Thu phóng)
    
    
    ref_depth, ref_mask = load_template_data(template_dir, plate_type)
    if ref_depth is None:
        _log_info("template load failed branch")
        # Fallback nếu không load được template
        depth_plate = np.full((img_h, img_w), camera_h_ref, dtype=np.float32)
        return depth_plate

    M = estimate_affine_from_shape(ref_mask, plate_mask, plate_type)
    
    if M is not None:
        _log_info("affine fit branch")
        depth_warped = cv2.warpAffine(ref_depth, M, (img_w, img_h), flags=cv2.INTER_LINEAR)
        
        # 4. Hiệu chỉnh Z-Offset (Khớp cao độ thực tế)
        if x_s is not None and len(x_s) > 15:
            ref_vals = depth_warped[y_s, x_s]
            valid_idx = (ref_vals > 0) & (plate_mask[y_s, x_s] > 0)
            if np.any(valid_idx):
                z_offset = np.median(z_s[valid_idx] - ref_vals[valid_idx])
                depth_warped += z_offset
    else:
        _log_info("affine fit failed branch")
        # Fallback 2: Nếu không khớp được hình dạng
        depth_warped = np.full((img_h, img_w), camera_h_ref - 0.5, dtype=np.float32)

    # 5. Fusion & Smoothing
    depth_plate = np.full((img_h, img_w), camera_h_ref, dtype=np.float32)
    valid_mask = (plate_mask > 0) & (depth_warped > 0)
    depth_plate[valid_mask] = depth_warped[valid_mask]
    
    # Ưu tiên các điểm anchor thực tế
    if x_s is not None:
        depth_plate[y_s, x_s] = z_s
        
    # Lọc mịn bề mặt để loại bỏ nhiễu răng cưa
    depth_plate = cv2.bilateralFilter(depth_plate.astype(np.float32), d=7, sigmaColor=0.3, sigmaSpace=10)
    depth_plate[plate_mask == 0] = camera_h_ref
    if np.any(plate_mask > 0):
        mean_plate_z = np.mean(depth_plate[plate_mask > 0])
    return depth_plate


def fill_outside_food_bilateral(height_map: np.ndarray, food_mask: np.ndarray, plate_mask: np.ndarray) -> np.ndarray:
    outside_mask = (food_mask > 0) & (plate_mask == 0)
    valid_mask = (food_mask > 0) & (plate_mask > 0)

    temp = height_map.copy()
    temp[~valid_mask] = 0

    filtered = cv2.bilateralFilter(
        temp.astype(np.float32),
        d=9,
        sigmaColor=25,
        sigmaSpace=9,
    )

    height_map[outside_mask] = filtered[outside_mask]
    return height_map


def flatten_instances(ingredient_masks: dict[str, list[np.ndarray]]) -> tuple[list[np.ndarray], list[str]]:
    instance_masks = []
    instance_labels = []
    for name, masks in ingredient_masks.items():
        for mask in masks:
            instance_masks.append(mask.astype(bool))
            instance_labels.append(name)
    return instance_masks, instance_labels


def get_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return xs.min(), ys.min(), xs.max(), ys.max()


def bbox_overlap(b1: tuple[int, int, int, int] | None, b2: tuple[int, int, int, int] | None) -> bool:
    if b1 is None or b2 is None:
        return False
    x1_min, y1_min, x1_max, y1_max = b1
    x2_min, y2_min, x2_max, y2_max = b2
    return not (
        x1_max < x2_min
        or x2_max < x1_min
        or y1_max < y2_min
        or y2_max < y1_min
    )


def merge_overlapping_instances(instance_masks: list[np.ndarray], overlap_thresh: float = 0.01) -> list[np.ndarray]:
    n_masks = len(instance_masks)
    if n_masks == 0:
        return []
    bboxes = [get_bbox(mask) for mask in instance_masks]
    adj: list[list[int]] = [[] for _ in range(n_masks)]

    for i in range(n_masks):
        for j in range(i + 1, n_masks):
            if not bbox_overlap(bboxes[i], bboxes[j]):
                continue
            mask_a = instance_masks[i]
            mask_b = instance_masks[j]
            inter = np.logical_and(mask_a, mask_b).sum()
            if inter == 0:
                continue
            union = np.logical_or(mask_a, mask_b).sum()
            if union == 0:
                continue
            iou = inter / union
            if iou > overlap_thresh:
                adj[i].append(j)
                adj[j].append(i)

    visited = [False] * n_masks
    groups: list[list[int]] = []
    for i in range(n_masks):
        if visited[i]:
            continue
        queue = [i]
        visited[i] = True
        group = [i]
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
                    group.append(v)
        groups.append(group)

    merged_instances = []
    for group in groups:
        merged_mask = np.zeros_like(instance_masks[0], dtype=bool)
        for idx in group:
            merged_mask |= instance_masks[idx]
        if np.sum(merged_mask) > 20:
            merged_instances.append(merged_mask)
    return merged_instances


def merge_masks_and_instances(
    segmentation_results: dict,
    food_bboxes: dict,
    full_image_shape: tuple[int, int] | tuple[int, int, int],
    overlap_thresh: float = 0.01,
) -> tuple[dict[str, np.ndarray], dict[str, list[np.ndarray]]]:
    _log_info("enter merge_masks_and_instances")
    height, width = full_image_shape[:2]
    instance_ingredient_masks: dict[str, list[np.ndarray]] = {}

    for crop_id, ingredient_masks in segmentation_results.items():
        if crop_id not in food_bboxes:
            continue
        x1, y1, x2, y2 = food_bboxes[crop_id]
        bbox_h, bbox_w = y2 - y1, x2 - x1
        for ing_name, crop_mask in ingredient_masks:
            ing_name = ing_name.lower().strip()
            instance_ingredient_masks.setdefault(ing_name, [])
            binary = (crop_mask > 0).astype(np.uint8)
            if binary.shape[:2] != (bbox_h, bbox_w):
                binary = cv2.resize(binary, (bbox_w, bbox_h), interpolation=cv2.INTER_NEAREST)
            full_mask = np.zeros((height, width), dtype=bool)
            full_mask[y1:y2, x1:x2] = binary.astype(bool)
            if np.sum(full_mask) < 20:
                continue
            instance_ingredient_masks[ing_name].append(full_mask)

    for ing_name in instance_ingredient_masks:
        instance_ingredient_masks[ing_name] = merge_overlapping_instances(
            instance_ingredient_masks[ing_name], overlap_thresh=overlap_thresh
        )

    global_ingredient_masks: dict[str, np.ndarray] = {}
    for ing_name, masks in instance_ingredient_masks.items():
        global_mask = np.zeros((height, width), dtype=np.uint8)
        for mask in masks:
            global_mask = np.maximum(global_mask, mask.astype(np.uint8))
        global_ingredient_masks[ing_name] = global_mask

    return global_ingredient_masks, instance_ingredient_masks


def get_clean_plate_samples(
    depth_map: np.ndarray,
    plate_mask: np.ndarray,
    food_mask: np.ndarray,
    camera_h_ref: float,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    _log_info("enter get_clean_plate_samples")
    kernel_food = np.ones((7, 7), np.uint8)
    food_dilated = cv2.dilate(food_mask.astype(np.uint8), kernel_food, iterations=1)

    kernel_plate = np.ones((5, 5), np.uint8)
    plate_eroded = cv2.erode(plate_mask.astype(np.uint8), kernel_plate, iterations=1)

    is_valid_zone = (
        (food_dilated == 0)
        & (depth_map >= MIN_PLATE_DEPTH_CM)
        & (depth_map <= camera_h_ref)
        & (~np.isnan(depth_map))
        & (plate_eroded > 0)
    )

    y_idx, x_idx = np.where(is_valid_zone)
    z_values = depth_map[y_idx, x_idx]

    if len(z_values) < 50:
        _log_info("insufficient clean samples branch")
        return None, None, None
    return x_idx, y_idx, z_values


def complete_depth_instance(
    mask: np.ndarray,
    depth_food: np.ndarray,
    depth_below: np.ndarray,
    missing_mask: np.ndarray,
) -> np.ndarray:
    reliable_mask = mask & (~missing_mask)
    core = binary_erosion(reliable_mask, iterations=2)
    if np.sum(core) < 10:
        core = reliable_mask if np.sum(reliable_mask) >= 10 else mask

    ys, xs = np.where(core)
    z = depth_food[core]
    if len(z) < 10:
        return depth_food.copy()

    A = np.c_[np.ones(len(xs)), xs, ys, xs**2, xs * ys, ys**2]
    try:
        coeffs = np.linalg.lstsq(A, z, rcond=None)[0]
        height, width = depth_food.shape
        yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        pred = (
            coeffs[0]
            + coeffs[1] * xx
            + coeffs[2] * yy
            + coeffs[3] * (xx**2)
            + coeffs[4] * (xx * yy)
            + coeffs[5] * (yy**2)
        )
        pred = np.maximum(pred, depth_food)
        pred = np.minimum(pred, depth_below)
    except Exception:
        return depth_food.copy()

    depth_completed = depth_food.copy()
    fill_zone = mask & missing_mask
    depth_completed[fill_zone] = pred[fill_zone]
    return depth_completed
