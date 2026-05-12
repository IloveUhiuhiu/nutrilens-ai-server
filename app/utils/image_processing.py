from __future__ import annotations

from app.utils.image import (
    resize_with_padding,
    decode_image_bytes,
    crop_image,
    warp_affine,
    template_match,
    fill_outside_food_bilateral,
)
from app.utils.depth import (
    inpaint_plate_depth,
    get_clean_plate_samples,
    complete_depth_instance,
)
from app.utils.segmentation import (
    flatten_instances,
    merge_overlapping_instances,
    merge_masks_and_instances,
)
from app.utils.geometry import get_bbox, bbox_overlap

__all__ = [
    "resize_with_padding",
    "decode_image_bytes",
    "crop_image",
    "warp_affine",
    "template_match",
    "fill_outside_food_bilateral",
    "inpaint_plate_depth",
    "get_clean_plate_samples",
    "complete_depth_instance",
    "flatten_instances",
    "merge_overlapping_instances",
    "merge_masks_and_instances",
    "get_bbox",
    "bbox_overlap",
]
)
from app.utils.depth import (
    inpaint_plate_depth,
    get_clean_plate_samples,
    complete_depth_instance,
)
from app.utils.segmentation import (
    flatten_instances,
    merge_overlapping_instances,
    merge_masks_and_instances,
)
from app.utils.geometry import get_bbox, bbox_overlap


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
