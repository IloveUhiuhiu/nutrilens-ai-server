from __future__ import annotations

import os
import numpy as np
import cv2
from scipy.ndimage import binary_erosion
from app.core.constants import MIN_PLATE_DEPTH_CM
from app.utils.common import get_logger
from app.utils.geometry import estimate_affine_from_shape

logger = get_logger(__name__)

def load_template_data(template_dir: str, plate_type: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    logger.info("enter load_template_data")
    target_dir = os.path.join(template_dir, plate_type)
    if not os.path.exists(target_dir):
        logger.info(f"template directory not found: {target_dir}")
        return None, None
    try:
        files = os.listdir(target_dir)
        depth_file = [f for f in files if "depth" in f.lower()][0]
        mask_file = [f for f in files if "mask" in f.lower()][0]
        ref_depth = cv2.imread(os.path.join(target_dir, depth_file), cv2.IMREAD_UNCHANGED).astype(np.float32) / 100.0
        ref_mask = cv2.imread(os.path.join(target_dir, mask_file), cv2.IMREAD_GRAYSCALE)
        return ref_depth, ref_mask
    except Exception as exc:
        logger.error(f"load_template_data exception: {exc}")
        return None, None

def get_clean_plate_samples(
    depth_map: np.ndarray,
    plate_mask: np.ndarray,
    food_mask: np.ndarray,
    camera_h_ref: float,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    logger.info("enter get_clean_plate_samples")
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
        logger.info("insufficient clean samples branch")
        return None, None, None
    return x_idx, y_idx, z_values

def inpaint_plate_depth(
    depth_map: np.ndarray,
    plate_mask: np.ndarray,
    food_mask: np.ndarray,
    plate_type: str | None,
    camera_h_ref: float,
    template_dir: str = "templates",
) -> np.ndarray:
    logger.info("enter inpaint_plate_depth")
    img_h, img_w = depth_map.shape
    x_s, y_s, z_s = get_clean_plate_samples(depth_map, plate_mask, food_mask, camera_h_ref)
    if plate_type in ["plate_flat", None]:
        logger.info("flat plate branch")
        depth_plate = np.full((img_h, img_w), camera_h_ref, dtype=np.float32)
        plate_z_level = np.median(z_s) if (x_s is not None and len(z_s) > 0) else camera_h_ref
        depth_plate[plate_mask > 0] = plate_z_level
        return depth_plate
    ref_depth, ref_mask = load_template_data(template_dir, plate_type)
    if ref_depth is None:
        logger.info("template load failed branch")
        return np.full((img_h, img_w), camera_h_ref, dtype=np.float32)
    M = estimate_affine_from_shape(ref_mask, plate_mask, plate_type)
    if M is not None:
        logger.info("affine fit branch")
        depth_warped = cv2.warpAffine(ref_depth, M, (img_w, img_h), flags=cv2.INTER_LINEAR)
        if x_s is not None and len(x_s) > 15:
            ref_vals = depth_warped[y_s, x_s]
            valid_idx = (ref_vals > 0) & (plate_mask[y_s, x_s] > 0)
            if np.any(valid_idx):
                z_offset = np.median(z_s[valid_idx] - ref_vals[valid_idx])
                depth_warped += z_offset
    else:
        logger.info("affine fit failed branch")
        depth_warped = np.full((img_h, img_w), camera_h_ref - 0.5, dtype=np.float32)
    depth_plate = np.full((img_h, img_w), camera_h_ref, dtype=np.float32)
    valid_mask = (plate_mask > 0) & (depth_warped > 0)
    depth_plate[valid_mask] = depth_warped[valid_mask]
    if x_s is not None:
        depth_plate[y_s, x_s] = z_s
    depth_plate = cv2.bilateralFilter(depth_plate.astype(np.float32), d=7, sigmaColor=0.3, sigmaSpace=10)
    depth_plate[plate_mask == 0] = camera_h_ref
    return depth_plate

def complete_depth_instance(
    mask: np.ndarray,
    depth_food: np.ndarray,
    depth_below: np.ndarray,
    missing_mask: np.ndarray,
) -> np.ndarray:
    logger.info("enter complete_depth_instance")
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

def compute_instance_heights(instance_masks, sorted_idx, instance_depth_maps, depth_plate):
    logger.info("enter compute_instance_heights")
    H, W = depth_plate.shape
    height_instances = [np.zeros((H, W), dtype=np.float32) for _ in range(len(instance_masks))]
    current_floor = depth_plate.copy()
    for idx in reversed(sorted_idx):
        mask_i = instance_masks[idx]
        surface_i = instance_depth_maps[idx]
        h_i_full = current_floor - surface_i
        height_instances[idx] = np.where(mask_i, np.maximum(h_i_full, 0), 0)
        current_floor[mask_i] = surface_i[mask_i]
    return height_instances
