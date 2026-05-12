from __future__ import annotations
import numpy as np
from scipy.ndimage import binary_erosion
import logging

logger = logging.getLogger(__name__)

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
