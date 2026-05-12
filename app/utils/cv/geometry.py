from __future__ import annotations

import numpy as np
import cv2
from app.utils.common import get_logger

logger = get_logger(__name__)

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

def get_contour(mask: np.ndarray):
    logger.info("enter get_contour")
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)

def fit_shape(mask: np.ndarray, plate_type: str):
    logger.info("enter fit_shape")
    cnt = get_contour(mask)
    if cnt is None or len(cnt) < 20:
        return None
    if plate_type in ["plate_small", "plate_plastic", "plate_round", "plate_ceramic", "bowl_metal"]:
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        return ("circle", np.array([cx, cy], dtype=np.float32), r)
    if plate_type in ["plate_square", "plate_rectangular"]:
        (cx, cy), (w, h), angle = cv2.minAreaRect(cnt)
        return ("rect", np.array([cx, cy], dtype=np.float32), w, h, angle)
    if plate_type == "plate_oval" and len(cnt) >= 5:
        (cx, cy), (MA, ma), angle = cv2.fitEllipse(cnt)
        return ("ellipse", np.array([cx, cy], dtype=np.float32), MA, ma, angle)
    (cx, cy), (w, h), angle = cv2.minAreaRect(cnt)
    return ("rect", np.array([cx, cy], dtype=np.float32), w, h, angle)

def build_affine(c_ref, c_cur, angle_deg, sx=1.0, sy=1.0) -> np.ndarray:
    logger.info("enter build_affine")
    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
    S = np.array([[sx, 0], [0, sy]], dtype=np.float32)
    A = R @ S
    t = c_cur - A @ c_ref
    return np.hstack([A, t.reshape(2, 1)]).astype(np.float32)

def estimate_affine_from_shape(ref_mask: np.ndarray, cur_mask: np.ndarray, plate_type: str) -> np.ndarray | None:
    logger.info("enter estimate_affine_from_shape")
    ref = fit_shape(ref_mask, plate_type)
    cur = fit_shape(cur_mask, plate_type)
    if ref is None or cur is None:
        return None
    if ref[0] == "circle" and cur[0] == "circle":
        _, c_ref, r_ref = ref
        _, c_cur, r_cur = cur
        s = (r_cur / r_ref) if r_ref > 1e-6 else 1.0
        return build_affine(c_ref, c_cur, angle_deg=0.0, sx=s, sy=s)
    if ref[0] == "rect" and cur[0] == "rect":
        _, c_ref, w_ref, h_ref, a_ref = ref
        _, c_cur, w_cur, h_cur, a_cur = cur
        sx = (w_cur / w_ref) if w_ref > 1e-6 else 1.0
        sy = (h_cur / h_ref) if h_ref > 1e-6 else 1.0
        return build_affine(c_ref, c_cur, angle_deg=(a_cur - a_ref), sx=sx, sy=sy)
    if ref[0] == "ellipse" and cur[0] == "ellipse":
        _, c_ref, MA_ref, ma_ref, a_ref = ref
        _, c_cur, MA_cur, ma_cur, a_cur = cur
        sx = (MA_cur / MA_ref) if MA_ref > 1e-6 else 1.0
        sy = (ma_cur / ma_ref) if ma_ref > 1e-6 else 1.0
        return build_affine(c_ref, c_cur, angle_deg=(a_cur - a_ref), sx=sx, sy=sy)
    return None
