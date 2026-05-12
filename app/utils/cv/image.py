from __future__ import annotations

from typing import Tuple
import cv2
import numpy as np
from app.utils.common import get_logger

logger = get_logger(__name__)

def resize_with_padding(image: np.ndarray, size: int) -> np.ndarray:
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
    logger.info("enter decode_image_bytes")
    if not image_bytes:
        logger.info("empty image_bytes branch")
        return np.zeros((0, 0, 3), dtype=np.uint8)
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        logger.info("image decode failed branch")
        return np.zeros((0, 0, 3), dtype=np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def crop_image(image: np.ndarray, bbox: list[float] | tuple[float, float, float, float]) -> np.ndarray:
    logger.info("enter crop_image")
    if image.size == 0:
        return image
    x1, y1, x2, y2 = [int(max(0, v)) for v in bbox]
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)
    return image[y1:y2, x1:x2]

def warp_affine(image: np.ndarray, matrix: np.ndarray, output_shape: Tuple[int, int]) -> np.ndarray:
    if image.size == 0:
        return image
    out_h, out_w = output_shape
    return cv2.warpAffine(image, matrix[:2], (out_w, out_h), flags=cv2.INTER_NEAREST)

def template_match(depth_map: np.ndarray, template: np.ndarray) -> Tuple[int, int]:
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
