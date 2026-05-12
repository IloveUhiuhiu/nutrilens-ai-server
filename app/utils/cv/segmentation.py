from __future__ import annotations

import numpy as np
import cv2
from app.utils.cv.geometry import get_bbox, bbox_overlap

def flatten_instances(ingredient_masks: dict[str, list[np.ndarray]]) -> tuple[list[np.ndarray], list[str]]:
    instance_masks = []
    instance_labels = []
    for name, masks in ingredient_masks.items():
        for mask in masks:
            instance_masks.append(mask.astype(bool))
            instance_labels.append(name)
    return instance_masks, instance_labels

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
