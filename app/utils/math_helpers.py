from __future__ import annotations

import numpy as np
from collections import deque
from scipy.ndimage import binary_erosion

def kahn_topological_sort(graph: dict[str, list[str]]) -> list[str]:
    """Thuật toán Kahn chuẩn cho các node dạng chuỗi."""
    indegree = {node: 0 for node in graph}
    for deps in graph.values():
        for dep in deps:
            indegree[dep] = indegree.get(dep, 0) + 1
            
    queue = deque([node for node, deg in indegree.items() if deg == 0])
    order = []
    
    while queue:
        node = queue.popleft()
        order.append(node)
        for dep in graph.get(node, []):
            indegree[dep] -= 1
            if indegree[dep] == 0:
                queue.append(dep)
                
    if len(order) != len(indegree):
        raise ValueError("Cycle detected in dependency graph")
    return order

def infer_instance_order(instance_masks: list[np.ndarray], depth_food: np.ndarray) -> tuple[list[int], bool]:
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
    """
    Sử dụng Polynomial Fitting bậc 2 để ước tính bề mặt bị che khuất.
    """
    reliable_mask = mask & (~missing_mask)
    
    # Erosion để lấy lõi tin cậy
    core = binary_erosion(reliable_mask, iterations=2)
    if np.sum(core) < 10:
        core = reliable_mask if np.sum(reliable_mask) >= 10 else mask

    ys, xs = np.where(core)
    z = depth_food[core]

    if len(z) < 10:
        return depth_food.copy()

    # Fit Polynomial bậc 2: [1, x, y, x^2, xy, y^2]
    A = np.c_[np.ones(len(xs)), xs, ys, xs**2, xs*ys, ys**2]
    try:
        coeffs = np.linalg.lstsq(A, z, rcond=None)[0]
        H, W = depth_food.shape
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        
        pred = (coeffs[0] + coeffs[1]*xx + coeffs[2]*yy +
                coeffs[3]*(xx**2) + coeffs[4]*(xx*yy) + coeffs[5]*(yy**2))
        
        # Ràng buộc vật lý: Nằm giữa vật che và mặt sàn bên dưới
        pred = np.maximum(pred, depth_food)  
        pred = np.minimum(pred, depth_below) 
        
        depth_completed = depth_food.copy()
        fill_zone = mask & missing_mask
        depth_completed[fill_zone] = pred[fill_zone]
        
        return depth_completed
    except:
        return depth_food.copy()

def compute_instance_heights(instance_masks, sorted_idx, instance_depth_maps, depth_plate):
    """
    Tính chiều cao từng lớp bằng logic mặt sàn động (Bottom -> Top).
    """
    H, W = depth_plate.shape
    height_instances = [np.zeros((H, W), dtype=np.float32) for _ in range(len(instance_masks))]
    current_floor = depth_plate.copy()

    # Duyệt ngược từ đáy lên đỉnh
    for idx in reversed(sorted_idx):
        mask_i = instance_masks[idx]
        surface_i = instance_depth_maps[idx] 
        
        # Chiều cao = Sàn hiện tại - Bề mặt vật i
        h_i_full = current_floor - surface_i
        height_instances[idx] = np.where(mask_i, np.maximum(h_i_full, 0), 0)
        
        # Cập nhật sàn mới cho lớp trên
        current_floor[mask_i] = surface_i[mask_i]

    return height_instances

def area_adaptive(pixel_area_ref: float, depth: float, camera_height_ref: float) -> float:
    """Công thức tính diện tích thích nghi bù trừ phối cảnh."""
    if camera_height_ref <= 0 or pixel_area_ref <= 0:
        raise ValueError("camera_height_ref and pixel_area_ref must be positive")
    # Sử dụng bình phương tỉ lệ độ sâu CM
    return pixel_area_ref * (depth / camera_height_ref) ** 2