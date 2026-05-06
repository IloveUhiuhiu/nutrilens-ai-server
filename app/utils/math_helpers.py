from __future__ import annotations
import os
import numpy as np
from collections import deque
from scipy.ndimage import binary_erosion
import cv2
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

def load_template_data(template_dir: str, plate_type: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Nạp dữ liệu depth và mask từ thư mục template dựa trên loại đĩa.
    """
    target_dir = os.path.join(template_dir, plate_type)
    if not os.path.exists(target_dir):
        print("Khong toi tai : ", target_dir)
        return None, None

    try:
        files = os.listdir(target_dir)
        depth_file = [f for f in files if 'depth' in f.lower()][0]
        mask_file = [f for f in files if 'mask' in f.lower()][0]
        print("depth_file", depth_file)
        print("mask_file", mask_file)
        # Đọc depth (chia 100 để về đơn vị cm) và mask (grayscale)
        ref_depth = cv2.imread(os.path.join(target_dir, depth_file), cv2.IMREAD_UNCHANGED).astype(np.float32) / 100.0
        ref_mask = cv2.imread(os.path.join(target_dir, mask_file), cv2.IMREAD_GRAYSCALE)
        
        return ref_depth, ref_mask
    except Exception as e:
        print("Looi ne: ", e)
        return None, None

def get_contour(mask: np.ndarray):
    """Lấy đường bao lớn nhất của mask."""
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)

def fit_shape(mask: np.ndarray, plate_type: str):
    """Phân loại và khớp hình dạng học (Circle, Rect, Ellipse)."""
    cnt = get_contour(mask)
    if cnt is None or len(cnt) < 20:
        return None

    # Các loại đĩa tròn
    if plate_type in ["plate_small", "plate_plastic", "plate_round", "plate_ceramic", "bowl_metal"]:
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        return ("circle", np.array([cx, cy], dtype=np.float32), r)

    # Các loại đĩa vuông/chữ nhật
    if plate_type in ["plate_square", "plate_rectangular"]:
        (cx, cy), (w, h), angle = cv2.minAreaRect(cnt)
        return ("rect", np.array([cx, cy], dtype=np.float32), w, h, angle)

    # Đĩa hình oval
    if plate_type == "plate_oval" and len(cnt) >= 5:
        (cx, cy), (MA, ma), angle = cv2.fitEllipse(cnt)
        return ("ellipse", np.array([cx, cy], dtype=np.float32), MA, ma, angle)

    # Mặc định dùng Rectangle cho các loại khác
    (cx, cy), (w, h), angle = cv2.minAreaRect(cnt)
    return ("rect", np.array([cx, cy], dtype=np.float32), w, h, angle)

def build_affine(c_ref, c_cur, angle_deg, sx=1.0, sy=1.0) -> np.ndarray:
    """Xây dựng ma trận Affine 2x3 từ các tham số xoay, tỉ lệ và dịch chuyển."""
    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    
    # Ma trận Rotation & Scaling kết hợp
    # $M = \begin{bmatrix} sx \cdot \cos\theta & -sy \cdot \sin\theta & tx \\ sx \cdot \sin\theta & sy \cdot \cos\theta & ty \end{bmatrix}$
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
    S = np.array([[sx, 0], [0, sy]], dtype=np.float32)
    A = R @ S
    
    # Tính toán vector dịch chuyển (translation)
    t = c_cur - A @ c_ref
    return np.hstack([A, t.reshape(2, 1)]).astype(np.float32)

def estimate_affine_from_shape(ref_mask: np.ndarray, cur_mask: np.ndarray, plate_type: str) -> np.ndarray | None:
    """
    Ước tính ma trận biến đổi Affine giữa Template và thực tế dựa trên đặc điểm hình dạng.
    """
    ref = fit_shape(ref_mask, plate_type)
    cur = fit_shape(cur_mask, plate_type)
    
    if ref is None or cur is None:
        return None

    # Khớp hình tròn
    if ref[0] == "circle" and cur[0] == "circle":
        _, c_ref, r_ref = ref
        _, c_cur, r_cur = cur
        s = (r_cur / r_ref) if r_ref > 1e-6 else 1.0
        return build_affine(c_ref, c_cur, angle_deg=0.0, sx=s, sy=s)

    # Khớp hình chữ nhật
    if ref[0] == "rect" and cur[0] == "rect":
        _, c_ref, w_ref, h_ref, a_ref = ref
        _, c_cur, w_cur, h_cur, a_cur = cur
        sx = (w_cur / w_ref) if w_ref > 1e-6 else 1.0
        sy = (h_cur / h_ref) if h_ref > 1e-6 else 1.0
        return build_affine(c_ref, c_cur, angle_deg=(a_cur - a_ref), sx=sx, sy=sy)

    # Khớp hình Oval
    if ref[0] == "ellipse" and cur[0] == "ellipse":
        _, c_ref, MA_ref, ma_ref, a_ref = ref
        _, c_cur, MA_cur, ma_cur, a_cur = cur
        sx = (MA_cur / MA_ref) if MA_ref > 1e-6 else 1.0
        sy = (ma_cur / ma_ref) if ma_ref > 1e-6 else 1.0
        return build_affine(c_ref, c_cur, angle_deg=(a_cur - a_ref), sx=sx, sy=sy)

    return None