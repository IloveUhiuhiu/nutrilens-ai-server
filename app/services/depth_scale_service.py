from __future__ import annotations

import numpy as np

from app.services.base import ServiceBase

# Half-size (px) of the local window sampled around the exact pixel the AR
# raycast measured. Small enough to stay a true point-estimate, large enough
# to absorb single-pixel depth-model noise.
ANCHOR_PATCH_RADIUS = 5
MIN_PATCH_SAMPLES = 8

# Fallback when the AR-measured pixel itself isn't usable (out of frame, or
# not enough valid depth around it) — whole clean-plate-region median.
MIN_ANCHOR_SAMPLES = 50

# Minimum table-region pixels to trust the table-derived camera_h_ref below
# — same order as MIN_ANCHOR_SAMPLES, since this is also a median over a
# (much larger) masked region.
MIN_TABLE_SAMPLES = 50


class DepthScaleResolver(ServiceBase):
    """Hybrid Depth Pipeline — scale resolution.

    CASE A: anchors a relative/metric depth map to an absolute scale using the
    measured camera-to-object distance (AR raycast onto the plate/table plane),
    so the metric depth — and therefore the estimated volume — reflects the real
    geometry instead of a fixed `max_depth` assumption.

    The AR distance can be measured at several candidate points per frame —
    native ring-searches outward for screen pixels currently landing on the
    detected table/plate plane (see ArKitPlatformView.swift /
    ArPlatformView.kt) and reports all of them, since it has no way to know
    on-device which pixel will turn out to overlap food once segmentation
    runs server-side. This resolver tries each candidate in priority order
    and uses the first whose patch doesn't land on food — not an aggregate
    over the whole plate, which has no defined spatial correspondence to
    where any single distance was actually measured.
    ``scale = distance / depth_map[chosen_candidate_pixel]`` (median of a
    small patch around it, for robustness against single-pixel model noise).
    """

    service_name = "depth_scale"

    def anchor_with_distance(
        self,
        depth_map: np.ndarray,
        plate_mask: np.ndarray,
        food_mask: np.ndarray,
        anchor_distance_cm: float | None,
        anchor_pixel: tuple[float, float] | None = None,
        anchor_candidates: list[tuple[float, float, float]] | None = None,
    ) -> tuple[np.ndarray, float, str]:
        """Đầu vào: depth (cm), mask đĩa/thức ăn, khoảng cách tuyệt đối (cm)
        (đường cũ, 1 điểm duy nhất - giữ để tương thích app cũ), hoặc
        `anchor_candidates` (đường mới): danh sách (pixel_x, pixel_y,
        distance_cm) mà native đã raycast được trong frame đó, xếp theo thứ
        tự ưu tiên của native (ngoài vào trong) - native không biết pixel nào
        sẽ là food (segment chỉ chạy ở server, sau khi đã chụp), nên không
        thể tự chọn "điểm tốt nhất" lúc capture; việc chọn được dồn về đây,
        sau khi đã có food_mask thật.
        Đầu ra: (depth đã anchor, scale, nguồn scale)."""
        candidates = self._normalize_candidates(anchor_distance_cm, anchor_pixel, anchor_candidates)
        if not candidates:
            return depth_map, 1.0, "da2_metric"

        reference, distance_cm, source = self._reference_depth(depth_map, plate_mask, food_mask, candidates)
        if reference is None or reference <= 0:
            self._log_info("anchor: no usable reference depth, keeping DA2 metric")
            return depth_map, 1.0, "da2_metric"

        scale = float(distance_cm) / reference
        self._log_info(
            f"anchor[{source}]: scale={scale:.4f} (ref={reference:.2f}cm -> {distance_cm:.2f}cm)"
        )
        return depth_map * scale, scale, "ar_absolute"

    def _normalize_candidates(
        self,
        anchor_distance_cm: float | None,
        anchor_pixel: tuple[float, float] | None,
        anchor_candidates: list[tuple[float, float, float]] | None,
    ) -> list[tuple[float | None, float | None, float]]:
        """Quy về 1 danh sách candidate duy nhất, ưu tiên `anchor_candidates`
        (đường mới, nhiều điểm) nếu có; nếu không có thì dùng đường cũ (1
        điểm, hoặc chỉ có distance không pixel - app rất cũ)."""
        if anchor_candidates:
            return [(x, y, d) for (x, y, d) in anchor_candidates if d and d > 0]
        if anchor_distance_cm and anchor_distance_cm > 0:
            px, py = anchor_pixel if anchor_pixel is not None else (None, None)
            return [(px, py, float(anchor_distance_cm))]
        return []

    def _reference_depth(
        self,
        depth_map: np.ndarray,
        plate_mask: np.ndarray,
        food_mask: np.ndarray,
        candidates: list[tuple[float | None, float | None, float]],
    ) -> tuple[float | None, float | None, str]:
        """Thử lần lượt từng candidate theo đúng thứ tự được gửi lên, trả về
        candidate ĐẦU TIÊN đọc được patch hợp lệ (không rơi vào food).
        Nếu không candidate nào dùng được, fallback sang median vùng đĩa
        sạch, dùng distance_cm của candidate đầu tiên (gần nguồn nhất theo
        thứ tự ưu tiên của native)."""
        valid = np.isfinite(depth_map) & (depth_map > 0)
        h, w = depth_map.shape[:2]
        plate_clean = plate_mask.astype(bool) & ~food_mask.astype(bool)
        non_food = ~food_mask.astype(bool)

        for idx, (px, py, distance_cm) in enumerate(candidates):
            if px is None or py is None:
                continue
            col = int(round(px))
            row = int(round(py))
            if not (0 <= col < w and 0 <= row < h):
                self._log_info(f"anchor candidate[{idx}] out of bounds: pixel=({col},{row}) image=({w}x{h})")
                continue

            r0, r1 = max(0, row - ANCHOR_PATCH_RADIUS), min(h, row + ANCHOR_PATCH_RADIUS + 1)
            c0, c1 = max(0, col - ANCHOR_PATCH_RADIUS), min(w, col + ANCHOR_PATCH_RADIUS + 1)
            patch_valid = valid[r0:r1, c0:c1]
            patch_depth = depth_map[r0:r1, c0:c1]

            # Đọc bất kỳ pixel không-phải-food trong patch (bàn hoặc đĩa) —
            # bàn là mặt tham chiếu hợp lệ tương đương đĩa, vì AR raycast
            # luôn đo trên một mặt phẳng ngang đã được xác nhận (không phải
            # food). Chỉ loại trừ food, vì bề mặt food nhô cao hơn mặt sàn
            # nên không phản ánh đúng khoảng cách AR đã đo tại candidate này.
            patch_non_food = non_food[r0:r1, c0:c1]
            samples = patch_depth[patch_valid & patch_non_food]
            if samples.size >= MIN_PATCH_SAMPLES:
                return float(np.median(samples)), distance_cm, f"anchor_candidate[{idx}]"
            self._log_info(
                f"anchor candidate[{idx}] miss: pixel=({col},{row}) image=({w}x{h}) "
                f"patch_valid={int(patch_valid.sum())} patch_non_food={int(patch_non_food.sum())} "
                f"samples={samples.size} (need {MIN_PATCH_SAMPLES})"
            )

        # Fallback: không candidate nào rơi vào vùng không-phải-food — median
        # cả vùng đĩa sạch là proxy thô hơn nhưng vẫn hợp lý, vì đĩa được coi
        # là tương đối phẳng. Dùng distance_cm của candidate đầu tiên (ưu
        # tiên cao nhất theo thứ tự native gửi lên).
        samples = depth_map[plate_clean & valid]
        fallback_distance = candidates[0][2] if candidates else None
        if samples.size < MIN_ANCHOR_SAMPLES or fallback_distance is None:
            return None, None, "insufficient"
        return float(np.median(samples)), fallback_distance, "plate_median_fallback"

    def derive_table_height(
        self,
        depth_map: np.ndarray,
        plate_mask: np.ndarray,
        food_mask: np.ndarray,
        plate_detected: bool = True,
    ) -> float | None:
        """Suy ra camera_h_ref thực tế từ vùng mặt bàn trên depth map ĐÃ
        SCALE — tách biệt hoàn toàn về không gian với anchor pixel dùng để
        tính scale: anchor đọc 1 patch nhỏ gần đúng điểm AR đo; ở đây đọc
        median cả vùng bàn nền quan sát được.

        plate_detected=False nghĩa là YOLO không tìm thấy đĩa — lúc đó
        `plate_mask` đầu vào đã bị nutrition_pipeline.py ghi đè thành toàn
        khung (ones) để phục vụ inpaint_plate_depth, KHÔNG phản ánh đĩa
        thật. Theo đúng quy ước "không có đĩa thì coi món ăn đặt trực tiếp
        trên mặt bàn" đã có sẵn ở đó, trường hợp này coi TOÀN BỘ phần
        không-phải-thức-ăn là mặt bàn, không lấy giao với plate_mask (sẽ
        rỗng nếu lấy giao).

        Trả None nếu mặt bàn không đủ pixel hợp lệ (vd thức ăn/đĩa chiếm
        toàn khung) — caller tự fallback về giá trị cũ."""
        valid = np.isfinite(depth_map) & (depth_map > 0)
        if plate_detected:
            table_mask = ~plate_mask.astype(bool) & ~food_mask.astype(bool)
        else:
            table_mask = ~food_mask.astype(bool)
        samples = depth_map[table_mask & valid]
        if samples.size < MIN_TABLE_SAMPLES:
            return None
        return float(np.median(samples))


__all__ = ["DepthScaleResolver"]
