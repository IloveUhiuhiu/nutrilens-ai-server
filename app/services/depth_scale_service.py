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


class DepthScaleResolver(ServiceBase):
    """Hybrid Depth Pipeline — scale resolution.

    CASE A: anchors a relative/metric depth map to an absolute scale using the
    measured camera-to-object distance (AR raycast onto the plate/table plane),
    so the metric depth — and therefore the estimated volume — reflects the real
    geometry instead of a fixed `max_depth` assumption.

    The AR distance is measured at one specific point: wherever the native
    anchor search found a pixel currently landing on the detected table/plate
    plane (no longer always the principal point — see ArKitPlatformView.swift
    / ArPlatformView.kt). The anchor must read the depth model's prediction at
    that *same* pixel — not an aggregate over the whole plate, which has no
    defined spatial correspondence to where the distance was actually
    measured. ``scale = distance / depth_map[anchor_pixel]`` (median of a
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
    ) -> tuple[np.ndarray, float, str]:
        """Đầu vào: depth (cm), mask đĩa/thức ăn, khoảng cách tuyệt đối (cm),
        pixel (x, y) mà tia raycast AR đã đo (không còn cố định là tâm khung).
        Đầu ra: (depth đã anchor, scale, nguồn scale)."""
        if not anchor_distance_cm or anchor_distance_cm <= 0:
            return depth_map, 1.0, "da2_metric"

        reference, source = self._reference_depth(depth_map, plate_mask, food_mask, anchor_pixel)
        if reference is None or reference <= 0:
            self._log_info("anchor: no usable reference depth, keeping DA2 metric")
            return depth_map, 1.0, "da2_metric"

        scale = float(anchor_distance_cm) / reference
        self._log_info(
            f"anchor[{source}]: scale={scale:.4f} (ref={reference:.2f}cm -> {anchor_distance_cm:.2f}cm)"
        )
        return depth_map * scale, scale, "ar_absolute"

    def _reference_depth(
        self,
        depth_map: np.ndarray,
        plate_mask: np.ndarray,
        food_mask: np.ndarray,
        anchor_pixel: tuple[float, float] | None,
    ) -> tuple[float | None, str]:
        """Lấy depth tham chiếu tại đúng pixel đã đo (ưu tiên), fallback sang
        median cả vùng đĩa nếu pixel đó không dùng được."""
        valid = np.isfinite(depth_map) & (depth_map > 0)
        h, w = depth_map.shape[:2]
        plate_clean = plate_mask.astype(bool) & ~food_mask.astype(bool)

        if anchor_pixel is not None:
            col = int(round(anchor_pixel[0]))
            row = int(round(anchor_pixel[1]))
            if 0 <= col < w and 0 <= row < h:
                r0, r1 = max(0, row - ANCHOR_PATCH_RADIUS), min(h, row + ANCHOR_PATCH_RADIUS + 1)
                c0, c1 = max(0, col - ANCHOR_PATCH_RADIUS), min(w, col + ANCHOR_PATCH_RADIUS + 1)
                patch_valid = valid[r0:r1, c0:c1]
                patch_depth = depth_map[r0:r1, c0:c1]

                # Only ever read plate (non-food) pixels inside the patch —
                # never widen to "any valid pixel in the patch", since that
                # could be the food's own surface sitting right at the
                # anchor point (e.g. the anchor landed near the food's edge).
                # If the plate mask doesn't reach the patch, fall through to
                # the whole-plate median below instead of silently anchoring
                # against food depth.
                patch_plate = plate_clean[r0:r1, c0:c1]
                samples = patch_depth[patch_valid & patch_plate]
                if samples.size >= MIN_PATCH_SAMPLES:
                    return float(np.median(samples)), "anchor_pixel"

        # Fallback: AR didn't give us (or we couldn't use) the exact pixel —
        # whole clean-plate-region median is a coarser but still reasonable
        # proxy, since the plate is assumed roughly planar.
        samples = depth_map[plate_clean & valid]
        if samples.size < MIN_ANCHOR_SAMPLES:
            return None, "insufficient"
        return float(np.median(samples)), "plate_median_fallback"


__all__ = ["DepthScaleResolver"]
