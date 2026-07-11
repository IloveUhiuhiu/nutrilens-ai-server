import numpy as np

from app.services.geometry_service import GeometryService


def test_pixel_area_uses_metric_depth_and_intrinsics():
    service = GeometryService()

    area_map = service._pixel_area_from_metric_depth(
        np.array([[40.0, 20.0]], dtype=np.float32),
        {"fx": 2000, "fy": 1000},
    )

    np.testing.assert_allclose(area_map, np.array([[0.0008, 0.0002]], dtype=np.float32))
