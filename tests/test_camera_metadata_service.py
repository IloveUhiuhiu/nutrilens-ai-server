from app.services.camera_metadata_service import (
    DEFAULT_CAMERA_HEIGHT_CM,
    CameraMetadataService,
)


def test_mobile_intrinsics_metadata_uses_default_camera_height():
    service = CameraMetadataService()

    camera_height = service.derive_camera_height_cm(
        {
            "width": 3024,
            "height": 4032,
            "fx": 2685.2,
            "fy": 2688.4,
        }
    )

    assert camera_height == DEFAULT_CAMERA_HEIGHT_CM


def test_root_intrinsics_can_be_derived():
    service = CameraMetadataService()

    intrinsics = service.derive_intrinsics(
        {
            "width": 3024,
            "height": 4032,
            "fx": 2000,
            "fy": 1000,
            "cx": 1512,
            "cy": 2016,
        },
    )

    assert intrinsics == {
        "fx": 2000.0,
        "fy": 1000.0,
        "cx": 1512.0,
        "cy": 2016.0,
    }
