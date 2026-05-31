from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.core.config import Settings
from app.services.base import ServiceBase


class CloudinaryStorage(ServiceBase):
    service_name = "cloudinary_storage"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.local_dir = Path(settings.mask_local_dir)
        self.folder = settings.cloudinary_mask_folder.strip("/")

    def save_component_mask(self, mask: np.ndarray, job_id: str, component_id: str) -> str:
        """Chức năng: lưu mask nguyên liệu lên Cloudinary. Đầu vào: mask, job_id, component_id. Đầu ra: URL/path mask."""
        local_path = self._write_local_mask(mask, job_id, component_id)
        if not self._is_cloudinary_configured():
            return str(local_path)
        return self._upload_to_cloudinary(local_path, job_id, component_id)

    def _write_local_mask(self, mask: np.ndarray, job_id: str, component_id: str) -> Path:
        """Chức năng: ghi mask PNG tạm. Đầu vào: mask. Đầu ra: local path."""
        output_dir = self.local_dir / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{component_id}.png"
        mask_png = (mask.astype(np.uint8) * 255) if mask.max() <= 1 else mask.astype(np.uint8)
        cv2.imwrite(str(output_path), mask_png)
        return output_path

    def _is_cloudinary_configured(self) -> bool:
        """Chức năng: kiểm tra cấu hình Cloudinary. Đầu vào: settings. Đầu ra: bool."""
        return all(
            [
                self.settings.cloudinary_cloud_name,
                self.settings.cloudinary_api_key,
                self.settings.cloudinary_api_secret,
            ]
        )

    def _upload_to_cloudinary(self, local_path: Path, job_id: str, component_id: str) -> str:
        """Chức năng: upload file local lên Cloudinary. Đầu vào: path/job/component. Đầu ra: secure URL."""
        import cloudinary
        import cloudinary.uploader

        cloudinary.config(
            cloud_name=self.settings.cloudinary_cloud_name,
            api_key=self.settings.cloudinary_api_key,
            api_secret=self.settings.cloudinary_api_secret,
            secure=True,
        )
        public_id = f"{self.folder}/{job_id}/{component_id}"
        result = cloudinary.uploader.upload(
            str(local_path),
            public_id=public_id,
            overwrite=True,
            resource_type="image",
        )
        return result["secure_url"]
