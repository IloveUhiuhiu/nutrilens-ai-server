from __future__ import annotations

from app.schemas.response import AIAnalysisResponse
from app.services.storage_service import CloudinaryStorage


class AIAnalysisResponseBuilder:
    """Chức năng: build response phân tích AI. Đầu vào: pipeline output. Đầu ra: AIAnalysisResponse."""

    def __init__(self, storage: CloudinaryStorage) -> None:
        self.storage = storage

    def build(self, pipeline_data: dict, model_version: str, latency_ms: int, job_id: str) -> AIAnalysisResponse:
        """Chức năng: chuẩn hóa output pipeline sang contract BE. Đầu vào: pipeline data. Đầu ra: response thô để BE tính dinh dưỡng."""
        geometry_data = pipeline_data["geometry_data"]
        global_masks = pipeline_data["segments"].get("global_masks", {})

        components = []
        for index, item in enumerate(geometry_data.get("geometry", [])):
            name = item.get("ingredient") or ""
            component_id = f"comp_{index + 1:03d}"
            components.append(
                {
                    "component_id": component_id,
                    "component_name": name,
                    "mask_path": self._mask_path(global_masks.get(name), job_id, component_id),
                    "volume": float(item.get("volume_cm3") or 0),
                }
            )

        return AIAnalysisResponse(
            model_version=model_version,
            latency_ms=latency_ms,
            components=components,
        )


    def _mask_path(self, mask, job_id: str, component_id: str) -> str:
        """Chức năng: lưu mask nếu có. Đầu vào: mask và id. Đầu ra: URL/path."""
        if mask is None:
            return ""
        return self.storage.save_component_mask(mask, job_id, component_id)
