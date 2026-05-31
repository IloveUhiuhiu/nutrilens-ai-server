from __future__ import annotations

from app.schemas.response import BackendNutritionResponse
from app.services.storage_service import CloudinaryStorage


class BackendNutritionResponseBuilder:
    """Chức năng: build response theo contract backend. Đầu vào: pipeline output. Đầu ra: BackendNutritionResponse."""

    def __init__(self, storage: CloudinaryStorage) -> None:
        self.storage = storage

    def build(self, pipeline_data: dict, model_version: str, latency_ms: int, job_id: str, nutrition_db: dict) -> BackendNutritionResponse:
        """Chức năng: chuẩn hóa output pipeline sang contract BE. Đầu vào: pipeline data. Đầu ra: response."""
        geometry_data = pipeline_data["geometry_data"]
        nutrition_results = pipeline_data["nutrition_results"]
        nutrition_by_label = nutrition_results.get("ingredients", {})
        db_foods = nutrition_db.get("foods", {})
        global_masks = pipeline_data["segments"].get("global_masks", {})

        components = []
        for index, item in enumerate(geometry_data.get("geometry", [])):
            name = item.get("ingredient") or ""
            nutrient = nutrition_by_label.get(name, {})
            matched_name = nutrient.get("matched_name") or name
            db_record = db_foods.get(matched_name, {})
            component_id = f"comp_{index + 1:03d}"
            components.append(
                {
                    "component_id": component_id,
                    "component_name": name,
                    "physical_data_id": self._physical_data_id(db_record),
                    "mask_path": self._mask_path(global_masks.get(name), job_id, component_id),
                    "volume": float(item.get("volume_cm3") or 0),
                    "weight": float(nutrient.get("mass_g") or 0),
                    "calories": float(nutrient.get("calories_kcal") or 0),
                    "protein": float(nutrient.get("protein_g") or 0),
                    "carbs": float(nutrient.get("carbs_g") or 0),
                    "fat": float(nutrient.get("fat_g") or 0),
                }
            )

        total = nutrition_results.get("total", {})
        return BackendNutritionResponse(
            model_version=model_version,
            latency_ms=latency_ms,
            totals={
                "calories": float(total.get("calories_kcal") or 0),
                "protein": float(total.get("protein_g") or 0),
                "carbs": float(total.get("carbs_g") or 0),
                "fat": float(total.get("fat_g") or 0),
                "weight": float(total.get("mass_g") or 0),
            },
            components=components,
        )

    def _mask_path(self, mask, job_id: str, component_id: str) -> str:
        """Chức năng: lưu mask nếu có. Đầu vào: mask và id. Đầu ra: URL/path."""
        if mask is None:
            return ""
        return self.storage.save_component_mask(mask, job_id, component_id)

    def _physical_data_id(self, db_record: dict) -> str:
        """Chức năng: lấy id nguyên liệu BE. Đầu vào: record nutrition. Đầu ra: id hoặc rỗng."""
        value = db_record.get("physical_data_id") or db_record.get("backend_physical_data_id") or db_record.get("id")
        return str(value) if str(value).startswith("igr_") else ""
