from __future__ import annotations

import requests

from app.core.config import Settings
from app.services.base import ServiceBase


class NutritionRepository(ServiceBase):
    service_name = "nutrition_repository"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def load(self) -> dict:
        """Chức năng: nạp dữ liệu dinh dưỡng từ internal API backend. Đầu vào: settings. Đầu ra: dict foods."""
        rows = self._fetch_ingredients()
        foods = {}
        for row in rows:
            record = self._row_to_food_record(row)
            for key in self._food_keys(row):
                foods[key] = record

        return {
            "metadata": {
                "source": "nutrilens-backend-api",
                "path": self.settings.backend_ingredients_path,
                "count": len(rows),
            },
            "foods": foods,
        }

    def _fetch_ingredients(self) -> list[dict]:
        """Chức năng: gọi internal API backend lấy IngredientPhysicalData. Đầu vào: config backend. Đầu ra: list row."""
        if not self.settings.backend_internal_api_key:
            raise ValueError("BACKEND_INTERNAL_API_KEY is not configured.")

        url = f"{self.settings.backend_base_url.rstrip('/')}{self.settings.backend_ingredients_path}"
        response = requests.get(
            url,
            headers={"X-Internal-API-Key": self.settings.backend_internal_api_key},
            timeout=self.settings.backend_api_timeout,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data")
        if not isinstance(data, list):
            raise ValueError("Backend ingredients response data must be a list.")
        return data

    def _row_to_food_record(self, row: dict) -> dict:
        """Chức năng: đổi IngredientPhysicalData API row sang format NutritionService. Đầu vào: row. Đầu ra: record."""
        return {
            "physical_data_id": row["id"],
            "vi_name": row["vi_name"],
            "en_name": row["en_name"],
            "density": float(row["density"] or 0),
            "cal": float(row["cal_per_100g"] or 0) / 100,
            "fat": float(row["fat_per_100g"] or 0) / 100,
            "carbs": float(row["carb_per_100g"] or 0) / 100,
            "protein": float(row["protein_per_100g"] or 0) / 100,
            "fdc_id_ref": row.get("fdc_id_ref") or "",
            "source": "nutrilens-backend-api",
        }

    def _food_keys(self, row: dict) -> list[str]:
        """Chức năng: tạo key match tên nguyên liệu. Đầu vào: row API. Đầu ra: list key."""
        keys = []
        for value in (row.get("en_name"), row.get("vi_name")):
            value = (value or "").strip()
            if value:
                keys.append(value)
        return keys
