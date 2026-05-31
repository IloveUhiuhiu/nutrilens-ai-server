from __future__ import annotations

from pydantic import BaseModel, Field


class BackendNutritionTotals(BaseModel):
    calories: float = Field(..., ge=0)
    protein: float = Field(..., ge=0)
    carbs: float = Field(..., ge=0)
    fat: float = Field(..., ge=0)
    weight: float = Field(..., ge=0)


class BackendNutritionComponent(BaseModel):
    component_id: str
    component_name: str
    physical_data_id: str = ""
    mask_path: str = ""
    volume: float = Field(..., ge=0)
    weight: float = Field(..., ge=0)
    calories: float = Field(..., ge=0)
    protein: float = Field(..., ge=0)
    carbs: float = Field(..., ge=0)
    fat: float = Field(..., ge=0)


class BackendNutritionResponse(BaseModel):
    model_version: str
    latency_ms: int = Field(..., ge=0)
    totals: BackendNutritionTotals
    components: list[BackendNutritionComponent]
