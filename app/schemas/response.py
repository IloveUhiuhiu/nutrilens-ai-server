from __future__ import annotations

from pydantic import BaseModel, Field


class NutritionItem(BaseModel):
    ingredient: str
    matched_name: str
    mass_g: float = Field(..., ge=0)
    calories_kcal: float = Field(..., ge=0)
    confidence: float = Field(..., ge=0, le=1)


class NutritionSummary(BaseModel):
    total_mass_g: float = Field(..., ge=0)
    total_calories_kcal: float = Field(..., ge=0)


class NutritionResponse(BaseModel):
    items: list[NutritionItem]
    summary: NutritionSummary
    device: str
    processing_time_s: float
