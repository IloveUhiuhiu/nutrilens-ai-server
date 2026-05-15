from __future__ import annotations

from pydantic import BaseModel, Field


class NutritionItem(BaseModel):
    ingredient: str
    matched_name: str
    confidence: float = Field(..., ge=0, le=1)
    mass_g: float = Field(..., ge=0)
    calories_kcal: float = Field(..., ge=0)
    protein_g: float = Field(..., ge=0)
    fat_g: float = Field(..., ge=0)
    carbs_g: float = Field(..., ge=0)


class NutritionSummary(BaseModel):
    total_mass_g: float = Field(..., ge=0)
    total_calories_kcal: float = Field(..., ge=0)
    total_protein_g: float = Field(..., ge=0)
    total_fat_g: float = Field(..., ge=0)
    total_carbs_g: float = Field(..., ge=0)


class NutritionResponse(BaseModel):
    items: list[NutritionItem]
    summary: NutritionSummary
    device: str
    processing_time_s: float


class NutritionDebugInfo(BaseModel):
    images: dict[str, str]  # key -> base64 encoded image
    texts: dict[str, str]
    processing_time_s: float
    device: str


class NutritionDebugResponse(NutritionResponse):
    debug_info: NutritionDebugInfo
