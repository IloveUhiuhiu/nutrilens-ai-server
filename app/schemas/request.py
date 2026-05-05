from __future__ import annotations

from pydantic import BaseModel, Field


class NutritionRequest(BaseModel):
    camera_height_ref: float = Field(..., gt=0)
    pixel_area_ref: float = Field(..., gt=0)
