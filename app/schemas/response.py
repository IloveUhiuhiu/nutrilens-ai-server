from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class AIAnalysisComponent(BaseModel):
    component_id: str
    component_name: str
    mask_path: str = ""
    volume: float = Field(..., ge=0)


class AIAnalysisResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_version: str
    latency_ms: int = Field(..., ge=0)
    components: list[AIAnalysisComponent]
