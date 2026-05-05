"""AI service layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ModelBundle:
	yolo_food: Any
	yolo_plate: Any
	qwen3_vl: Any
	sam3: Any
	depth_anything: Any
	device: str
