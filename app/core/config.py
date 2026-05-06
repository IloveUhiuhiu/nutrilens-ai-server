from __future__ import annotations

from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Tự động đọc file .env và map các biến không phân biệt hoa thường
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        case_sensitive=False 
    )

    # --- Cấu hình chung ---
    device: Literal["auto", "cpu", "cuda"] = "auto"
    log_level: str = "INFO"
    nutrition_db_path: str = "app/db/nutrition_db.json"

    # --- YOLO Detection (Food & Plate) ---
    yolo_food_weights: str = "weights/yolo/food_yolo.pt"
    yolo_food_conf: float = 0.5
    
    yolo_plate_weights: str = "weights/yolo/plate_yolo_seg.pt"
    yolo_plate_conf: float = 0.8

    # --- VLM Extraction (Qwen3-VL) ---
    qwen3vl_weights: str = "weights/vlm/qwen3vl-4bit"

    # --- SAM3 LoRA Segmentation ---
    sam3_config_path: str = "app/services/sam3/food_config.yaml"
    sam3_weights: str = "weights/sam3/sam3_lora.pth"
    sam3_conf: float = 0.75

    # --- Depth Estimation (DepthAnythingV2) ---
    depth_encoder: Literal["vits", "vitb", "vitl", "vitg"] = "vits"
    depthanything_weights: str = "weights/depth/depth_anything_v2_vits.pth"
    templates_dir: str = "templates"

    @property
    def device_resolved(self) -> str:
        """Tự động xác định thiết bị tính toán nếu để là 'auto'."""
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"