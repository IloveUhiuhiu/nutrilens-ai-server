from __future__ import annotations

import asyncio
import logging
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1.nutrition import router as nutrition_router
from app.core.config import Settings
from app.core.logging import configure_logging
from app.services import ModelBundle
from app.services.depth import load_depth_anything
from app.services.detection import load_yolo_food, load_yolo_plate
from app.services.extraction import load_qwen3_vl
from app.services.segmentation import load_sam3

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Khởi tạo cấu hình và Logging
    settings = Settings()
    configure_logging(settings.log_level)
    device = settings.device_resolved
    logger.info("[DEBUG] NutriLens AI Server starting on device: %s", device)

    # 2. Nạp các Model Bundles kèm tham số conf (Ngưỡng tin cậy)
    # Tách biệt weights và conf cho từng model để tối ưu độ chính xác
    models = ModelBundle(
        yolo_food=load_yolo_food(
            settings.yolo_food_weights, 
            device, 
            conf=settings.yolo_food_conf
        ),
        yolo_plate=load_yolo_plate(
            settings.yolo_plate_weights, 
            device, 
            conf=settings.yolo_plate_conf
        ),
        qwen3_vl=load_qwen3_vl(
            settings.qwen3vl_weights, 
            device
        ),
        sam3=load_sam3(
            settings.sam3_config_path, 
            settings.sam3_weights, 
            device, 
            conf=settings.sam3_conf
        ),
        depth_anything=load_depth_anything(
            settings.depthanything_weights, 
            device, 
            encoder=settings.depth_encoder  # 'vits' hoặc 'vitb'
        ),
        device=device,
    )

    # 3. Nạp Nutrition Database vào RAM (Chỉ thực hiện 1 lần duy nhất)
    logger.info("[DEBUG] Pre-loading Nutrition Database from %s", settings.nutrition_db_path)
    try:
        with open(settings.nutrition_db_path, "r", encoding="utf-8") as f:
            nutrition_db = json.load(f)
    except Exception as e:
        logger.error("[ERROR] Failed to load nutrition database: %s", e)
        # Fallback về dict trống nếu lỗi để tránh crash server
        nutrition_db = {}

    # 4. Lưu trữ trạng thái vào app.state để truy cập từ Router
    app.state.settings = settings
    app.state.device = device
    app.state.models = models
    app.state.nutrition_db = nutrition_db
    app.state.gpu_lock = asyncio.Lock() # Đảm bảo an toàn tài nguyên GPU khi xử lý đa luồng

    yield
    
    # Dọn dẹp tài nguyên khi tắt server (nếu cần)
    logger.info("[DEBUG] NutriLens AI Server shutting down...")


app = FastAPI(
    title="NutriLens AI Server", 
    description="Hệ thống ước tính dinh dưỡng tự động sử dụng Computer Vision & VLM",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(nutrition_router, prefix="/v1")