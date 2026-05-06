from __future__ import annotations

import gc
import time
import torch
from fastapi import APIRouter, File, Form, Request, UploadFile, HTTPException
import numpy as np
from app.schemas.response import NutritionResponse, NutritionSummary
from app.services.depth import estimate_depth
from app.services.detection import detect_food_and_plate
from app.services.extraction import extract_ingredients
from app.services.geometry import compute_geometry
from app.services.nutrition import estimate_nutrition
from app.services.segmentation import segment_ingredients
from app.utils.image_processing import decode_image_bytes
import logging
router = APIRouter(tags=["nutrition"])

logger = logging.getLogger(__name__)

def _ensure_logging() -> None:
    if not logging.getLogger().handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(filename)s - %(funcName)s - %(message)s")
        handler.setFormatter(formatter)
        logging.getLogger().addHandler(handler)
    logger.setLevel(logging.INFO)

def _log_info(message: str) -> None:
    _ensure_logging()
    logger.info(message)

def _log_error(message: str) -> None:
    _ensure_logging()
    logger.error(message)

def _normalize_items(nutrition_results: dict) -> list[dict]:
    _log_info("normalize items")
    items_raw = list(nutrition_results["ingredients"].values())
    items: list[dict] = []
    for item in items_raw:
        matched_name = item.get("matched_name") or item.get("ingredient") or item.get("name") or "unknown"
        items.append({
            "ingredient": item.get("ingredient") or matched_name,
            "matched_name": matched_name,
            "confidence": float(item.get("confidence", 1.0)),
            "mass_g": float(item.get("mass_g", 0.0)),
            "calories_kcal": float(item.get("calories_kcal", 0.0)),
            "protein_g": float(item.get("protein_g", 0.0)),
            "fat_g": float(item.get("fat_g", 0.0)),
            "carbs_g": float(item.get("carbs_g", 0.0)),
        })
    return items

def _build_summary(total: dict) -> NutritionSummary:
    _log_info("build summary")
    return NutritionSummary(
        total_mass_g=total["mass_g"],
        total_calories_kcal=total["calories_kcal"],
        total_protein_g=total.get("protein_g", 0.0),
        total_fat_g=total.get("fat_g", 0.0),
        total_carbs_g=total.get("carbs_g", 0.0),
    )

def _run_step(step_name: str, func, *args, **kwargs):
    _log_info(f"run step: {step_name}")
    try:
        return func(*args, **kwargs)
    except Exception as exc:
        _log_error(f"step failed: {step_name}")
        raise HTTPException(
            status_code=500,
            detail=f"Step failed in {step_name}: {exc}",
        ) from exc

@router.post("/nutrition/analyze", response_model=NutritionResponse)
async def analyze_nutrition(
    request: Request,
    file: UploadFile = File(...),
    camera_height_ref: float = Form(...),
    pixel_area_ref: float = Form(...),
) -> NutritionResponse:
    _log_info("enter analyze_nutrition")
    try:
        start = time.perf_counter()
        image_bytes = await file.read()

        # 1. Lấy tài nguyên từ app.state (Đã nạp sẵn ở lifespan)
        models = request.app.state.models
        nutrition_db = request.app.state.nutrition_db
        device = request.app.state.device
        gpu_lock = request.app.state.gpu_lock

        async with gpu_lock:
            # 2. Phát hiện thực phẩm và vật chứa (YOLOv11)
            detections = _run_step(
                "detection.detect_food_and_plate",
                detect_food_and_plate,
                image_bytes,
                models.yolo_food,
                models.yolo_plate,
            )

            plate_type = detections["plate_mask"].get("class")
         
            # 3. Trích xuất thành phần theo từng Box ID (Qwen3-VL)
            ingredients_map = _run_step(
                "extraction.extract_ingredients",
                extract_ingredients,
                image_bytes,
                detections["food_boxes"],
                models.qwen3_vl,
            )

            # 4. Phân đoạn nguyên liệu chi tiết (SAM3 LoRA)
            segments = _run_step(
                "segmentation.segment_ingredients",
                segment_ingredients,
                image_bytes,
                ingredients_map,
                models.sam3,
                detections["food_boxes"],
            )

            image_rgb = _run_step(
                "image_processing.decode_image_bytes",
                decode_image_bytes,
                image_bytes,
            )

            orig_h, orig_w = image_rgb.shape[:2]
            food_mask_combined = np.zeros((orig_h, orig_w), dtype=np.uint8)
            
            # Lấy tất cả mask từ kết quả segmentation để hợp nhất
            for mask in segments["global_masks"].values():
                food_mask_combined = np.maximum(food_mask_combined, mask)

            # 5. Ước tính độ sâu và Inpaint đĩa
            depth_data = _run_step(
                "depth.estimate_depth",
                estimate_depth,
                image_bytes=image_bytes,
                plate_mask=detections["plate_mask"]["mask"],
                food_mask=food_mask_combined,
                plate_type=plate_type,
                camera_h_ref=camera_height_ref,
                depth_bundle=models.depth_anything,
                templates_dir=request.app.state.settings.templates_dir,
            )

            # 6. Tính toán hình học nâng cao (Stacking & Depth Completion)
            geometry = _run_step(
                "geometry.compute_geometry",
                compute_geometry,
                segments,
                depth_map=depth_data["depth_map"],
                depth_plate=depth_data["plate_depth"],
                camera_height_ref=camera_height_ref,
                pixel_area_ref=pixel_area_ref,
            )

            # 7. Tra cứu dinh dưỡng từ RAM Database (Fuzzy Matching)
            nutrition_results = _run_step(
                "nutrition.estimate_nutrition",
                estimate_nutrition,
                geometry,
                nutrition_db.get("foods", {}),
            )

        items = _normalize_items(nutrition_results)
        summary = _build_summary(nutrition_results["total"])

        # 9. Giải phóng tài nguyên
        if device == "cuda":
            _log_info("cuda cleanup branch")
            torch.cuda.empty_cache()
        gc.collect()

        return NutritionResponse(
            items=items,
            summary=summary,
            device=device,
            processing_time_s=time.perf_counter() - start,
        )
    except HTTPException:
        raise
    except Exception as exc:
        _log_error("unhandled error in analyze_nutrition")
        raise HTTPException(status_code=500, detail=str(exc)) from exc