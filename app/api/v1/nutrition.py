from __future__ import annotations

import gc
import time
import torch
from pathlib import Path
from fastapi import APIRouter, File, Form, Request, UploadFile, HTTPException
from app.schemas.response import NutritionResponse, NutritionSummary
from app.services.depth_service import DepthService
from app.services.detection_service import DetectionService
from app.services.extraction_service import ExtractionService
from app.services.geometry_service import GeometryService
from app.services.nutrition_service import NutritionService
from app.services.segmentation_service import SegmentationService
from app.services.nutrition_pipeline import NutritionPipeline
from app.utils.visualization.debug import DebugVisualizer
from app.exceptions import AppError, ValidationError
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
    except AppError as exc:
        _log_error(f"app error in step: {step_name}")
        raise HTTPException(status_code=500, detail=exc.to_detail()) from exc
    except Exception as exc:
        _log_error(f"step failed: {step_name}")
        raise HTTPException(
            status_code=500,
            detail={"code": "internal_error", "message": str(exc), "detail": {"step": step_name}},
        ) from exc

detection_service = DetectionService()
extraction_service = ExtractionService()
segmentation_service = SegmentationService()
depth_service = DepthService()
geometry_service = GeometryService()
nutrition_service = NutritionService()
pipeline = NutritionPipeline(
    detection_service,
    extraction_service,
    segmentation_service,
    depth_service,
    geometry_service,
    nutrition_service,
)

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
        debugger = DebugVisualizer()
        image_bytes = await file.read()

        if not image_bytes:
            raise ValidationError("Empty upload file", {"filename": file.filename})

        dish_id = Path(file.filename).stem

        # 1. Lấy tài nguyên từ app.state (Đã nạp sẵn ở lifespan)
        models = request.app.state.models
        nutrition_db = request.app.state.nutrition_db
        ground_truth = request.app.state.ground_truth
        device = request.app.state.device
        gpu_lock = request.app.state.gpu_lock

        async with gpu_lock:
            pipeline_data = _run_step(
                "nutrition_pipeline.run_pipeline",
                pipeline.run_pipeline,
                image_bytes=image_bytes,
                models=models,
                nutrition_db=nutrition_db,
                camera_height_ref=camera_height_ref,
                pixel_area_ref=pixel_area_ref,
                templates_dir=request.app.state.settings.templates_dir,
            )

            image_rgb = pipeline_data["image_rgb"]
            detections = pipeline_data["detections"]
            ingredients_map = pipeline_data["ingredients_map"]
            segments = pipeline_data["segments"]
            food_mask_combined = pipeline_data["food_mask_combined"]
            depth_data = pipeline_data["depth_data"]
            geometry_data = pipeline_data["geometry_data"]
            nutrition_results = pipeline_data["nutrition_results"]

            debug_visuals = getattr(
                request.app.state.settings,
                "debug_visuals",
                False
            )
            if debug_visuals:
                debugger.run_debug_visuals(
                    dish_id=dish_id,
                    image_rgb=image_rgb,
                    detections=detections,
                    ingredients_map=ingredients_map,
                    segments=segments,
                    food_mask_combined=food_mask_combined,
                    depth_data=depth_data,
                    geometry_data=geometry_data,
                    nutrition_results=nutrition_results,
                    ground_truth=ground_truth,
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
    except AppError as exc:
        _log_error("handled app error in analyze_nutrition")
        raise HTTPException(status_code=500, detail=exc.to_detail()) from exc
    except Exception as exc:
        _log_error("unhandled error in analyze_nutrition")
        raise HTTPException(
            status_code=500,
            detail={"code": "internal_error", "message": str(exc)},
        ) from exc