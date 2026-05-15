from __future__ import annotations

import gc
import time
import torch
import cv2
import numpy as np
from pathlib import Path
from fastapi import APIRouter, File, Form, Request, UploadFile, HTTPException
from app.schemas.response import NutritionResponse, NutritionSummary, NutritionDebugResponse, NutritionDebugInfo
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

@router.post("/nutrition/analyze", response_model=NutritionDebugResponse)
async def analyze_nutrition(
    request: Request,
    file: UploadFile = File(...),
    camera_height_ref: float = Form(...),
    pixel_area_ref: float = Form(...),
) -> NutritionDebugResponse:
    _log_info("enter analyze_nutrition")
    try:
        start = time.perf_counter()
        debugger = DebugVisualizer(to_memory=True)
        image_bytes = await file.read()

        if not image_bytes:
            raise ValidationError("Empty upload file", {"filename": file.filename})

        dish_id = Path(file.filename).stem
        _log_info(f"Processing dish: {dish_id}")

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
                debugger=debugger,
            )

            nutrition_results = pipeline_data["nutrition_results"]
            image_rgb = pipeline_data.get("image_rgb")
            detections = pipeline_data.get("detections")
            geometry_data = pipeline_data.get("geometry_data")
            depth_data = pipeline_data.get("depth_data")

        items = _normalize_items(nutrition_results)
        summary = _build_summary(nutrition_results["total"])

        # Add essential debug information (reduced duplicate images)
        if image_rgb is not None:
            debugger.save_image_rgb("00_original_input.png", image_rgb)
        
        # Save detection boxes overlay
        if detections and image_rgb is not None:
            debugger.save_detection_boxes("01_detection_boxes.png", image_rgb, detections.get("food_boxes", []))
        
        # Save plate mask
        if detections and detections.get("plate_mask", {}).get("mask") is not None:
            debugger.save_mask("02_plate_mask.png", detections["plate_mask"]["mask"])
        
        # Save topological order overlay (more important than individual depth maps)
        topo_overlay = None
        if geometry_data:
            # Save geometry details
            debugger.save_json("geometry_details.json", {
                "results": geometry_data.get("geometry", []),
                "topological_order": geometry_data.get("topological_order", []),
                "cycle_detected": geometry_data.get("cycle_detected", False),
            })
            
            # Save topological order overlay
            try:
                if image_rgb is not None and geometry_data.get("instance_masks") and geometry_data.get("instance_labels"):
                    topo_overlay = debugger.save_topological_order_overlay(
                        "03_topological_order_overlay.png",
                        image_rgb,
                        geometry_data["instance_masks"],
                        geometry_data["instance_labels"],
                        geometry_data.get("topological_order", [])
                    )
                    _log_info("Topological order overlay saved")
            except Exception as e:
                _log_error(f"Failed to save topological overlay: {str(e)}")
        
        # Save only essential JSON data
        debugger.save_json("nutrition_results.json", {
            "total": nutrition_results.get("total", {}),
            "ingredients": nutrition_results.get("ingredients", {})
        })
        
        # Create lightweight dashboard (no depth maps - avoid duplication)
        try:
            if image_rgb is not None:
                plate_mask = detections.get("plate_mask", {}).get("mask") if detections else None
                
                debugger.save_dashboard(
                    filename="04_dashboard.png",
                    original_rgb=image_rgb,
                    boxes_rgb=image_rgb.copy() if detections and detections.get("food_boxes") else None,
                    plate_mask=plate_mask,
                    ingredient_overlay=image_rgb.copy(),
                    plate_depth=None,  # Removed to reduce duplicates
                    merged_depth=None,  # Removed to reduce duplicates
                    topo_overlay=topo_overlay,  # Already saved separately
                    nutrition_results=nutrition_results,
                    geometry_results=geometry_data.get("geometry", []) if geometry_data else [],
                    food_heights=None,
                    ground_truth=ground_truth,
                    dish_id=dish_id
                )
                _log_info("Dashboard saved successfully")
        except Exception as e:
            _log_error(f"Dashboard creation failed: {str(e)}")
            # Don't fail the entire request if dashboard fails
        
        # Single comprehensive report (avoid duplication)
        report = f"""=== NUTRILENS AI DEBUG REPORT ===
Dish ID: {dish_id}
Device: {device}
Processing Time: {time.perf_counter() - start:.2f}s

INPUT PARAMETERS:
  Camera Height Reference: {camera_height_ref} cm
  Pixel Area Reference: {pixel_area_ref} cm²

DETECTION RESULTS:
  Food Boxes Found: {len(detections.get("food_boxes", [])) if detections else 0}
  Plate Detected: {detections.get("plate_mask", {}).get("class") if detections else "N/A"}

NUTRITIONAL SUMMARY:
  Total Mass: {summary.total_mass_g:.2f} g
  Total Calories: {summary.total_calories_kcal:.2f} kcal
  Total Protein: {summary.total_protein_g:.2f} g
  Total Fat: {summary.total_fat_g:.2f} g
  Total Carbs: {summary.total_carbs_g:.2f} g

DETECTED INGREDIENTS ({len(items)}):"""
        for i, item in enumerate(items, 1):
            report += f"\n  {i}. {item['ingredient']} → {item['matched_name']}\n     Mass: {item['mass_g']:.2f}g | Calories: {item['calories_kcal']:.2f}kcal | Confidence: {item['confidence']*100:.1f}%"
        
        debugger.save_text("05_report.txt", report)
        _log_info("Debug report saved")

        # Get debug info from debugger (optimized - fewer duplicates)
        debug_results = debugger.get_results_as_dict()
        debug_info = NutritionDebugInfo(
            images=debug_results.get("images", {}),
            texts=debug_results.get("texts", {}),
            processing_time_s=time.perf_counter() - start,
            device=device,
        )

        # Clean up resources
        if device == "cuda":
            _log_info("cuda cleanup branch")
            torch.cuda.empty_cache()
        gc.collect()

        _log_info(f"Analysis complete. Generated {len(debug_info.images)} images and {len(debug_info.texts)} text files")

        return NutritionDebugResponse(
            items=items,
            summary=summary,
            device=device,
            processing_time_s=time.perf_counter() - start,
            debug_info=debug_info,
        )
    except HTTPException:
        raise
    except AppError as exc:
        _log_error("handled app error in analyze_nutrition")
        raise HTTPException(status_code=500, detail=exc.to_detail()) from exc
    except Exception as exc:
        _log_error(f"unhandled error in analyze_nutrition: {str(exc)}")
        import traceback
        _log_error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={"code": "internal_error", "message": str(exc)},
        ) from exc


@router.post("/nutrition/analyze_debug", response_model=NutritionDebugResponse)
async def analyze_nutrition_debug(
    request: Request,
    file: UploadFile = File(...),
    camera_height_ref: float = Form(...),
    pixel_area_ref: float = Form(...),
) -> NutritionDebugResponse:
    """Alias endpoint - delegates to analyze_nutrition for consistency"""
    _log_info("enter analyze_nutrition_debug (alias -> analyze_nutrition)")
    return await analyze_nutrition(request, file, camera_height_ref, pixel_area_ref)