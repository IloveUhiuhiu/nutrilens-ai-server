from __future__ import annotations

import gc
import logging
import time

import torch
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from app.exceptions import AppError, ValidationError
from app.schemas.request import AnalyzeNutritionInput, parse_json_form
from app.schemas.response import AIAnalysisResponse
from app.services.camera_metadata_service import CameraMetadataService
from app.services.depth_service import DepthService
from app.services.detection_service import DetectionService
from app.services.extraction_service import ExtractionService
from app.services.geometry_service import GeometryService
from app.services.nutrition_pipeline import NutritionPipeline
from app.services.response_builder import AIAnalysisResponseBuilder
from app.services.segmentation_service import SegmentationService
from app.services.storage_service import CloudinaryStorage


router = APIRouter(tags=["nutrition"])
logger = logging.getLogger(__name__)


def _log_error(message: str) -> None:
    logger.exception(message)


def _run_step(step_name: str, func, *args, **kwargs):
    """Chức năng: chạy một bước và map lỗi. Đầu vào: tên bước và callable. Đầu ra: kết quả bước.
    Status code lấy từ exc.status_code (do exception class quyết định) thay vì suy luận lại từ exc.code,
    để các lỗi nghiệp vụ mới (no_food_detected, no_plate_detected,...) có status đúng (422) thay vì rơi về 500."""
    try:
        return func(*args, **kwargs)
    except AppError as exc:
        _log_error(f"app error in step: {step_name} (error_code={exc.code})")
        detail = exc.to_detail()
        detail["context"].setdefault("step", step_name)
        raise HTTPException(status_code=exc.status_code, detail=detail) from exc
    except Exception as exc:
        # Lỗi không lường trước: log đầy đủ traceback ở server, nhưng KHÔNG trả str(exc)
        # (có thể chứa chi tiết nội bộ thư viện) thẳng ra client.
        _log_error(f"step failed: {step_name} (unexpected {type(exc).__name__})")
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": "internal_error",
                "message": "An unexpected error occurred while processing the request.",
                "context": {"step": step_name},
            },
        ) from exc


detection_service = DetectionService()
extraction_service = ExtractionService()
segmentation_service = SegmentationService()
depth_service = DepthService()
geometry_service = GeometryService()
camera_metadata_service = CameraMetadataService()
pipeline = NutritionPipeline(
    detection_service,
    extraction_service,
    segmentation_service,
    depth_service,
    geometry_service,
)


@router.post("/analyze", response_model=AIAnalysisResponse)
async def analyze_nutrition(
    request: Request,
    image: UploadFile = File(...),
    job_id: str = Form(...),
    camera_metadata: str = Form(...),
    depth_map: UploadFile | None = File(default=None),
) -> AIAnalysisResponse:
    """Chức năng: API phân tích dinh dưỡng. Đầu vào: multipart ảnh/depth/metadata. Đầu ra: response BE."""
    try:
        start = time.perf_counter()
        analyze_input = await _parse_analyze_input(
            image=image,
            depth_map=depth_map,
            job_id=job_id,
            camera_metadata=camera_metadata,
        )
        camera_height_ref = camera_metadata_service.derive_camera_height_cm(
            analyze_input.camera_metadata,
        )
        camera_intrinsics = camera_metadata_service.derive_intrinsics(
            analyze_input.camera_metadata,
        )
        has_absolute_depth, anchor_distance_cm = camera_metadata_service.derive_absolute_distance(
            analyze_input.camera_metadata,
        )
        anchor_pixel = camera_metadata_service.derive_anchor_pixel(
            analyze_input.camera_metadata,
        )
        anchor_candidates = camera_metadata_service.derive_anchor_candidates(
            analyze_input.camera_metadata,
        )

        models = request.app.state.models
        device = request.app.state.device
        gpu_lock = request.app.state.gpu_lock

        async with gpu_lock:
            pipeline_data = _run_step(
                "nutrition_pipeline.run_pipeline",
                pipeline.run_pipeline,
                image_bytes=analyze_input.image_bytes,
                models=models,
                camera_height_ref=camera_height_ref,
                camera_intrinsics=camera_intrinsics,
                templates_dir=request.app.state.settings.templates_dir,
                depth_bytes=analyze_input.depth_bytes,
                depth_metadata=analyze_input.depth_metadata,
                has_absolute_depth=has_absolute_depth,
                anchor_distance_cm=anchor_distance_cm,
                anchor_pixel=anchor_pixel,
                anchor_candidates=anchor_candidates,
            )
            if getattr(request.app.state.settings, "debug_visuals", False):
                # Debug visuals chỉ phục vụ mục đích kỹ thuật, không thuộc response trả
                # về cho client - nếu thất bại thì chỉ log lại, không làm hỏng request.
                try:
                    _run_debug_visuals(request, analyze_input.dish_id, pipeline_data)
                except Exception:
                    _log_error(f"debug visuals failed for dish_id={analyze_input.dish_id}, ignoring")

        response = _run_step(
            "response_builder.build",
            AIAnalysisResponseBuilder(CloudinaryStorage(request.app.state.settings)).build,
            pipeline_data=pipeline_data,
            model_version=request.app.state.settings.model_version,
            latency_ms=int((time.perf_counter() - start) * 1000),
            job_id=analyze_input.dish_id,
        )

        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        return response
    except HTTPException:
        raise
    except AppError as exc:
        _log_error(f"handled app error in analyze_nutrition (error_code={exc.code})")
        raise HTTPException(status_code=exc.status_code, detail=exc.to_detail()) from exc
    except Exception as exc:
        _log_error(f"unhandled error in analyze_nutrition (unexpected {type(exc).__name__})")
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": "internal_error",
                "message": "An unexpected error occurred while processing the request.",
                "context": {},
            },
        ) from exc


async def _parse_analyze_input(
    image: UploadFile,
    depth_map: UploadFile | None,
    job_id: str,
    camera_metadata: str,
) -> AnalyzeNutritionInput:
    """Chức năng: parse multipart thành input nội bộ. Đầu vào: form fields. Đầu ra: AnalyzeNutritionInput."""
    image_bytes = await image.read()
    if not image_bytes:
        logger.warning(f"step=request_validation failed: empty upload file (filename={image.filename})")
        raise ValidationError("Empty upload file", {"filename": image.filename})

    parsed_camera_metadata = parse_json_form(camera_metadata)
    parsed_depth_metadata = parsed_camera_metadata.get("depth") or {}
    if not isinstance(parsed_depth_metadata, dict):
        logger.warning("step=request_validation failed: camera_metadata.depth is not a JSON object")
        raise ValidationError("camera_metadata.depth must be a JSON object", {"field": "camera_metadata.depth"})

    depth_bytes = None
    if depth_map:
        depth_bytes = await depth_map.read()
        if depth_map.filename:
            from pathlib import Path

            parsed_depth_metadata.setdefault("file_extension", Path(depth_map.filename).suffix.lower())

    return AnalyzeNutritionInput(
        image_bytes=image_bytes,
        image_filename=image.filename or "",
        job_id=job_id,
        camera_metadata=parsed_camera_metadata,
        depth_metadata=parsed_depth_metadata,
        depth_bytes=depth_bytes,
    )


def _run_debug_visuals(request: Request, dish_id: str, pipeline_data: dict) -> None:
    """Chức năng: chạy debug visualization nếu bật. Đầu vào: request, dish_id, pipeline output. Đầu ra: None."""
    from app.utils.visualization.debug import DebugVisualizer

    DebugVisualizer().run_debug_visuals(
        dish_id=dish_id,
        image_rgb=pipeline_data["image_rgb"],
        detections=pipeline_data["detections"],
        ingredients_map=pipeline_data["ingredients_map"],
        segments=pipeline_data["segments"],
        food_mask_combined=pipeline_data["food_mask_combined"],
        depth_data=pipeline_data["depth_data"],
        geometry_data=pipeline_data["geometry_data"],
    )
