from __future__ import annotations

import gc
import time
import torch
from fastapi import APIRouter, File, Form, Request, UploadFile
import numpy as np
from app.schemas.response import NutritionResponse, NutritionSummary
from app.services.depth import estimate_depth
from app.services.detection import detect_food_and_plate
from app.services.extraction import extract_ingredients
from app.services.geometry import compute_geometry
from app.services.nutrition import estimate_nutrition
from app.services.segmentation import segment_ingredients
from app.utils.image_processing import decode_image_bytes
router = APIRouter(tags=["nutrition"])

@router.post("/nutrition/analyze", response_model=NutritionResponse)
async def analyze_nutrition(
    request: Request,
    file: UploadFile = File(...),
    camera_height_ref: float = Form(...),
    pixel_area_ref: float = Form(...),
) -> NutritionResponse:
    start = time.perf_counter()
    image_bytes = await file.read()

    # 1. Lấy tài nguyên từ app.state (Đã nạp sẵn ở lifespan)
    models = request.app.state.models
    nutrition_db = request.app.state.nutrition_db
    device = request.app.state.device
    gpu_lock = request.app.state.gpu_lock

    async with gpu_lock:
        # 2. Phát hiện thực phẩm và vật chứa (YOLOv11)
        # Trả về food_boxes (kèm ID) và plate_mask
        detections = detect_food_and_plate(image_bytes, models.yolo_food, models.yolo_plate)
    
        plate_type = detections["plate_mask"].get("class")
        print(plate_type)
        # 3. Trích xuất thành phần theo từng Box ID (Qwen3-VL)
        ingredients_map = extract_ingredients(image_bytes, detections["food_boxes"], models.qwen3_vl)
    
        # 4. Phân đoạn nguyên liệu chi tiết (SAM3 LoRA)
        # Kết quả bao gồm global_masks và instance_masks để xử lý stacking
        segments = segment_ingredients(
            image_bytes, 
            ingredients_map, 
            models.sam3, 
            detections["food_boxes"]
        )
  
        image_rgb = decode_image_bytes(image_bytes)

        orig_h, orig_w = image_rgb.shape[:2]
        food_mask_combined = np.zeros((orig_h, orig_w), dtype=np.uint8)
        
        # Lấy tất cả mask từ kết quả segmentation để hợp nhất
        for mask in segments["global_masks"].values():
            food_mask_combined = np.maximum(food_mask_combined, mask)

        # 5. Ước tính độ sâu và Inpaint đĩa
        depth_data = estimate_depth(
            image_bytes=image_bytes,
            plate_mask=detections["plate_mask"]["mask"],
            food_mask=food_mask_combined,
            plate_type=plate_type,
            camera_h_ref=camera_height_ref,
            depth_bundle=models.depth_anything,
            templates_dir=request.app.state.settings.templates_dir
        )
        
        # 6. Tính toán hình học nâng cao (Stacking & Depth Completion)
        # Cần truyền cả depth_map (mặt trên) và plate_depth (mặt sàn)
        geometry = compute_geometry(
            segments,
            depth_map=depth_data["depth_map"],
            depth_plate=depth_data["plate_depth"],
            camera_height_ref=camera_height_ref,
            pixel_area_ref=pixel_area_ref,
        )
        
        # 7. Tra cứu dinh dưỡng từ RAM Database (Fuzzy Matching)
        nutrition_results = estimate_nutrition(geometry, nutrition_db.get("foods", {}))

    # 8. Tổng hợp kết quả từ Nutrition Service
    items_raw = list(nutrition_results["ingredients"].values())

    # Chuẩn hóa về schema (ingredient, confidence bắt buộc)
    items = []
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
    total = nutrition_results["total"]


    summary = NutritionSummary(
        total_mass_g=total["mass_g"],
        total_calories_kcal=total["calories_kcal"],
        total_protein_g=total.get("protein_g", 0.0),
        total_fat_g=total.get("fat_g", 0.0),
        total_carbs_g=total.get("carbs_g", 0.0),
    )

    # 9. Giải phóng tài nguyên
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return NutritionResponse(
        items=items,
        summary=summary,
        device=device,
        processing_time_s=time.perf_counter() - start,
    )