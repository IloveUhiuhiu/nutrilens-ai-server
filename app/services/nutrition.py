from __future__ import annotations

import logging
import time
import difflib 
from typing import Any

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

def find_best_ingredient_match(target_name, database_keys, similarity_cutoff=0.6):
    _log_info("enter find_best_ingredient_match")
    """
    Tìm kiếm tên nguyên liệu phù hợp nhất.
    """
    target_name = target_name.lower().strip()
    
    # Tạo bản map lowercase để so khớp chính xác hơn
    db_map = {k.lower(): k for k in database_keys}
    
    # 1. Kiểm tra khớp hoàn toàn (sau khi đã chuẩn hóa)
    if target_name in db_map:
        _log_info("exact match branch")
        return db_map[target_name]
    
    # 2. Substring Match: Rất quan trọng cho VLM (ví dụ: "boiled egg" -> "egg")
    for norm_key in db_map:
        if norm_key in target_name or target_name in norm_key:
            _log_info("substring match branch")
            return db_map[norm_key]

    # 3. Sử dụng logic so khớp mờ
    best_matches = difflib.get_close_matches(
        target_name, 
        list(db_map.keys()), 
        n=1, 
        cutoff=similarity_cutoff
    )
    
    return db_map[best_matches[0]] if best_matches else None

def estimate_nutrition(geometry_results: list[dict], db_dict: dict) -> dict[str, Any]:
    _log_info("enter estimate_nutrition")
    start = time.perf_counter()
    try:
        if not geometry_results:
            _log_info("empty geometry_results branch")
            return {"ingredients": {}, "total": {
                "mass_g": 0.0, "calories_kcal": 0.0, "protein_g": 0.0, "fat_g": 0.0, "carbs_g": 0.0
            }}
            
        db_keys = list(db_dict.keys())
        nutrition_details = {}
        total_summary = {
            "mass_g": 0.0, "calories_kcal": 0.0, "protein_g": 0.0, "fat_g": 0.0, "carbs_g": 0.0
        }

        for item in geometry_results:
     
            ing_name = item.get("ingredient") 
            if not ing_name:
                _log_info("missing ingredient name branch")
                continue

            matched_key = find_best_ingredient_match(ing_name, db_keys, similarity_cutoff=0.6)
            
            if not matched_key:
                _log_info(f"no match found for ingredient: {ing_name}")
                continue

            ing_info = db_dict[matched_key]
            density = float(ing_info.get("density", 1.0))
            volume = float(item.get("volume_cm3", 0.0))
            mass = volume * density
            
            # Tính toán dựa trên đơn vị 1g
            ing_nutrients = {
                "matched_name": matched_key,
                "volume_cm3": round(volume, 2),
                "mass_g": round(mass, 2),
                "calories_kcal": round(mass * float(ing_info.get("cal", 0.0)), 2),
                "protein_g": round(mass * float(ing_info.get("protein", 0.0)), 2),
                "fat_g": round(mass * float(ing_info.get("fat", 0.0)), 2),
                "carbs_g": round(mass * float(ing_info.get("carbs", 0.0)), 2),
            }
            
            nutrition_details[ing_name] = ing_nutrients
            
            # Cộng dồn
            for key in ["mass_g", "calories_kcal", "protein_g", "fat_g", "carbs_g"]:
                total_summary[key] += ing_nutrients[key]

        # Làm tròn kết quả tổng
        total_summary = {k: round(v, 2) for k, v in total_summary.items()}

        return {
            "ingredients": nutrition_details,
            "total": total_summary
        }

    except Exception:
        _log_error("nutrition_service failed")
        raise
    finally:
        _log_info("nutrition_service finished")