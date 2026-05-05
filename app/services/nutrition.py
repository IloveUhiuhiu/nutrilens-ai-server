from __future__ import annotations

import json
import logging
import time
from difflib import get_close_matches
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

def estimate_nutrition(geometry_results: list[dict], db_dict: dict) -> dict[str, Any]:
    
    start = time.perf_counter()
    try:
        if not geometry_results:
            return {"ingredients": {}, "total": {}}
        db_keys = list(db_dict.keys())
        
        nutrition_details = {}
        total_summary = {
            "mass_g": 0.0,
            "calories_kcal": 0.0,
            "protein_g": 0.0,
            "fat_g": 0.0,
            "carbs_g": 0.0
        }

        for item in geometry_results:
            ing_name = item["ingredient"]
            
            # 2. Fuzzy Matching (So khớp tên gần nhất)
            match = get_close_matches(ing_name, db_keys, n=1, cutoff=0.6)
            matched_key = match[0] if match else None
            
            if not matched_key:
                logger.warning(f"[WARN] No match found for ingredient: {ing_name}")
                continue

            # 3. Trích xuất thông số từ DB (Mặc định đơn vị trên 1g)
            ing_info = db_dict[matched_key]
            density = float(ing_info.get("density", 1.0))
            
            # 4. Tính toán Khối lượng (g) = Thể tích (cm3) * Khối lượng riêng (g/cm3)
            volume = float(item.get("volume_cm3", 0.0))
            mass = volume * density
            
            # 5. Tính toán chi tiết (Mass * Nutrient_per_1g)
            # Notebook: "Chỉ số trong DB nên được hiểu là giá trị trên 1 gram"
            ing_nutrients = {
                "matched_name": matched_key,
                "volume_cm3": round(volume, 2),
                "mass_g": round(mass, 2),
                "avg_height_cm": round(float(item.get("avg_height_cm", 0.0)), 2),
                "calories_kcal": round(mass * float(ing_info.get("cal", 0.0)), 2),
                "protein_g": round(mass * float(ing_info.get("protein", 0.0)), 2),
                "fat_g": round(mass * float(ing_info.get("fat", 0.0)), 2),
                "carbs_g": round(mass * float(ing_info.get("carbs", 0.0)), 2),
                "confidence": 0.9
            }
            
            nutrition_details[ing_name] = ing_nutrients
            
            # 6. Cộng dồn tổng số
            total_summary["mass_g"] += ing_nutrients["mass_g"]
            total_summary["calories_kcal"] += ing_nutrients["calories_kcal"]
            total_summary["protein_g"] += ing_nutrients["protein_g"]
            total_summary["fat_g"] += ing_nutrients["fat_g"]
            total_summary["carbs_g"] += ing_nutrients["carbs_g"]

        # Làm tròn kết quả tổng cuối cùng
        for key in total_summary:
            total_summary[key] = round(total_summary[key], 2)

        return {
            "ingredients": nutrition_details,
            "total": total_summary
        }

    except Exception:
        logger.exception("[ERROR] nutrition_service failed")
        raise
    finally:
        logger.info("[DEBUG] nutrition_service finished in %.2fs", time.perf_counter() - start)