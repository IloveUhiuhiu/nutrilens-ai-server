from __future__ import annotations

import logging
import time
import torch
import gc
import numpy as np
from PIL import Image
from typing import Any

from app.utils.image_processing import crop_image, decode_image_bytes

logger = logging.getLogger(__name__)

def load_qwen3_vl(weights_path: str, device: str) -> dict:
    """
    Nạp mô hình Qwen3-VL sử dụng tối ưu hóa Unsloth.
    Trả về một bundle gồm model và tokenizer.
    """
    logger.info("[DEBUG] Loading Qwen3-VL (Unsloth 4-bit) on %s", device)
    from unsloth import FastVisionModel
    
    # Nạp model với cấu hình 4-bit để tiết kiệm VRAM cho server
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=weights_path,
        load_in_4bit=True,
    )
    FastVisionModel.for_inference(model)
    model.to(device)
    
    return {
        "model": model, 
        "tokenizer": tokenizer, 
        "device": device
    }

def _run_vlm_inference(bundle: dict, crop_np: np.ndarray) -> list[str]:
    """
    Thực hiện nhận diện nguyên liệu cho một vùng ảnh đơn lẻ.
    """
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    
    # Chuyển đổi sang PIL Image như yêu cầu của mẫu VLM
    pil_image = Image.fromarray(crop_np)
    
    # Cấu trúc hội thoại
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Analyze the ingredients in this dish. Output ONLY a comma-separated list of ingredients."}
        ]
    }]
    
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        pil_image, 
        prompt, 
        return_tensors="pt"
    ).to(model.device)
    
    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=128,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id
    )

    full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    if "assistant" in full_text.lower():
        raw_response = full_text.lower().split("assistant")[-1].strip()
    else:
        raw_response = full_text.lower().strip()
        
    # Làm sạch chuỗi văn bản
    clean_response = raw_response.replace("\n", " ").rstrip('.')
    
    # Chuyển thành danh sách nguyên liệu
    ingredients = [
        item.strip() for item in clean_response.split(",") 
        if len(item.strip()) > 1
    ]
    
    # Giải phóng bộ nhớ đệm ngay lập tức 
    del inputs, generated_ids
    
    return ingredients

def extract_ingredients(
    image_bytes: bytes,
    food_boxes: list[dict],
    qwen3_bundle: dict,
) -> dict[str, list[str]]:
    """
    Duyệt qua các Box thực phẩm đã detect, cắt ảnh và dùng VLM để lấy nguyên liệu.
    Trả về: { "food_0": ["rice", "chicken"], "food_1": ["salad"] }
    """
    start = time.perf_counter()
    logger.info("[DEBUG] Starting extraction_service...")
    
    try:
        # Giải mã ảnh gốc
        image = decode_image_bytes(image_bytes)
        ingredient_map: dict[str, list[str]] = {}
        
        # Chế độ inference để tăng tốc và giảm bộ nhớ
        with torch.inference_mode():
            for box in food_boxes:
                box_id = box["id"] 
                bbox = box["bbox"]
                
                logger.info("[DEBUG] Extracting ingredients for %s...", box_id)
                
                # Cắt vùng ảnh chứa thực phẩm
                crop = crop_image(image, bbox)
                
                # Nhận diện nguyên liệu
                ingredients = _run_vlm_inference(qwen3_bundle, crop)
                ingredient_map[box_id] = ingredients
        
        return ingredient_map
        
    except Exception:
        logger.exception("[ERROR] extraction_service failed")
        raise
    finally:
        gc.collect()
        logger.info("[DEBUG] extraction_service finished in %.2fs", time.perf_counter() - start)