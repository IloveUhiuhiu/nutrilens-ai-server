from __future__ import annotations

import logging
import time
import torch
import gc
import numpy as np
from PIL import Image
from unsloth import FastVisionModel
from app.utils.image import crop_image, decode_image_bytes
from app.services.base import ServiceBase

logger = logging.getLogger(__name__)

class ExtractionService(ServiceBase):
    service_name = "extraction"

    def __init__(self) -> None:
        pass

    def load_qwen3_vl(self, weights_path: str, device: str) -> dict:
        self._log_info(f"loading qwen3_vl on {device}")
        """
        Nạp mô hình Qwen3-VL sử dụng tối ưu hóa Unsloth.
        Trả về một bundle gồm model và tokenizer.
        """
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

    def _run_vlm_inference(self, bundle: dict, crop_np: np.ndarray) -> list[str]:
        self._log_info("enter _run_vlm_inference")
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
            self._log_info("assistant branch in response")
            raw_response = full_text.lower().split("assistant")[-1].strip()
        else:
            self._log_info("no assistant branch in response")
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
        self,
        image_bytes: bytes,
        food_boxes: list[dict],
        qwen3_bundle: dict,
    ) -> dict[str, list[str]]:
        self._log_info("enter extract_ingredients")
        """
        Duyệt qua các Box thực phẩm đã detect, cắt ảnh và dùng VLM để lấy nguyên liệu.
        Trả về: { "food_0": ["rice", "chicken"], "food_1": ["salad"] }
        """
        start = time.perf_counter()
        
        try:
            # Giải mã ảnh gốc
            image = decode_image_bytes(image_bytes)
            ingredient_map: dict[str, list[str]] = {}
            
            # Chế độ inference để tăng tốc và giảm bộ nhớ
            with torch.inference_mode():
                self._log_info("torch inference_mode branch")
                for box in food_boxes:
                    box_id = box["id"] 
                    bbox = box["bbox"]
                    
                    self._log_info(f"extracting ingredients for {box_id}")
                    
                    # Cắt vùng ảnh chứa thực phẩm
                    crop = crop_image(image, bbox)
                    
                    # Nhận diện nguyên liệu
                    ingredients = self._run_vlm_inference(qwen3_bundle, crop)
                    ingredient_map[box_id] = ingredients
            
            return ingredient_map
            
        except Exception:
            self._log_error("extraction_service failed")
            raise
        finally:
            self._log_info("extraction_service finished")
            gc.collect()

__all__ = ["ExtractionService"]