from __future__ import annotations

import logging
import os
import time

# Unsloth's torch.compile path monkeypatches torch.nn.Conv2d.forward (and other
# norm/conv modules) globally for the whole process to speed up Qwen3-VL's vision
# tower. That breaks every other Conv2d-based model loaded in this same process
# (YOLO, SAM3, DepthAnythingV2) with "CUDNN_STATUS_NOT_INITIALIZED" once it kicks
# in. Disable it before unsloth is imported; Qwen3-VL still runs correctly, just
# without that specific speedup.
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")

import torch
import gc
import numpy as np
from PIL import Image
from unsloth import FastVisionModel
from app.exceptions import AppError, ImageCropError, InferenceError, ModelLoadError, NoIngredientsIdentifiedError
from app.utils.cv.image import crop_image, decode_image_bytes
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
        try:
            # Nạp model với cấu hình 4-bit để tiết kiệm VRAM cho server
            model, tokenizer = FastVisionModel.from_pretrained(
                model_name=weights_path,
                load_in_4bit=True,
            )
            FastVisionModel.for_inference(model)
            model.to(device)
        except Exception as exc:
            self._log_error(f"model_load failed: Qwen3-VL weights_path={weights_path} device={device}")
            raise ModelLoadError(
                "Failed to load the ingredient extraction model (Qwen3-VL).",
                {"model": "qwen3_vl", "weights_path": weights_path},
            ) from exc

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
        self._log_info("step=extraction: enter extract_ingredients")
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

                    self._log_info(f"step=extraction: extracting ingredients for {box_id}")

                    try:
                        # Cắt vùng ảnh chứa thực phẩm
                        crop = crop_image(image, bbox)

                        # Nhận diện nguyên liệu
                        ingredients = self._run_vlm_inference(qwen3_bundle, crop)
                    except ImageCropError as exc:
                        self._log_error(f"step=extraction failed: cannot crop food box {box_id} bbox={bbox}")
                        raise ImageCropError(str(exc), {"step": "extraction", "box_id": box_id, "bbox": bbox}) from exc
                    except torch.cuda.OutOfMemoryError as exc:
                        self._log_error(f"step=extraction failed: GPU out of memory while running Qwen3-VL on box {box_id}")
                        raise InferenceError(
                            "extraction",
                            "GPU ran out of memory while extracting ingredients.",
                            {"step": "extraction", "box_id": box_id, "reason": "cuda_oom"},
                        ) from exc
                    except Exception as exc:
                        self._log_error(
                            f"step=extraction failed: Qwen3-VL inference raised {type(exc).__name__} for box {box_id}"
                        )
                        raise InferenceError(
                            "extraction",
                            "Ingredient extraction model (Qwen3-VL) failed to run.",
                            {"step": "extraction", "box_id": box_id},
                        ) from exc

                    ingredient_map[box_id] = ingredients

            boxes_with_ingredients = sum(1 for v in ingredient_map.values() if v)
            if food_boxes and boxes_with_ingredients == 0:
                self._log_warning(
                    f"step=extraction completed: Qwen3-VL identified zero ingredients across "
                    f"{len(food_boxes)} food box(es)"
                )
                raise NoIngredientsIdentifiedError(detail={"step": "extraction", "food_boxes": len(food_boxes)})

            self._log_info(
                f"step=extraction completed: ingredients found for {boxes_with_ingredients}/{len(food_boxes)} food box(es)"
            )

            return ingredient_map

        except AppError:
            raise
        except Exception as exc:
            self._log_error(f"step=extraction failed: unexpected error {type(exc).__name__}")
            raise InferenceError("extraction", "Ingredient extraction failed due to an internal error.", {"step": "extraction"}) from exc
        finally:
            self._log_info("step=extraction: extraction_service finished")
            gc.collect()

__all__ = ["ExtractionService"]