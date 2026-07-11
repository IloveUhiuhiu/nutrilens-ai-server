from __future__ import annotations

import sys
from pathlib import Path
import time
import os
import torch
import gc
from PIL import Image
from app.exceptions import AppError, ImageCropError, InferenceError, ModelLoadError, NoSegmentsProducedError
from app.utils.cv.image import decode_image_bytes, crop_image
from app.utils.cv.segmentation import merge_masks_and_instances
from app.services.base import ServiceBase

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
SAM3_DIR = MODELS_DIR / "SAM3_LoRA"
if str(SAM3_DIR) not in sys.path:
    sys.path.append(str(SAM3_DIR))
os.environ.setdefault("SAM3_ASSETS_DIR", str(SAM3_DIR / "sam3" / "assets"))
from infer_sam import SAM3LoRAInference

class SegmentationService(ServiceBase):
    service_name = "segmentation"

    def __init__(self) -> None:
        pass

    def load_sam3(self, config_path: str, weights_path: str, device: str, conf: float = 0.75) -> dict:
        self._log_info(f"loading sam3 on {device} with conf={conf}")
        """
        Khởi tạo SAM3 LoRA và đóng gói vào bundle kèm ngưỡng tin cậy conf.
        """
        try:
            # detection_threshold trong SAM3 tương đương với conf
            inferencer = SAM3LoRAInference(
                config_path=config_path,
                weights_path=weights_path,
                device=device,
                detection_threshold=conf,
                nms_iou_threshold=0.5
            )
        except Exception as exc:
            self._log_error(
                f"model_load failed: SAM3 LoRA config_path={config_path} weights_path={weights_path} device={device}"
            )
            raise ModelLoadError(
                "Failed to load the segmentation model (SAM3 LoRA).",
                {"model": "sam3", "config_path": config_path, "weights_path": weights_path},
            ) from exc

        return {
            "model": inferencer,
            "device": device,
            "conf": conf,  # Lưu lại ngưỡng mặc định
        }

    def segment_ingredients(
        self,
        image_bytes: bytes,
        ingredients_map: dict[str, list[str]],
        sam3_bundle: dict,
        food_boxes: list[dict],
    ) -> dict:
        self._log_info("step=segmentation: enter segment_ingredients")
        """
        Phân đoạn nguyên liệu sử dụng ngưỡng conf từ bundle hoặc tham số ghi đè.
        """
        start = time.perf_counter()


        threshold = sam3_bundle["conf"]
        temp_path = "/tmp/temp_crop.png"
        segmentation_results = {}

        try:
            image_rgb = decode_image_bytes(image_bytes)
            full_image_shape = image_rgb.shape

            inferencer = sam3_bundle["model"]
            inferencer.model.eval()
            self._log_info("inferencer eval mode")

            with torch.no_grad():
                self._log_info("torch no_grad branch")
                for box in food_boxes:
                    box_id = box["id"]
                    bbox = box["bbox"]
                    target_ingredients = ingredients_map.get(box_id, [])

                    if not target_ingredients:
                        self._log_info(f"no target ingredients for {box_id}")
                        segmentation_results[box_id] = []
                        continue

                    prompts = [ing.strip().lower() for ing in target_ingredients if ing.strip()]

                    try:
                        # Cắt ảnh theo vùng thực phẩm
                        crop_np = crop_image(image_rgb, bbox)
                        Image.fromarray(crop_np).save(temp_path)

                        # Chạy inference trên file tạm
                        predictions = inferencer.predict(temp_path, text_prompts=prompts)
                    except ImageCropError as exc:
                        self._log_error(f"step=segmentation failed: cannot crop food box {box_id} bbox={bbox}")
                        raise ImageCropError(str(exc), {"step": "segmentation", "box_id": box_id, "bbox": bbox}) from exc
                    except torch.cuda.OutOfMemoryError as exc:
                        self._log_error(f"step=segmentation failed: GPU out of memory while running SAM3 on box {box_id}")
                        raise InferenceError(
                            "segmentation",
                            "GPU ran out of memory while segmenting ingredients.",
                            {"step": "segmentation", "box_id": box_id, "reason": "cuda_oom"},
                        ) from exc
                    except Exception as exc:
                        self._log_error(
                            f"step=segmentation failed: SAM3 inference raised {type(exc).__name__} "
                            f"for box {box_id} prompts={prompts}"
                        )
                        raise InferenceError(
                            "segmentation",
                            "Ingredient segmentation model (SAM3) failed to run.",
                            {"step": "segmentation", "box_id": box_id},
                        ) from exc

                    masks_for_this_crop = []
                    for p_idx, prompt_text in enumerate(prompts):
                        res = predictions.get(p_idx)

                        if res and res['num_detections'] > 0:
                            self._log_info(f"detections found for {box_id}:{prompt_text}")
                            masks = res.get('masks')
                            scores = res.get('scores')

                            if masks is not None and scores is not None:
                                for m, score in zip(masks, scores):
                                    # Kiểm tra ngưỡng tin cậy cho từng instance
                                    if float(score) >= threshold:
                                        self._log_info(f"score pass threshold for {box_id}:{prompt_text}")
                                        masks_for_this_crop.append((prompt_text, m))
                        else:
                            self._log_info(f"no detections for {box_id}:{prompt_text}")

                    segmentation_results[box_id] = masks_for_this_crop

            # Hợp nhất các mask về hệ tọa độ ảnh gốc
            food_bboxes_map = {box["id"]: box["bbox"] for box in food_boxes}
            global_masks, instance_masks = merge_masks_and_instances(
                segmentation_results,
                food_bboxes_map,
                full_image_shape
            )

            if food_boxes and not global_masks:
                self._log_warning("step=segmentation completed: zero ingredient masks produced by SAM3")
                raise NoSegmentsProducedError(detail={"step": "segmentation", "food_boxes": len(food_boxes)})

            self._log_info(
                f"step=segmentation completed: global_masks={len(global_masks)}, "
                f"instance_groups={len(instance_masks)}"
            )

            return {
                "global_masks": global_masks,
                "instance_masks": instance_masks
            }

        except AppError:
            raise
        except Exception as exc:
            self._log_error(f"step=segmentation failed: unexpected error {type(exc).__name__}")
            raise InferenceError("segmentation", "Ingredient segmentation failed due to an internal error.", {"step": "segmentation"}) from exc
        finally:
            self._log_info("step=segmentation: segmentation_service finished")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            gc.collect()