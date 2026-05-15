from __future__ import annotations

import base64
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from io import BytesIO
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

class DebugVisualizer:
    def __init__(self, root_dir: str = "debug_outputs", to_memory: bool = True):
        self.to_memory = to_memory  # Also return to frontend
        self.images_bytes: dict[str, bytes] = {}  # For frontend
        self.texts: dict[str, str] = {}  # For frontend
        
        # ALWAYS create disk output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(root_dir) / timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[DebugVisualizer] Output directory: {self.output_dir}")

    def _safe_savefig(self, filename: str):
        """Save figure to both memory (for frontend) and disk"""
        plt.tight_layout()
        
        # Save to disk (always)
        disk_path = str(self.output_dir / filename)
        plt.savefig(disk_path, dpi=150, bbox_inches="tight")
        
        # Save to memory (for frontend)
        if self.to_memory:
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches="tight")
            buf.seek(0)
            self.images_bytes[filename] = buf.getvalue()
        
        plt.close()

    def _save_or_store_image(self, filename: str, data: bytes):
        """Save image bytes to both disk and memory"""
        # Save to disk (always)
        disk_path = self.output_dir / filename
        with open(disk_path, "wb") as f:
            f.write(data)
        
        # Save to memory (for frontend)
        if self.to_memory:
            self.images_bytes[filename] = data

    def _safe_imshow(self, ax, image, title: str, cmap: str | None = None):
        if image is None:
            ax.axis("off")
            ax.set_title(f"{title} (missing)")
            return
        ax.imshow(image, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")

    def save_image_rgb(self, filename: str, image_rgb: np.ndarray):
        if image_rgb is None or image_rgb.size == 0:
            return

        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        is_success, buffer = cv2.imencode(".png", image_bgr)
        if is_success:
            self._save_or_store_image(filename, buffer.tobytes())

    def save_mask(self, filename: str, mask: np.ndarray):
        if mask is None or mask.size == 0:
            return

        mask_vis = (mask.astype(np.uint8) * 255)
        is_success, buffer = cv2.imencode(".png", mask_vis)
        if is_success:
            self._save_or_store_image(filename, buffer.tobytes())

    def save_depth(self, filename: str, depth_map: np.ndarray):
        if depth_map is None or depth_map.size == 0:
            return

        plt.figure(figsize=(8, 8))
        plt.imshow(
            depth_map,
            cmap="inferno",
            vmin=0.0,
            vmax=40.0
        )
        plt.colorbar(label="Depth (cm)")
        plt.axis("off")
        plt.title(filename)
        
        # Save to disk (always)
        disk_path = str(self.output_dir / filename)
        plt.savefig(disk_path, bbox_inches="tight", pad_inches=0, dpi=150)
        
        # Save to memory (for frontend)
        if self.to_memory:
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches="tight", pad_inches=0, dpi=150)
            buf.seek(0)
            self.images_bytes[filename] = buf.getvalue()
        
        plt.close()

    def save_text(self, filename: str, content: str):
        """Save text to both disk and memory"""
        # Save to disk (always)
        disk_path = self.output_dir / filename
        with open(disk_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        # Save to memory (for frontend)
        if self.to_memory:
            self.texts[filename] = content

    def save_json(self, filename: str, data):
        """Save JSON to both disk and memory"""
        content = json.dumps(data, indent=2, ensure_ascii=False)
        
        # Save to disk (always)
        disk_path = self.output_dir / filename
        with open(disk_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        # Save to memory (for frontend)
        if self.to_memory:
            self.texts[filename] = content

    def get_results_as_dict(self) -> dict:
        """Returns all stored debug info as a dictionary with base64 encoded images."""
        return {
            "images": {
                Path(key).stem: base64.b64encode(value).decode("utf-8")
                for key, value in self.images_bytes.items()
            },
            "texts": {
                Path(key).stem: value
                for key, value in self.texts.items()
            }
        }

    def save_detection_boxes(self, filename: str, image_rgb: np.ndarray, boxes: list[dict]):
        img = image_rgb.copy()

        for box in boxes:
            x1, y1, x2, y2 = box["bbox"]

            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                img,
                box["id"],
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )

        self.save_image_rgb(filename, img)
        return img

    def save_global_masks_overlay(
        self,
        filename: str,
        image_rgb: np.ndarray,
        masks: dict[str, np.ndarray]
    ):
        overlay = image_rgb.copy()

        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]

        for idx, (name, mask) in enumerate(masks.items()):
            color = colors[idx % len(colors)]

            colored_mask = np.zeros_like(overlay)
            colored_mask[mask > 0] = color

            overlay = cv2.addWeighted(
                overlay,
                1.0,
                colored_mask,
                0.4,
                0
            )

        self.save_image_rgb(filename, overlay)
        return overlay
    
    def save_topological_order_overlay(
        self,
        filename: str | None,
        image_rgb: np.ndarray,
        instance_masks: list[np.ndarray],
        instance_labels: list[str],
        sorted_idx: list[int]
    ):
        overlay = image_rgb.copy()

        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]

        for order, idx in enumerate(sorted_idx):
            mask = instance_masks[idx]
            label = instance_labels[idx]

            color = colors[order % len(colors)]

            # tô màu mask
            colored = np.zeros_like(overlay)
            colored[mask > 0] = color

            overlay = cv2.addWeighted(
                overlay,
                1.0,
                colored,
                0.4,
                0
            )

            # tìm center để vẽ text
            ys, xs = np.where(mask > 0)

            if len(xs) == 0:
                continue

            cx = int(np.mean(xs))
            cy = int(np.mean(ys))

            text = f"{order}: {label}"

            cv2.putText(
                overlay,
                text,
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        if filename:
            self.save_image_rgb(filename, overlay)
        return overlay

    def save_dashboard(
        self,
        filename: str,
        original_rgb: np.ndarray,
        boxes_rgb: np.ndarray,
        plate_mask: np.ndarray,
        ingredient_overlay: np.ndarray,
        plate_depth: np.ndarray,
        merged_depth: np.ndarray,
        topo_overlay: np.ndarray,
        nutrition_results: dict,
        geometry_results: list[dict],
        food_heights: np.ndarray,
        ground_truth=None,
        dish_id: str | None = None,
    ):
        """
        Lưu dashboard tổng hợp toàn bộ pipeline phân tích dinh dưỡng.
        """

        fig = plt.figure(figsize=(24, 18))

        gs = GridSpec(
            4,
            3,
            figure=fig,
            hspace=0.25,
            wspace=0.15
        )

        # =========================================================
        # 1. ORIGINAL IMAGE
        # =========================================================
        ax1 = fig.add_subplot(gs[0, 0])
        self._safe_imshow(ax1, original_rgb, "Original Image")

        # =========================================================
        # 2. DETECTION BOXES
        # =========================================================
        ax2 = fig.add_subplot(gs[0, 1])
        self._safe_imshow(ax2, boxes_rgb, "Detection Boxes")

        # =========================================================
        # 3. PLATE MASK
        # =========================================================
        ax3 = fig.add_subplot(gs[0, 2])
        self._safe_imshow(ax3, plate_mask, "Plate Mask", cmap="gray")

        # =========================================================
        # 4. INGREDIENT MASKS
        # =========================================================
        ax4 = fig.add_subplot(gs[1, 0])
        self._safe_imshow(ax4, ingredient_overlay, "Ingredient Masks")

        # =========================================================
        # 5. RESTORED PLATE DEPTH
        # =========================================================
        ax5 = fig.add_subplot(gs[1, 1])
        if plate_depth is not None:
            im5 = ax5.imshow(plate_depth, cmap="inferno", vmin=0, vmax=40)
            ax5.set_title("Restored Plate Depth")
            ax5.axis("off")
            plt.colorbar(im5, ax=ax5, fraction=0.046)
        else:
            self._safe_imshow(ax5, None, "Restored Plate Depth")

        # =========================================================
        # 6. MERGED DEPTH
        # =========================================================
        ax6 = fig.add_subplot(gs[1, 2])
        if merged_depth is not None:
            im6 = ax6.imshow(merged_depth, cmap="inferno", vmin=0, vmax=40)
            ax6.set_title("Merged Food + Plate Depth")
            ax6.axis("off")
            plt.colorbar(im6, ax=ax6, fraction=0.046)
        else:
            self._safe_imshow(ax6, None, "Merged Food + Plate Depth")

        # =========================================================
        # 7. TOPOLOGICAL ORDER
        # =========================================================
        ax7 = fig.add_subplot(gs[2, 0])
        self._safe_imshow(ax7, topo_overlay, "Topological Order")

        # =========================================================
        # 8. NUTRITION TABLE
        # =========================================================
        ax8 = fig.add_subplot(gs[2, 1:])
        ax8.axis("off")

        lines = []

        for item in geometry_results:

            ing = item.get("ingredient", "unknown")

            nutrition = nutrition_results[
                "ingredients"
            ].get(ing, {})

            line = (
                f"{ing:<18} | "
                f"Vol={item.get('volume_cm3', 0):7.1f} cm3 | "
                f"Mass={nutrition.get('mass_g', 0):7.1f} g | "
                f"Cal={nutrition.get('calories_kcal', 0):7.1f} kcal"
            )

            lines.append(line)

        total = nutrition_results["total"]

        pred_mass = float(
            total.get("mass_g", 0.0)
        )

        pred_cal = float(
            total.get("calories_kcal", 0.0)
        )

        lines.append("")
        lines.append(
            f"TOTAL MASS      : {pred_mass:.2f} g"
        )

        lines.append(
            f"TOTAL CALORIES  : {pred_cal:.2f} kcal"
        )

        ax8.text(
            0.01,
            0.98,
            "\n".join(lines),
            fontsize=11,
            va="top",
            family="monospace"
        )

        ax8.set_title("Nutrition Summary")

        # =========================================================
        # 9. PREDICTION / GROUND TRUTH
        # =========================================================
        ax9 = fig.add_subplot(gs[3, 0])
        ax9.axis("off")

        metric_lines = []

        metric_lines.append(
            f"Dish ID          : {dish_id}"
        )

        metric_lines.append("")

        metric_lines.append(
            f"Pred Mass        : {pred_mass:.2f} g"
        )

        metric_lines.append(
            f"Pred Calories    : {pred_cal:.2f} kcal"
        )

        gt_found = False

        # =========================================================
        # CHECK GROUND TRUTH CSV
        # =========================================================
        if (
            ground_truth is not None
            and hasattr(ground_truth, "empty")
            and not ground_truth.empty
            and dish_id is not None
        ):

            gt_row = ground_truth[
                ground_truth["dish_id"] == dish_id
            ]

            if not gt_row.empty:

                gt_found = True

                gt_mass = float(
                    gt_row.iloc[0]["total_mass"]
                )

                gt_cal = float(
                    gt_row.iloc[0]["total_calories"]
                )

                mae_mass = abs(
                    pred_mass - gt_mass
                )

                mae_cal = abs(
                    pred_cal - gt_cal
                )

                mape_mass = (
                    mae_mass / gt_mass * 100.0
                    if gt_mass > 1e-6 else 0.0
                )

                mape_cal = (
                    mae_cal / gt_cal * 100.0
                    if gt_cal > 1e-6 else 0.0
                )

                metric_lines.append("")
                metric_lines.append(
                    f"GT Mass          : {gt_mass:.2f} g"
                )

                metric_lines.append(
                    f"GT Calories      : {gt_cal:.2f} kcal"
                )

                metric_lines.append("")

                metric_lines.append(
                    f"Mass MAPE        : {mape_mass:.2f} %"
                )

                metric_lines.append(
                    f"Calories MAPE    : {mape_cal:.2f} %"
                )

        ax9.text(
            0.05,
            0.95,
            "\n".join(metric_lines),
            fontsize=13,
            va="top",
            family="monospace"
        )

        if gt_found:
            ax9.set_title(
                "Prediction vs Ground Truth"
            )
        else:
            ax9.set_title(
                "Prediction Summary"
            )

        # =========================================================
        # 10. HEIGHT HISTOGRAM
        # =========================================================
        ax10 = fig.add_subplot(gs[3, 1:])
        if food_heights is None or food_heights.size == 0:
            ax10.set_title("Food Height Histogram (no data)")
            ax10.axis("off")
        else:
            valid_heights = food_heights[food_heights > 0]
            if valid_heights.size == 0:
                ax10.set_title("Food Height Histogram (no data)")
                ax10.axis("off")
            else:
                p_low = np.percentile(valid_heights, 1)
                p_high = np.percentile(valid_heights, 99)
                valid_heights = np.clip(valid_heights, p_low, p_high)
                ax10.hist(valid_heights.flatten(), bins=40)
                ax10.set_title(
                    f"Food Height Histogram\nClamp [{p_low:.2f}, {p_high:.2f}] cm"
                )
                ax10.set_xlabel("Height (cm)")
                ax10.set_ylabel("Pixel Count")

        self._safe_savefig(filename)

    def run_debug_visuals(
        self,
        dish_id: str,
        image_rgb: np.ndarray,
        detections: dict,
        ingredients_map: dict,
        segments: dict,
        food_mask_combined: np.ndarray,
        depth_data: dict,
        geometry_data: dict,
        nutrition_results: dict,
        ground_truth,
    ):
        self.save_image_rgb(
            "01_original.jpg",
            image_rgb
        )

        boxes_overlay = self.save_detection_boxes(
            "02_detection_boxes.jpg",
            image_rgb,
            detections["food_boxes"]
        )

        if detections["plate_mask"]["mask"] is not None:
            self.save_mask(
                "03_plate_mask.png",
                detections["plate_mask"]["mask"]
            )

        self.save_json(
            "04_extracted_ingredients.json",
            ingredients_map
        )

        ingredient_overlay = self.save_global_masks_overlay(
            "05_global_masks.png",
            image_rgb,
            segments["global_masks"]
        )

        merged_depth = depth_data["plate_depth"].copy()
        food_mask = food_mask_combined > 0
        merged_depth[food_mask] = depth_data["depth_map"][food_mask]

        food_heights = np.zeros_like(
            depth_data["depth_map"],
            dtype=np.float32
        )
        food_heights[food_mask] = (
            depth_data["plate_depth"][food_mask]
            - depth_data["depth_map"][food_mask]
        )
        food_heights = np.clip(food_heights, 0, None)

        self.save_depth(
            "06_depth_map.png",
            depth_data["depth_map"]
        )
        self.save_depth(
            "07_plate_restored.png",
            depth_data["plate_depth"]
        )

        geometry = geometry_data["geometry"]
        topo_overlay = self.save_topological_order_overlay(
            "08_topological_order.png",
            image_rgb,
            geometry_data["instance_masks"],
            geometry_data["instance_labels"],
            geometry_data["topological_order"]
        )
        self.save_json(
            "09_geometry.json",
            geometry
        )

        self.save_json(
            "10_nutrition.json",
            nutrition_results
        )

        self.save_dashboard(
            filename="dashboard.png",
            original_rgb=image_rgb,
            boxes_rgb=boxes_overlay,
            plate_mask=detections["plate_mask"]["mask"],
            ingredient_overlay=ingredient_overlay,
            plate_depth=depth_data["plate_depth"],
            merged_depth=merged_depth,
            topo_overlay=topo_overlay,
            nutrition_results=nutrition_results,
            geometry_results=geometry,
            food_heights=food_heights,
            ground_truth=ground_truth,
            dish_id=dish_id
        )