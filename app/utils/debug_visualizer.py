from __future__ import annotations

import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

class DebugVisualizer:
    def __init__(self, root_dir: str = "debug_outputs"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(root_dir) / timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_image_rgb(self, filename: str, image_rgb: np.ndarray):
        if image_rgb is None or image_rgb.size == 0:
            return

        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(self.output_dir / filename), image_bgr)

    def save_mask(self, filename: str, mask: np.ndarray):
        mask_vis = (mask.astype(np.uint8) * 255)
        cv2.imwrite(str(self.output_dir / filename), mask_vis)

    def save_depth(self, filename: str, depth_map: np.ndarray):
        plt.figure(figsize=(8, 8))
        plt.imshow(
            depth_map,
            cmap="inferno",
            vmin=0.0,
            vmax=40.0
        )
        plt.colorbar(label="Depth (cm)")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(
            str(self.output_dir / filename),
            bbox_inches="tight",
            pad_inches=0
        )
        plt.close()

    def save_text(self, filename: str, content: str):
        with open(self.output_dir / filename, "w", encoding="utf-8") as f:
            f.write(content)

    def save_json(self, filename: str, data):
        with open(self.output_dir / filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

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
    
    def save_topological_order_overlay(
        self,
        filename: str,
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

        ax1.imshow(original_rgb)

        ax1.set_title("Original Image")
        ax1.axis("off")

        # =========================================================
        # 2. DETECTION BOXES
        # =========================================================
        ax2 = fig.add_subplot(gs[0, 1])

        ax2.imshow(boxes_rgb)

        ax2.set_title("Detection Boxes")
        ax2.axis("off")

        # =========================================================
        # 3. PLATE MASK
        # =========================================================
        ax3 = fig.add_subplot(gs[0, 2])

        ax3.imshow(
            plate_mask,
            cmap="gray"
        )

        ax3.set_title("Plate Mask")
        ax3.axis("off")

        # =========================================================
        # 4. INGREDIENT MASKS
        # =========================================================
        ax4 = fig.add_subplot(gs[1, 0])

        ax4.imshow(ingredient_overlay)

        ax4.set_title("Ingredient Masks")
        ax4.axis("off")

        # =========================================================
        # 5. RESTORED PLATE DEPTH
        # =========================================================
        ax5 = fig.add_subplot(gs[1, 1])

        im5 = ax5.imshow(
            plate_depth,
            cmap="inferno",
            vmin=0,
            vmax=40
        )

        ax5.set_title("Restored Plate Depth")
        ax5.axis("off")

        plt.colorbar(
            im5,
            ax=ax5,
            fraction=0.046
        )

        # =========================================================
        # 6. MERGED DEPTH
        # =========================================================
        ax6 = fig.add_subplot(gs[1, 2])

        im6 = ax6.imshow(
            merged_depth,
            cmap="inferno",
            vmin=0,
            vmax=40
        )

        ax6.set_title("Merged Food + Plate Depth")
        ax6.axis("off")

        plt.colorbar(
            im6,
            ax=ax6,
            fraction=0.046
        )

        # =========================================================
        # 7. TOPOLOGICAL ORDER
        # =========================================================
        ax7 = fig.add_subplot(gs[2, 0])

        ax7.imshow(topo_overlay)

        ax7.set_title("Topological Order")
        ax7.axis("off")

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

        valid_heights = food_heights[
            food_heights > 0
        ]

        if len(valid_heights) > 0:

            ax10.hist(
                valid_heights.flatten(),
                bins=40
            )

        ax10.set_title(
            "Food Height Histogram"
        )

        ax10.set_xlabel(
            "Height (cm)"
        )

        ax10.set_ylabel(
            "Pixel Count"
        )

        # =========================================================
        # SAVE
        # =========================================================
        plt.tight_layout()

        plt.savefig(
            str(self.output_dir / filename),
            dpi=250,
            bbox_inches="tight"
        )

        plt.close()