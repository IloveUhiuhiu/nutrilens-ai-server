from __future__ import annotations

import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from matplotlib import pyplot as plt

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