import argparse
import json
import os
from typing import Dict, Tuple, List

import pandas as pd
import requests


def _load_predictions(annotation_path: str) -> Dict[str, Dict[str, float]]:
    with open(annotation_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "predictions" in data:
        data = data["predictions"]
    preds: Dict[str, Dict[str, float]] = {}
    for item in data:
        dish_id = str(item.get("dish_id"))
        preds[dish_id] = {
            "mass": float(item.get("mass", 0.0)),
            "calo": float(item.get("calo", 0.0)),
        }
    return preds


def _load_ground_truth(csv_path: str) -> Dict[str, Dict[str, float]]:
    df = pd.read_csv(csv_path)
    required = {"dish_id", "mass", "calo"}
    if not required.issubset(set(df.columns)):
        missing = required - set(df.columns)
        raise ValueError(f"Missing required columns in GT CSV: {missing}")
    gt: Dict[str, Dict[str, float]] = {}
    for _, row in df.iterrows():
        dish_id = str(row["dish_id"])
        gt[dish_id] = {
            "mass": float(row["mass"]),
            "calo": float(row["calo"]),
        }
    return gt


def _mae(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _mape(errors: List[Tuple[float, float]]) -> float:
    if not errors:
        return 0.0
    return sum(abs(e) / gt for e, gt in errors if gt > 0) / len([gt for _, gt in errors if gt > 0]) if any(gt > 0 for _, gt in errors) else 0.0


def _call_api(api_url: str, image_path: str, camera_height_ref: float, pixel_area_ref: float) -> Dict[str, float]:
    with open(image_path, "rb") as f:
        files = {"file": f}
        data = {
            "camera_height_ref": str(camera_height_ref),
            "pixel_area_ref": str(pixel_area_ref),
        }
        resp = requests.post(api_url, files=files, data=data, timeout=300)
        resp.raise_for_status()
        payload = resp.json()
    summary = payload.get("summary", {})
    return {
        "mass": float(summary.get("total_mass_g", 0.0)),
        "calo": float(summary.get("total_calories_kcal", 0.0)),
    }


def evaluate(images_dir: str, gt_csv_path: str, api_url: str, camera_height_ref: float, pixel_area_ref: float) -> Dict[str, float]:
    gt = _load_ground_truth(gt_csv_path)

    mass_abs_errors: List[float] = []
    calo_abs_errors: List[float] = []
    mass_mape_terms: List[Tuple[float, float]] = []
    calo_mape_terms: List[Tuple[float, float]] = []

    for dish_id, gt_vals in gt.items():
        image_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = os.path.join(images_dir, f"{dish_id}{ext}")
            if os.path.exists(candidate):
                image_path = candidate
                break
        if image_path is None:
            print(f"[WARN] Missing image for dish_id: {dish_id}")
            continue

        pred = _call_api(api_url, image_path, camera_height_ref, pixel_area_ref)

        mass_err = pred["mass"] - gt_vals["mass"]
        calo_err = pred["calo"] - gt_vals["calo"]

        mass_abs_errors.append(abs(mass_err))
        calo_abs_errors.append(abs(calo_err))

        mass_mape_terms.append((mass_err, gt_vals["mass"]))
        calo_mape_terms.append((calo_err, gt_vals["calo"]))

    results = {
        "mass_mae": _mae(mass_abs_errors),
        "mass_mape": _mape(mass_mape_terms),
        "calo_mae": _mae(calo_abs_errors),
        "calo_mape": _mape(calo_mape_terms),
        "num_samples": len(mass_abs_errors),
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate mass and calorie predictions.")
    parser.add_argument("--images_dir", type=str, required=True, help="Path to image folder.")
    parser.add_argument("--gt_csv", type=str, required=True, help="Path to GT CSV with dish_id, mass, calo.")
    parser.add_argument("--api_url", type=str, required=True, help="Nutrition analyze API URL.")
    parser.add_argument("--camera_height_ref", type=float, required=True, help="Camera height reference (cm).")
    parser.add_argument("--pixel_area_ref", type=float, required=True, help="Pixel area reference (cm2).")
    args = parser.parse_args()

    results = evaluate(
        images_dir=args.images_dir,
        gt_csv_path=args.gt_csv,
        api_url=args.api_url,
        camera_height_ref=args.camera_height_ref,
        pixel_area_ref=args.pixel_area_ref,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
