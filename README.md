# nutrilens-ai-server

# Deep Learning-based Ingredient-level Calorie Estimation using Foundation Vision Models

## Abstract
This project presents an end-to-end pipeline for ingredient-level calorie estimation from a single food image. The system integrates object detection, vision-language reasoning, instance segmentation, monocular depth estimation, geometric volume computation, and nutritional quantification based on physical food databases.

## Methodology
The analysis is performed through six tightly coupled stages:

1. **Stage 1 — Object Detection:** YOLO is used to localize food regions and reduce background noise, constraining downstream computation.  
2. **Stage 2 — Semantic Reasoning:** Qwen3-VL infers ingredient semantics from visual features to generate a list of candidate ingredients.  
3. **Stage 3 — Instance Segmentation:** SAM3 produces pixel-level masks for each inferred ingredient within detected food regions.  
4. **Stage 4 — Monocular Depth Estimation:** Depth Anything V2 reconstructs a depth map from a single 2D image to recover 3D structure.  
5. **Stage 5 — Volume Estimation:** Geometric integration combines segmentation masks and depth to estimate ingredient volumes.  
6. **Stage 6 — Nutritional Quantification:** Volumes are mapped to mass and energy using density and nutritional databases.

## Author
- **Đặng Phúc Long** — Class 22T_DT4 — Faculty Information Technology — Email: dangphuclong2019@gmail.com — Phone: 0366646801  
- **Nguyễn Đức Nhã** — Class 22T_KHDL  
- **Trương Bùi Diễn** — Class 24T_KHDL  

## Configuration
Environment variables are stored in `.env`. Set `DEVICE=auto` to use CUDA if available.

Nutrition data is loaded from NutriLens backend internal API:

```env
BACKEND_BASE_URL=http://127.0.0.1:8000
BACKEND_INTERNAL_API_KEY=
BACKEND_INGREDIENTS_PATH=/api/v1/nutrients/internal/ingredients/
BACKEND_API_TIMEOUT=10
```

The AI server maps backend fields as follows:

- `density` -> `density`
- `cal_per_100g / 100` -> `cal`
- `protein_per_100g / 100` -> `protein`
- `fat_per_100g / 100` -> `fat`
- `carb_per_100g / 100` -> `carbs`
- `id` -> `physical_data_id`

## Running the API
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Tests
```bash
pytest
```

## API
`POST /v1/analyze`

Form data:
- `image`: image upload
- `depth_map`: optional depth map upload (`.npy`, `.png`, `.exr`, ...)
- `job_id`: backend inference job id
- `camera_metadata`: JSON string with camera height, intrinsics/pixel area, and optional `depth` metadata

If `depth_map` is present, the server uses client depth. Otherwise it keeps the original Depth Anything V2 flow.

Example `camera_metadata`:

```json
{
  "device_model": "iPhone 15",
  "camera_type": "wide",
  "camera_height_mm": 400,
  "intrinsics": {
    "fx": 2850.2,
    "fy": 2851.7,
    "cx": 2016.0,
    "cy": 1512.0
  },
  "depth": {
    "depth_unit": "meter",
    "source": "client_depth_model",
    "width_px": 4032,
    "height_px": 3024
  }
}
```

Returns the backend contract:

```json
{
  "model_version": "seg-nutrition-v1",
  "latency_ms": 1240,
  "totals": {
    "calories": 520.5,
    "protein": 28.4,
    "carbs": 62.1,
    "fat": 15.7,
    "weight": 430.0
  },
  "components": [
    {
      "component_id": "comp_001",
      "component_name": "Cơm trắng",
      "physical_data_id": "igr_abcd1234",
      "mask_path": "https://res.cloudinary.com/.../comp_001.png",
      "volume": 180.5,
      "weight": 235.0,
      "calories": 305.5,
      "protein": 5.2,
      "carbs": 67.1,
      "fat": 0.6
    }
  ]
}
```

Cloudinary mask upload env:

```env
CLOUDINARY_CLOUD_NAME=
CLOUDINARY_API_KEY=
CLOUDINARY_API_SECRET=
CLOUDINARY_MASK_FOLDER=inference/masks
MASK_LOCAL_DIR=logs/masks
```
