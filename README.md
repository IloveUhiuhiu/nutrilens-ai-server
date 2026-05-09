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
`POST /v1/nutrition/analyze`

Form data:
- `file`: image upload  
- `camera_height_ref`: float  
- `pixel_area_ref`: float  

Returns nutrition totals and per-ingredient estimates.
