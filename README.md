# nutrilens-ai-server

NutriLens AI Server is a FastAPI backend for food detection, ingredient extraction, depth estimation, and nutrition analysis. This repository provides the project skeleton and environment configuration for production deployment.

## 📦 Project structure

```
nutrilens-ai-server/
├── app/
│   ├── main.py
│   ├── api/
│   ├── core/
│   ├── services/
│   ├── utils/
│   ├── schemas/
│   └── db/
├── weights/
├── templates/
├── tests/
├── logs/
├── .env
├── .gitignore
└── requirements.txt
```

## 🔧 Configuration

Environment variables are stored in `.env`. Set `DEVICE=auto` to use CUDA if available.

## 🚀 Running the API

Install dependencies and start the server:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 🧪 Tests

Run the test suite:

```bash
pytest
```

## 📡 API

`POST /v1/nutrition/analyze`

Form data:

- `file`: image upload
- `camera_height_ref`: float
- `pixel_area_ref`: float

Returns nutrition totals and per-ingredient estimates.
