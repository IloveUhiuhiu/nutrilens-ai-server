# Nutrilens AI - Web Frontend Guide

## 📋 Overview

The Nutrilens AI system now includes a complete web-based frontend built with vanilla HTML, CSS, and JavaScript. This interface allows users to:

- ✅ Upload dish images
- ✅ Configure reference parameters
- ✅ Monitor real-time processing with progress visualization
- ✅ View detailed nutrition analysis results
- ✅ Debug with intermediate processing images
- ✅ Export comprehensive debug data to disk

## 🚀 Getting Started

### 1. Start the Server

```bash
cd /teamspace/studios/this_studio/nutrilens-ai-server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Access the Web Interface

Open your browser and navigate to:
```
http://localhost:8000
```

## 🎨 Frontend Features

### Input Form (Left Column)

**Upload Section:**
- Select your dish image file
- Supported formats: JPEG, PNG, etc.

**Reference Parameters:**
- **Camera Height Reference**: Distance from camera to reference surface
  - Options: cm (centimeters) or mm (millimeters)
  - Default: 30 cm
  
- **Pixel Area Reference**: Calibration value for pixel-to-area conversion
  - Options: cm² or mm²
  - Default: 0.000625 cm²

**Action Buttons:**
- 👁️ **Preview**: Show image preview and metadata
- 🚀 **Process**: Start the analysis pipeline

### Image Preview (Right Column)

After clicking "Preview":
- Displays the uploaded image
- Shows file information:
  - Filename
  - Image dimensions (width × height)
  - File size
  - Resolution (PPI)

### Processing UI

While analysis is running:

**Loading Spinner:**
- Visual feedback showing the analysis is in progress

**Progress Bar:**
- Graphical representation of overall progress
- Current step name display
- Elapsed time counter

**Steps Log:**
- Real-time pipeline step tracking
- Status icons for each step:
  - ◯ Pending
  - ⟳ Processing (animated)
  - ✓ Completed
  - ✕ Error
- Step descriptions and timing

### Results Tabs

After analysis completes, view results in different tabs:

#### 1. 📊 Summary Tab
- Total nutrition values in card format:
  - Total Mass
  - Total Calories
  - Total Protein
  - Total Fat
  - Total Carbohydrates

#### 2. 📋 Details Tab
- Detailed table showing:
  - Ingredient name
  - Matched database entry
  - Mass (grams)
  - Calories (kcal)
  - Protein (grams)
  - Fat (grams)
  - Carbohydrates (grams)
  - Confidence score (%)

#### 3. 🔍 Debug Tab
- Visual gallery of processing pipeline intermediate images:
  - Original image
  - Food detection boxes
  - Plate mask
  - Extracted ingredients
  - Instance masks
  - Combined food mask
  - Depth map (raw)
  - Plate depth (inpainted)
  - Geometry results
  - Nutrition results

Each image is labeled with its step in the pipeline.

#### 4. 📝 Raw Data Tab
- Raw JSON response from the API
- Useful for developers and debugging
- Full response data in formatted JSON

## 📱 Responsive Design

The frontend is fully responsive:
- **Desktop**: Multi-column layout
- **Tablet**: Adjusted grid layout
- **Mobile**: Single column layout

## 🛠️ API Endpoints

### Main Analysis Endpoint (with debug info)
```
POST /api/v1/nutrition/analyze_debug
```

**Request:**
```multipart/form-data
- file: <image file>
- camera_height_ref: <float> (in cm)
- pixel_area_ref: <float> (in cm²)
```

**Response:**
```json
{
  "items": [
    {
      "ingredient": "rice",
      "matched_name": "rice",
      "confidence": 0.95,
      "mass_g": 120.5,
      "calories_kcal": 130.6,
      "protein_g": 2.7,
      "fat_g": 0.3,
      "carbs_g": 28.2
    }
  ],
  "summary": {
    "total_mass_g": 250.0,
    "total_calories_kcal": 350.0,
    "total_protein_g": 12.5,
    "total_fat_g": 5.0,
    "total_carbs_g": 45.0
  },
  "device": "cuda",
  "processing_time_s": 45.23,
  "debug_info": {
    "images": {
      "original": "base64_encoded_image...",
      "detection": "base64_encoded_image...",
      ...
    },
    "texts": {
      "ingredients": "json_string..."
    },
    "processing_time_s": 45.23,
    "device": "cuda"
  }
}
```

## 📁 Directory Structure

```
nutrilens-ai-server/
├── templates/
│   └── index.html          # Main HTML interface
├── static/
│   ├── css/
│   │   └── style.css       # All styling
│   └── js/
│       └── app.js          # Frontend logic & API communication
├── app/
│   ├── main.py             # FastAPI application setup
│   ├── api/
│   │   └── v1/
│   │       └── nutrition.py # API endpoints (includes analyze_debug)
│   ├── services/
│   │   └── nutrition_pipeline.py # Pipeline with debug support
│   ├── schemas/
│   │   └── response.py     # Response models with debug info
│   └── utils/
│       └── visualization/
│           └── debug.py    # DebugVisualizer with in-memory support
```

## 🔧 Configuration

Edit `.env` file to configure:

```properties
# Device (cuda or cpu)
DEVICE=cuda

# Reference Parameters Defaults
CAMERA_HEIGHT_REF=30      # cm
PIXEL_AREA_REF=0.000625   # cm²

# Model paths
YOLO_FOOD_WEIGHTS=weights/yolo/food_yolo.pt
YOLO_PLATE_WEIGHTS=weights/yolo/plate_yolo_seg.pt
QWEN3VL_WEIGHTS=weights/qwen3vl
SAM3_CONFIG_PATH=weights/sam3/food_config.yaml
SAM3_WEIGHTS=weights/sam3/sam3_lora.pt
DEPTH_ENCODER=vits
DEPTHANYTHING_WEIGHTS=weights/da2/depth_anything_v2_vits
```

## 💡 Tips & Tricks

1. **Unit Conversion**: The frontend automatically converts between cm/mm and cm²/mm²
2. **Real-time Progress**: The progress bar updates every 0.5 seconds
3. **Error Handling**: Network errors and API errors are displayed in alert boxes
4. **Debug Images**: All intermediate processing steps are captured and displayed
5. **Performance**: Processing time varies based on image size and device (GPU/CPU)

## 🐛 Troubleshooting

### Images Not Loading
- Check browser console (F12) for CORS errors
- Ensure static files are properly mounted in FastAPI

### Processing Fails
- Check `/api/v1/nutrition/analyze_debug` endpoint is responding
- Verify model weights are loaded correctly
- Check server logs for detailed error messages

### Slow Processing
- Use CUDA device if available (check Device info)
- Reduce image resolution
- Check GPU memory usage

## 🎯 Next Steps

- Add more visualization charts (nutrition pie charts, etc.)
- Implement image cropping/editing before processing
- Add batch processing support
- Implement results history/database
- Add recipe suggestions based on detected ingredients
- Export results to PDF/CSV

---

For more information, see the main [README.md](./README.md)
