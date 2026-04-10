# Brain Tumor MRI Classifier

## Overview
End-to-end AI pipeline to classify brain MRI scans into 4 categories:
- **Glioma**
- **Meningioma**
- **No Tumor**
- **Pituitary Tumor**

**Model**: EfficientNetB0 (Transfer Learning + Fine-tuning)  
**Backend**: FastAPI + Uvicorn  
**Frontend**: Modern drag-and-drop web UI  

---

## Project Structure

```
Tumor Classification/
├── Dataset/
│   ├── Training/   (glioma, meningioma, notumor, pituitary)
│   └── Testing/    (glioma, meningioma, notumor, pituitary)
├── train_model.py          ← Run this first
├── model/
│   └── tumor_classifier.keras
├── plots/
│   ├── phase1_history.png
│   ├── phase2_history.png
│   └── confusion_matrix.png
├── app/
│   ├── __init__.py
│   ├── main.py             ← FastAPI app
│   ├── predict.py          ← Inference logic
│   └── static/
│       └── index.html      ← Web UI
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### 1. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Step 1 — Train the Model

```bash
python train_model.py
```

This will:
- Train EfficientNetB0 in **two phases** (feature extraction → fine-tuning)
- Save the best model to `model/tumor_classifier.keras`
- Save training plots to `plots/`
- Print a full classification report and confusion matrix

---

## Step 2 — Run the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The server starts at: **http://localhost:8000**

| URL | Description |
|-----|-------------|
| `http://localhost:8000/` | Web UI |
| `http://localhost:8000/docs` | Swagger API docs |
| `http://localhost:8000/health` | Health check |
| `http://localhost:8000/predict` | POST — image upload |

---

## API Usage

### `POST /predict`

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@path/to/mri.jpg"
```

**Response:**
```json
{
  "predicted_class": "glioma",
  "confidence": 97.43,
  "label": "Glioma",
  "description": "A tumor that originates in the glial cells...",
  "severity": "High",
  "color": "#ef4444",
  "scores": {
    "glioma": 97.43,
    "meningioma": 1.02,
    "notumor": 0.98,
    "pituitary": 0.57
  },
  "filename": "mri.jpg",
  "inference_time_ms": 45.2
}
```

---

## Notes
- Model must be trained before running the API
- Recommended: NVIDIA GPU for faster training (CPU works too)
- For production deployment, use `gunicorn` with `uvicorn` workers
