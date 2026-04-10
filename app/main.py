"""
main.py — FastAPI Application
==============================
Brain Tumor MRI Classification API

Endpoints:
  GET  /           → Serve frontend UI
  GET  /health     → Health check
  POST /predict    → Upload MRI image → get prediction
  GET  /docs       → Swagger UI (auto-generated)
"""

import sys
import time
import logging
from contextlib import asynccontextmanager
from pathlib import Path

# Add root directory to path to allow running directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.predict import load_model, predict

# ── Logging ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
MAX_FILE_SIZE_MB = 10


# ── Lifespan (startup / shutdown) ─────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting Brain Tumor Classifier API...")
    try:
        load_model()
        logger.info("✅ Model ready.")
    except FileNotFoundError as e:
        logger.error(f"❌ {e}")
    yield
    logger.info("🛑 Shutting down.")


# ── App ───────────────────────────────────────────────────
app = FastAPI(
    title="Brain Tumor MRI Classifier",
    description=(
        "Upload a brain MRI scan and get an AI-powered classification into: "
        "Glioma, Meningioma, No Tumor, or Pituitary Tumor."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Routes ────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_ui():
    """Serve the main frontend UI."""
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>Frontend not found.</h1>", status_code=404)
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/health", tags=["Status"])
async def health_check():
    """Returns API health status and model availability."""
    model_path = Path(__file__).parent.parent / "model" / "tumor_classifier.keras"
    return {
        "status": "ok",
        "model_loaded": model_path.exists(),
        "version": "1.0.0",
    }


@app.post("/predict", tags=["Prediction"])
async def predict_tumor(file: UploadFile = File(..., description="Brain MRI image (JPG/PNG)")):
    """
    Classify a brain MRI image.

    - **file**: MRI scan image (JPEG, PNG, WebP, BMP)

    Returns predicted class, confidence score, and per-class probabilities.
    """
    # ── Validate content type ─────────────────────────────
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Use JPEG, PNG, WebP, or BMP.",
        )

    # ── Read & size-check ─────────────────────────────────
    image_bytes = await file.read()
    if len(image_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {MAX_FILE_SIZE_MB} MB.",
        )
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file received.")

    # ── Run inference ─────────────────────────────────────
    try:
        start = time.perf_counter()
        result = predict(image_bytes)
        elapsed = round((time.perf_counter() - start) * 1000, 1)

        logger.info(
            f"Prediction: {result['predicted_class']} "
            f"({result['confidence']:.1f}%) in {elapsed}ms — file: {file.filename}"
        )

        return JSONResponse({
            **result,
            "filename": file.filename,
            "inference_time_ms": elapsed,
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
