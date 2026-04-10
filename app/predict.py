"""
predict.py — Inference Module
==============================
Loads the trained model once and exposes a predict() function.
"""

import io
import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# ── Constants ─────────────────────────────────────────────
MODEL_PATH  = Path(__file__).parent.parent / "model" / "tumor_classifier.keras"
IMG_SIZE    = (224, 224)
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

CLASS_INFO = {
    "glioma": {
        "label": "Glioma",
        "description": "A tumor that originates in the glial cells of the brain or spine.",
        "severity": "High",
        "color": "#ef4444",
    },
    "meningioma": {
        "label": "Meningioma",
        "description": "A tumor that arises from the meninges — the membranes surrounding the brain and spinal cord.",
        "severity": "Medium",
        "color": "#f97316",
    },
    "notumor": {
        "label": "No Tumor",
        "description": "No tumor detected in the MRI scan.",
        "severity": "None",
        "color": "#22c55e",
    },
    "pituitary": {
        "label": "Pituitary Tumor",
        "description": "A tumor that forms in the pituitary gland at the base of the brain.",
        "severity": "Medium",
        "color": "#a855f7",
    },
}

# ── Model Singleton ───────────────────────────────────────
_model = None


def load_model() -> keras.Model:
    """Load model into memory (called once at startup)."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Please run `python train_model.py` first."
            )
        print(f"[LOAD] Loading model from: {MODEL_PATH}")
        _model = keras.models.load_model(str(MODEL_PATH))
        print("[OK] Model loaded successfully.")
    return _model


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Convert raw image bytes -> float array (1, 224, 224, 3) in [0, 1] range.
    The model contains an internal Rescaling(2, -1) layer that maps [0,1] -> [-1,1]
    as required by MobileNetV2 — so we only divide by 255 here."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0   # [0, 1]
    return np.expand_dims(arr, axis=0)


def predict(image_bytes: bytes) -> dict:
    """
    Run inference on a single MRI image.

    Returns:
        {
            "predicted_class": str,
            "confidence": float,           # 0–100
            "label": str,
            "description": str,
            "severity": str,
            "color": str,
            "scores": {class: confidence, ...}
        }
    """
    model = load_model()
    tensor = preprocess_image(image_bytes)
    probs  = model.predict(tensor, verbose=0)[0]          # shape (4,)

    pred_idx   = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx]) * 100

    scores = {cls: round(float(prob) * 100, 2) for cls, prob in zip(CLASS_NAMES, probs)}

    return {
        "predicted_class": pred_class,
        "confidence": round(confidence, 2),
        **CLASS_INFO[pred_class],
        "scores": scores,
    }
