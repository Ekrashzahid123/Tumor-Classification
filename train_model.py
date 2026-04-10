"""
Brain Tumor MRI Classification - Model Training Script
=======================================================
Dataset : 4 classes -> glioma, meningioma, notumor, pituitary
Model   : MobileNetV2 (Transfer Learning + Fine-tuning)
          - Faster & lighter than EfficientNetB0
          - Works reliably with TF 2.20 / Keras 3
Output  : model/tumor_classifier.keras + training plots
"""

import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
)
from sklearn.metrics import classification_report, confusion_matrix

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
DATASET_DIR   = BASE_DIR / "Dataset"
TRAIN_DIR     = DATASET_DIR / "Training"
TEST_DIR      = DATASET_DIR / "Testing"
MODEL_DIR     = BASE_DIR / "model"
PLOTS_DIR     = BASE_DIR / "plots"

MODEL_PATH    = MODEL_DIR / "tumor_classifier.keras"
HISTORY_PATH  = MODEL_DIR / "training_history.json"

IMG_SIZE      = (224, 224)
BATCH_SIZE    = 32
EPOCHS_PHASE1 = 15        # Feature extraction (frozen backbone)
EPOCHS_PHASE2 = 20        # Fine-tuning (top layers unfrozen)
LR_PHASE1     = 1e-3
LR_PHASE2     = 1e-5
SEED          = 42

CLASS_NAMES   = ["glioma", "meningioma", "notumor", "pituitary"]

# ─────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────
MODEL_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Force UTF-8 output so Windows terminal does not choke on special chars
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

print(f"TensorFlow version : {tf.__version__}")
print(f"GPUs available     : {len(tf.config.list_physical_devices('GPU'))}")
print(f"Backbone           : MobileNetV2 (imagenet)")
print(f"Training dir       : {TRAIN_DIR}")
print(f"Testing dir        : {TEST_DIR}")
print("-" * 55)


# ─────────────────────────────────────────────────────────────
# Data Generators
# MobileNetV2 preprocess_input expects [-1, 1].
# We rescale [0,255] -> [0,1] in the generator, then the model
# head converts [0,1] -> [-1,1] via a Rescaling layer.
# ─────────────────────────────────────────────────────────────
def build_generators():
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.10,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.85, 1.15],
        fill_mode="nearest",
        validation_split=0.15,
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        subset="training",
        seed=SEED,
        shuffle=True,
    )

    val_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        subset="validation",
        seed=SEED,
        shuffle=False,
    )

    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=False,
    )

    print(f"\n[OK] Training samples   : {train_gen.samples}")
    print(f"[OK] Validation samples : {val_gen.samples}")
    print(f"[OK] Test samples       : {test_gen.samples}")
    print(f"[OK] Class indices      : {train_gen.class_indices}\n")

    return train_gen, val_gen, test_gen


# ─────────────────────────────────────────────────────────────
# Model Architecture
# ─────────────────────────────────────────────────────────────
def build_model(num_classes: int = 4):
    """
    MobileNetV2 backbone + custom classification head.
    Input  : [0, 1] float (from generator rescale=1/255)
    Layer 1: Rescaling [0,1] -> [-1,1]  (MobileNetV2 expects this)
    Backbone: MobileNetV2 (frozen initially)
    Head   : GAP -> BN -> Dropout -> Dense(256) -> BN -> Dropout -> Softmax
    """
    inputs = keras.Input(shape=(*IMG_SIZE, 3), name="input_image")

    # Convert [0,1] -> [-1,1] as MobileNetV2 expects
    x = layers.Rescaling(scale=2.0, offset=-1.0, name="rescale_to_mobilenet")(inputs)

    # MobileNetV2 backbone — frozen initially
    backbone = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3),
        alpha=1.0,
    )
    backbone.trainable = False

    x = backbone(x, training=False)

    # Classification head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Dropout(0.4, name="drop1")(x)
    x = layers.Dense(256, activation="relu", name="dense1")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Dropout(0.3, name="drop2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="BrainTumorClassifier")
    return model, backbone


# ─────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────
def get_callbacks(phase: int) -> list:
    return [
        ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-8,
            verbose=1,
        ),
        CSVLogger(
            str(MODEL_DIR / f"phase{phase}_log.csv"),
            append=False,
        ),
    ]


# ─────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────
def plot_history(history, phase: int):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Training History - Phase {phase}", fontsize=14, fontweight="bold")

    axes[0].plot(history.history["accuracy"],     label="Train Acc",  linewidth=2)
    axes[0].plot(history.history["val_accuracy"], label="Val Acc",    linewidth=2)
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["loss"],     label="Train Loss", linewidth=2)
    axes[1].plot(history.history["val_loss"], label="Val Loss",   linewidth=2)
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = PLOTS_DIR / f"phase{phase}_history.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] History plot saved -> {path}")


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    path = PLOTS_DIR / "confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Confusion matrix saved -> {path}")


# ─────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────
def evaluate_model(model, test_gen):
    print("\n" + "=" * 55)
    print("EVALUATION ON TEST SET")
    print("=" * 55)

    test_gen.reset()
    loss, accuracy = model.evaluate(test_gen, verbose=1)
    print(f"\n[OK] Test Loss     : {loss:.4f}")
    print(f"[OK] Test Accuracy : {accuracy * 100:.2f}%\n")

    test_gen.reset()
    preds  = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_gen.classes

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    plot_confusion_matrix(y_true, y_pred)
    return loss, accuracy


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    # ── Data ──────────────────────────────────────────────────
    train_gen, val_gen, test_gen = build_generators()

    # ── Build model ───────────────────────────────────────────
    model, backbone = build_model(num_classes=len(CLASS_NAMES))
    model.summary()

    # ════════════════════════════════════════════════════════════
    # Phase 1 : Feature Extraction  (backbone frozen)
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 55)
    print("PHASE 1: Feature Extraction  (Frozen MobileNetV2)")
    print("=" * 55)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR_PHASE1),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_PHASE1,
        callbacks=get_callbacks(phase=1),
        verbose=1,
    )
    plot_history(history1, phase=1)

    # ════════════════════════════════════════════════════════════
    # Phase 2 : Fine-Tuning  (top 30 layers of backbone unfrozen)
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 55)
    print("PHASE 2: Fine-Tuning  (Top Backbone Layers Unfrozen)")
    print("=" * 55)

    backbone.trainable = True
    for layer in backbone.layers[:-30]:
        layer.trainable = False

    trainable_count = sum(1 for l in backbone.layers if l.trainable)
    print(f"Unfreezing last 30 of {len(backbone.layers)} backbone layers.")
    print(f"Total trainable backbone layers: {trainable_count}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR_PHASE2),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_PHASE2,
        callbacks=get_callbacks(phase=2),
        verbose=1,
    )
    plot_history(history2, phase=2)

    # ── Merge & save history ──────────────────────────────────
    full_history = {}
    for key in history1.history:
        full_history[key] = (
            history1.history[key] + history2.history.get(key, [])
        )
    with open(HISTORY_PATH, "w") as f:
        json.dump(full_history, f, indent=2)
    print(f"\n[SAVED] Training history -> {HISTORY_PATH}")

    # ── Load best model & evaluate ────────────────────────────
    print(f"\n[LOAD] Loading best model from: {MODEL_PATH}")
    best_model = keras.models.load_model(str(MODEL_PATH))
    evaluate_model(best_model, test_gen)

    print("\n" + "=" * 55)
    print("[DONE] Training Complete!")
    print(f"       Model saved  -> {MODEL_PATH}")
    print(f"       Plots saved  -> {PLOTS_DIR}/")
    print("=" * 55)


if __name__ == "__main__":
    main()
