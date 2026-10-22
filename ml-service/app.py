import json
import os
from pathlib import Path
from typing import List
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_PATH = Path(os.getenv("PULSEAI_MODEL", MODEL_DIR / "ecg_cnn.keras"))
LABEL_PATH = Path(os.getenv("PULSEAI_LABELS", MODEL_DIR / "label_map.json"))
WINDOW_SIZE = int(os.getenv("PULSEAI_WINDOW_SIZE", "187"))
app = FastAPI(title="PulseAI ML Service", version="1.0.0", description="Educational ECG classification inference service. Not for clinical use.")
model = None
label_map = {"0": "N", "1": "V", "2": "A", "3": "Other"}

class PredictRequest(BaseModel):
    signal: List[float] = Field(min_length=100, max_length=2_000_000)

def load_model():
    global model, label_map
    if LABEL_PATH.exists():
        try:
            label_map = json.loads(LABEL_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    if MODEL_PATH.exists():
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(MODEL_PATH)
        except Exception as exc:
            print(f"Could not load model {MODEL_PATH}: {exc}")
            model = None

@app.on_event("startup")
def startup():
    load_model()

@app.get("/health")
def health():
    return {"available": True, "model_loaded": model is not None, "model_path": str(MODEL_PATH), "message": "Ready for research inference" if model is not None else "No trained model found; train the model before inference."}

def normalize_window(w):
    w = np.asarray(w, dtype=np.float32)
    w = w - np.mean(w)
    std = np.std(w)
    if std > 1e-7:
        w = w / std
    return w

def segment_signal(signal, size=187):
    # Baseline educational segmenter: evenly spaced windows. For production research,
    # replace with validated R-peak detection and record-specific sampling metadata.
    arr = np.asarray(signal, dtype=np.float32)
    if not np.all(np.isfinite(arr)):
        raise ValueError("Signal contains NaN or infinite values.")
    # windows with 187 samples; step 187 prevents accidental overlapping-window inflation.
    windows = []
    for start in range(0, len(arr) - size + 1, size):
        windows.append(normalize_window(arr[start:start+size]))
    return np.asarray(windows, dtype=np.float32)

@app.post("/predict")
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="No trained model is loaded. Run the training pipeline and copy ecg_cnn.keras and label_map.json into ml-service/models/.")
    try:
        windows = segment_signal(req.signal, WINDOW_SIZE)
        if len(windows) == 0:
            raise HTTPException(status_code=400, detail=f"Signal must contain at least {WINDOW_SIZE} samples.")
        x = windows[..., np.newaxis]
        probs = model.predict(x, verbose=0)
        pred_ids = np.argmax(probs, axis=1)
        labels = [label_map.get(str(int(i)), str(int(i))) for i in pred_ids]
        return {"predictions": [{"segment": int(i), "class": labels[i], "class_id": int(pred_ids[i]), "scores": [float(v) for v in probs[i]]} for i in range(len(labels))],
                "segments_processed": int(len(windows)), "window_size": WINDOW_SIZE,
                "notice": "Research output only. Even a loaded model is not a clinical diagnostic tool."}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not process signal: {exc}") from exc
