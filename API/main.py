from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, Any
import joblib
import pandas as pd
import numpy as np
import os
import json
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="Breast Cancer Multi-Model API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # project root
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

# ---------- Load feature names ----------
feature_names_path = os.path.join(ARTIFACTS_DIR, "feature_names.csv")
if not os.path.exists(feature_names_path):
    raise RuntimeError(f"feature_names.csv not found at {feature_names_path}")

feature_names = pd.read_csv(feature_names_path, header=None)[0].tolist()

# ---------- Load models registry ----------
registry_path = os.path.join(ARTIFACTS_DIR, "models_registry.json")
if not os.path.exists(registry_path):
    raise RuntimeError(f"models_registry.json not found at {registry_path}")

with open(registry_path) as f:
    registry = json.load(f)

# ---------- Load all models ----------
MODELS: Dict[str, Any] = {}
for model_id, info in registry.items():
    model_path = os.path.join(ARTIFACTS_DIR, info["file"])
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model file not found: {model_path}")
    MODELS[model_id] = joblib.load(model_path)

class BreastCancerInput(BaseModel):
    data: Dict[str, float]

@app.get("/models")
def list_models():
    return registry

@app.post("/predict")
def predict(
    input_data: BreastCancerInput,
    model_id: str = Query("random_forest", description="Which model to use"),
):
    if model_id not in MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model_id: {model_id}")

    model = MODELS[model_id]

    provided = set(input_data.data.keys())
    required = set(feature_names)
    missing = required - provided
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing features: {sorted(missing)}",
        )

    x_vals = [input_data.data[f] for f in feature_names]
    X = np.array(x_vals, dtype=float).reshape(1, -1)

    y_pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0].tolist() if hasattr(model, "predict_proba") else None

    return {
        "model_id": model_id,
        "model_name": registry[model_id]["name"],
        "prediction": int(y_pred),
        "probabilities": proba,
    }
@app.get("/metrics")
def get_metrics():
    """Return metrics.json (computed in the notebook)."""
    metrics_path = os.path.join(ARTIFACTS_DIR, "metrics.json")
    if not os.path.exists(metrics_path):
        return {}
    with open(metrics_path) as f:
        return json.load(f)


@app.get("/features")
def get_features():
    """Return the ordered list of feature names."""
    return {"features": feature_names}

@app.get("/feature-ranges")
def get_feature_ranges():
    path = os.path.join(ARTIFACTS_DIR, "random_feature_ranges.json")
    if not os.path.exists(path):
        raise HTTPException(500, "random_feature_ranges.json missing")
    with open(path) as f:
        return json.load(f)
@app.get("/dataset-info")
def get_dataset_info():
    path = os.path.join(ARTIFACTS_DIR, "dataset_info.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)
@app.get("/evaluation-table")
def get_evaluation_table():
    path = os.path.join(ARTIFACTS_DIR, "evaluation_table.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)
