"""Estimators for FaceVitals: Blood Pressure (BP) plus Cholesterol and
Glucose screening estimates, from Random Forest (RF) models with
transparent formula fallbacks."""


import os
import pickle
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve()
CANDIDATE_PATHS = [os.environ.get("MODEL_PATH", ""),
                   str(HERE.parents[2] / "models" / "vitals_models.pkl"),
                   str(HERE.parents[1] / "models" / "vitals_models.pkl"),
                   "models/vitals_models.pkl"]
MODELS = None
for path in CANDIDATE_PATHS:
    if path and os.path.exists(path):
        try:
            with open(path, "rb") as f:
                MODELS = pickle.load(f)
            break
        except Exception:
            MODELS = None


def build_features(feat, profile):
    """One row of 11 numbers, in the EXACT order the models were trained on."""
    sex = 1.0 if str(profile.get("sex", "M")).upper().startswith("M") else 0.0
    return np.array([[
        feat["heart_rate_bpm"], feat["hrv_sdnn_ms"], feat["hrv_rmssd_ms"],
        feat["upstroke"] * 1000, feat["perfusion_index"] * 100,
        float(profile.get("age", 35)), float(profile.get("bmi", 24)),
        sex, float(profile.get("smoker", 0)),
        float(profile.get("family_history", 0)),
        float(profile.get("activity_level", 2)),
    ]])


def estimate_bp(feat, profile):
    """Blood Pressure (BP) estimate with category. Screening only."""
    x = build_features(feat, profile)
    if MODELS and "bp" in MODELS:
        sbp, dbp = MODELS["bp"].predict(x)[0]
    else:
        age, bmi, hr = x[0, 5], x[0, 6], x[0, 0]
        sbp = 92 + 0.55 * age + 0.9 * (bmi - 22) + 0.18 * (hr - 70)
        dbp = 60 + 0.32 * age + 0.6 * (bmi - 22) + 0.10 * (hr - 70)
    sbp = float(np.clip(sbp, 85, 200))
    dbp = float(np.clip(dbp, 50, 130))
    if sbp < 120 and dbp < 80: category = "normal"
    elif sbp < 130 and dbp < 80: category = "elevated"
    elif sbp < 140 or dbp < 90: category = "high - stage 1"
    else: category = "high - stage 2"
    return {"systolic": round(sbp), "diastolic": round(dbp), "unit": "mmHg",
            "category": category, "type": "model estimate (screening)"}


def category_of(value, normal_below, high_from):
    if value < normal_below: return "normal"
    if value < high_from: return "borderline"
    return "high"


def estimate_metabolic(feat, profile):
    """Cholesterol and Glucose screening estimates with categories."""
    x = build_features(feat, profile)
    if MODELS and "metabolic" in MODELS:
        chol, glucose = MODELS["metabolic"].predict(x)[0]
    else:
        age, bmi = x[0, 5], x[0, 6]
        fam, act = x[0, 9], x[0, 10]
        chol = 155 + 0.85 * age + 3.2 * (bmi - 22) + 18 * fam - 5 * act
        glucose = 82 + 0.35 * age + 1.9 * (bmi - 22) + 10 * fam - 3 * act
    chol = float(np.clip(chol, 110, 340))
    glucose = float(np.clip(glucose, 65, 320))
    return {
        "cholesterol_total": {"value": round(chol), "unit": "mg/dL",
                              "category": category_of(chol, 200, 240),
                              "type": "screening estimate - confirm with lipid profile"},
        "blood_glucose_fasting_est": {"value": round(glucose), "unit": "mg/dL",
                                      "category": category_of(glucose, 100, 126),
                                      "type": "screening estimate - confirm with fasting lab test"},
    }
    
