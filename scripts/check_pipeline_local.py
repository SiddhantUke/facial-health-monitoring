"""Local reproduction of /api/analyze (no server) - finds the crashing stage."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.estimators import estimate_bp, estimate_metabolic
from app.groq_insight import generate_insight
from app.signal_processing import extract_features, quality_verdict


def make_trace(hr_bpm, seconds=30, fs=30, noise=0.4, seed=1):
    t = np.arange(0, seconds, 1 / fs)
    pulse = 0.5 + 0.5 * np.sin(2 * np.pi * (hr_bpm / 60) * t)
    return (np.array([110.0, 95.0, 80.0]) + pulse[:, None] * np.array([2.0, 6.0, 2.5])
            + np.random.default_rng(seed).normal(0, noise, (len(t), 3)))


profile = {"age": 40, "sex": "M", "bmi": 26, "smoker": 0,
           "family_history": 1, "activity_level": 1}

print("stage 1: extract_features")
feat = extract_features(make_trace(72), 30.0)
print("  ok ->", feat["heart_rate_bpm"], feat["signal_quality"],
      quality_verdict(feat["signal_quality"]))

print("stage 2: estimators")
bp = estimate_bp(feat, profile)
met = estimate_metabolic(feat, profile)
print("  bp keys:", sorted(bp.keys()))
print("  bp:", bp)
print("  met:", met)

print("stage 3: assemble vitals (same code as main.py)")
vitals = {
    "heart_rate_bpm": feat["heart_rate_bpm"],
    "hrv": {"sdnn_ms": feat["hrv_sdnn_ms"], "rmssd_ms": feat["hrv_rmssd_ms"],
            "pnn50_pct": feat["hrv_pnn50_pct"]},
    "blood_pressure": {"systolic": bp["systolic"], "diastolic": bp["diastolic"],
                       "category": bp["category"], "type": bp["type"]},
    "cholesterol_total": met["cholesterol_total"],
    "blood_glucose_fasting_est": met["blood_glucose_fasting_est"],
    "signal_quality": feat["signal_quality"],
}
print("  ok")

print("stage 4: generate_insight")
ins = generate_insight(vitals)
print("  ok ->", ins["source"])
print("ALL STAGES PASSED")
