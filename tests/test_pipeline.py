"""Guard tests for the FaceVitals pipeline."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.signal_processing import extract_features
from app.estimators import estimate_bp, estimate_metabolic

PROFILE = {"age": 40, "sex": "M", "bmi": 26, "smoker": 0,
           "family_history": 1, "activity_level": 1}


def make_trace(hr_bpm, seconds=30, fs=30, noise=0.4, seed=1):
    t = np.arange(0, seconds, 1 / fs)
    pulse = 0.5 + 0.5 * np.sin(2 * np.pi * (hr_bpm / 60) * t)
    return (np.array([110.0, 95.0, 80.0]) + pulse[:, None] * np.array([2.0, 6.0, 2.5])
            + np.random.default_rng(seed).normal(0, noise, (len(t), 3)))


def test_heart_rate_ground_truth():
    feat = extract_features(make_trace(72), 30.0)
    assert abs(feat["heart_rate_bpm"] - 72) <= 4
    assert feat["signal_quality"] > 0.4


def test_heart_rate_frequency_sweep():
    for bpm in (55, 90, 120):
        feat = extract_features(make_trace(bpm), 30.0)
        assert abs(feat["heart_rate_bpm"] - bpm) <= 3


def test_estimator_output_ranges():
    feat = extract_features(make_trace(72), 30.0)
    bp = estimate_bp(feat, PROFILE)
    met = estimate_metabolic(feat, PROFILE)
    assert 85 <= bp["systolic"] <= 200 and 50 <= bp["diastolic"] <= 130
    assert 110 <= met["cholesterol_total"]["value"] <= 340
    assert 65 <= met["blood_glucose_fasting_est"]["value"] <= 320



