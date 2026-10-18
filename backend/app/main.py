"""FaceVitals API server.

Endpoints:
  GET  /api/health   - liveness + model-file check
  POST /api/analyze  - RGB trace + profile -> vitals + LLM insight

Run (from the backend folder):  python -m uvicorn app.main:app --port 8000
Interactive docs:               http://localhost:8000/docs
"""
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse
from app.estimators import estimate_bp, estimate_metabolic
from app.groq_insight import generate_insight
from app.signal_processing import extract_features, quality_verdict

app = FastAPI(title="FaceVitals API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

MODEL_FILE = Path(__file__).resolve().parents[2] / "models" / "vitals_models.pkl"


class Profile(BaseModel):
    age: int = 35
    sex: str = "M"
    bmi: float = 24.0
    smoker: int = 0
    family_history: int = 0
    activity_level: int = 2


class AnalyzeRequest(BaseModel):
    rgb: list[list[float]]
    fps: float = 30.0
    profile: Profile = Field(default_factory=Profile)


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "FaceVitals API",
            "models_file_present": MODEL_FILE.exists()}


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    rgb = np.asarray(req.rgb, dtype=float)
    if rgb.ndim != 2 or rgb.shape[1] != 3 or len(rgb) < 150:
        return {"status": "error",
                "message": "rgb must be rows of [r, g, b], at least 5 seconds of frames."}
    feat = extract_features(rgb, float(req.fps))
    verdict = quality_verdict(feat["signal_quality"])
    if feat["signal_quality"] < 0.5:
        return {"status": "rescan",
                "message": "Signal quality too low - improve lighting, stay still, re-scan.",
                "signal_quality": feat["signal_quality"], "quality_verdict": verdict}
    profile = req.profile.model_dump()
    bp = estimate_bp(feat, profile)
    met = estimate_metabolic(feat, profile)
    vitals = {
        "heart_rate_bpm": feat["heart_rate_bpm"],
        "hrv": {"sdnn_ms": feat["hrv_sdnn_ms"], "rmssd_ms": feat["hrv_rmssd_ms"],
                "pnn50_pct": feat["hrv_pnn50_pct"]},
        "blood_pressure": {"systolic": bp["systolic"], "diastolic": bp["diastolic"],
                   "category": bp["category"], "type": bp["type"]},
        "cholesterol_total": met["cholesterol_total"],
        "blood_glucose_fasting_est": met["blood_glucose_fasting_est"],
        "signal_quality": feat["signal_quality"],
        "measurement": "HR measured from face video (rPPG); BP estimated by model; "
                       "cholesterol/glucose are screening estimates - confirm with a lab test.",
    }
    return {"status": "ok", "quality_verdict": verdict, "vitals": vitals,
            "insight": generate_insight(vitals)}

STATIC_PAGE = Path(__file__).resolve().parent / "static" / "demo.html"

@app.get("/")
def home():
    return FileResponse(STATIC_PAGE)

