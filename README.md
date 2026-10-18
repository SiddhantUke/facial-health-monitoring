# FaceVitals — Non-Contact Facial Health Monitoring

A web-based system that estimates Heart Rate (HR), Blood Pressure (BP),
cholesterol and blood glucose from a 30-second webcam face scan.
No contact sensors: the pulse is recovered from subtle skin-colour changes
using remote Photoplethysmography (rPPG).

**What each number is:** HR is *measured* from the pulse signal.
BP is a *model estimate*. Cholesterol and glucose are *screening estimates*
(stratified from pulse-wave + profile features) — confirm with a lab test.

## How it works

1. The browser samples forehead + cheek Regions of Interest (ROIs) and sends
   only the averaged RGB trace (900 x 3 numbers, ~54 KB) — video never
   leaves the device.
2. Plane-Orthogonal-to-Skin (POS) rPPG + Fast Fourier Transform (FFT)
   extracts HR and Heart Rate Variability (HRV); a Signal Quality Index
   (SQI) below 0.5 rejects the scan instead of showing unreliable vitals.
3. Two RandomForest models (11 features, ~1.8 MB) estimate BP, cholesterol
   and glucose, with physiological clamping and clinical categories.
4. GPT-OSS-120B (openai/gpt-oss-120b) via the Groq free tier explains the
   results in plain language (strict JSON, never diagnoses, never changes
   a number); a deterministic template covers offline use.

## Run it

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r backend\requirements.txt
    copy .env.example .env        # then put your GROQ_API_KEY in .env
    python scripts\train_models.py
    pytest tests -q
    cd backend
    python -m uvicorn app.main:app --port 8000

Open http://localhost:8000/ — webcam scan, or "Demo scan (no camera)".

## Verified results (Phase-1 synthetic data, 4,000 records)

- Synthetic 72 bpm scan recovered at exactly 72.0 bpm (quality 0.858);
  frequency sweep 50–140 bpm within ~1 bpm; video-file path: 72.2 bpm.
- Held-out Mean Absolute Error (MAE): BP 4.8 / 3.7 mmHg,
  cholesterol 12.0 mg/dL, glucose 6.1 mg/dL.
- Tests: 3 passed (ground truth, frequency sweep, output ranges).

## Structure

backend/app/ — main.py (API + demo page), signal_processing.py,
estimators.py, groq_insight.py, video_processing.py, static/demo.html
models/ — vitals_models.pkl · scripts/ — training + checks
tests/ — guard tests · dataset_full.csv — Phase-1 dataset

## Scope

Screening and awareness tool, not a medical device. Phase 1 is trained on
synthetic data generated from published physiological relationships;
Phase 2 is a paired study against cuff and laboratory references.
