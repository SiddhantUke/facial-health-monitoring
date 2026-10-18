import os
import sys
import time

import numpy as np
import streamlit as st

st.set_page_config(page_title="FaceVitals", page_icon="🫀", layout="centered")

if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))

from app.signal_processing import extract_features
from app.estimators import estimate_bp, estimate_metabolic
from app.groq_insight import generate_insight

st.title("🫀 FaceVitals — Non-Contact Facial Health Monitoring")
st.caption("Heart rate is measured from the facial blood-volume pulse (rPPG). Blood pressure is a model estimate; cholesterol and glucose are screening estimates, not lab measurements.")

with st.sidebar:
    st.header("Profile")
    age = st.number_input("Age", 10, 100, 40)
    sex = st.selectbox("Sex", ["M", "F"])
    bmi = st.number_input("BMI", 12.0, 50.0, 26.0, step=0.1)
    smoker = 1 if st.checkbox("Smoker") else 0
    family = 1 if st.checkbox("Family history of heart disease", value=True) else 0
    activity = st.slider("Activity level (0 low - 2 high)", 0, 2, 1)

profile = {"age": int(age), "sex": sex, "bmi": float(bmi),
           "smoker": smoker, "family_history": family, "activity_level": int(activity)}


def show_results(trace, fps):
    features = extract_features(np.asarray(trace, dtype=float), float(fps))
    if features.get("signal_quality", 0) < 0.5:
        st.warning("Signal quality too low (" + str(round(features.get("signal_quality", 0), 3)) + "). Please re-scan in better light, holding still.")
        return
    bp = estimate_bp(features, profile)
    met = estimate_metabolic(features, profile)
    chol = met.get("cholesterol", {})
    gluc = met.get("glucose", {})
    vitals = {
        "heart_rate_bpm": features.get("heart_rate_bpm"),
        "hrv_sdnn_ms": features.get("hrv_sdnn_ms"),
        "hrv_rmssd_ms": features.get("hrv_rmssd_ms"),
        "systolic_bp": bp.get("systolic"),
        "diastolic_bp": bp.get("diastolic"),
        "cholesterol_mg_dl": chol.get("value"),
        "glucose_mg_dl": gluc.get("value"),
    }
    c1, c2, c3 = st.columns(3)
    c1.metric("Heart Rate", str(vitals["heart_rate_bpm"]) + " bpm")
    c2.metric("Blood Pressure", str(vitals["systolic_bp"]) + "/" + str(vitals["diastolic_bp"]), bp.get("category", ""))
    c3.metric("Signal Quality", str(round(features.get("signal_quality", 0), 3)))
    c4, c5, c6 = st.columns(3)
    c4.metric("Cholesterol (screening)", str(vitals["cholesterol_mg_dl"]) + " mg/dL", chol.get("category", ""))
    c5.metric("Glucose (screening)", str(vitals["glucose_mg_dl"]) + " mg/dL", gluc.get("category", ""))
    c6.metric("HRV - SDNN / RMSSD", str(vitals["hrv_sdnn_ms"]) + " / " + str(vitals["hrv_rmssd_ms"]) + " ms")
    try:
        insight = generate_insight(vitals, profile)
        st.subheader("Insight — " + str(insight.get("source", "")))
        st.info(insight.get("summary", ""))
    except Exception:
        st.info("Insight unavailable right now; the vitals above are complete.")


def synthetic_trace():
    fps = 30.0
    t = np.arange(900) / fps
    pulse = np.sin(2 * np.pi * 1.2 * t) + 0.4 * np.sin(2 * np.pi * 2.4 * t + 0.5)
    pulse = pulse / np.max(np.abs(pulse))
    rng = np.random.default_rng(42)
    trace = np.column_stack([
        150 + 4 * pulse + rng.normal(0, 0.4, 900),
        110 + 9 * pulse + rng.normal(0, 0.4, 900),
        90 + 3 * pulse + rng.normal(0, 0.4, 900),
    ])
    return trace, fps


mode = st.radio("Scan mode", ["Demo scan (no camera)", "Upload a face video", "Live webcam"], horizontal=True)

if mode == "Demo scan (no camera)":
    st.write("Generates the standard 72 bpm synthetic face signal and runs the full pipeline on it.")
    if st.button("Run demo scan"):
        trace, fps = synthetic_trace()
        show_results(trace, fps)

elif mode == "Upload a face video":
    up = st.file_uploader("MP4 / MOV video of a face, 20-30 seconds", type=["mp4", "mov"])
    if up is not None and st.button("Analyze video"):
        import tempfile
        from app.video_processing import extract_trace_from_video
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(up.getbuffer())
            path = tmp.name
        trace, fps = extract_trace_from_video(path)
        show_results(trace, fps)

else:
    import av
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

    class TraceCollector(VideoProcessorBase):
        def __init__(self):
            self.samples = []

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            h, w, _ = img.shape
            boxes = [(0.42, 0.08, 0.58, 0.20), (0.30, 0.45, 0.42, 0.60), (0.58, 0.45, 0.70, 0.60)]
            means = []
            for (x1, y1, x2, y2) in boxes:
                roi = img[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]
                means.append(roi.mean(axis=(0, 1)))
            b, g, r = np.mean(means, axis=0)
            self.samples.append([r, g, b])
            return frame

    st.write("Start the camera, sit still facing the light for ~30 seconds, then stop and analyze.")
    ctx = webrtc_streamer(key="facevitals", video_processor_factory=TraceCollector,
                          media_stream_constraints={"video": True, "audio": False})
    if ctx.video_processor and st.button("Analyze collected frames"):
        samples = ctx.video_processor.samples
        if len(samples) < 150:
            st.warning("Only " + str(len(samples)) + " frames collected - let the camera run longer.")
        else:
            show_results(np.array(samples), 30.0)
