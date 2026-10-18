"""Plain-language insight layer for FaceVitals.

Uses openai/gpt-oss-120b on the Groq free tier when GROQ_API_KEY is
set; otherwise returns a deterministic offline template with the same
structure. The model explains results - it never produces numbers.
"""
import json
import os

from dotenv import load_dotenv
load_dotenv()


SYSTEM_PROMPT = (
    "You are a health-information assistant inside a screening app. "
    "Explain the given vital-sign results in simple, calm language. "
    "Never diagnose a disease. Never change or invent numbers. "
    "Always remind the user these are non-contact screening estimates, "
    "not a medical diagnosis, and lab tests confirm cholesterol/glucose. "
    "Reply with ONLY a JSON object: {\"summary\": str, \"points\": [str, str, str], "
    "\"next_steps\": [str, str]} - each string under 22 words."
)


def fallback_insight(vitals):
    """Deterministic template used whenever the LLM is unavailable."""
    bp = vitals["blood_pressure"]
    chol = vitals["cholesterol_total"]
    glucose = vitals["blood_glucose_fasting_est"]
    return {
        "summary": (f"Your heart rate is {vitals['heart_rate_bpm']} bpm and estimated "
                    f"blood pressure is {bp['systolic']}/{bp['diastolic']} mmHg, "
                    f"both from a 30-second face scan."),
        "points": [
            f"Blood pressure category: {bp['category']}.",
            f"Cholesterol screening estimate: {chol['value']} mg/dL ({chol['category']}).",
            f"Glucose screening estimate: {glucose['value']} mg/dL ({glucose['category']}).",
        ],
        "next_steps": [
            "Repeat the scan in good light for a second reading.",
            "Confirm cholesterol and glucose with a standard lab test.",
        ],
        "source": "offline template (no GROQ_API_KEY set)",
    }


def generate_insight(vitals):
    """LLM explanation via Groq when a key exists; template otherwise.
    Never raises - any failure falls back to the template."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return fallback_insight(vitals)
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": json.dumps(vitals)}],
            temperature=0.3, max_tokens=800, reasoning_effort="low")

        data = json.loads(response.choices[0].message.content)
        data["source"] = "Groq / openai/gpt-oss-120b"
        return data
    except Exception:
        return fallback_insight(vitals)
