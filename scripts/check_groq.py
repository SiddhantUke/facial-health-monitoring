"""Diagnostic: call Groq exactly like generate_insight does, but with no
try/except - so the real output or the real error becomes visible."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from dotenv import load_dotenv
from groq import Groq

from app.groq_insight import SYSTEM_PROMPT

load_dotenv()

vitals = {"heart_rate_bpm": 72.0,
          "blood_pressure": {"systolic": 117, "diastolic": 73, "category": "normal"},
          "cholesterol_total": {"value": 216, "category": "borderline"},
          "blood_glucose_fasting_est": {"value": 114, "category": "borderline"}}

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[{"role": "system", "content": SYSTEM_PROMPT},
              {"role": "user", "content": json.dumps(vitals)}],
    temperature=0.3, max_tokens=800, reasoning_effort="low")

content = response.choices[0].message.content
choice = response.choices[0]
print("finish_reason:", choice.finish_reason)
print("usage:", response.usage)
content = choice.message.content
print("RAW MODEL OUTPUT:")
print(content)
print("PARSED:", json.loads(content))
