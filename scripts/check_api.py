"""End-to-end API check: sends a synthetic 72 bpm trace to /api/analyze."""
import json

import numpy as np
import requests


def make_trace(hr_bpm, seconds=30, fs=30, noise=0.4, seed=1):
    t = np.arange(0, seconds, 1 / fs)
    pulse = 0.5 + 0.5 * np.sin(2 * np.pi * (hr_bpm / 60) * t)
    return (np.array([110.0, 95.0, 80.0]) + pulse[:, None] * np.array([2.0, 6.0, 2.5])
            + np.random.default_rng(seed).normal(0, noise, (len(t), 3)))


payload = {"rgb": make_trace(72).tolist(), "fps": 30.0,
           "profile": {"age": 40, "sex": "M", "bmi": 26, "smoker": 0,
                       "family_history": 1, "activity_level": 1}}
print("health:", requests.get("http://localhost:8000/api/health").json())
resp = requests.post("http://localhost:8000/api/analyze", json=payload)
print("status code:", resp.status_code)
print(resp.text[:1500])
