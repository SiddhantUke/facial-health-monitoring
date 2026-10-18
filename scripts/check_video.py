"""Make a synthetic pulsing video, then recover its heart rate from the file."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.signal_processing import extract_features
from app.video_processing import extract_trace_from_video

t = np.arange(0, 30, 1 / 30)
pulse = 0.5 + 0.5 * np.sin(2 * np.pi * (72 / 60) * t)
frames = np.array([110.0, 95.0, 80.0]) + pulse[:, None] * np.array([2.0, 6.0, 2.5])

out = Path("test_face.mp4")
writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
for rgb in frames:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[:, :] = rgb[::-1]  # RGB -> BGR for OpenCV
    writer.write(frame)
writer.release()

trace, fps = extract_trace_from_video(out)
feat = extract_features(trace, fps)
print("trace:", trace.shape, "fps:", fps)
print("recovered HR:", feat["heart_rate_bpm"], "| quality:", feat["signal_quality"])
