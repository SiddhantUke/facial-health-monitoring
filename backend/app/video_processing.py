"""Video file -> RGB trace: sample the forehead + cheek regions of every
frame and average them. Output format is identical to the browser trace,
so the whole pipeline works unchanged."""
import cv2
import numpy as np

ROIS = [  # (x0, y0, x1, y1) as fractions of frame size
    (0.42, 0.08, 0.58, 0.20),   # forehead
    (0.30, 0.45, 0.42, 0.60),   # left cheek
    (0.58, 0.45, 0.70, 0.60),   # right cheek
]


def extract_trace_from_video(path, max_seconds=35):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rows = []
    while len(rows) < int(fps * max_seconds):
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        means = []
        for x0, y0, x1, y1 in ROIS:
            roi = frame[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
            means.append(roi.reshape(-1, 3).mean(axis=0))
        rows.append(np.mean(means, axis=0)[::-1])  # average the 3 ROIs; BGR -> RGB
    cap.release()
    if len(rows) < 150:
        raise RuntimeError("video too short - need at least 5 seconds of frames")
    return np.array(rows), float(fps)
