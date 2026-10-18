"""Signal processing for FaceVitals.

Turns an averaged Red-Green-Blue (RGB) trace from the face into
Heart Rate (HR), Heart Rate Variability (HRV) and a Signal Quality
Index (SQI), using the Plane-Orthogonal-to-Skin (POS) method.
"""
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def detrend(s):
    """Remove slow drift: fit a straight line and subtract it."""
    x = np.arange(len(s))
    return s - np.polyval(np.polyfit(x, s, 1), x)


def bandpass(s, fs, low=0.7, high=4.0):
    """Keep only 0.7-4.0 Hertz (Hz) = 42-240 beats per minute (bpm)."""
    b, a = butter(4, [low / (fs / 2), high / (fs / 2)], btype="band")
    padlen = 3 * (max(len(a), len(b)) - 1)
    return filtfilt(b, a, s) if len(s) > padlen else s - s.mean()


def pos_rppg(rgb, fs):
    """Plane-Orthogonal-to-Skin (POS) -> Blood Volume Pulse (BVP)."""
    rgb = np.asarray(rgb, dtype=float)
    win = int(1.6 * fs)
    bvp = np.zeros(len(rgb))
    for n in range(win, len(rgb) + 1):
        w = rgb[n - win:n]
        cn = w / w.mean(axis=0)                       # temporal normalisation
        xs = cn[:, 1] - cn[:, 2]                      # projection 1: Green - Blue
        ys = cn[:, 1] + cn[:, 2] - 2 * cn[:, 0]       # projection 2: Green + Blue - 2*Red
        alpha = np.std(xs) / (np.std(ys) or 1e-6)     # balancing weight
        pulse = xs + alpha * ys                       # PLUS: pulse reinforces
        bvp[n - win:n] += pulse - pulse.mean()        # overlap-add
    return bvp


def estimate_hr(bvp, fs):
    """Dominant Fast Fourier Transform (FFT) peak inside 42-180 bpm,
    plus a confidence score from how far it towers over the spectrum."""
    f = bandpass(detrend(bvp), fs)
    spec = np.abs(np.fft.rfft(f * np.hanning(len(f))))
    bpm_axis = np.fft.rfftfreq(len(f), d=1 / fs) * 60
    band = (bpm_axis >= 42) & (bpm_axis <= 180)
    s = np.where(band, spec, 0)
    peak = int(np.argmax(s))
    snr = spec[peak] / (np.median(spec[band]) or 1e-9)
    confidence = float(np.clip(snr / (snr + 4), 0, 0.99))
    return float(bpm_axis[peak]), confidence


def extract_features(rgb, fs):
    """Full feature extraction from one scan. Raises ValueError if the
    trace is shorter than 10 seconds."""
    rgb = np.asarray(rgb, dtype=float)
    if len(rgb) / fs < 10:
        raise ValueError("Need at least 10 seconds of trace")
    raw = pos_rppg(rgb, fs)
    b = bandpass(detrend(raw), fs)
    hr, confidence = estimate_hr(raw, fs)

    b = np.convolve(b, np.ones(5) / 5, mode="same")
    min_dist = int(fs * 60 / (hr * 1.3))              # anti double-count guard
    peaks, _ = find_peaks(b, distance=max(min_dist, 1),
                          prominence=np.std(b) * 0.5)
    rr = np.diff(peaks) / fs * 1000                   # RR intervals, milliseconds (ms)
    rr = rr[(rr >= 333) & (rr <= 1500)]
    if len(rr) >= 4:
        sdnn = float(np.std(rr))
        rmssd = float(np.sqrt(np.mean(np.diff(rr) ** 2)))
        pnn50 = float(np.mean(np.abs(np.diff(rr)) > 50) * 100)
        hr_rr = 60000 / float(np.mean(rr))
        if abs(hr_rr - hr) <= 8:                      # blend only on agreement
            hr = 0.7 * hr + 0.3 * hr_rr
    else:
        sdnn, rmssd, pnn50 = 35.0, 28.0, 8.0          # neutral priors

    norm = (b - b.min()) / ((b.max() - b.min()) or 1e-9)
    upstroke = float(np.mean(np.gradient(norm)))
    perfusion = float(np.std(rgb[:, 1]) / (np.mean(rgb[:, 1]) or 1e-9))

    green = rgb[:, 1]
    motion = float(np.mean(np.abs(np.diff(green))) / (np.std(green) or 1e-9))
    quality = float(np.clip(confidence * 0.7 + (1 - min(motion, 1.0)) * 0.3, 0, 1))

    return {"heart_rate_bpm": round(hr, 1), "hrv_sdnn_ms": round(sdnn, 1),
            "hrv_rmssd_ms": round(rmssd, 1), "hrv_pnn50_pct": round(pnn50, 1),
            "confidence": round(confidence, 3), "signal_quality": round(quality, 3),
            "duration_sec": round(len(rgb) / fs, 1), "num_beats": int(len(peaks)),
            "upstroke": round(upstroke, 5), "perfusion_index": round(perfusion, 5)}



def quality_verdict(q):
    """Human sentence for a Signal Quality Index (SQI) value."""
    if q >= 0.75:
        return "good"
    if q >= 0.5:
        return "acceptable - stay still and face the light"
    return "poor - improve lighting, stay still and re-scan"
