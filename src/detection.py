import numpy as np
from scipy import signal as sps

BAND = (0.1, 200)
FS = 1000
FRAC = 0.1
B, A = sps.butter(3, list(BAND), btype="bandpass", fs=FS)

# Order must match the column order used during model training
FEATURE_KEYS = [
    "amplitude_uv", "duration_s", "rise_time_s", "decay_time_s",
    "slope_uv_per_ms", "area_uv_ms", "half_width_s",
]

_classifier = None


def mad_noise_std(x):
    return float(np.median(np.abs(x - np.median(x))) / 0.6745)


def bandpass_filter(data):
    return sps.filtfilt(B, A, data)


def refine_peaks(data, candidates, window_s=0.150):
    if len(candidates) == 0:
        return []
    half = int(window_s * FS)
    centered = data - np.median(data)
    refined = []
    for peak in candidates:
        lo = max(0, int(peak) - half)
        hi = min(len(data), int(peak) + half)
        chunk = centered[lo:hi]
        if len(chunk) == 0:
            continue
        refined.append(lo + int(np.argmin(chunk)))
    return refined


def _interp_crossing(sig, i, j, thresh):
    """Fractional sample index where sig crosses thresh between samples i and j."""
    denom = sig[j] - sig[i]
    if denom == 0:
        return float(i)
    return i + (thresh - sig[i]) / denom


def _robust_right_boundary(signal, start, thresh, min_consec=5):
    """Walk right until signal stays >= thresh for min_consec consecutive samples."""
    consec = 0
    i = start
    while i < len(signal) - 1:
        if signal[i] >= thresh:
            consec += 1
            if consec >= min_consec:
                return i - consec + 1
        else:
            consec = 0
        i += 1
    return len(signal) - 1


def compute_features(signal, peak_idx, max_overshoot_s=1, overshoot_frac=0.2, boundary_consec=5, detection=False):
    peak = signal[peak_idx]
    thresh = peak * FRAC

    left = peak_idx
    while left > 0 and signal[left] < thresh:
        left -= 1

    right = _robust_right_boundary(signal, peak_idx, thresh, min_consec=boundary_consec)

    chunk = signal[left:right]

    if detection:
        amplitude_uv = float(abs(peak))
    else:
        idx_duration = right - left
        half = int(idx_duration * 0.5)
        left_baseline  = signal[max(0, left - half) : left]
        right_baseline = signal[right : min(len(signal), right + half)]
        baseline = np.concatenate([left_baseline, right_baseline])
        baseline_mean = float(np.mean(baseline))
        amplitude_uv = float(abs(peak - baseline_mean))

    # Fractional boundary positions for sub-ms time precision
    left_f  = _interp_crossing(signal, left, left + 1, thresh) if left > 0 else float(left)
    right_f = _interp_crossing(signal, right - 1, right, thresh) if 0 < right < len(signal) else float(right)

    duration_s   = (right_f - left_f) / FS
    rise_time_s  = (peak_idx - left_f) / FS
    decay_time_s = (right_f - peak_idx) / FS

    W = 5
    sample_period_ms = 1000.0 / FS
    slope_uv_per_ms = float(np.max(np.abs(chunk[W:] - chunk[:-W])) / (W * sample_period_ms))

    area_uv_ms = float(np.sum(np.abs(chunk)))
    
    cap = min(len(signal), right + int(max_overshoot_s * FS))
    if cap > right:
        pos_region = signal[right:cap]
        pos_peak_local = int(np.argmax(pos_region))
        pos_peak = float(pos_region[pos_peak_local])
        if pos_peak >= amplitude_uv * overshoot_frac:
            pos_peak_idx = right + pos_peak_local
            pos_thresh = pos_peak * FRAC
            pos_right = pos_peak_idx
            while pos_right < len(signal) - 1 and signal[pos_right] > pos_thresh:
                pos_right += 1
            pos_lobe = signal[right:pos_right]
            area_uv_ms += float(np.sum(pos_lobe[pos_lobe > 0]))

    hw_thresh = peak * 0.5
    hw_left = peak_idx
    while hw_left > 0 and signal[hw_left] < hw_thresh:
        hw_left -= 1
    hw_right = peak_idx
    while hw_right < len(signal) - 1 and signal[hw_right] < hw_thresh:
        hw_right += 1

    hw_left_f  = _interp_crossing(signal, hw_left, hw_left + 1, hw_thresh) if hw_left > 0 else float(hw_left)
    hw_right_f = _interp_crossing(signal, hw_right - 1, hw_right, hw_thresh) if 0 < hw_right < len(signal) else float(hw_right)

    half_width_s = (hw_right_f - hw_left_f) / FS

    return {
        "amplitude_uv":    amplitude_uv,
        "duration_s":      duration_s,
        "rise_time_s":     rise_time_s,
        "decay_time_s":    decay_time_s,
        "slope_uv_per_ms": slope_uv_per_ms,
        "area_uv_ms":      area_uv_ms,
        "half_width_s":    half_width_s,
    }

def _load_classifier(path=None):
    global _classifier
    if _classifier is None:
        import joblib, os, sys
        if path is None:
            base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            path = os.path.join(base, "event_classifier.joblib")
        _classifier = joblib.load(path)
    return _classifier


def detect_events(
    data,
    height_k=5.0,
    prominence_k=4.0,
    min_distance_s=0.5,
    min_width_s=0.020,
    refine_window_s=0.150,
    debug=False,
):
    if data is None or len(data) == 0:
        return []

    clf = _load_classifier()

    detect_signal = bandpass_filter(data)
    noise_std = mad_noise_std(detect_signal)
    if noise_std == 0:
        return []

    target = -detect_signal
    h_thresh = height_k * noise_std
    p_thresh = prominence_k * noise_std

    candidates, _ = sps.find_peaks(
        target,
        height=h_thresh,
        prominence=p_thresh,
        distance=max(1, int(min_distance_s * FS)),
        width=max(1, int(min_width_s * FS)),
    )

    if debug:
        print(f"[debug] noise_std={noise_std:.3f}  h_thresh={h_thresh:.3f}  p_thresh={p_thresh:.3f}")
        print(f"[debug] find_peaks candidates: {len(candidates)}")

    refined = refine_peaks(detect_signal, candidates, window_s=refine_window_s)

    if debug:
        print(f"[debug] after refine_peaks: {len(refined)}")

    if not refined:
        return []

    feature_matrix = np.array([
        [compute_features(detect_signal, idx, detection=True)[k] for k in FEATURE_KEYS]
        for idx in refined
    ])

    predictions = clf.predict(feature_matrix)
    passed = [idx for idx, pred in zip(refined, predictions) if pred == 1]

    if debug:
        print(f"[debug] ML classifier: {len(passed)} passed, {len(refined) - len(passed)} rejected")

    return passed
