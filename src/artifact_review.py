"""
Cross-channel artifact detection and review dialog.

find_coincident_events  — identifies events that land on the exact same sample
                          index across two or more channels.
ArtifactReviewDialog    — stacked-trace Qt dialog; user steps through each
                          coincident group and chooses to remove or keep.
"""

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import mea_io
import detection

DECIMATION = 25
WINDOW_S   = 0.5   # ±500 ms of signal shown around each artifact

_COLORS = [
    "#2980b9", "#e67e22", "#27ae60", "#8e44ad",
    "#c0392b", "#16a085", "#d35400", "#2c3e50",
]


def find_coincident_events(events_per_channel, fs=1000, tolerance_ms=10):
    """
    Return list of (representative_sample, {ch_idx: sample_idx}) for every
    cluster of events that fall within `tolerance_ms` of each other across
    two or more channels, sorted ascending.

    The dict value is the actual per-channel sample index closest to the
    cluster anchor — use it when removing events from the session.

    A greedy sweep is used: events are sorted globally, and any event within
    tolerance_ms of the current cluster's first event is merged into it.

    Parameters
    ----------
    events_per_channel : dict[int, list[int]]
        {ch_idx: [sample_indices]}
    fs : float
        Decimated sampling rate (samples/s).
    tolerance_ms : float
        Maximum time difference in ms to consider two events coincident.
    """
    tol = int(tolerance_ms * fs / 1000)

    # Flatten to (sample_idx, ch_idx) and sort by sample
    all_events = sorted(
        (sample_idx, ch_idx)
        for ch_idx, events in events_per_channel.items()
        for sample_idx in events
    )

    groups = []
    i = 0
    while i < len(all_events):
        anchor = all_events[i][0]
        cluster_samples = []
        cluster_channels = {}
        j = i
        while j < len(all_events) and all_events[j][0] - anchor <= tol:
            s, ch = all_events[j]
            # Keep only the closest sample per channel within the window
            if ch not in cluster_channels or abs(s - anchor) < abs(cluster_channels[ch] - anchor):
                cluster_channels[ch] = s
            cluster_samples.append(s)
            j += 1
        if len(cluster_channels) >= 2:
            representative = int(np.median(cluster_samples))
            groups.append((representative, dict(cluster_channels)))
        i = j

    return groups


class ArtifactReviewDialog(QDialog):
    """
    Step through each coincident-event group on a stacked trace plot.
    Removals are applied to session.channels immediately on confirmation.
    """

    def __init__(self, session, filename, fs, groups, label_map, parent=None):
        """
        Parameters
        ----------
        session   : Session
        filename  : str — path to the HDF5 recording
        fs        : float — decimated sampling rate
        groups    : list of (sample_idx, [ch_idx, ...])
        label_map : dict[int, str] — ch_idx -> electrode label
        """
        super().__init__(parent)
        self.setWindowTitle("Cross-Channel Artifact Review")
        self.setModal(True)
        self.resize(960, 560)

        self.session   = session
        self.filename  = filename
        self.fs        = fs
        self.groups    = groups
        self.label_map = label_map
        self.current   = 0
        self._data     = {}  # lazy-loaded: ch_idx -> filtered array (or None on error)

        self._build_ui()
        self._show_current()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)

        self.header = QLabel()
        self.header.setAlignment(Qt.AlignCenter)
        self.header.setStyleSheet(
            "font-weight: bold; font-size: 13px; padding: 6px; "
            "background: #fef3cd; border-radius: 4px;"
        )
        root.addWidget(self.header)

        self.figure = Figure(figsize=(9, 4))
        self.canvas = FigureCanvas(self.figure)
        root.addWidget(self.canvas, stretch=1)

        self.info_label = QLabel()
        self.info_label.setAlignment(Qt.AlignCenter)
        root.addWidget(self.info_label)

        btn_row = QHBoxLayout()
        self.remove_btn = QPushButton("Remove from all channels")
        self.remove_btn.setStyleSheet("font-weight: bold; color: #c0392b;")
        self.keep_btn = QPushButton("Keep")
        self.done_btn = QPushButton("Done reviewing")
        self.done_btn.setStyleSheet("font-weight: bold;")
        for b in (self.remove_btn, self.keep_btn, self.done_btn):
            btn_row.addWidget(b)
        root.addLayout(btn_row)

        self.remove_btn.clicked.connect(self._on_remove)
        self.keep_btn.clicked.connect(self._on_keep)
        self.done_btn.clicked.connect(self.accept)

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------
    def _get_channel_data(self, ch_idx):
        """Load and cache a single channel's filtered signal on first access."""
        if ch_idx not in self._data:
            try:
                raw, _ = mea_io.read_and_decimate(self.filename, ch_idx, decimation=DECIMATION)
                self._data[ch_idx] = detection.bandpass_filter(raw)
            except (IndexError, OSError):
                self._data[ch_idx] = None
        return self._data[ch_idx]

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _show_current(self):
        sample_idx, ch_samples = self.groups[self.current]
        n = len(self.groups)

        loaded_chs = [c for c in ch_samples if self._get_channel_data(c) is not None]
        labels = ", ".join(self.label_map.get(c, str(c)) for c in loaded_chs)
        self.header.setText(
            f"Potential artifact  {self.current + 1} of {n}"
            f"  |  Sample {sample_idx}"
            f"  |  Electrodes: {labels}"
        )

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        half   = int(WINDOW_S * self.fs)
        traces = []
        max_rng = 0.0

        for ch_idx in ch_samples:
            sig = self._get_channel_data(ch_idx)
            if sig is None:
                continue
            lo    = max(0, sample_idx - half)
            hi    = min(len(sig), sample_idx + half)
            chunk = sig[lo:hi]
            t     = (np.arange(lo, hi) - sample_idx) / self.fs * 1000
            rng   = float(np.ptp(chunk)) if len(chunk) else 1.0
            max_rng = max(max_rng, rng)
            traces.append((t, chunk, ch_idx))

        spacing = (max_rng or 100.0) * 1.5
        yticks, ylabels = [], []

        for i, (t, chunk, ch_idx) in enumerate(traces):
            offset = i * spacing
            ax.plot(t, chunk + offset, linewidth=0.8, color=_COLORS[i % len(_COLORS)])
            yticks.append(offset)
            ylabels.append(self.label_map.get(ch_idx, str(ch_idx)))

        ax.axvline(0, color="red", linestyle="--", alpha=0.7, linewidth=1)
        ax.set_xlabel("Time relative to artifact (ms)")
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)
        ax.set_title("Stacked filtered traces, red line marks the coincident sample")
        self.figure.tight_layout()
        self.canvas.draw()

        self.info_label.setText(
            f"{len(ch_samples)} electrode(s) share an event within the artifact window"
        )

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _on_remove(self):
        _, ch_samples = self.groups[self.current]
        for ch_idx, actual_sample in ch_samples.items():
            key = str(ch_idx)
            if key in self.session.channels:
                evs = self.session.channels[key]["validated_events"]
                if actual_sample in evs:
                    evs.remove(actual_sample)
        self._advance()

    def _on_keep(self):
        self._advance()

    def _advance(self):
        if self.current < len(self.groups) - 1:
            self.current += 1
            self._show_current()
        else:
            self.accept()
