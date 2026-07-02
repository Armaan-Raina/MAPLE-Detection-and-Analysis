"""
MEA Event Visualizer — channel-selective detection workflow.

Flow:
  1. User opens a file → "please wait" dialog during h5 metadata read + RMS computation.
  2. Channel selection dialog: toggle buttons per electrode label + Select All.
  3. Manual review of each channel: validate or reject each detected candidate.
  4. Navigate channels freely with Prev / Next Channel buttons.
  5. Export validated events to Excel at any point during the session.

Keyboard shortcuts:
  V      Validate current event
  R      Invalidate (reject) current event
  U      Undo last action
  ← / →  Previous / Next event
"""

import sys
import os

import h5py
from PyQt5.QtCore import Qt, QTimer, QEvent
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLineEdit, QLabel, QSlider,
    QFileDialog, QMessageBox, QDialog,
    QDialogButtonBox, QScrollArea,
    QProgressBar, QComboBox,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import numpy as np

import mea_io
import detection
from artifact_review import find_coincident_events, ArtifactReviewDialog
from event_pool import EventPool
from session import Session
from exporter import export_events_excel


DECIMATION = 25


# ======================================================================
# File metadata helpers
# ======================================================================

def _read_file_metadata(filename):
    """Return (good_indices, good_labels, fs) from an MCS HDF5 file."""
    with h5py.File(filename, "r") as f:
        info_ds = f[mea_io.INFO_PATH][:]
        n_total = f[mea_io.SIGNAL_PATH].shape[0]
        tick_us = float(info_ds[0]["Tick"])
        fs      = 1e6 / tick_us
        good_indices, good_labels = [], []
        for i in range(n_total):
            label = mea_io._decode_label(info_ds[i]["Label"])
            if label != mea_io.REFERENCE_LABEL:
                good_indices.append(i)
                good_labels.append(label)
    return good_indices, good_labels, fs


# ======================================================================
# Helper dialogs
# ======================================================================

class WaitDialog(QDialog):
    """Modal 'please wait' overlay during blocking operations."""

    def __init__(self, message="Please wait...", parent=None):
        super().__init__(parent)
        self.setWindowTitle(" ")
        self.setModal(True)
        self.setWindowFlags(Qt.Dialog | Qt.CustomizeWindowHint | Qt.WindowTitleHint)
        layout = QVBoxLayout(self)
        lbl = QLabel(message)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("font-size: 15px; padding: 24px 40px;")
        layout.addWidget(lbl)
        self.adjustSize()


class ChannelSelectionDialog(QDialog):
    """Grid of toggle buttons — one per electrode label — to pick analysis channels."""

    COLS = 8

    def __init__(self, labels, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Channels for Analysis")
        self._labels = labels
        self._buttons = []

        outer = QVBoxLayout(self)

        instr = QLabel(
            "Select the electrodes you want to analyze, then click Confirm.\n"
            "Highlighted (blue) channels will be included."
        )
        instr.setWordWrap(True)
        instr.setStyleSheet("padding: 6px;")
        outer.addWidget(instr)

        ctrl_row = QHBoxLayout()
        all_btn  = QPushButton("Select All")
        none_btn = QPushButton("Deselect All")
        all_btn.clicked.connect(self._select_all)
        none_btn.clicked.connect(self._deselect_all)
        ctrl_row.addWidget(all_btn)
        ctrl_row.addWidget(none_btn)
        ctrl_row.addStretch()
        outer.addLayout(ctrl_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        grid = QGridLayout(container)
        grid.setSpacing(5)

        # Use physical MEA grid positions if every label is a 2-digit row-col coordinate
        def _mea_pos(lbl):
            if len(lbl) == 2 and lbl.isdigit():
                r, c = int(lbl[0]), int(lbl[1])
                if 1 <= r <= 8 and 1 <= c <= 8:
                    return c - 1, r - 1  # row = second digit, col = first digit
            return None

        positions   = [_mea_pos(lbl) for lbl in labels]
        use_mea_grid = all(p is not None for p in positions)

        for i, label in enumerate(labels):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setChecked(True)
            btn.setFixedSize(68, 38)
            btn.setStyleSheet("""
                QPushButton {
                    background: #d5d5d5;
                    border-radius: 5px;
                    font-size: 12px;
                }
                QPushButton:checked {
                    background: #3d7dc8;
                    color: white;
                    font-weight: bold;
                }
                QPushButton:hover {
                    border: 2px solid #555;
                }
            """)
            self._buttons.append(btn)
            if use_mea_grid:
                grid.addWidget(btn, *positions[i])
            else:
                grid.addWidget(btn, i // self.COLS, i % self.COLS)

        container.setLayout(grid)
        scroll.setWidget(container)
        scroll.setMinimumHeight(200)
        outer.addWidget(scroll)

        count_row = QHBoxLayout()
        self._count_label = QLabel()
        count_row.addWidget(self._count_label)
        count_row.addStretch()
        outer.addLayout(count_row)

        for b in self._buttons:
            b.toggled.connect(self._update_count)
        self._update_count()

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.button(QDialogButtonBox.Ok).setText("Confirm")
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        outer.addWidget(btns)

        self.resize(640, 420)

    def _select_all(self):
        for b in self._buttons:
            b.setChecked(True)

    def _deselect_all(self):
        for b in self._buttons:
            b.setChecked(False)

    def _update_count(self):
        n = sum(1 for b in self._buttons if b.isChecked())
        self._count_label.setText(f"{n} of {len(self._buttons)} channels selected")

    def _on_accept(self):
        if not any(b.isChecked() for b in self._buttons):
            QMessageBox.warning(self, "No channels selected",
                                "Please select at least one channel.")
            return
        self.accept()

    def selected_label_indices(self):
        """Indices (into the label list) of checked buttons."""
        return [i for i, b in enumerate(self._buttons) if b.isChecked()]


class SearchDialog(QDialog):
    """Standalone viewer: enter a channel label and time (s) to inspect that data."""

    def __init__(self, filename, label_map, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Search Channel / Time")
        self.filename = filename
        self._label_to_idx = {v: k for k, v in label_map.items()}

        layout = QVBoxLayout(self)

        search_row = QHBoxLayout()
        search_row.addWidget(QLabel("Channel:"))
        self.channel_input = QLineEdit()
        self.channel_input.setPlaceholderText("e.g. 47")
        search_row.addWidget(self.channel_input)
        search_row.addWidget(QLabel("Time (s):"))
        self.time_input = QLineEdit()
        self.time_input.setPlaceholderText("e.g. 12.5")
        search_row.addWidget(self.time_input)
        self.view_btn = QPushButton("View")
        search_row.addWidget(self.view_btn)
        layout.addLayout(search_row)

        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        self.figure = Figure(figsize=(8, 3))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        layout.addWidget(self.canvas, stretch=1)

        self.view_btn.clicked.connect(self._on_view)
        self.channel_input.returnPressed.connect(self._on_view)
        self.time_input.returnPressed.connect(self._on_view)
        self.resize(820, 420)

    def _on_view(self):
        label    = self.channel_input.text().strip()
        time_str = self.time_input.text().strip()

        if label not in self._label_to_idx:
            self.status_label.setText(f"Channel '{label}' not found.")
            return

        try:
            time_s = float(time_str)
            if time_s < 0:
                raise ValueError
        except ValueError:
            self.status_label.setText("Enter a valid non-negative time in seconds.")
            return

        ch_idx = self._label_to_idx[label]
        try:
            data, fs = mea_io.read_and_decimate(self.filename, ch_idx, decimation=DECIMATION)
        except (IndexError, OSError) as e:
            self.status_label.setText(f"Error loading channel: {e}")
            return

        center = int(time_s * fs)
        if center >= len(data):
            self.status_label.setText(
                f"Time {time_s:.3f} s exceeds recording length ({len(data) / fs:.1f} s)."
            )
            return

        window = int(1.0 * fs)
        start  = max(0, center - window)
        end    = min(len(data), center + window)
        chunk  = data[start:end]
        t      = (np.arange(start, end) - center) / fs

        self.ax.clear()
        self.ax.plot(t, chunk, linewidth=0.6)
        self.ax.axvline(0, color="red", linestyle="--", alpha=0.5)
        self.ax.set_xlabel("Time relative to search point (s)")
        self.ax.set_ylabel("Voltage (µV)")
        self.ax.set_title(f"Electrode {label}  —  t = {time_s:.3f} s")

        self.canvas.draw()
        self.status_label.setText(
            f"Showing ±1 s around t = {time_s:.3f} s  |  Electrode {label}"
        )


# ======================================================================
# Main window
# ======================================================================

class EventVisualizer(QMainWindow):

    DEFAULT_WINDOW_MS = 1000
    MIN_WINDOW_MS     = 100
    MAX_WINDOW_MS     = 1500

    def __init__(self):
        super().__init__()

        # Per-file state (reset on every load)
        self.filename            = None
        self.session             = None
        self.selected_ch_indices = []
        self.channel_pos         = -1
        self._label_map          = {}

        # RMS noise per channel
        self._channel_rms = {}
        self._mean_rms    = None

        # Per-channel state
        self.current_ch_idx   = None
        self.current_ch_label = None
        self.data             = None
        self.filtered_data    = None
        self.show_filtered    = False
        self.sampling_rate    = None
        self.pool             = EventPool()
        self.phase            = "idle"
        self._comments        = {}

        # Unified action stack for single-key undo ("validate" | "invalidate")
        self._action_stack = []

        # Cached downsampled signal for the overview strip
        self._overview_t   = None
        self._overview_sig = None

        # Ensure artifact review only runs once per session
        self._artifact_review_done = False

        self.window_ms = self.DEFAULT_WINDOW_MS

        # Timer to auto-clear the save indicator after 3 s
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.timeout.connect(lambda: self.save_indicator.setText(""))

        self._build_ui()
        self._connect_signals()
        self._update_button_states()

    # ======================================================================
    # UI construction
    # ======================================================================
    def _build_ui(self):
        self.setWindowTitle("MAPLE Detection Software")
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # ── File row ──────────────────────────────────────────────────────
        file_row = QHBoxLayout()
        self.browse_btn = QPushButton("Browse")
        file_row.addWidget(self.browse_btn)
        file_row.addWidget(QLabel("File:"))
        self.filename_edit = QLineEdit()
        self.filename_edit.setPlaceholderText("path/to/recording.h5")
        file_row.addWidget(self.filename_edit, stretch=3)
        self.load_btn = QPushButton("Load")
        file_row.addWidget(self.load_btn)
        self.search_btn = QPushButton("Search")
        self.search_btn.setEnabled(False)
        file_row.addWidget(self.search_btn)
        root.addLayout(file_row)

        # ── Channel header banner ─────────────────────────────────────────
        self.channel_header = QLabel("No file loaded")
        self.channel_header.setAlignment(Qt.AlignCenter)
        self.channel_header.setStyleSheet(
            "font-weight: bold; font-size: 13px; padding: 7px; "
            "background: #eef2fb; border-radius: 5px;"
        )
        root.addWidget(self.channel_header)

        # ── Channel progress bar ──────────────────────────────────────────
        self.channel_progress = QProgressBar()
        self.channel_progress.setMaximumHeight(12)
        self.channel_progress.setTextVisible(False)
        self.channel_progress.setValue(0)
        self.channel_progress.setStyleSheet(
            "QProgressBar { border: 1px solid #bbb; border-radius: 5px; background: #f0f0f0; }"
            "QProgressBar::chunk { background: #3d7dc8; border-radius: 5px; }"
        )
        root.addWidget(self.channel_progress)

        # ── Noisy channel badge (hidden by default) ───────────────────────
        self.noisy_badge = QLabel("")
        self.noisy_badge.setAlignment(Qt.AlignCenter)
        self.noisy_badge.setStyleSheet(
            "background: #f5c842; color: #5a3e00; font-weight: bold; "
            "padding: 3px 8px; border-radius: 4px; font-size: 12px;"
        )
        self.noisy_badge.setVisible(False)
        root.addWidget(self.noisy_badge)

        # ── Matplotlib canvas (overview strip + main plot) ─────────────────
        self.figure = Figure(figsize=(8, 5), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        gs = GridSpec(5, 1, figure=self.figure, hspace=0.55)
        self.ax_overview = self.figure.add_subplot(gs[0])
        self.ax          = self.figure.add_subplot(gs[1:])
        self.ax_overview.set_visible(False)  # hidden until a channel is loaded
        root.addWidget(self.canvas, stretch=1)

        # ── Info label (event position, coloured by pool mode) ────────────
        self.info_label = QLabel("")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("border-radius: 4px; padding: 3px 8px;")
        root.addWidget(self.info_label)

        # ── View toggle row ───────────────────────────────────────────────
        view_toggle_row = QHBoxLayout()
        self.toggle_view_btn    = QPushButton("View Validated (0)")
        self.filter_toggle_btn  = QPushButton("Overlay Filtered Signal")
        self.filter_toggle_btn.setCheckable(True)
        self.filter_toggle_btn.setEnabled(False)
        view_toggle_row.addWidget(self.toggle_view_btn)
        view_toggle_row.addWidget(self.filter_toggle_btn)
        root.addLayout(view_toggle_row)

        # ── Window slider ─────────────────────────────────────────────────
        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel("± window (ms):"))
        self.window_slider = QSlider(Qt.Horizontal)
        self.window_slider.setRange(self.MIN_WINDOW_MS, self.MAX_WINDOW_MS)
        self.window_slider.setValue(self.DEFAULT_WINDOW_MS)
        self.window_slider.setFocusPolicy(Qt.NoFocus)
        slider_row.addWidget(self.window_slider, stretch=1)
        self.window_label = QLabel(f"{self.DEFAULT_WINDOW_MS} ms")
        self.window_label.setFixedWidth(70)
        slider_row.addWidget(self.window_label)
        root.addLayout(slider_row)

        # ── Event action buttons ──────────────────────────────────────────
        button_row = QHBoxLayout()
        self.prev_btn       = QPushButton("Prev [←]")
        self.next_btn       = QPushButton("Next [→]")
        self.validate_btn   = QPushButton("Validate [V]")
        self.undo_btn       = QPushButton("Undo [U]")
        self.invalidate_btn = QPushButton("Reject [R]")
        self.validate_btn.setStyleSheet("QPushButton:enabled { color: #1a7a2e; font-weight: bold; }")
        self.invalidate_btn.setStyleSheet("QPushButton:enabled { color: #a02020; font-weight: bold; }")
        for b in (self.prev_btn, self.next_btn, self.validate_btn,
                  self.undo_btn, self.invalidate_btn):
            button_row.addWidget(b)
        root.addLayout(button_row)

        # ── Comment row ───────────────────────────────────────────────────
        comment_row = QHBoxLayout()
        comment_row.addWidget(QLabel("Tag:"))
        self.comment_tags = QComboBox()
        self.comment_tags.addItems(["— tag —", "Clean", "Borderline", "Artifact", "Possible"])
        self.comment_tags.setFixedWidth(110)
        self.comment_tags.setEnabled(False)
        comment_row.addWidget(self.comment_tags)
        comment_row.addWidget(QLabel("Comment:"))
        self.comment_input = QLineEdit()
        self.comment_input.setPlaceholderText(
            "Type a comment here (click enter to remove your cursor from this box)"
        )
        self.comment_input.setEnabled(False)
        comment_row.addWidget(self.comment_input, stretch=1)
        root.addLayout(comment_row)

        # ── Workflow row ──────────────────────────────────────────────────
        workflow_row = QHBoxLayout()
        self.prev_channel_btn = QPushButton("◀ Prev Channel")
        self.prev_channel_btn.setStyleSheet("font-weight: bold;")
        self.next_channel_btn = QPushButton("Next Channel ▶")
        self.next_channel_btn.setStyleSheet("font-weight: bold;")
        workflow_row.addWidget(self.prev_channel_btn)
        workflow_row.addWidget(self.next_channel_btn)
        self.export_btn = QPushButton("Export to Excel")
        self.export_btn.setEnabled(False)
        workflow_row.addWidget(self.export_btn)
        workflow_row.addStretch()
        self.save_indicator = QLabel("")
        self.save_indicator.setStyleSheet("color: #28a745; font-size: 11px; padding: 2px 8px;")
        workflow_row.addWidget(self.save_indicator)
        root.addLayout(workflow_row)

        # Prevent buttons from stealing keyboard focus so arrow keys always
        # reach the window's keyPressEvent regardless of what was last clicked.
        for _btn in (
            self.browse_btn, self.load_btn, self.search_btn,
            self.prev_btn, self.next_btn,
            self.validate_btn, self.undo_btn, self.invalidate_btn,
            self.toggle_view_btn, self.filter_toggle_btn,
            self.prev_channel_btn, self.next_channel_btn, self.export_btn,
        ):
            _btn.setFocusPolicy(Qt.NoFocus)
        self.comment_tags.setFocusPolicy(Qt.NoFocus)
        self.canvas.setFocusPolicy(Qt.NoFocus)

    def _connect_signals(self):
        self.load_btn.clicked.connect(self.on_load)
        self.browse_btn.clicked.connect(self.on_browse)
        self.search_btn.clicked.connect(self.on_search)
        self.prev_btn.clicked.connect(self.on_previous)
        self.next_btn.clicked.connect(self.on_next)
        self.validate_btn.clicked.connect(self.on_validate)
        self.undo_btn.clicked.connect(self.on_undo)
        self.invalidate_btn.clicked.connect(self.on_invalidate)
        self.window_slider.valueChanged.connect(self.on_window_changed)
        self.toggle_view_btn.clicked.connect(self.on_toggle_view)
        self.filter_toggle_btn.clicked.connect(self.on_toggle_filter)
        self.prev_channel_btn.clicked.connect(self.on_prev_channel)
        self.next_channel_btn.clicked.connect(self.on_next_channel)
        self.export_btn.clicked.connect(self.on_export)
        self.comment_tags.currentTextChanged.connect(self._on_tag_selected)
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)
        self.comment_input.installEventFilter(self)

    def eventFilter(self, obj, event):
        if obj is self.comment_input and event.type() == QEvent.KeyPress:
            key  = event.key()
            mods = event.modifiers()
            if key in (Qt.Key_Escape, Qt.Key_Return, Qt.Key_Enter):
                self.comment_input.clearFocus()
                return True
            if mods == Qt.NoModifier:
                if key == Qt.Key_V:
                    self.on_validate()
                    return True
                if key == Qt.Key_R:
                    self.on_invalidate()
                    return True
                if key == Qt.Key_U:
                    self.on_undo()
                    return True
        return super().eventFilter(obj, event)

    # ======================================================================
    # Keyboard shortcuts
    # ======================================================================
    def keyPressEvent(self, event):
        # Don't intercept arrow keys while filename field has focus
        if self.filename_edit.hasFocus():
            super().keyPressEvent(event)
            return

        mods = event.modifiers()
        key  = event.key()

        if mods == Qt.NoModifier:
            if key == Qt.Key_V:
                self.on_validate()
            elif key == Qt.Key_R:
                self.on_invalidate()
            elif key == Qt.Key_U:
                self.on_undo()
            elif key == Qt.Key_Left:
                self.on_previous()
            elif key == Qt.Key_Right:
                self.on_next()
            else:
                super().keyPressEvent(event)
        else:
            super().keyPressEvent(event)

    # ======================================================================
    # File loading
    # ======================================================================
    def on_load(self):
        filename = self.filename_edit.text().strip()
        if not os.path.isfile(filename):
            self.info_label.setText("Error: file does not exist.")
            return

        wait = WaitDialog("Reading file, please wait...", parent=self)
        wait.show()
        wait.repaint()
        QApplication.processEvents()

        try:
            good_indices, good_labels, fs = _read_file_metadata(filename)
        except (KeyError, OSError) as e:
            wait.close()
            self.info_label.setText(f"Error reading file: {e}")
            return
        wait.close()

        self.filename     = filename
        self._channel_rms = {}
        self._mean_rms    = None

        dlg = ChannelSelectionDialog(good_labels, parent=self)
        if dlg.exec_() != QDialog.Accepted:
            return

        sel_label_indices        = dlg.selected_label_indices()
        self.selected_ch_indices = [good_indices[i] for i in sel_label_indices]

        decimated_fs = fs / DECIMATION

        self.session = Session.load_or_new(filename, decimated_fs, DECIMATION)
        self._label_map = {
            idx: label
            for idx, label in zip(good_indices, good_labels)
        }
        self.channel_pos   = 0
        self.export_btn.setEnabled(True)   # allow export any time after loading
        self.search_btn.setEnabled(True)
        self.filename_edit.setEnabled(False)
        self.browse_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
        self.show_filtered = False
        self.filter_toggle_btn.setChecked(False)

        n_sel = len(self.selected_ch_indices)
        self.channel_progress.setMaximum(n_sel)
        self.channel_progress.setValue(0)
        self.info_label.setText(f"{n_sel} channel(s) selected  |  Starting review...")
        self._load_channel_at_pos()

    def on_browse(self):
        start_dir = ""
        current = self.filename_edit.text().strip()
        if current and os.path.isdir(os.path.dirname(current)):
            start_dir = os.path.dirname(current)
        path, _ = QFileDialog.getOpenFileName(
            self, "Select recording file", start_dir, "HDF5 (*.h5)"
        )
        if path:
            self.filename_edit.setText(path)

    def on_search(self):
        if not self.filename or not self._label_map:
            return
        dlg = SearchDialog(self.filename, self._label_map, parent=self)
        dlg.exec_()

    # ======================================================================
    # Channel-level flow
    # ======================================================================
    def _load_channel_at_pos(self):
        if self.channel_pos >= len(self.selected_ch_indices):
            self._finish_session()
            return
        self._load_channel(self.selected_ch_indices[self.channel_pos])

    def _load_channel(self, ch_idx):
        ch_label = self._label_map[ch_idx]

        wait = WaitDialog(
            f"Running detection on channel {ch_label}, please wait",
            parent=self,
        )
        wait.show()
        wait.repaint()
        QApplication.processEvents()

        try:
            data, fs = mea_io.read_and_decimate(
                self.filename, ch_idx, decimation=DECIMATION
            )
        except (IndexError, OSError) as e:
            wait.close()
            self.info_label.setText(f"Skipping electrode {ch_label}: {e}")
            self.channel_pos += 1
            self._load_channel_at_pos()
            return

        self.current_ch_idx   = ch_idx
        self.current_ch_label = ch_label
        self.data             = data
        self.sampling_rate    = fs
        self.phase            = "manual"
        self._action_stack.clear()

        self.session.start_channel(ch_idx, ch_label)

        all_candidates = detection.detect_events(self.data)
        wait.close()

        # Restore pool state if this channel was previously completed
        ch_state = self.session.channel_state(ch_idx)
        if ch_state and ch_state["stage"] == "complete":
            validated_set = set(ch_state["validated_events"])
            self.pool.candidates = [c for c in all_candidates if c not in validated_set]
            self.pool.validated  = sorted(validated_set)
            self.pool.mode       = "candidates"
            self.pool.position   = 0
            self.pool._saved_positions = {"candidates": 0, "validated": 0}
            self.pool.validate_undo.clear()
            self.pool.invalidate_undo.clear()
            self.session.channels[str(ch_idx)]["stage"] = "in_progress"
            self._comments = {
                int(k): v for k, v in ch_state.get("comments", {}).items()
            }
        else:
            self.pool.reset(all_candidates)
            self._comments = {}

        self.filtered_data = detection.bandpass_filter(self.data)
        self.comment_input.clear()
        self.comment_tags.setCurrentIndex(0)

        # Cache RMS for noise comparison
        self._channel_rms[ch_idx] = detection.mad_noise_std(self.filtered_data)
        if len(self._channel_rms) > 1:
            self._mean_rms = float(np.mean(list(self._channel_rms.values())))

        # Cache downsampled signal for the overview strip
        step = max(1, len(self.data) // 5000)
        self._overview_t   = np.arange(0, len(self.data), step) / self.sampling_rate
        self._overview_sig = self.data[::step]

        # Update progress bar
        n_pos   = self.channel_pos + 1
        n_total = len(self.selected_ch_indices)
        self.channel_progress.setValue(n_pos)

        self._update_channel_header(
            f"Channel  {n_pos} of {n_total}"
            f"  |  Electrode  {ch_label}"
            f"  |  {len(all_candidates)} candidate event(s)"
        )

        # Noisy channel badge (persists while on channel; replaces one-time popup)
        ch_rms = self._channel_rms[ch_idx]
        if self._mean_rms and ch_rms > 1.5 * self._mean_rms:
            self.noisy_badge.setText(
                f"⚠  High Noise  —  Channel RMS {ch_rms:.1f} µV  "
                f"vs. mean {self._mean_rms:.1f} µV, review with caution"
            )
            self.noisy_badge.setVisible(True)
        else:
            self.noisy_badge.setVisible(False)

        self.ax_overview.set_visible(True)
        self.refresh()

    def _save_current_channel(self):
        """Persist the current channel's review state to the session JSON."""
        if self.current_ch_idx is None or self.phase != "manual":
            return
        self.session.finish_channel(
            self.current_ch_idx,
            list(self.pool.validated),
            list(self.pool.candidates),
            self._comments,
        )
        self.session.save()
        self._show_saved()

    def _show_saved(self):
        self.save_indicator.setText("Session saved")
        self._save_timer.start(3000)

    def on_next_channel(self):
        if self.phase != "manual":
            return
        self.phase = "idle"
        self._save_current_channel()
        if self.channel_pos >= len(self.selected_ch_indices) - 1:
            self._finish_session()
            return
        self.channel_pos += 1
        self._load_channel_at_pos()

    def on_prev_channel(self):
        if self.phase != "manual" or self.channel_pos <= 0:
            return
        self._save_current_channel()
        self.channel_pos -= 1
        self._load_channel_at_pos()

    def _run_artifact_review(self):
        """Run cross-channel coincidence check and show the review dialog if needed.

        Safe to call multiple times — skips silently after the first run.
        Returns the groups list (may be empty).
        """
        if self._artifact_review_done:
            return []

        wait = WaitDialog("Checking for cross-channel artifacts, please wait...", parent=self)
        wait.show()
        wait.repaint()
        QApplication.processEvents()

        try:
            events_per_channel = {
                int(k): v["validated_events"]
                for k, v in self.session.channels.items()
                if v.get("validated_events")
            }
            groups = find_coincident_events(events_per_channel, fs=self.session.fs)
        except Exception as e:
            wait.close()
            QMessageBox.critical(self, "Artifact check error", str(e))
            return []

        wait.close()

        if groups:
            dlg = ArtifactReviewDialog(
                self.session, self.filename, self.session.fs,
                groups, self._label_map, parent=self,
            )
            dlg.exec_()
            self.session.save()

        self._artifact_review_done = True
        return groups

    def _finish_session(self):
        groups = self._run_artifact_review()

        self.phase = "done"
        self.filtered_data = None
        self.noisy_badge.setVisible(False)
        self.ax_overview.set_visible(False)
        self._update_channel_header("All channels processed, ready to export")

        n_total = sum(
            len(ch.get("validated_events", []))
            for ch in self.session.channels.values()
        )
        n_chs_with_events = sum(
            1 for ch in self.session.channels.values()
            if ch.get("validated_events")
        )
        artifact_note = (
            f"{len(groups)} artifact group(s) reviewed" if groups
            else f"artifact check skipped (need ≥2 channels with events, have {n_chs_with_events})"
            if n_chs_with_events < 2
            else "no cross-channel artifacts detected"
        )

        self.info_label.setText(
            f"Done.  {len(self.session.channels)} channel(s) processed  |  "
            f"{n_total} total event(s)  |  {artifact_note}  |  "
            f"Click 'Export to Excel' to save results."
        )
        self.info_label.setStyleSheet("border-radius: 4px; padding: 3px 8px;")
        self.ax.clear()
        self.canvas.draw()
        self._update_button_states()

    # ======================================================================
    # Excel export
    # ======================================================================
    def on_export(self):
        # Save current channel state so the export includes any unsaved work
        self._save_current_channel()

        # Run artifact review if all channels are done and it hasn't run yet
        if self.phase == "done":
            self._run_artifact_review()

        default = os.path.splitext(self.filename)[0] + "_events.xlsx"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Excel Report", default, "Excel (*.xlsx)"
        )
        if not path:
            return

        wait = WaitDialog("Exporting events, please wait...", parent=self)
        wait.show()
        wait.repaint()
        QApplication.processEvents()
        try:
            export_events_excel(
                path,
                self.session,
                self.filename,
                self.session.fs,
            )
        except Exception as e:
            wait.close()
            QMessageBox.critical(self, "Export error", str(e))
            return
        wait.close()
        QMessageBox.information(
            self, "Export complete",
            f"Events exported to:\n{path}"
        )

    # ======================================================================
    # Plotting & state display
    # ======================================================================
    def update_plot(self, _draw=True):
        self.ax.clear()
        if self.data is None or len(self.data) == 0:
            if _draw:
                self.canvas.draw()
            return

        idx = self.pool.current_index()
        if idx is None:
            if _draw:
                self.canvas.draw()
            return

        lo, hi = detection.BAND
        n      = self._ms_to_samples(self.window_ms)
        start  = max(0, idx - n)
        end    = min(idx + n, len(self.data))
        t      = (np.arange(start, end) - idx) / self.sampling_rate * 1000

        # Always plot raw signal
        self.ax.plot(t, self.data[start:end],
                     linewidth=0.7, color="steelblue", label="raw")

        # Overlay filtered signal when toggled
        if self.show_filtered and self.filtered_data is not None:
            self.ax.plot(t, self.filtered_data[start:end],
                         linewidth=1.0, color="#e05c00", alpha=0.85,
                         label=f"filtered [{lo}–{hi} Hz]")
            self.ax.legend(loc="upper right", fontsize=7)

        self.ax.axvline(0, color="red", linestyle="--", alpha=0.5)
        self.ax.set_xlabel("Time (ms)")
        self.ax.set_ylabel("Voltage (µV)")

        time_s  = idx / self.sampling_rate
        self.ax.set_title(
            f"Electrode {self.current_ch_label}  —  "
            f"Event {self.pool.position + 1} / {len(self.pool.current_pool())}"
            f"  (t = {time_s:.3f} s)"
        )

        # Populate comment box with saved comment for this event
        if not self.comment_input.hasFocus():
            comment = self._comments.get(idx, "")
            self.comment_input.blockSignals(True)
            self.comment_input.setText(comment)
            self.comment_input.blockSignals(False)
            _TAG_OPTIONS = ("Clean", "Borderline", "Artifact", "Possible")
            self.comment_tags.blockSignals(True)
            self.comment_tags.setCurrentText(
                comment if comment in _TAG_OPTIONS else "— tag —"
            )
            self.comment_tags.blockSignals(False)

        if _draw:
            self.canvas.draw()

    def _update_overview(self, _draw=True):
        """Redraw the channel overview strip with event markers and current position."""
        self.ax_overview.clear()
        if self._overview_t is None or self.phase != "manual":
            if _draw:
                self.canvas.draw()
            return

        self.ax_overview.plot(
            self._overview_t, self._overview_sig,
            linewidth=0.3, color="#888", rasterized=True
        )

        for ev in self.pool.validated:
            self.ax_overview.axvline(
                ev / self.sampling_rate, color="#28a745", alpha=0.6, linewidth=0.8
            )
        for ev in self.pool.candidates:
            self.ax_overview.axvline(
                ev / self.sampling_rate, color="#cc3333", alpha=0.4, linewidth=0.6
            )

        idx = self.pool.current_index()
        if idx is not None:
            self.ax_overview.axvline(
                idx / self.sampling_rate, color="blue", linewidth=1.5, alpha=0.9
            )

        self.ax_overview.set_yticks([])
        self.ax_overview.tick_params(labelsize=6)
        self.ax_overview.set_xlabel("Time (s)", fontsize=7)
        self.ax_overview.set_title("Signal Overview", fontsize=7, loc="left")
        self.ax_overview.legend(
            handles=[
                Line2D([0], [0], color="#28a745", linewidth=1.5, label="validated"),
                Line2D([0], [0], color="#cc3333", linewidth=1.5, label="candidate"),
                Line2D([0], [0], color="blue",    linewidth=1.5, label="current"),
            ],
            loc="upper right", fontsize=6, framealpha=0.7,
        )

        if _draw:
            self.canvas.draw()

    def _on_canvas_click(self, event):
        """Click on the overview strip to jump to the nearest event."""
        if event.inaxes is not self.ax_overview:
            return
        if event.xdata is None or self.phase != "manual":
            return

        clicked_sample = int(event.xdata * self.sampling_rate)
        all_evs = self.pool.candidates + list(self.pool.validated)
        if not all_evs:
            return

        nearest = min(all_evs, key=lambda x: abs(x - clicked_sample))

        if nearest in self.pool.candidates:
            if self.pool.mode != "candidates":
                self.pool.toggle_mode()
            self.pool.position = self.pool.candidates.index(nearest)
        else:
            if self.pool.mode != "validated":
                self.pool.toggle_mode()
            self.pool.position = list(self.pool.validated).index(nearest)

        self.refresh()

    def refresh(self):
        if self.pool.mode == "candidates":
            self.validate_btn.setText("Validate [V]")
            self.toggle_view_btn.setText(
                f"View Validated ({len(self.pool.validated)})"
            )
            self.info_label.setStyleSheet(
                "background: #ddeeff; border-radius: 4px; padding: 3px 8px;"
            )
        else:
            self.validate_btn.setText("Unvalidate [V]")
            self.toggle_view_btn.setText(
                f"View Candidates ({len(self.pool.candidates)})"
            )
            self.info_label.setStyleSheet(
                "background: #ddffee; border-radius: 4px; padding: 3px 8px;"
            )

        pool  = self.pool.current_pool()
        n_val = len(self.pool.validated)
        n_can = len(self.pool.candidates)

        if not pool:
            if self.pool.mode == "candidates":
                label = f"All candidates reviewed  |  {n_val} validated"
            else:
                label = f"No events validated yet  |  {n_can} candidate(s) remaining"
        elif self.pool.mode == "candidates":
            label = f"Candidate {self.pool.position + 1} of {n_can}  |  {n_val} validated"
        else:
            label = f"Validated {self.pool.position + 1} of {n_val}  |  {n_can} candidate(s) remaining"

        self.info_label.setText(label)

        # Single canvas draw after both plots are updated
        self.update_plot(_draw=False)
        self._update_overview(_draw=False)
        self.canvas.draw()

        self._update_button_states()

    def _update_channel_header(self, text):
        self.channel_header.setText(text)

    # ======================================================================
    # Button handlers
    # ======================================================================
    def on_validate(self):
        idx         = self.pool.current_index()
        mode_before = self.pool.mode
        if self.pool.validate():
            if mode_before == "candidates" and idx is not None:
                comment = self.comment_input.text().strip()
                if comment:
                    self._comments[idx] = comment
            self._action_stack.append("validate")
            self.comment_input.clear()
            self.comment_tags.setCurrentIndex(0)
            self.refresh()
            self._save_current_channel()

    def on_invalidate(self):
        if self.pool.invalidate():
            self._action_stack.append("invalidate")
            self.comment_input.clear()
            self.comment_tags.setCurrentIndex(0)
            self.refresh()
            self._save_current_channel()

    def on_undo(self):
        if not self._action_stack:
            return
        last = self._action_stack.pop()
        if last == "validate":
            if self.pool.undo_validate():
                self.comment_input.clear()
                self.comment_tags.setCurrentIndex(0)
                self.refresh()
                self._save_current_channel()
            else:
                self._action_stack.append(last)  # put back — shouldn't happen
        elif last == "invalidate":
            if self.pool.undo_invalidate():
                self.refresh()
                self._save_current_channel()
            else:
                self._action_stack.append(last)

    def on_next(self):
        if self.pool.next():
            self.refresh()

    def on_previous(self):
        if self.pool.prev():
            self.refresh()

    def on_window_changed(self, value):
        self.window_ms = value
        self.window_label.setText(f"{value} ms")
        if self.data is not None:
            self.update_plot()

    def on_toggle_view(self):
        self.pool.toggle_mode()
        self.refresh()

    def on_toggle_filter(self):
        self.show_filtered = self.filter_toggle_btn.isChecked()
        if self.data is not None:
            self.update_plot()

    def _on_tag_selected(self, text):
        if text and text != "— tag —":
            self.comment_input.setText(text)

    # ======================================================================
    # Helpers
    # ======================================================================
    def _ms_to_samples(self, ms):
        if self.sampling_rate is None:
            return 0
        return int(round(ms * self.sampling_rate / 1000.0))

    def _update_button_states(self):
        pool            = self.pool.current_pool()
        has_events      = bool(pool)
        pos             = self.pool.position
        in_active_phase = self.phase == "manual"

        self.prev_btn.setEnabled(
            in_active_phase and has_events and pos > 0)
        self.next_btn.setEnabled(
            in_active_phase and has_events and pos < len(pool) - 1)
        self.validate_btn.setEnabled(
            in_active_phase and has_events)
        self.invalidate_btn.setEnabled(
            in_active_phase and has_events)
        self.undo_btn.setEnabled(
            in_active_phase and bool(self._action_stack))
        self.toggle_view_btn.setEnabled(in_active_phase)
        self.filter_toggle_btn.setEnabled(in_active_phase)
        self.comment_input.setEnabled(in_active_phase)
        self.comment_tags.setEnabled(in_active_phase)
        self.prev_channel_btn.setEnabled(
            in_active_phase and self.channel_pos > 0)
        self.next_channel_btn.setEnabled(in_active_phase)


def main():
    app = QApplication(sys.argv)
    win = EventVisualizer()
    win.resize(1100, 720)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
