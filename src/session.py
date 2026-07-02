"""
Session: persistence layer for a multi-channel detection workflow.

Stores everything in a JSON sidecar next to the recording. Survives crashes,
supports resumption, and is human-readable for debugging.

JSON structure:
{
  "filename": "...",
  "fs": 1000.0,
  "decimation": 25,
  "channels": {
    "0": {
        "label": "47",
        "stage": "in_progress" | "complete",
        "validated_events": [123, 456, ...],
        "rejected_events":  [789, ...]
    },
    ...
  }
}
"""

import json
import os

def session_path_for(recording_filename):
    """Return the JSON path used to persist a session for a given .h5 file."""
    base, _ = os.path.splitext(recording_filename)
    return f"{base}_session.json"


class Session:
    """In-memory session state with load/save to JSON sidecar."""

    def __init__(self, recording_filename, fs, decimation):
        self.recording_filename = recording_filename
        self.session_filename = session_path_for(recording_filename)
        self.fs = float(fs)
        self.decimation = int(decimation)
        self.channels = {}

    # ------------------------------------------------------------------
    # Load / save
    # ------------------------------------------------------------------
    @classmethod
    def load_or_new(cls, recording_filename, fs, decimation):
        """Return an existing session if its JSON exists, else a fresh one."""
        path = session_path_for(recording_filename)
        if os.path.isfile(path):
            with open(path, "r") as f:
                payload = json.load(f)
            s = cls(payload["filename"], payload["fs"], payload["decimation"])
            s.channels = payload.get("channels", {})
            return s
        return cls(recording_filename, fs, decimation)

    def save(self):
        payload = {
            "filename": self.recording_filename,
            "fs": self.fs,
            "decimation": self.decimation,
            "channels": self.channels,
        }
        tmp = self.session_filename + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, self.session_filename)

    # ------------------------------------------------------------------
    # Per-channel state
    # ------------------------------------------------------------------
    def channel_state(self, ch_idx):
        return self.channels.get(str(ch_idx))

    def channel_stage(self, ch_idx):
        st = self.channel_state(ch_idx)
        return st["stage"] if st else None

    def start_channel(self, ch_idx, label):
        """Initialize bookkeeping for a channel."""
        key = str(ch_idx)
        if key not in self.channels:
            self.channels[key] = {
                "label": str(label),
                "stage": "in_progress",
                "validated_events": [],
                "rejected_events": [],
                "comments": {},
            }

    def finish_channel(self, ch_idx, validated, rejected, comments=None):
        """Record manual review results for a channel."""
        key = str(ch_idx)
        if key not in self.channels:
            raise ValueError(f"Channel {ch_idx} not started")
        self.channels[key]["validated_events"] = list(map(int, validated))
        self.channels[key]["rejected_events"] = list(map(int, rejected))
        self.channels[key]["comments"] = {
            str(k): v for k, v in (comments or {}).items()
        }
        self.channels[key]["stage"] = "complete"
