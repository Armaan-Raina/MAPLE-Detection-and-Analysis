"""
MEA file I/O.

Reading MultiChannel Systems HDF5 exports: discovering channels (with
reference-electrode exclusion by label), loading single channels lazily,
and converting to microvolts.
"""

import h5py
import numpy as np
from scipy import signal as sps


# Paths inside MCS HDF5 exports
SIGNAL_PATH = "Data/Recording_0/AnalogStream/Stream_0/ChannelData"
INFO_PATH   = "Data/Recording_0/AnalogStream/Stream_0/InfoChannel"

# Electrode 15 is always the reference
REFERENCE_LABEL = "15"



def _decode_label(raw: bytes) -> str:
    """
    Args:
        raw: Potentially raw (sequence of bytes) input to be decoded

    Returns:
        The string conversion of the raw argument
    """

    if isinstance(raw, bytes):
        raw = raw.decode()
    return str(raw).strip()



def read_channel(filename, idx):
    """Read a single channel from an MCS HDF5 file, scaled to microvolts.

    Returns (signal_uv, fs).
    """
    with h5py.File(filename, "r") as f:
        signal_ds = f[SIGNAL_PATH]
        info = f[INFO_PATH][:]

        n_channels = signal_ds.shape[0]
        if idx < 0 or idx >= n_channels:
            raise IndexError(
                f"Channel index {idx} out of range (file has {n_channels})"
            )

        raw = signal_ds[idx, :].astype(np.float32)

        # MCS scaling: V = raw * ConversionFactor * 10^Exponent
        conv = float(info[idx]["ConversionFactor"])
        exp = int(info[idx]["Exponent"])
        raw *= conv * (10.0 ** exp)

        unit = info[idx]["Unit"]
        if isinstance(unit, bytes):
            unit = unit.decode()
        if unit == "V":
            raw *= 1e6

        tick_us = float(info[idx]["Tick"])
        fs = 1e6 / tick_us

    return raw, fs

#HEAVY decimation
def read_and_decimate(filename, idx, decimation=25):
    """Read a channel and decimate. Returns (data, fs_after_decimation)."""
    raw, fs = read_channel(filename, idx)
    data = sps.decimate(raw, decimation).astype(np.float32)
    return data, fs / decimation