"""CZ raw-volume loader.

Extracted from ``sessions/03c_onset_features/iterations/iter08_cz_prior.py``.
Only ``load_cz_volume`` is promoted; the surface-fitting research code stays
in the session directory.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile


def load_cz_volume(s) -> np.ndarray:
    """Load the CZ 488 raw TIFF for subject ``s`` as a float32 (Z, Y, X) array.

    Searches ``s.coreg_dir`` for ``*reg-dim-swapped.ome.tif`` first, then
    ``*zstack.tif`` as fallback — the two naming conventions that appear in
    the benchmark dataset.  Squeezes leading size-1 dimensions to ensure the
    output is exactly 3-D.
    """
    files = (
        list(s.coreg_dir.glob("*reg-dim-swapped.ome.tif"))
        or list(s.coreg_dir.glob("*zstack.tif"))
    )
    if not files:
        raise FileNotFoundError(
            f"No CZ TIFF found in {s.coreg_dir} for subject {s.subject_id}"
        )
    arr = tifffile.imread(str(files[0]))
    # Some OME-TIFFs carry extra leading size-1 T/C dimensions.
    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(
            f"CZ TIFF for {s.subject_id} has unexpected shape {arr.shape} "
            f"(expected 3-D ZYX after squeezing)"
        )
    return arr.astype(np.float32, copy=False)
