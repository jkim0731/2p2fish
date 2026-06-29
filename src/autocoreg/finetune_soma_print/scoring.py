"""Vectorised soma-print scoring of one CZ descriptor vs many HCR candidates.

Extracted from the original locked-frame protocol (now archived at
autocoreg.archive.locked_benchmark).
"""
from __future__ import annotations

import numpy as np


def _score_soma_per_gt(
    cz_vec: np.ndarray, hcr_vecs_stack: np.ndarray, n: int,
) -> np.ndarray:
    """Vectorised soma score for one GT-CZ vs many HCR candidates.

    cz_vec: (m_cz, 3)
    hcr_vecs_stack: (n_cands, m_hcr, 3)
    Returns: (n_cands,) — mean of n-smallest distances over m_cz × m_hcr per cand.
    """
    diff = cz_vec[None, :, None, :] - hcr_vecs_stack[:, None, :, :]
    d = np.linalg.norm(diff, axis=-1)  # (n_cands, m_cz, m_hcr)
    flat = d.reshape(d.shape[0], -1)  # (n_cands, m_cz*m_hcr)
    if flat.shape[1] < n:
        return np.full(flat.shape[0], np.inf, dtype=np.float32)
    order = np.argpartition(flat, n - 1, axis=1)[:, :n]
    rows = np.arange(flat.shape[0])[:, None]
    return flat[rows, order].mean(axis=1).astype(np.float32)
