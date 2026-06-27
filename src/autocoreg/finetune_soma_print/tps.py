"""Thin-plate-spline warp + soma-print neighbour scoring / anchor-vote.

Production helpers for the soma-print matcher, extracted from the original
iterative protocol (now archived at autocoreg.archive.iterative_matcher).
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import Rbf
from scipy.spatial import cKDTree

from autocoreg.finetune_soma_print.descriptor import cell_vectors


def fit_tps(src_zyx: np.ndarray, dst_zyx: np.ndarray) -> dict | None:
    """Per-axis thin-plate Rbf src → dst."""
    if len(src_zyx) < 4:
        return None
    try:
        rz = Rbf(src_zyx[:, 0], src_zyx[:, 1], src_zyx[:, 2],
                  dst_zyx[:, 0], function="thin_plate")
        ry = Rbf(src_zyx[:, 0], src_zyx[:, 1], src_zyx[:, 2],
                  dst_zyx[:, 1], function="thin_plate")
        rx = Rbf(src_zyx[:, 0], src_zyx[:, 1], src_zyx[:, 2],
                  dst_zyx[:, 2], function="thin_plate")
    except Exception:
        return None
    return dict(rbf_z=rz, rbf_y=ry, rbf_x=rx, src=src_zyx, dst=dst_zyx)

def apply_tps(tps: dict, pts_zyx: np.ndarray) -> np.ndarray:
    z = tps["rbf_z"](pts_zyx[:, 0], pts_zyx[:, 1], pts_zyx[:, 2])
    y = tps["rbf_y"](pts_zyx[:, 0], pts_zyx[:, 1], pts_zyx[:, 2])
    x = tps["rbf_x"](pts_zyx[:, 0], pts_zyx[:, 1], pts_zyx[:, 2])
    return np.column_stack([z, y, x])

def soma_score_with_neighbour_indices(
    cz_zyx: np.ndarray, hcr_zyx: np.ndarray,
    cand_pairs: list[tuple[int, int]],
    *, m_cz: int, m_hcr: int, n: int,
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Score each candidate pair AND return the n best (cz_nbr, hcr_nbr)
    index pairs in the original CZ / HCR neighbour lists so we can run
    the anchor-vote later.
    """
    cz_vecs = cell_vectors(cz_zyx, m=m_cz)
    hcr_vecs = cell_vectors(hcr_zyx, m=m_hcr)
    cz_tree = cKDTree(cz_zyx)
    hcr_tree = cKDTree(hcr_zyx)
    _, cz_nbr_idx = cz_tree.query(cz_zyx, k=min(m_cz + 1, len(cz_zyx)))
    _, hcr_nbr_idx = hcr_tree.query(hcr_zyx, k=min(m_hcr + 1, len(hcr_zyx)))
    cz_nbr_idx = cz_nbr_idx[:, 1:]
    hcr_nbr_idx = hcr_nbr_idx[:, 1:]
    scores = np.full(len(cand_pairs), np.inf, dtype=np.float32)
    best_pairs_per_cand: list[tuple[np.ndarray, np.ndarray]] = []
    for k, (i, j) in enumerate(cand_pairs):
        ci = cz_vecs[i]
        hj = hcr_vecs[j]
        if ci.shape[0] == 0 or hj.shape[0] == 0:
            best_pairs_per_cand.append((np.empty(0, int), np.empty(0, int)))
            continue
        diff = ci[:, None, :] - hj[None, :, :]
        d = np.linalg.norm(diff, axis=-1)
        flat = d.ravel()
        if flat.size < n:
            best_pairs_per_cand.append((np.empty(0, int), np.empty(0, int)))
            continue
        order = np.argpartition(flat, n - 1)[:n]
        scores[k] = float(flat[order].mean())
        rows = order // d.shape[1]
        cols = order % d.shape[1]
        rows = np.clip(rows, 0, cz_nbr_idx.shape[1] - 1)
        cols = np.clip(cols, 0, hcr_nbr_idx.shape[1] - 1)
        best_pairs_per_cand.append(
            (cz_nbr_idx[i, rows], hcr_nbr_idx[j, cols])
        )
    return scores, best_pairs_per_cand

def anchor_vote(
    best_pair_indices: list[tuple[np.ndarray, np.ndarray]],
    cand_row_pairs: list[tuple[int, int]],
    active_row_pairs: set[tuple[int, int]],
) -> np.ndarray:
    """Fraction of (cz_nbr, hcr_nbr) cells implied by the soma score's
    n-best vector pairs that are themselves in the active match set."""
    out = np.zeros(len(cand_row_pairs), dtype=np.float32)
    for k, (cz_idx, hcr_idx) in enumerate(best_pair_indices):
        if cz_idx.size == 0:
            continue
        hits = sum(
            int((int(a), int(b)) in active_row_pairs)
            for a, b in zip(cz_idx, hcr_idx)
        )
        out[k] = hits / cz_idx.size
    return out
