"""Thin-plate-spline warp + soma-print neighbour scoring / anchor-vote.

Production helpers for the soma-print matcher, extracted from the original
iterative protocol (now archived at autocoreg.archive.iterative_matcher).
"""
from __future__ import annotations

import os
import numpy as np
from scipy.interpolate import Rbf
from scipy.spatial import cKDTree, ConvexHull

from autocoreg.finetune_soma_print.descriptor import cell_vectors


# --- Bounded-extrapolation control (see fit_tps/apply_tps) ------------------
# The thin-plate kernel r^2*log(r) grows super-linearly, so *outside* the
# anchor cloud the warp extrapolates without bound. That unbounded drag is the
# mechanism by which a slab of cells lying just beyond the anchor hull (e.g. the
# deepest CZ z-plane whose true HCR partners fall outside the reliable overlap)
# gets pulled by a large, coherent vector onto a translated constellation and
# then locked in as anchors. We therefore blend the exact thin-plate warp
# (trustworthy INSIDE the anchor source hull) toward a robust global affine
# continuation (bounded) OUTSIDE it. w=1 strictly inside the hull -> the warp is
# BIT-IDENTICAL to the pure-TPS behaviour for every interior point (hence no
# change to any in-hull / benchmark match); w->0 beyond it.
TPS_EXTRAP_TAU_UM = 40.0     # blend decay length (um) once past the grace zone
# GRACE ZONE: keep the exact thin-plate warp (w=1) until this far OUTSIDE the hull,
# then decay to affine. Normal near-hull extrapolation (a real peripheral cell sits
# ≲1 candidate-radius beyond the anchor hull) is left UNTOUCHED — so the fix is inert
# on well-covered/sparse pools — while the pathological far drag (a deep orphan sheet
# pulled ~130µm out) is still bounded. Sized to the adaptive_r_cand floor (~50µm), not
# a magic constant. (margin<0 ⇒ grace outside the hull; the blend uses clip(sd+margin,0,∞).)
TPS_EXTRAP_MARGIN_UM = -50.0


def _fit_robust_affine(src_zyx: np.ndarray, dst_zyx: np.ndarray,
                       n_iter: int = 2) -> np.ndarray:
    """Robust affine A (4x3, row-vec: dst ~= [src,1] @ A) via IRLS, so the
    ~majority of correct anchors dominate and a small coherent bad block barely
    moves it. Same robust-linear pattern used elsewhere in the pipeline."""
    X = np.column_stack([src_zyx, np.ones(len(src_zyx))])
    w = np.ones(len(src_zyx))
    A = np.linalg.lstsq(X, dst_zyx, rcond=None)[0]
    for _ in range(n_iter):
        res = np.linalg.norm(dst_zyx - X @ A, axis=1)
        s = np.median(res) + 1e-6
        w = 1.0 / (1.0 + (res / (2.5 * s)) ** 2)
        Wr = np.sqrt(w)[:, None]
        A = np.linalg.lstsq(X * Wr, dst_zyx * Wr, rcond=None)[0]
    return A


def fit_tps(src_zyx: np.ndarray, dst_zyx: np.ndarray) -> dict | None:
    """Per-axis thin-plate Rbf src → dst, plus a robust global affine and the
    anchor-source convex hull so ``apply_tps`` can bound extrapolation."""
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
    try:
        affine = _fit_robust_affine(src_zyx, dst_zyx)
    except Exception:
        affine = None
    try:
        hull_eqs = ConvexHull(src_zyx).equations  # (n_faces, 4); a.x + b = signed dist
    except Exception:
        hull_eqs = None
    return dict(rbf_z=rz, rbf_y=ry, rbf_x=rx, src=src_zyx, dst=dst_zyx,
                affine=affine, hull_eqs=hull_eqs,
                tau=TPS_EXTRAP_TAU_UM, margin=TPS_EXTRAP_MARGIN_UM)


def apply_tps(tps: dict, pts_zyx: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts_zyx, dtype=float)
    tps_pred = np.column_stack([
        tps["rbf_z"](pts[:, 0], pts[:, 1], pts[:, 2]),
        tps["rbf_y"](pts[:, 0], pts[:, 1], pts[:, 2]),
        tps["rbf_x"](pts[:, 0], pts[:, 1], pts[:, 2]),
    ])
    # Legacy warps (or degenerate hull), or the bound disabled -> pure thin-plate.
    affine = tps.get("affine"); hull_eqs = tps.get("hull_eqs")
    if affine is None or hull_eqs is None or os.environ.get("MFISH_TPS_BOUND", "1") == "0":
        return tps_pred
    tau = float(tps.get("tau", TPS_EXTRAP_TAU_UM))
    margin = float(tps.get("margin", TPS_EXTRAP_MARGIN_UM))
    affine_pred = np.column_stack([pts, np.ones(len(pts))]) @ affine
    # signed distance to the anchor-source hull: >0 outside, <=0 inside.
    sd = (pts @ hull_eqs[:, :3].T + hull_eqs[:, 3]).max(axis=1)
    over = np.clip(sd + margin, 0.0, None)
    w = np.exp(-(over / tau) ** 2)          # w=1 inside hull -> exact TPS (bit-identical)
    return w[:, None] * tps_pred + (1.0 - w)[:, None] * affine_pred

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
