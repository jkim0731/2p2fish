"""3D Soma-print (Wang et al., 2026 — adapted) for cell matching.

API
---
``cell_vectors(points_zyx_um, m, pool_mask=None)``
    Returns, per cell, the m × 3 relative-position vectors to its m
    nearest neighbours in ``points_zyx_um`` (restricted to ``pool_mask``
    if given).  Self is excluded from the neighbour pool.

``score_pair(cz_vecs_i, hcr_vecs_j, n)``
    Wang's score: among the n_cz × n_hcr pairwise vector distances,
    select the n pairs with the smallest distances and return their
    mean.  Lower = better.  This is the raw (unpenalised) score —
    Round 0 in our setting (no displacement penalty, per user).

``apply_penalty(soma_raw, displacement_um, form, R_pen)``
    Multiplies ``soma_raw`` by a displacement-penalty factor.  Used
    only at round ≥ 1; round 0 calls pass ``form='off'``.

Notes
-----
* Implements the n-best-from-(m_cz × m_hcr) cost in O(m_cz · m_hcr +
  k log k) where k = m_cz · m_hcr; this matches Wang's description
  exactly (NOT Hungarian-of-m followed by n-best).
* All distances and centroids are in µm.  Anisotropic scaling is
  not applied here — we assume the input is already in the same µm
  frame on both sides.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


# --------------------------------------------------------------------------- #
# Per-cell vector set
# --------------------------------------------------------------------------- #
def cell_vectors(
    points_zyx_um: np.ndarray,
    *,
    m: int,
    pool_idx: np.ndarray | None = None,
) -> list[np.ndarray]:
    """For each point in ``points_zyx_um``, return its m nearest-neighbour
    displacement vectors (shape (m_i, 3); m_i ≤ m if pool is small).

    If ``pool_idx`` is given (1-D int array, indices into the same array),
    neighbours are drawn only from ``pool_idx``; self is always excluded.
    """
    pts = np.ascontiguousarray(points_zyx_um, dtype=float)
    if pool_idx is None:
        pool = pts
        pool_to_global = np.arange(len(pts))
    else:
        pool_to_global = np.asarray(pool_idx, dtype=int)
        pool = pts[pool_to_global]
    tree = cKDTree(pool)
    k_query = min(m + 1, len(pool))
    if k_query == 0:
        return [np.empty((0, 3)) for _ in pts]
    d, idx = tree.query(pts, k=k_query)
    if k_query == 1:
        idx = idx[:, None]
        d = d[:, None]
    out: list[np.ndarray] = []
    for i in range(len(pts)):
        nbr_pool_rows = idx[i]
        # Drop self-match: when querying its own pool, the first hit is self
        # (distance == 0). Otherwise drop any hit that is the same global
        # index as `i`.
        global_rows = pool_to_global[nbr_pool_rows]
        keep = global_rows != i
        nbr = pool[nbr_pool_rows[keep]][:m]
        out.append(nbr - pts[i])
    return out


# --------------------------------------------------------------------------- #
# Pairwise score
# --------------------------------------------------------------------------- #
def score_pair(cz_vecs_i: np.ndarray, hcr_vecs_j: np.ndarray, *, n: int) -> float:
    """Mean of the n smallest pairwise vector distances between ``cz_vecs_i``
    (shape (m_cz, 3)) and ``hcr_vecs_j`` (shape (m_hcr, 3)).

    Wang's n-best-from-(m_cz × m_hcr).  Returns +inf if either side is
    empty or n > m_cz · m_hcr.
    """
    if cz_vecs_i.size == 0 or hcr_vecs_j.size == 0:
        return float("inf")
    # Pairwise Euclidean distance matrix (m_cz, m_hcr)
    diff = cz_vecs_i[:, None, :] - hcr_vecs_j[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    flat = dist.ravel()
    if flat.size < n:
        return float("inf")
    # Partial sort: smallest n
    idx = np.argpartition(flat, n - 1)[:n]
    return float(flat[idx].mean())


def score_many_to_many(
    cz_vecs: list[np.ndarray],
    hcr_vecs: list[np.ndarray],
    candidate_pairs: list[tuple[int, int]],
    *,
    n: int,
) -> np.ndarray:
    """Vectorised wrapper: score every (cz_row, hcr_row) pair in
    ``candidate_pairs``.  Returns a 1-D array of soma_raw values (µm)
    in the same order.
    """
    out = np.empty(len(candidate_pairs), dtype=float)
    for k, (i, j) in enumerate(candidate_pairs):
        out[k] = score_pair(cz_vecs[i], hcr_vecs[j], n=n)
    return out


# --------------------------------------------------------------------------- #
# Displacement penalty (round ≥ 1 only)
# --------------------------------------------------------------------------- #
def apply_penalty(
    soma_raw: np.ndarray,
    displacement_um: np.ndarray,
    *,
    form: str = "off",
    R_pen: float = 50.0,
) -> np.ndarray:
    """Multiplicative displacement penalty (Wang 2026, generalised).

    ``form`` ∈ {"off", "linear", "exp", "quadratic"}:
      * off       — return soma_raw unchanged
      * linear    — soma_raw * (1 + d / R_pen)
      * exp       — soma_raw * exp(d / R_pen)
      * quadratic — soma_raw * (1 + d / R_pen) ** 2
    """
    form = form.lower()
    if form == "off":
        return soma_raw
    d = np.asarray(displacement_um, dtype=float)
    if form == "linear":
        return soma_raw * (1.0 + d / R_pen)
    if form == "exp":
        return soma_raw * np.exp(d / R_pen)
    if form == "quadratic":
        return soma_raw * (1.0 + d / R_pen) ** 2
    raise ValueError(f"unknown penalty form {form!r}")
