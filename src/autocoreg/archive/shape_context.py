"""3D log-spherical shape context (Belongie 2002, adapted to 3D).

Per-cell signature: histogram over (r, elevation, azimuth) of relative
positions of *other cells in a filtered pool* within radius ``R_outer``.

* r:           log-spaced bins on [r_inner, R_outer] µm
* elevation θ: equal bins on [0, π] from the local pia-normal
* azimuth φ:   equal circular bins on [-π, π] in the pia-plane

Comparison is χ² with bounded azimuth shift to absorb the residual
in-plane rotation (locked frame leaves θ_residual ≤ ~5°; az_shift_max_bins
defaults to 1).

Pia-normal source is the analytic gradient of the cached
``surfaces_iter08`` polynomial — same as S12 used.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


# --------------------------------------------------------------------------- #
# Geometry: pia-normal and pia-plane basis per cell
# --------------------------------------------------------------------------- #
def pia_normal_xyz(surface: dict, xy_pts: np.ndarray) -> np.ndarray:
    """Pia surface model: z(x, y) = a·x + b·y + c + p·x² + q·xy + r·y².
    Gradient: (∂z/∂x, ∂z/∂y) = (a + 2px + qy, b + qx + 2ry).
    Normal pointing into tissue: (-∂z/∂x, -∂z/∂y, 1) / ||·||.

    Returns (N, 3) unit normals in (x, y, z) order. xy_pts is (N, 2) in (x, y).
    """
    x = np.asarray(xy_pts[:, 0], dtype=float)
    y = np.asarray(xy_pts[:, 1], dtype=float)
    a = float(surface["a"]); b = float(surface["b"])
    p = float(surface["p"]); q = float(surface["q"]); r = float(surface["r"])
    dz_dx = a + 2.0 * p * x + q * y
    dz_dy = b + q * x + 2.0 * r * y
    n = np.column_stack([-dz_dx, -dz_dy, np.ones_like(x)])
    n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
    return n


def pia_basis_xyz(normals_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each pia-normal, build an orthonormal (e_az, e_el, n) basis.

    e_az and e_el span the pia-plane.  Chosen by picking the projection of
    the global +x axis onto the pia plane (well-defined since the pia
    normal is never aligned with +x in our coordinate system).

    Returns (e_az, e_el, n) each of shape (N, 3) in (x, y, z) order.
    """
    n = normals_xyz
    ref = np.tile(np.array([1.0, 0.0, 0.0]), (len(n), 1))
    # Make ref orthogonal to n
    e_az = ref - (np.einsum("ij,ij->i", ref, n))[:, None] * n
    e_az_len = np.linalg.norm(e_az, axis=1, keepdims=True)
    # Guard against degenerate case
    bad = e_az_len[:, 0] < 1e-6
    if bad.any():
        ref2 = np.tile(np.array([0.0, 1.0, 0.0]), (bad.sum(), 1))
        e_az_bad = ref2 - (np.einsum("ij,ij->i", ref2, n[bad]))[:, None] * n[bad]
        e_az[bad] = e_az_bad
        e_az_len = np.linalg.norm(e_az, axis=1, keepdims=True)
    e_az = e_az / (e_az_len + 1e-12)
    e_el = np.cross(n, e_az)  # right-handed
    return e_az, e_el, n


# --------------------------------------------------------------------------- #
# Histogram
# --------------------------------------------------------------------------- #
def build_histograms(
    points_zyx_um: np.ndarray,
    *,
    surface: dict,
    pool_idx: np.ndarray | None = None,
    r_inner: float,
    R_outer: float,
    r_bins: int,
    theta_bins: int,
    phi_bins: int,
) -> np.ndarray:
    """Build per-cell 3D log-spherical histograms.

    Inputs are zyx µm.  ``surface`` is the HCR pia surface dict (used for
    local pia-normal at each cell's (x, y)).

    Returns shape (N, r_bins · theta_bins · phi_bins) histograms.
    L1-normalised per cell.
    """
    pts = np.ascontiguousarray(points_zyx_um, dtype=float)
    if pool_idx is None:
        pool_to_global = np.arange(len(pts))
        pool_pts = pts
    else:
        pool_to_global = np.asarray(pool_idx, dtype=int)
        pool_pts = pts[pool_to_global]

    tree = cKDTree(pool_pts)
    pts_xyz = pts[:, [2, 1, 0]]
    xy = np.column_stack([pts_xyz[:, 0], pts_xyz[:, 1]])
    normals_xyz = pia_normal_xyz(surface, xy)
    e_az, e_el, e_n = pia_basis_xyz(normals_xyz)

    # log-spaced r-edges
    r_edges = np.geomspace(r_inner, R_outer, r_bins + 1)
    theta_edges = np.linspace(0.0, np.pi, theta_bins + 1)
    phi_edges = np.linspace(-np.pi, np.pi, phi_bins + 1)

    n_pts = len(pts)
    D = r_bins * theta_bins * phi_bins
    H = np.zeros((n_pts, D), dtype=float)

    for i in range(n_pts):
        nbr_idx = tree.query_ball_point(pool_pts[i] if pool_idx is None else pts[i],
                                        r=R_outer)
        if not nbr_idx:
            continue
        nbr_global = pool_to_global[nbr_idx]
        # drop self
        keep = nbr_global != i
        nbr_global = nbr_global[keep]
        if nbr_global.size == 0:
            continue
        # displacement in xyz µm
        d_xyz = pts_xyz[nbr_global] - pts_xyz[i]
        r = np.linalg.norm(d_xyz, axis=1)
        keep_r = (r >= r_inner) & (r <= R_outer)
        if not keep_r.any():
            continue
        d_xyz = d_xyz[keep_r]
        r = r[keep_r]
        # project onto pia-plane basis at cell i
        comp_az = d_xyz @ e_az[i]
        comp_el = d_xyz @ e_el[i]
        comp_n = d_xyz @ e_n[i]
        # elevation = angle from pia normal (0 = along normal, π = anti-normal)
        cos_th = np.clip(comp_n / (r + 1e-12), -1.0, 1.0)
        theta = np.arccos(cos_th)
        # azimuth in pia plane
        phi = np.arctan2(comp_el, comp_az)

        # bin indices
        rb = np.searchsorted(r_edges, r, side="right") - 1
        tb = np.searchsorted(theta_edges, theta, side="right") - 1
        pb = np.searchsorted(phi_edges, phi, side="right") - 1
        # Clip exact edges
        rb = np.clip(rb, 0, r_bins - 1)
        tb = np.clip(tb, 0, theta_bins - 1)
        pb = np.clip(pb, 0, phi_bins - 1)
        flat = (rb * theta_bins + tb) * phi_bins + pb
        np.add.at(H[i], flat, 1.0)

    # L1 normalise per cell
    s = H.sum(axis=1, keepdims=True)
    nz = s[:, 0] > 0
    H[nz] = H[nz] / s[nz]
    return H


# --------------------------------------------------------------------------- #
# Comparison with bounded azimuth shift
# --------------------------------------------------------------------------- #
def _chi2(h_a: np.ndarray, h_b: np.ndarray, *, eps: float = 1e-8) -> float:
    num = (h_a - h_b) ** 2
    den = h_a + h_b + eps
    return float((num / den).sum())


def score_pair(
    h_a: np.ndarray,
    h_b: np.ndarray,
    *,
    r_bins: int,
    theta_bins: int,
    phi_bins: int,
    az_shift_max_bins: int = 1,
) -> float:
    """Pairwise χ² between two flat (r·θ·φ) histograms with bounded
    azimuth shift (cyclic in φ).  Returns the minimum χ² over shifts.
    """
    if h_a.sum() == 0 or h_b.sum() == 0:
        return float("inf")
    A = h_a.reshape(r_bins, theta_bins, phi_bins)
    B = h_b.reshape(r_bins, theta_bins, phi_bins)
    best = float("inf")
    for shift in range(-az_shift_max_bins, az_shift_max_bins + 1):
        Bs = np.roll(B, shift, axis=2)
        chi2 = _chi2(A.ravel(), Bs.ravel())
        if chi2 < best:
            best = chi2
    return best


def score_many_to_many(
    H_cz: np.ndarray,
    H_hcr: np.ndarray,
    candidate_pairs: list[tuple[int, int]],
    *,
    r_bins: int,
    theta_bins: int,
    phi_bins: int,
    az_shift_max_bins: int = 1,
) -> np.ndarray:
    """Compute χ² with bounded azimuth shift for ``candidate_pairs``.
    Returns a 1-D array of χ² values, same order as ``candidate_pairs``.

    Fallback path; prefer :func:`score_dense_gt_to_pool` when the pair
    set is *every GT-CZ × every HCR_pool cell* (vectorisable).
    """
    A_full = H_cz.reshape(-1, r_bins, theta_bins, phi_bins)
    B_full = H_hcr.reshape(-1, r_bins, theta_bins, phi_bins)
    shifts = list(range(-az_shift_max_bins, az_shift_max_bins + 1))
    out = np.full(len(candidate_pairs), np.inf, dtype=float)
    for k, (i, j) in enumerate(candidate_pairs):
        A = A_full[i]
        B = B_full[j]
        if A.sum() == 0 or B.sum() == 0:
            continue
        best = np.inf
        for sh in shifts:
            Bs = np.roll(B, sh, axis=2)
            num = (A - Bs) ** 2
            den = A + Bs + 1e-8
            chi2 = float(np.sum(num / den))
            if chi2 < best:
                best = chi2
        out[k] = best
    return out


def score_dense_gt_to_pool(
    H_cz_gt: np.ndarray,
    H_hcr: np.ndarray,
    *,
    r_bins: int,
    theta_bins: int,
    phi_bins: int,
    az_shift_max_bins: int = 1,
    chunk_size: int = 64,
    eps: float = 1e-8,
) -> np.ndarray:
    """Compute the full (n_gt × n_hcr) χ² matrix with bounded azimuth shift.

    Vectorised over HCR cells per CZ chunk.  Returns float32 matrix, with
    +inf where either histogram has zero mass.
    """
    A_full = H_cz_gt.astype(np.float32).reshape(-1, r_bins, theta_bins, phi_bins)
    B_full = H_hcr.astype(np.float32).reshape(-1, r_bins, theta_bins, phi_bins)
    n_gt = A_full.shape[0]
    n_hcr = B_full.shape[0]
    D = r_bins * theta_bins * phi_bins
    shifts = list(range(-az_shift_max_bins, az_shift_max_bins + 1))
    A_sum_zero = (A_full.reshape(n_gt, D).sum(axis=1) == 0)
    B_sum_zero = (B_full.reshape(n_hcr, D).sum(axis=1) == 0)
    out = np.full((n_gt, n_hcr), np.inf, dtype=np.float32)

    # Build all shifted HCR variants once (3 × n_hcr × D float32)
    B_shifted = [np.roll(B_full, sh, axis=3).reshape(n_hcr, D) for sh in shifts]

    for c0 in range(0, n_gt, chunk_size):
        c1 = min(c0 + chunk_size, n_gt)
        A_chunk = A_full[c0:c1].reshape(c1 - c0, D)  # (chunk, D)
        # Best χ² across shifts for this chunk
        best = np.full((c1 - c0, n_hcr), np.inf, dtype=np.float32)
        for Bs in B_shifted:
            # Pairwise (chunk, n_hcr, D) — keep in float32
            diff = A_chunk[:, None, :] - Bs[None, :, :]
            ssum = A_chunk[:, None, :] + Bs[None, :, :] + eps
            chi2 = ((diff * diff) / ssum).sum(axis=-1)
            np.minimum(best, chi2, out=best)
        out[c0:c1] = best

    # Mask invalid entries (zero-mass on either side)
    bad_rows = A_sum_zero[:, None]
    bad_cols = B_sum_zero[None, :]
    out[bad_rows | bad_cols] = np.inf
    return out
