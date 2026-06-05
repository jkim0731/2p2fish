"""Analysis functions for benchmark coregistration data.

Implements:
- per-subject summary statistics
- anisotropic expansion factor (CZ in-vivo -> HCR ex-vivo) via Procrustes on landmarks
- pia surface estimation from ROI centroids with out-of-tissue filtering
- depth-from-surface density profiles
- additional features (nearest-neighbor distances, affine residuals, etc.)

Only landmark/coreg data is used for characterization. It is NOT used to develop
the automatic registration algorithm (see dev protocol).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .benchmark_data_loader import (
    SubjectData,
    cz_px_to_um,
    hcr_px_to_um,
    landmark_pairs_um,
)


# ===========================================================
# Summary table
# ===========================================================
def summary_row(s: SubjectData) -> dict:
    cz = s.cz_centroids
    hcr = s.hcr_centroids
    lm = s.landmarks_qced

    cz_um = cz_px_to_um(cz[["z_px", "y_px", "x_px"]].values, s)
    hcr_um = hcr_px_to_um(hcr[["z_px", "y_px", "x_px"]].values, s)

    def extent_um(a: np.ndarray) -> tuple[float, float, float]:
        return tuple((a[:, i].max() - a[:, i].min()) for i in range(3))  # z, y, x

    cz_ext = extent_um(cz_um) if len(cz_um) else (0, 0, 0)
    hcr_ext = extent_um(hcr_um) if len(hcr_um) else (0, 0, 0)

    n_manual = int(lm["ids"].astype(str).str.startswith("Pt-").sum()) if not lm.empty else 0
    n_active = int(lm["active"].sum()) if not lm.empty else 0

    n_gfp = len(s.hcr_gfp_df) if not s.hcr_gfp_df.empty else 0
    n_hcr = len(hcr)

    coreg_rate = len(s.coreg_table) / len(cz) if len(cz) else float("nan")

    return {
        "subject": s.subject_id,
        "cz_cells": len(cz),
        "hcr_total_cells": n_hcr,
        "hcr_gfp_cells": n_gfp,
        "gfp_feature": s.gfp_feature_name,
        "gfp_fraction": n_gfp / n_hcr if n_hcr else float("nan"),
        "cz_xy_um_per_px": s.cz_xy_um,
        "hcr_xy_um_per_px": s.hcr_xy_um,
        "cz_extent_um": tuple(round(v, 0) for v in cz_ext),   # (z, y, x)
        "hcr_extent_um": tuple(round(v, 0) for v in hcr_ext),
        "matched_cells": len(s.coreg_table),
        "match_rate": coreg_rate,
        "n_manual_landmarks": n_manual,
        "n_active_landmarks": n_active,
        "n_landmark_iters": sum(
            1 for p in s.landmark_iter_files if "qced" in p.name and "iter" in p.name
        ),
    }


# ===========================================================
# Procrustes with anisotropic per-axis scaling
# ===========================================================
@dataclass
class ProcrustesFit:
    R: np.ndarray        # rotation (3x3)
    scales: np.ndarray   # per-axis scale after rotation (3,)
    translation: np.ndarray  # (3,)
    rotation_angle_z_deg: float
    residuals_um: np.ndarray  # (N, 3) hcr_predicted - hcr_actual
    rms_um: float
    mean_um: float
    max_um: float


def fit_anisotropic_similarity(src: np.ndarray, dst: np.ndarray) -> ProcrustesFit:
    """Fit dst ≈ R @ (scales * (src - mean_src)) + mean_dst + t

    Approach: (1) compute the rigid rotation via Procrustes without scaling,
    (2) rotate src, then fit a per-axis diagonal scale + translation to dst.
    This captures the ~180-degree xy rotation plus anisotropic expansion.

    Arrays are shape (N, 3) with columns (x, y, z).
    """
    if len(src) < 4:
        raise ValueError("Need at least 4 landmark pairs")

    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean

    # Unscaled rotation via Procrustes (Kabsch)
    H = src_c.T @ dst_c  # 3x3
    U, _, Vt = np.linalg.svd(H)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt  # src -> dst rotation

    # After rotation, fit per-axis scale + translation to match dst
    src_rot = src_c @ R  # (N, 3)
    # Per-axis: dst_c[:,i] = s_i * src_rot[:,i]  =>  s_i = dot / sum_sq
    scales = np.zeros(3)
    for i in range(3):
        denom = (src_rot[:, i] ** 2).sum()
        scales[i] = (src_rot[:, i] * dst_c[:, i]).sum() / denom if denom > 0 else 0.0

    pred = src_rot * scales + dst_mean
    residuals = pred - dst

    # rotation angle around z (from R entries)
    # R should be approximately diag(cos, cos, 1) with sin entries around z
    angle_z = math.degrees(math.atan2(R[1, 0], R[0, 0]))

    rms = float(np.sqrt((residuals ** 2).sum(axis=1).mean()))
    mean = float(np.linalg.norm(residuals, axis=1).mean())
    mx = float(np.linalg.norm(residuals, axis=1).max())

    return ProcrustesFit(
        R=R,
        scales=scales,
        translation=dst_mean - R @ (src_mean * scales),
        rotation_angle_z_deg=angle_z,
        residuals_um=residuals,
        rms_um=rms,
        mean_um=mean,
        max_um=mx,
    )


def expansion_from_landmarks(s: SubjectData) -> ProcrustesFit | None:
    cz_um, hcr_um = landmark_pairs_um(s, active_only=True)
    if len(cz_um) < 4:
        return None
    return fit_anisotropic_similarity(cz_um, hcr_um)


# ===========================================================
# Pia surface estimation
# ===========================================================
def filter_in_tissue(points_um: np.ndarray, radius_um: float, min_neighbors: int):
    """Remove isolated points (likely out-of-tissue false positives).

    For each point, count neighbors within radius. Keep points with >= min_neighbors.
    Returns a boolean mask over points_um.
    """
    if len(points_um) == 0:
        return np.zeros(0, dtype=bool)
    tree = cKDTree(points_um)
    counts = tree.query_ball_point(points_um, r=radius_um, return_length=True)
    return np.asarray(counts) >= min_neighbors


def estimate_pia_surface(
    points_um: np.ndarray,
    xy_tile_um: float = 100.0,
    top_frac: float = 0.05,
    density_radius_um: float | None = None,
    density_min_neighbors: int = 5,
):
    """Fit a tilted-plane pia surface z = a*x + b*y + c in physical um.

    points_um: (N, 3) array with columns (x, y, z). Positive z is deep (away from pia).
    1. Filter out-of-tissue points by local density.
    2. In each (x, y) tile, take the top-surface cell (smallest z).
    3. Fit plane to the tile-tops via robust IRLS (Huber).
    Returns a dict with a, b, c, tilt_angle_deg, residual_std_um, n_tile_tops.
    """
    pts = np.asarray(points_um, dtype=float)
    if len(pts) < 50:
        return None

    # Auto-scale the density radius from the median 1-NN distance if not specified.
    if density_radius_um is None:
        tree_tmp = cKDTree(pts)
        d, _ = tree_tmp.query(pts, k=2)
        median_nn = float(np.median(d[:, 1]))
        density_radius_um = max(3.0 * median_nn, 30.0)

    mask = filter_in_tissue(pts, density_radius_um, density_min_neighbors)
    pts = pts[mask]
    if len(pts) < 50:
        return None

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    # Bin by (x, y) tile
    x_edges = np.arange(x.min(), x.max() + xy_tile_um, xy_tile_um)
    y_edges = np.arange(y.min(), y.max() + xy_tile_um, xy_tile_um)

    tile_pts = []
    for ix in range(len(x_edges) - 1):
        for iy in range(len(y_edges) - 1):
            m = (
                (x >= x_edges[ix])
                & (x < x_edges[ix + 1])
                & (y >= y_edges[iy])
                & (y < y_edges[iy + 1])
            )
            if m.sum() < 3:
                continue
            zs = z[m]
            # Use small quantile to get top surface (pia side)
            z_top = np.quantile(zs, top_frac)
            tile_pts.append(((x_edges[ix] + x_edges[ix + 1]) / 2,
                              (y_edges[iy] + y_edges[iy + 1]) / 2,
                              z_top))

    tile_pts = np.array(tile_pts)
    if len(tile_pts) < 10:
        return None

    # IRLS Huber
    X = np.column_stack([tile_pts[:, 0], tile_pts[:, 1], np.ones(len(tile_pts))])
    z_obs = tile_pts[:, 2]
    beta = np.linalg.lstsq(X, z_obs, rcond=None)[0]
    for _ in range(5):
        resid = z_obs - X @ beta
        sigma = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        if sigma < 1e-6:
            break
        w = np.where(np.abs(resid) <= 1.345 * sigma, 1.0, 1.345 * sigma / np.abs(resid))
        W = np.diag(w)
        beta = np.linalg.lstsq(W @ X, W @ z_obs, rcond=None)[0]

    a, b, c = beta
    resid = z_obs - X @ beta
    tilt = math.degrees(math.atan(math.hypot(a, b)))
    return {
        "a": float(a),
        "b": float(b),
        "c": float(c),
        "tilt_deg": float(tilt),
        "residual_std_um": float(np.std(resid)),
        "n_tile_tops": int(len(tile_pts)),
        "n_points_used": int(len(pts)),
    }


def _robust_plane_fit(xs_um, ys_um, zs_um, max_iter: int = 8):
    """IRLS-Huber plane fit z = a*x + b*y + c in microns."""
    X = np.column_stack([xs_um, ys_um, np.ones_like(xs_um)])
    beta = np.linalg.lstsq(X, zs_um, rcond=None)[0]
    for _ in range(max_iter):
        resid = zs_um - X @ beta
        sigma = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        if sigma < 1e-6:
            break
        w = np.where(np.abs(resid) <= 1.345 * sigma, 1.0, 1.345 * sigma / np.abs(resid))
        sw = np.sqrt(w)
        beta = np.linalg.lstsq(sw[:, None] * X, sw * zs_um, rcond=None)[0]
    a, b, c = beta
    tilt = math.degrees(math.atan(math.hypot(a, b)))
    return {
        "a": float(a),
        "b": float(b),
        "c": float(c),
        "tilt_deg": float(tilt),
        "residual_std_um": float(np.std(zs_um - X @ beta)),
    }


def estimate_pia_surface_from_image(
    image: np.ndarray,
    z_um: float,
    xy_um: float,
    relative_margin: float = 0.05,
    min_signal_abs: float = 50.0,
    min_thick_um: float = 15.0,
    xy_stride_um: float = 10.0,
    tile_um: float = 150.0,
    mad_cut: float = 3.0,
    bg_top_n_slices: int = 0,
    bg_sigma_mult: float = 0.0,
):
    """Estimate pia surface from the raw image intensity.

    For each (y, x) column, intensity is searched along z for the first voxel
    above a **per-column relative threshold**
    `thr = col_bg + relative_margin * (col_top - col_bg)`, where
    `col_bg = percentile(col, 10)` and `col_top = percentile(col, 95)`.
    The column is accepted only if `col_top - col_bg >= min_signal_abs`
    (so empty/padded columns are skipped) and if `min_thick_um` of z-voxels
    above the threshold are contiguous.

    Parameters
    ----------
    image : ndarray (Z, Y, X)
        Full 3D image, z ascending from the top (pia) to deep cortex.
    z_um, xy_um : float
        Voxel size of the image in microns (depends on zarr level).
    relative_margin : float
        Fraction of the column's dynamic range above `col_bg` at which to
        declare tissue presence.
    min_signal_abs : float
        Minimum `col_top - col_bg` for a column to be used (rejects columns
        with no tissue).
    min_thick_um : float
        Minimum contiguous above-threshold thickness to accept a column.
    xy_stride_um : float
        XY sub-sampling step for the per-column search.
    tile_um, mad_cut : float
        Robust outlier rejection: drop columns whose first-tissue z differs
        from the tile-median by more than `mad_cut * MAD`.
    bg_top_n_slices, bg_sigma_mult : legacy, unused.

    Returns dict with the plane coefficients and fit diagnostics.
    """
    Z, Y, X = image.shape
    if Z < 10:
        return None

    # Sub-sample XY to limit memory
    stride_x = max(1, int(round(xy_stride_um / xy_um)))
    stride_y = max(1, int(round(xy_stride_um / xy_um)))
    sub = image[:, ::stride_y, ::stride_x].astype(np.float32, copy=False)
    Zs, Ys, Xs = sub.shape

    # Per-column background and top percentile along z
    col_bg = np.percentile(sub, 10, axis=0)   # (Ys, Xs)
    col_top = np.percentile(sub, 95, axis=0)  # (Ys, Xs)
    col_thr = col_bg + relative_margin * (col_top - col_bg)
    col_valid = (col_top - col_bg) >= min_signal_abs

    above = sub > col_thr[None, :, :]
    # Drop invalid columns
    above &= col_valid[None, :, :]

    k = max(1, int(round(min_thick_um / z_um)))
    if Zs < k + 1:
        return None
    cum = np.cumsum(above.astype(np.int32), axis=0)
    win = cum[k - 1:].copy()
    win[1:] -= cum[:-k]  # window[z] = count(above) in z..z+k-1
    is_start = win == k
    has_any = is_start.any(axis=0)
    if not has_any.any():
        return None
    first_z = np.argmax(is_start, axis=0)  # (Ysub, Xsub)

    ys_sub, xs_sub = np.where(has_any)
    zs_um = first_z[ys_sub, xs_sub].astype(float) * z_um
    xs_um = xs_sub.astype(float) * stride_x * xy_um
    ys_um = ys_sub.astype(float) * stride_y * xy_um

    # Outlier rejection by tile median (same tile_um as centroid fit)
    if tile_um > 0 and len(zs_um):
        x_bin = (xs_um // tile_um).astype(int)
        y_bin = (ys_um // tile_um).astype(int)
        key = y_bin * 100000 + x_bin
        df = pd.DataFrame({"key": key, "z": zs_um})
        med = df.groupby("key")["z"].transform("median")
        dev = zs_um - med.values
        mad = 1.4826 * np.median(np.abs(dev - np.median(dev))) + 1.0
        mask = np.abs(dev) < mad_cut * mad
        xs_um = xs_um[mask]
        ys_um = ys_um[mask]
        zs_um = zs_um[mask]

    if len(zs_um) < 50:
        return None

    fit = _robust_plane_fit(xs_um, ys_um, zs_um)
    fit["n_columns"] = int(len(zs_um))
    fit["z_um"] = float(z_um)
    fit["xy_um"] = float(xy_um)
    fit["relative_margin"] = float(relative_margin)
    fit["min_signal_abs"] = float(min_signal_abs)
    return fit


def estimate_pia_surface_hybrid(
    hcr_centroids_xyz_um: np.ndarray,
    image_combined: np.ndarray,
    z_um: float,
    xy_um: float,
    *,
    density_min_neighbors: int = 6,
    xy_tile_um: float = 120.0,
    z_window_um: float = 100.0,
    relative_margin: float = 0.05,
    min_signal_abs: float = 0.1,
    min_thick_um: float = 15.0,
    xy_stride_um: float = 10.0,
):
    """Hybrid ROI-prior + image-refinement pia fitter.

    Steps (each step reuses the same primitives used by the stand-alone
    fitters; keeping them here for a self-contained implementation):
      1. Keep only HCR ROIs with ``>= density_min_neighbors`` neighbors
         within 30 um.
      2. Per (x, y) tile of ``xy_tile_um`` size, take the **minimum z** of
         surviving ROIs and IRLS-Huber plane fit.  This is the **coarse
         prior** ``z_prior(x, y)``.
      3. Compute the image-based first-crossing z per XY-subsampled column
         (same convention as :func:`estimate_pia_surface_from_image`).
      4. Restrict the fit to columns whose first-crossing z lies within
         ``z_prior +/- z_window_um`` of the prior.  IRLS-Huber plane fit.

    The hybrid rejects far-away image crossings (e.g., above-pia
    autofluorescence artefacts driving a locally-deep first crossing in
    otherwise-good columns) while keeping the image precision.
    """
    from scipy.ndimage import label as _scipy_label  # unused but keeps parity

    # 1) density filter
    pts = np.asarray(hcr_centroids_xyz_um, dtype=float)
    if len(pts) < 100:
        return None
    tree = cKDTree(pts)
    neighbors = tree.query_ball_point(pts, r=30.0, return_length=True)
    mask = neighbors >= density_min_neighbors
    if mask.sum() < 100:
        return None

    xs = pts[mask, 0]; ys = pts[mask, 1]; zs = pts[mask, 2]

    # 2) min-z per tile + plane fit -> prior
    x_edges = np.arange(xs.min(), xs.max() + xy_tile_um, xy_tile_um)
    y_edges = np.arange(ys.min(), ys.max() + xy_tile_um, xy_tile_um)
    samples = []
    for ix in range(len(x_edges) - 1):
        for iy in range(len(y_edges) - 1):
            m = ((xs >= x_edges[ix]) & (xs < x_edges[ix + 1])
                 & (ys >= y_edges[iy]) & (ys < y_edges[iy + 1]))
            if m.sum() < 3:
                continue
            samples.append((0.5 * (x_edges[ix] + x_edges[ix + 1]),
                            0.5 * (y_edges[iy] + y_edges[iy + 1]),
                            zs[m].min()))
    if len(samples) < 10:
        return None
    arr = np.asarray(samples)
    prior = _robust_plane_fit(arr[:, 0], arr[:, 1], arr[:, 2])

    # 3) image first-crossing map (same logic as the single-image fitter)
    Z, Y, X = image_combined.shape
    stride_x = max(1, int(round(xy_stride_um / xy_um)))
    stride_y = max(1, int(round(xy_stride_um / xy_um)))
    sub = image_combined[:, ::stride_y, ::stride_x].astype(np.float32, copy=False)
    Zs, Ys, Xs = sub.shape

    col_bg = np.percentile(sub, 10, axis=0)
    col_top = np.percentile(sub, 95, axis=0)
    col_thr = col_bg + relative_margin * (col_top - col_bg)
    col_valid = (col_top - col_bg) >= min_signal_abs
    above = (sub > col_thr[None, :, :]) & col_valid[None, :, :]

    k = max(1, int(round(min_thick_um / z_um)))
    if Zs < k + 1:
        return None
    cum = np.cumsum(above.astype(np.int32), axis=0)
    win = cum[k - 1:].copy()
    win[1:] -= cum[:-k]
    is_start = win == k
    has_any = is_start.any(axis=0)
    first_z = np.argmax(is_start, axis=0).astype(float) * z_um

    # 4) window restriction and final plane fit
    ys_grid = np.arange(Ys) * stride_y * xy_um
    xs_grid = np.arange(Xs) * stride_x * xy_um
    Xg, Yg = np.meshgrid(xs_grid, ys_grid)
    z_prior_grid = prior["a"] * Xg + prior["b"] * Yg + prior["c"]
    ok = has_any & (np.abs(first_z - z_prior_grid) <= z_window_um)
    if ok.sum() < 50:
        return None
    yi, xi = np.where(ok)
    fit = _robust_plane_fit(xs_grid[xi], ys_grid[yi], first_z[yi, xi])
    fit["method"] = "hybrid"
    fit["prior_c"] = float(prior["c"])
    fit["prior_tilt"] = float(prior["tilt_deg"])
    fit["z_window_um"] = float(z_window_um)
    fit["n_columns"] = int(ok.sum())
    return fit


def estimate_pia_surface_image_ceiling(
    xyz_um: np.ndarray,
    image_combined: np.ndarray,
    z_um: float,
    xy_um: float,
    *,
    relative_margin: float = 0.005,
    min_signal_abs: float = 0.05,
    safety_offset_um: float = 5.0,
    roi_xy_tile_um: float = 120.0,
    roi_q_frac: float = 0.0,
    roi_density_radius_um: float = 30.0,
    roi_density_min_neighbors: int = 3,
):
    """Session 03 v2 winner — low-margin image fit then lifted so that
    no (x, y) tile has ROIs above the plane.

    User's criterion (session 03 v2): ROI density at and above the pia
    surface must be close to 0. A pure image threshold places the plane
    near the autofluorescence transition but some segmentation ROIs (both
    real buffer junk and true outermost cells) still sit above it. This
    function:

    1. Fits a tilted plane from the image at a low `relative_margin`
       (default 0.5%) to track the outermost intensity edge.
    2. Bins the density-filtered ROIs into (x, y) tiles and takes the
       `roi_q_frac` quantile of z per tile (default 0 = absolute min).
    3. Measures the worst tile-deficit (plane z minus tile-min z where
       positive = plane sits below ROIs there) and lifts the entire
       plane by that deficit plus `safety_offset_um`.

    Net effect: the pia plane sits at or above every ROI cluster by
    at least `safety_offset_um`, preserving the image fit's tilt.
    """
    base = estimate_pia_surface_from_image(
        image_combined, z_um, xy_um,
        min_signal_abs=min_signal_abs,
        relative_margin=relative_margin,
    )
    if base is None:
        return None
    pts = np.asarray(xyz_um, dtype=float)
    if len(pts) < 50:
        return base
    keep = filter_in_tissue(
        pts, radius_um=roi_density_radius_um,
        min_neighbors=roi_density_min_neighbors,
    )
    pts = pts[keep]
    if len(pts) < 50:
        return base
    xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]
    x_bin = (xs // roi_xy_tile_um).astype(int)
    y_bin = (ys // roi_xy_tile_um).astype(int)
    key = y_bin * 100000 + x_bin
    df = pd.DataFrame({"key": key, "x": xs, "y": ys, "z": zs})
    g = df.groupby("key")
    agg = g.agg(x=("x", "median"), y=("y", "median"),
                z=("z", lambda v: float(np.quantile(v, roi_q_frac))),
                n=("z", "size"))
    agg = agg[agg["n"] >= 5]
    if len(agg) < 5:
        return base
    plane_z = base["a"] * agg["x"].values + base["b"] * agg["y"].values + base["c"]
    deficit = plane_z - agg["z"].values
    lift = float(max(0.0, deficit.max()))
    fit = dict(base)
    fit["c"] = float(fit["c"] - lift - safety_offset_um)
    fit["method"] = "image_ceiling"
    fit["lift_um"] = lift
    fit["safety_offset_um"] = float(safety_offset_um)
    fit["n_ceiling_tiles"] = int(len(agg))
    return fit


# Backward-compatible alias for callers that still import the old name.
estimate_pia_surface_image_trough = estimate_pia_surface_image_ceiling


def _per_column_top_in_band(
    vols_sub: list[np.ndarray],
    z_um: float,
    z_low: np.ndarray,
    z_high: np.ndarray,
    relative_margin: float,
    min_signal_abs_frac: float,
    min_thick_um: float,
):
    """Per-column top-of-signal (Y, X) µm map restricted to a per-column
    (z_low, z_high) band. NaN where no signal. Returns None if no columns
    pass.  See N22 notebook for derivation (session 03 v2 addendum).
    """
    Zs = vols_sub[0].shape[0]
    z_axis = np.arange(Zs, dtype=np.float32) * z_um
    band_mask = (z_axis[:, None, None] >= z_low[None]) & \
                (z_axis[:, None, None] <= z_high[None])
    evidence = vols_sub[0].copy()
    for v in vols_sub[1:]:
        np.maximum(evidence, v, out=evidence)
    in_band_ev = np.where(band_mask, evidence, np.nan).astype(np.float32)
    with np.errstate(all="ignore"):
        col_bg = np.nanpercentile(in_band_ev, 10, axis=0)
        col_max = np.nanpercentile(in_band_ev, 99, axis=0)
    dyn = col_max - col_bg
    global_dyn = float(np.nanpercentile(dyn, 95))
    abs_floor = float(min_signal_abs_frac * global_dyn)
    col_valid = np.isfinite(dyn) & (dyn >= abs_floor)
    thr = col_bg + relative_margin * dyn
    above = (evidence > thr[None, :, :]) & band_mask & col_valid[None, :, :]
    k = max(1, int(round(min_thick_um / z_um)))
    if Zs < k + 1:
        return None
    cum = np.cumsum(above.astype(np.int32), axis=0)
    win = cum[k - 1:].copy()
    win[1:] -= cum[:-k]
    is_start = win == k
    has_any = is_start.any(axis=0)
    if not has_any.any():
        return None
    first_z_idx = np.argmax(is_start, axis=0)
    top_z = np.full_like(col_bg, np.nan)
    top_z[has_any] = (first_z_idx[has_any] + (k - 1) // 2) * z_um
    return top_z


def _clamp_to_roi_envelope(
    fit: dict,
    xyz_um: np.ndarray,
    *,
    safety_offset_um: float = 3.0,
    within_tile_q: float = 0.02,
    roi_xy_tile_um: float = 120.0,
    roi_density_radius_um: float = 30.0,
    roi_density_min_neighbors: int = 3,
    min_tile_n: int = 5,
):
    """Lift a polynomial surface fit so no density-filtered ROI tile sits
    above it, minus a safety offset.

    ``within_tile_q`` is the per-tile quantile used as the tile's "ceiling
    z".  Default 0.02 mirrors the strict 2nd-percentile clamp used by
    ``estimate_pia_surface_image_ceiling``.  Set to 0.10 to tolerate
    shallow-outlier cells per tile (N22 fix for subject 790322).
    """
    pts = np.asarray(xyz_um, dtype=float)
    if len(pts) < 50:
        return fit, 0.0
    keep = filter_in_tissue(
        pts, radius_um=roi_density_radius_um,
        min_neighbors=roi_density_min_neighbors,
    )
    pts = pts[keep]
    if len(pts) < 50:
        return fit, 0.0
    xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]
    x_bin = (xs // roi_xy_tile_um).astype(int)
    y_bin = (ys // roi_xy_tile_um).astype(int)
    key = y_bin * 100000 + x_bin
    q_within = float(within_tile_q)
    df = pd.DataFrame({"key": key, "x": xs, "y": ys, "z": zs})
    agg = df.groupby("key").agg(
        x=("x", "median"), y=("y", "median"),
        z=("z", lambda v: float(np.quantile(v, q_within))),
        n=("z", "size"),
    )
    agg = agg[agg["n"] >= min_tile_n]
    if len(agg) == 0:
        return fit, 0.0
    tx, ty, tz = agg["x"].values, agg["y"].values, agg["z"].values
    plane_z = (fit["a"] * tx + fit["b"] * ty + fit["c"]
               + fit.get("p", 0) * tx * tx
               + fit.get("q", 0) * tx * ty
               + fit.get("r", 0) * ty * ty)
    deficit = plane_z - tz
    pos = deficit[deficit > 0]
    lift = float(pos.max()) if len(pos) else 0.0
    out = dict(fit)
    out["c"] = float(out["c"] - lift - safety_offset_um)
    return out, lift


def estimate_pia_surface_quantile_ceiling(
    xyz_um: np.ndarray,
    vols_sub: list[np.ndarray],
    z_um: float,
    xy_um_sub: float,
    xx: np.ndarray,
    yy: np.ndarray,
    anchor: dict,
    *,
    band_um: float = 150.0,
    relative_margin: float = 0.25,
    min_signal_abs_frac: float = 0.20,
    min_thick_um: float = 10.0,
    target_quantile: float = 0.70,
    n_iter: int = 30,
    safety_offset_um: float = 3.0,
    within_tile_q: float = 0.10,
):
    """Session 03 v2 winner — N22 (quantile_ceiling).

    Recipe:
      1. Evaluate ``anchor`` (typically ``estimate_pia_surface_image_ceiling``
         with ``safety_offset_um=0``) on the (xx, yy) grid.
      2. For each (y, x) column find the top-of-signal inside
         anchor_z ± ``band_um`` using a low relative margin and a
         fractional global dynamic-range floor.
      3. Fit a quadratic to those tops by IRLS quantile regression at
         ``target_quantile`` (q = 0.70 keeps 70 % of tops above the
         surface, pulling the fit deeper than a symmetric Huber fit).
      4. Re-clamp to the ROI envelope with a loose within-tile quantile
         (0.10 — the N22 fix) so shallow-outlier ROIs per 120 µm tile
         cannot drag the surface above the tissue.

    Compared with ``estimate_pia_surface_image_ceiling``, this variant
    was developed to fix subject 790322, whose per-column tops tracked
    the pia correctly but whose surface sat ~80 µm too shallow under the
    strict 2nd-percentile clamp. See:
      - sessions/03_surface_estimation_v2/log.md (N22 addendum)
      - notebooks/03_surface_estimation_iteration_v3.ipynb
    """
    if anchor is None or not vols_sub:
        return None
    xs_anchor = np.asarray(xx, dtype=float)
    ys_anchor = np.asarray(yy, dtype=float)
    anchor_zz = (anchor["a"] * xs_anchor + anchor["b"] * ys_anchor
                 + anchor["c"]
                 + anchor.get("p", 0) * xs_anchor * xs_anchor
                 + anchor.get("q", 0) * xs_anchor * ys_anchor
                 + anchor.get("r", 0) * ys_anchor * ys_anchor)
    z_low = anchor_zz - band_um
    z_high = anchor_zz + band_um
    top_z = _per_column_top_in_band(
        vols_sub, z_um, z_low, z_high,
        relative_margin, min_signal_abs_frac, min_thick_um,
    )
    if top_z is None:
        return None
    mask = np.isfinite(top_z)
    if int(mask.sum()) < 50:
        return None
    xs_fit = xs_anchor[mask].ravel()
    ys_fit = ys_anchor[mask].ravel()
    zs_fit = top_z[mask].ravel()

    x0 = float(xs_fit.mean()); y0 = float(ys_fit.mean())
    u = xs_fit - x0; v = ys_fit - y0
    X = np.column_stack([u, v, u * u, u * v, v * v, np.ones_like(u)])

    beta = np.linalg.lstsq(X, zs_fit, rcond=None)[0]
    q = float(target_quantile)
    for _ in range(n_iter):
        r = zs_fit - X @ beta
        abs_r = np.abs(r)
        eps = max(0.01 * float(np.median(abs_r) + 1e-6), 0.5)
        w = np.where(r >= 0, q, 1.0 - q) / np.clip(abs_r, eps, None)
        sw = np.sqrt(w)
        beta_new = np.linalg.lstsq(sw[:, None] * X, sw * zs_fit, rcond=None)[0]
        if np.linalg.norm(beta_new - beta) < 1e-6:
            beta = beta_new
            break
        beta = beta_new
    a_c, b_c, p_c, q_c, r_c, c_c = beta

    A = a_c - 2 * p_c * x0 - q_c * y0
    B = b_c - 2 * r_c * y0 - q_c * x0
    C = (c_c - a_c * x0 - b_c * y0
         + p_c * x0 * x0 + q_c * x0 * y0 + r_c * y0 * y0)
    fit = {
        "a": float(A), "b": float(B), "c": float(C),
        "p": float(p_c), "q": float(q_c), "r": float(r_c),
        "tilt_deg": float(math.degrees(math.atan(math.hypot(A, B)))),
    }
    fit, lift = _clamp_to_roi_envelope(
        fit, xyz_um,
        safety_offset_um=safety_offset_um,
        within_tile_q=within_tile_q,
    )
    fit["method"] = "quantile_ceiling"
    fit["target_quantile"] = float(target_quantile)
    fit["band_um"] = float(band_um)
    fit["within_tile_q"] = float(within_tile_q)
    fit["safety_offset_um"] = float(safety_offset_um)
    fit["clamp_lift_um"] = float(lift)
    fit["n_tops"] = int(mask.sum())
    return fit


def estimate_pia_surface_edge(
    image: np.ndarray,
    z_um: float,
    xy_um: float,
    smooth_sigma_um: float = 8.0,
    noise_k: float = 4.0,
    min_thick_um: float = 8.0,
    xy_stride_um: float = 10.0,
    tile_um: float = 150.0,
    mad_cut: float = 3.0,
    min_col_signal_abs: float = 0.05,
):
    """First-edge pia detector — finds where each column's intensity first
    rises above its own low-z noise floor.

    For each (y, x) column:
      1. Smooth along z (Gaussian sigma = smooth_sigma_um).
      2. Estimate a column background (`col_bg`) and its MAD-std (`col_sd`)
         from the LOWEST 30% of the column values (not the top z-slices — we
         avoid the zero-padding problem that affected the previous fitter).
      3. Declare "above" = smoothed > col_bg + noise_k * col_sd.
      4. Accept the column only if at least `min_col_signal_abs` of the total
         normalised intensity is produced.
      5. Require `min_thick_um` of contiguous above-threshold z.
      6. First such z is the pia candidate for that column.
    A tile-median outlier rejection + IRLS-Huber plane fit follow.

    Intended use is on an intensity-combined volume
    (see `load_hcr_combined`) but also works on single channels.
    """
    from scipy.ndimage import gaussian_filter1d

    Z, Y, X = image.shape
    if Z < 10:
        return None

    stride_x = max(1, int(round(xy_stride_um / xy_um)))
    stride_y = max(1, int(round(xy_stride_um / xy_um)))
    sub = image[:, ::stride_y, ::stride_x].astype(np.float32, copy=False)
    Zs, Ys, Xs = sub.shape

    sigma_px = max(0.5, smooth_sigma_um / z_um)
    smooth = gaussian_filter1d(sub, sigma=sigma_px, axis=0, mode="nearest")

    # Per-column background as 30th percentile (robust against padded top and
    # against tissue dominating the column).
    col_bg = np.percentile(smooth, 30, axis=0)
    col_mad = np.median(np.abs(smooth - col_bg[None, :, :]), axis=0)
    col_sd = 1.4826 * col_mad + 1e-3
    thr = col_bg + noise_k * col_sd  # (Ys, Xs)

    col_total = smooth.sum(axis=0)
    col_valid = col_total >= (min_col_signal_abs * Zs)

    above = (smooth > thr[None, :, :]) & col_valid[None, :, :]

    k = max(1, int(round(min_thick_um / z_um)))
    if Zs < k + 1:
        return None
    cum = np.cumsum(above.astype(np.int32), axis=0)
    win = cum[k - 1:].copy()
    win[1:] -= cum[:-k]
    is_start = win == k
    has_any = is_start.any(axis=0)
    if not has_any.any():
        return None
    first_z = np.argmax(is_start, axis=0)

    ys_sub, xs_sub = np.where(has_any)
    zs_um = first_z[ys_sub, xs_sub].astype(float) * z_um
    xs_um = xs_sub.astype(float) * stride_x * xy_um
    ys_um = ys_sub.astype(float) * stride_y * xy_um

    if tile_um > 0 and len(zs_um):
        x_bin = (xs_um // tile_um).astype(int)
        y_bin = (ys_um // tile_um).astype(int)
        key = y_bin * 100000 + x_bin
        df = pd.DataFrame({"key": key, "z": zs_um})
        med = df.groupby("key")["z"].transform("median")
        dev = zs_um - med.values
        mad = 1.4826 * np.median(np.abs(dev - np.median(dev))) + 1.0
        mask = np.abs(dev) < mad_cut * mad
        xs_um = xs_um[mask]
        ys_um = ys_um[mask]
        zs_um = zs_um[mask]

    if len(zs_um) < 50:
        return None

    fit = _robust_plane_fit(xs_um, ys_um, zs_um)
    fit["n_columns"] = int(len(zs_um))
    fit["z_um"] = float(z_um)
    fit["xy_um"] = float(xy_um)
    fit["method"] = "edge"
    fit["smooth_sigma_um"] = float(smooth_sigma_um)
    fit["noise_k"] = float(noise_k)
    return fit


def _nonparam_surface_z(surface: dict, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Evaluate a Gaussian-kernel smoothed surface at (x, y).
    `surface` must carry `type='gauss'`, `tile_x`, `tile_y`, `tile_z`
    (1-D arrays of anchor points), `sigma_um`, and `c` (global offset)."""
    tx = np.asarray(surface["tile_x"])
    ty = np.asarray(surface["tile_y"])
    tz = np.asarray(surface["tile_z"])
    sigma = float(surface["sigma_um"])
    c_off = float(surface.get("c", 0.0))
    # Compute in chunks to keep memory bounded.
    out = np.empty(len(x), dtype=float)
    chunk = 4096
    inv2s2 = 0.5 / (sigma * sigma)
    for start in range(0, len(x), chunk):
        end = min(start + chunk, len(x))
        dx = x[start:end, None] - tx[None, :]
        dy = y[start:end, None] - ty[None, :]
        d2 = dx * dx + dy * dy
        w = np.exp(-d2 * inv2s2)
        denom = w.sum(axis=1)
        denom = np.where(denom < 1e-12, 1.0, denom)
        out[start:end] = (w * tz[None, :]).sum(axis=1) / denom
    return out + c_off


def _hybrid_surface_z(surface: dict, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Quadratic base + Gaussian-smoothed tile residuals + scalar offset.

    Keys: a, b, p, q, r (quadratic base), tile_x, tile_y, tile_res
    (anchor residuals), sigma_um, c (final offset)."""
    a = surface["a"]; b = surface["b"]
    p = surface.get("p", 0.0); q = surface.get("q", 0.0); r = surface.get("r", 0.0)
    quad = a * x + b * y + p * x * x + q * x * y + r * y * y
    tx = np.asarray(surface["tile_x"])
    ty = np.asarray(surface["tile_y"])
    tr = np.asarray(surface["tile_res"])
    sigma = float(surface["sigma_um"])
    c_off = float(surface.get("c", 0.0))
    out = np.empty(len(x), dtype=float)
    chunk = 4096
    inv2s2 = 0.5 / (sigma * sigma)
    for start in range(0, len(x), chunk):
        end = min(start + chunk, len(x))
        dx = x[start:end, None] - tx[None, :]
        dy = y[start:end, None] - ty[None, :]
        d2 = dx * dx + dy * dy
        w = np.exp(-d2 * inv2s2)
        denom = w.sum(axis=1)
        denom = np.where(denom < 1e-12, 1.0, denom)
        out[start:end] = (w * tr[None, :]).sum(axis=1) / denom
    return quad + out + c_off


def _grid_surface_z(surface: dict, x, y):
    """Bilinear evaluation of a grid-type surface on an irregular (x, y)
    scatter. ``surface`` must contain keys ``grid`` (Y, X), ``xs_um``,
    ``ys_um``."""
    g = surface["grid"]; xs = surface["xs_um"]; ys = surface["ys_um"]
    x = np.asarray(x, dtype=np.float32); y = np.asarray(y, dtype=np.float32)
    ix = np.clip(np.searchsorted(xs, x) - 1, 0, len(xs) - 2)
    iy = np.clip(np.searchsorted(ys, y) - 1, 0, len(ys) - 2)
    fx = np.clip((x - xs[ix]) / (xs[ix + 1] - xs[ix]), 0.0, 1.0)
    fy = np.clip((y - ys[iy]) / (ys[iy + 1] - ys[iy]), 0.0, 1.0)
    return ((1 - fx) * (1 - fy) * g[iy, ix]
            + fx * (1 - fy) * g[iy, ix + 1]
            + (1 - fx) * fy * g[iy + 1, ix]
            + fx * fy * g[iy + 1, ix + 1])


def depth_from_surface(points_um: np.ndarray, surface: dict) -> np.ndarray:
    x, y, z = points_um[:, 0], points_um[:, 1], points_um[:, 2]
    if surface.get("type") == "gauss":
        return z - _nonparam_surface_z(surface, x, y)
    if surface.get("type") == "hybrid":
        return z - _hybrid_surface_z(surface, x, y)
    if surface.get("type") == "grid":
        return z - _grid_surface_z(surface, x, y)
    a, b, c = surface["a"], surface["b"], surface["c"]
    z_surf = a * x + b * y + c
    # Optional quadratic terms: z_surf += p*x^2 + q*x*y + r*y^2
    p = surface.get("p", 0.0)
    q = surface.get("q", 0.0)
    r = surface.get("r", 0.0)
    if p or q or r:
        z_surf = z_surf + p * x * x + q * x * y + r * y * y
    return z - z_surf


# ===========================================================
# Depth density profiles
# ===========================================================
def depth_profile(
    points_um: np.ndarray,
    surface: dict,
    bin_um: float = 20.0,
    depth_range: tuple[float, float] = (-100.0, 1500.0),
):
    """Cells per unit depth (in 20-um bins). Returns (bin_centers, density)."""
    if surface is None or len(points_um) == 0:
        return np.array([]), np.array([])
    depths = depth_from_surface(points_um, surface)
    edges = np.arange(depth_range[0], depth_range[1] + bin_um, bin_um)
    hist, _ = np.histogram(depths, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, hist.astype(float) / bin_um  # cells/um depth-bin


# ===========================================================
# Nearest-neighbor distance
# ===========================================================
def knn_distance_stats(points_um: np.ndarray, k: int = 5):
    if len(points_um) <= k:
        return None
    tree = cKDTree(points_um)
    d, _ = tree.query(points_um, k=k + 1)  # 0-th is self
    d = d[:, 1:]  # drop self
    return {
        "mean_1nn": float(d[:, 0].mean()),
        "median_1nn": float(np.median(d[:, 0])),
        "mean_knn": float(d.mean()),
    }


# ===========================================================
# Image slabs for pia visualization
# ===========================================================
def load_cz_y_slab(s: SubjectData, y_center_px: int | None = None, half_width_px: int = 20):
    """Load a Y-centered slab from the CZ OME-TIFF and return a (Z, X) MIP image.

    Returns (mip, y_center_px, z_per_um, x_per_um).
    """
    import tifffile

    files = list(s.coreg_dir.glob("*reg-dim-swapped.ome.tif"))
    if not files:
        files = list(s.coreg_dir.glob("*zstack.tif"))
    if not files:
        raise FileNotFoundError(f"No CZ OME-TIFF in {s.coreg_dir}")
    path = files[0]

    with tifffile.TiffFile(path) as tf:
        arr = tf.asarray()
    # Squeeze leading singleton axes (e.g. (1,1,Z,Y,X) -> (Z,Y,X))
    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim != 3:
        raise ValueError(f"Unexpected CZ TIF shape: {arr.shape}")

    Z, Y, X = arr.shape
    if y_center_px is None:
        y_center_px = Y // 2
    y0 = max(0, y_center_px - half_width_px)
    y1 = min(Y, y_center_px + half_width_px)
    mip = arr[:, y0:y1, :].max(axis=1)  # (Z, X)
    return mip, y_center_px, s.cz_z_um, s.cz_xy_um


def hcr_level_resolution(s: SubjectData, level: int) -> tuple[float, float]:
    """Return (xy_um, z_um) voxel size at the requested HCR zarr level.

    The fused pyramid preserves Z through level 2 (XY /4 from raw) and then
    halves both Z and XY at every level >= 3.
    """
    xy_um = s.hcr_xy_um * (2 ** (level - 2))
    z_um = s.hcr_z_um * (2 ** max(0, level - 2))
    return xy_um, z_um


def list_hcr_channels(s: SubjectData) -> list[str]:
    """Return the available HCR channel names (e.g. ['405','488','561','594'])."""
    zroot = s.hcr_dir / "image_tile_fusing" / "fused"
    if not zroot.exists():
        return []
    names = []
    for p in zroot.glob("channel_*.zarr"):
        try:
            names.append(p.name.split("_")[1].split(".")[0])
        except Exception:
            continue
    return sorted(names)


def load_hcr_volume(
    s: SubjectData,
    channel: str = "405",
    level: int = 4,
) -> tuple[np.ndarray, float, float]:
    """Load a full HCR pyramid level as (Z, Y, X) into memory.

    Level 4 is a good default for surface estimation: typical shape ~
    (250, 580, 580) at ~4 um XY / 4 um Z — fits in ~130 MB.
    Returns (volume, xy_um, z_um).
    """
    import zarr

    zarr_path = s.hcr_dir / "image_tile_fusing" / "fused" / f"channel_{channel}.zarr"
    if not zarr_path.exists():
        raise FileNotFoundError(f"No HCR fused zarr at {zarr_path}")
    z = zarr.open(str(zarr_path), mode="r")
    vol = np.asarray(z[str(level)][0, 0])
    xy_um, z_um = hcr_level_resolution(s, level)
    return vol, xy_um, z_um


def load_hcr_combined(
    s: SubjectData,
    channels: list[str] | None = None,
    level: int = 4,
) -> tuple[np.ndarray, float, float, list[str]]:
    """Load and combine multiple HCR channels into a normalized tissue-signal
    volume.  Each channel is background-subtracted and scaled by its 99.9th
    percentile before summation.  This amplifies the actual tissue boundary
    over autofluorescence variability and zero-padded edges.

    Returns (combined, xy_um, z_um, channels_used).
    """
    if channels is None:
        channels = list_hcr_channels(s)
    if not channels:
        raise FileNotFoundError("No HCR channels for combined load")

    combined: np.ndarray | None = None
    xy_um = z_um = None
    used: list[str] = []
    for ch in channels:
        try:
            vol, xy_um, z_um = load_hcr_volume(s, channel=ch, level=level)
        except FileNotFoundError:
            continue
        vol = vol.astype(np.float32, copy=False)
        vmin = float(np.percentile(vol, 50))
        vmax = float(np.percentile(vol, 99.9))
        span = max(vmax - vmin, 1.0)
        norm = np.clip((vol - vmin) / span, 0.0, None)
        combined = norm if combined is None else combined + norm
        used.append(ch)
    if combined is None:
        raise FileNotFoundError("Could not load any HCR channel")
    return combined, float(xy_um), float(z_um), used


def estimate_pia_surface_image_autoselect(
    s: SubjectData,
    *,
    level: int = 4,
    qcon_n_threshold: float = 0.9,
    v21_module=None,
):
    """Image-only pia surface with candidate-bank auto-selection.

    Fits the session-03 v2.1 image estimator on a small bank of
    (channel-subset × target_quantile) candidates, scores each in the
    reference combined HCR volume using
    :func:`score_surface_quality`, and returns the highest-Q candidate
    whose normalised contrast ``Qcon_n >= qcon_n_threshold``.

    Rationale (session 03c, iter 3–4): v2.1-default fails on 755252
    because it lands in the autofluorescence-to-pia gap.  The Q-score
    with contrast filter rejects both the gap and the
    deep-in-tissue-body false positives, and on the 6-subject HCR
    benchmark reproduces N22 onset quality (onset ≤ 2.5 µm, above
    frac ≤ 1.13 %) on all subjects without needing centroids.

    Parameters
    ----------
    s : SubjectData
    level : int
        HCR pyramid level.  Default 4 matches the surface-fitting
        resolution used elsewhere in the benchmark.
    qcon_n_threshold : float
        Minimum normalised contrast to keep a candidate.  0.9 from the
        session 03c analysis; in practice real-pia candidates land at
        ≥ 0.98 and tissue-body false positives at ≤ 0.55.
    v21_module : module, optional
        Pre-loaded `03_image_based_surface.py` module.  Loaded lazily
        via importlib if not provided (the module name starts with a
        digit so it cannot be imported directly).

    Returns
    -------
    (surface, info, scores)
        ``surface`` — quadratic dict ``{a, b, c, p, q, r}``
        ``info``    — ``{'selected': name, 'used_channels': [..]}``
        ``scores``  — per-candidate DataFrame with Q/Qabove/Qcon/Qcon_n,
                      ``pass``, ``selected``.
    """
    if v21_module is None:
        import importlib.util, sys
        mod_path = Path(__file__).parent / "03_image_based_surface.py"
        spec = importlib.util.spec_from_file_location("img_surface_v21", str(mod_path))
        v21_module = importlib.util.module_from_spec(spec)
        sys.modules.setdefault("img_surface_v21", v21_module)
        spec.loader.exec_module(v21_module)

    ref_vol, xy_um, z_um, ref_used = load_hcr_combined(s, level=level)
    channels = list_hcr_channels(s)
    # Pick the "middle" short-wavelength tissue channel (514 vs 561) per
    # acquisition — differs across subjects in the benchmark.
    mid = "514" if "514" in channels else ("561" if "561" in channels else None)

    subsets = [
        ("all_tq0.85",    None,                                        0.85),
        ("all_tq0.70",    None,                                        0.70),
        ("all_tq0.95",    None,                                        0.95),
    ]
    if mid is not None:
        subsets += [
            ("no405_tq0.85",  [c for c in ("488", mid, "594") if c in channels], 0.85),
            ("mid488_tq0.85", [c for c in ("488", mid)         if c in channels], 0.85),
        ]
    if "488" in channels:
        subsets.append(("only488_tq0.85", ["488"], 0.85))
    if "594" in channels:
        subsets.append(("only594_tq0.85", ["594"], 0.85))

    candidates: list[tuple[str, object]] = []
    per_name_used: dict[str, list[str]] = {}
    for name, chs, tq in subsets:
        try:
            cvol, cxy, czu, cused = load_hcr_combined(s, channels=chs, level=level)
            res = v21_module.estimate_surface_and_l2_image_based(
                cvol, czu, cxy, target_quantile=tq)
        except Exception:
            res = None
            cused = chs or []
        candidates.append((name, res))
        per_name_used[name] = cused

    sel_name, sel_res, scores = v21_module.auto_select_surface(
        candidates, ref_vol, z_um, xy_um, qcon_n_threshold=qcon_n_threshold)
    info = {
        "selected": sel_name,
        "used_channels": per_name_used.get(sel_name, []),
        "ref_channels": ref_used,
    }
    return (sel_res.surface if sel_res is not None else None), info, scores


def load_hcr_y_slab(
    s: SubjectData,
    channel: str = "488",
    y_center_um: float | None = None,
    half_width_um: float = 20.0,
    level: int = 2,
):
    """Load a Y-centered slab from the HCR zarr and return a (Z, X) MIP image.

    Level defaults to 2 (the same resolution as the cell_centroids.npy).
    Returns (mip, y_center_um, z_per_um, x_per_um).
    """
    import zarr

    zarr_path = s.hcr_dir / "image_tile_fusing" / "fused" / f"channel_{channel}.zarr"
    if not zarr_path.exists():
        raise FileNotFoundError(f"No HCR fused zarr at {zarr_path}")
    z = zarr.open(str(zarr_path), mode="r")
    arr = z[str(level)]  # shape (1, 1, Z, Y, X)
    _, _, Z, Y, X = arr.shape

    xy_um, z_um = hcr_level_resolution(s, level)

    if y_center_um is None:
        y_center_um = (Y * xy_um) / 2
    y_center_px = int(round(y_center_um / xy_um))
    half_px = max(1, int(round(half_width_um / xy_um)))
    y0 = max(0, y_center_px - half_px)
    y1 = min(Y, y_center_px + half_px)
    slab = np.asarray(arr[0, 0, :, y0:y1, :])
    mip = slab.max(axis=1)
    return mip, y_center_um, z_um, xy_um


def plot_pia_overlay(
    ax,
    mip: np.ndarray,
    z_um: float,
    x_um: float,
    surface: dict,
    y_slab_um: float,
    title: str = "",
    cmap: str = "gray",
    vmax_pct: float = 99.5,
):
    """Render a (Z, X) MIP with the fitted pia plane overlaid as a line.

    surface: {'a', 'b', 'c'} for z_surface = a*x + b*y + c in um.
    """
    Zn, Xn = mip.shape
    extent = (0, Xn * x_um, Zn * z_um, 0)  # origin top-left (z increases downward)

    vmin = np.percentile(mip, 1)
    vmax = np.percentile(mip, vmax_pct)
    ax.imshow(
        mip,
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        interpolation="nearest",
    )

    # pia line in (x_um, z_um) space at the slab's y.
    xs = np.linspace(0, Xn * x_um, 200)
    y0 = y_slab_um
    if surface.get("type") == "gauss":
        ys = np.full_like(xs, y0)
        zs = _nonparam_surface_z(surface, xs, ys)
    elif surface.get("type") == "hybrid":
        ys = np.full_like(xs, y0)
        zs = _hybrid_surface_z(surface, xs, ys)
    else:
        a, b, c = surface["a"], surface["b"], surface["c"]
        p = surface.get("p", 0.0)
        q = surface.get("q", 0.0)
        r = surface.get("r", 0.0)
        zs = (a * xs + b * y0 + c
              + p * xs * xs + q * xs * y0 + r * y0 * y0)
    ax.plot(xs, zs, color="red", lw=1.5, label="fitted pia")

    ax.set_xlabel("x (um)")
    ax.set_ylabel("z (um)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)


# ===========================================================
# Full per-subject analysis
# ===========================================================
def analyze_subject(
    s: SubjectData,
    hcr_channel_for_surface: str = "combined",
    hcr_level_for_surface: int = 4,
    hcr_surface_relative_margin: float = 0.05,
    cz_surface_relative_margin: float = 0.05,
    hcr_surface_method: str = "iter07",  # "iter07" | "quantile_ceiling" | "image_ceiling" | "hybrid" | "image" | "centroid"
    cz_surface_method: str = "iter08",  # "iter08" | "image_ceiling" | "image" | "centroid"
) -> dict:
    cz_um = cz_px_to_um(s.cz_centroids[["z_px", "y_px", "x_px"]].values, s)
    hcr_um = hcr_px_to_um(s.hcr_centroids[["z_px", "y_px", "x_px"]].values, s)
    # reorder to (x, y, z) for surface fitting (surface expects (x, y, z))
    cz_xyz = cz_um[:, [2, 1, 0]]
    hcr_xyz = hcr_um[:, [2, 1, 0]]

    # HCR GFP+ subset in xyz physical coords
    gfp_xyz = np.empty((0, 3))
    if not s.hcr_gfp_df.empty and "hcr_id" in s.hcr_gfp_df.columns:
        gfp_ids = set(s.hcr_gfp_df["hcr_id"].astype(int).tolist())
        mask = s.hcr_centroids["hcr_id"].astype(int).isin(gfp_ids).values
        gfp_xyz = hcr_xyz[mask]

    # Centroid-based (kept for comparison/fallback)
    cz_surface_centroid = estimate_pia_surface(cz_xyz)
    hcr_surface_centroid = estimate_pia_surface(hcr_xyz)

    # Image-based surface (primary)
    cz_surface_image = None
    try:
        import tifffile
        cz_tifs = list(s.coreg_dir.glob("*reg-dim-swapped.ome.tif"))
        if not cz_tifs:
            cz_tifs = list(s.coreg_dir.glob("*zstack.tif"))
        if cz_tifs:
            cz_img = tifffile.imread(str(cz_tifs[0]))
            while cz_img.ndim > 3 and cz_img.shape[0] == 1:
                cz_img = cz_img[0]
            cz_surface_image = estimate_pia_surface_from_image(
                cz_img, s.cz_z_um, s.cz_xy_um,
                min_signal_abs=50.0,
                relative_margin=cz_surface_relative_margin,
            )
    except Exception:
        pass

    hcr_surface_image = None
    hcr_surface_hybrid = None
    hcr_surface_image_ceiling = None
    hcr_surface_quantile_ceiling = None
    try:
        if hcr_channel_for_surface == "combined":
            vol, xy_um, z_um, _ = load_hcr_combined(
                s, level=hcr_level_for_surface
            )
            min_signal = 0.1  # sum of normalized channels (each 0..~1+)
        else:
            vol, xy_um, z_um = load_hcr_volume(
                s, channel=hcr_channel_for_surface, level=hcr_level_for_surface
            )
            min_signal = 100.0
        hcr_surface_image = estimate_pia_surface_from_image(
            vol, z_um, xy_um,
            min_signal_abs=min_signal,
            relative_margin=hcr_surface_relative_margin,
        )
        try:
            hcr_surface_hybrid = estimate_pia_surface_hybrid(
                hcr_xyz, vol, z_um, xy_um,
                relative_margin=hcr_surface_relative_margin,
                min_signal_abs=min_signal,
            )
        except Exception:
            hcr_surface_hybrid = None
        try:
            hcr_surface_image_ceiling = estimate_pia_surface_image_ceiling(
                hcr_xyz, vol, z_um, xy_um,
                relative_margin=0.005,
                min_signal_abs=min_signal / 2,
                safety_offset_um=5.0,
            )
        except Exception:
            hcr_surface_image_ceiling = None
        try:
            anchor_for_q = estimate_pia_surface_image_ceiling(
                hcr_xyz, vol, z_um, xy_um,
                relative_margin=0.005,
                min_signal_abs=min_signal / 2,
                safety_offset_um=0.0,
            )
            channels = list_hcr_channels(s)
            per_ch_vols = []
            for ch in channels:
                try:
                    v_ch, _xy_um_ch, _z_um_ch = load_hcr_volume(
                        s, channel=ch, level=hcr_level_for_surface,
                    )
                except FileNotFoundError:
                    continue
                per_ch_vols.append(v_ch.astype(np.float32, copy=False))
            if per_ch_vols and anchor_for_q is not None:
                xy_stride_um = 10.0
                sxy = max(1, int(round(xy_stride_um / xy_um)))
                vols_sub = [v[:, ::sxy, ::sxy] for v in per_ch_vols]
                _, Ys, Xs = vols_sub[0].shape
                xs_um_sub = np.arange(Xs) * sxy * xy_um
                ys_um_sub = np.arange(Ys) * sxy * xy_um
                xx_sub, yy_sub = np.meshgrid(xs_um_sub, ys_um_sub)
                hcr_surface_quantile_ceiling = estimate_pia_surface_quantile_ceiling(
                    hcr_xyz, vols_sub, z_um, sxy * xy_um,
                    xx_sub, yy_sub, anchor_for_q,
                )
        except Exception:
            hcr_surface_quantile_ceiling = None
    except Exception:
        pass

    # CZ surface — default is image_ceiling (low margin + per-tile ROI
    # safety lift), falling back to image-only if the CZ image is missing.
    cz_surface_image_ceiling = None
    if cz_surface_image is not None:
        try:
            cz_tifs = list(s.coreg_dir.glob("*reg-dim-swapped.ome.tif"))
            if not cz_tifs:
                cz_tifs = list(s.coreg_dir.glob("*zstack.tif"))
            if cz_tifs:
                import tifffile
                _cz_img = tifffile.imread(str(cz_tifs[0]))
                while _cz_img.ndim > 3 and _cz_img.shape[0] == 1:
                    _cz_img = _cz_img[0]
                cz_surface_image_ceiling = estimate_pia_surface_image_ceiling(
                    cz_xyz, _cz_img, s.cz_z_um, s.cz_xy_um,
                    relative_margin=0.02,
                    min_signal_abs=50.0,
                    safety_offset_um=3.0,
                    roi_xy_tile_um=80.0,
                    roi_density_min_neighbors=2,
                )
        except Exception:
            cz_surface_image_ceiling = None

    # iter07/08 promoted surfaces — cached JSON fits from
    # `surfaces_iter08.build_main_surface_store()`.  These are the
    # pipeline defaults; other methods remain available via the
    # `*_method` kwargs.  A cache miss falls through to recomputing
    # the fit from the CZ / HCR volumes.
    cz_surface_iter08 = None
    hcr_top_surface_iter07 = None
    hcr_bottom_surface_iter08 = None
    try:
        from .surfaces_iter08 import (
            get_cz_surface_iter08,
            get_hcr_top_surface_iter07,
            get_hcr_bottom_surface_iter08,
        )
        cz_surface_iter08 = get_cz_surface_iter08(s)
        hcr_top_surface_iter07 = get_hcr_top_surface_iter07(
            s, level=hcr_level_for_surface)
        hcr_bottom_surface_iter08 = get_hcr_bottom_surface_iter08(
            s, level=hcr_level_for_surface)
    except Exception:
        pass

    if cz_surface_method == "iter08" and cz_surface_iter08 is not None:
        cz_surface = cz_surface_iter08
    elif cz_surface_method == "image_ceiling" and cz_surface_image_ceiling is not None:
        cz_surface = cz_surface_image_ceiling
    else:
        cz_surface = cz_surface_image if cz_surface_image is not None else cz_surface_centroid

    if hcr_surface_method == "iter07" and hcr_top_surface_iter07 is not None:
        hcr_surface = hcr_top_surface_iter07
    elif (hcr_surface_method == "quantile_ceiling"
            and hcr_surface_quantile_ceiling is not None):
        hcr_surface = hcr_surface_quantile_ceiling
    elif hcr_surface_method == "image_ceiling" and hcr_surface_image_ceiling is not None:
        hcr_surface = hcr_surface_image_ceiling
    elif hcr_surface_method == "hybrid" and hcr_surface_hybrid is not None:
        hcr_surface = hcr_surface_hybrid
    elif hcr_surface_method == "image" and hcr_surface_image is not None:
        hcr_surface = hcr_surface_image
    elif hcr_top_surface_iter07 is not None:
        hcr_surface = hcr_top_surface_iter07
    elif hcr_surface_image_ceiling is not None:
        hcr_surface = hcr_surface_image_ceiling
    else:
        hcr_surface = (hcr_surface_image if hcr_surface_image is not None
                        else hcr_surface_centroid)

    procrustes = expansion_from_landmarks(s)

    cz_nn = knn_distance_stats(cz_xyz)
    hcr_nn = knn_distance_stats(hcr_xyz)
    gfp_nn = knn_distance_stats(gfp_xyz) if len(gfp_xyz) > 10 else None

    return {
        "summary": summary_row(s),
        "cz_surface": cz_surface,
        "hcr_surface": hcr_surface,
        "cz_surface_iter08": cz_surface_iter08,
        "hcr_top_surface_iter07": hcr_top_surface_iter07,
        "hcr_bottom_surface_iter08": hcr_bottom_surface_iter08,
        "cz_surface_image": cz_surface_image,
        "hcr_surface_image": hcr_surface_image,
        "hcr_surface_hybrid": hcr_surface_hybrid,
        "hcr_surface_image_ceiling": hcr_surface_image_ceiling,
        "hcr_surface_quantile_ceiling": hcr_surface_quantile_ceiling,
        "cz_surface_image_ceiling": cz_surface_image_ceiling,
        "cz_surface_centroid": cz_surface_centroid,
        "hcr_surface_centroid": hcr_surface_centroid,
        "procrustes": procrustes,
        "cz_nn": cz_nn,
        "hcr_nn": hcr_nn,
        "gfp_nn": gfp_nn,
        "cz_xyz": cz_xyz,
        "hcr_xyz": hcr_xyz,
        "gfp_xyz": gfp_xyz,
    }
