"""R1 (revised) — Scale-free coarse affine with surface-tilt alignment.

Replaces the superseded first-pass `r1_coarse_align.py`. Per the Grand
Plan (07 §R1, 2026-04-17 revision) this version uses no benchmark-derived
expansion priors. Scales are estimated from data, and the output is
*graceful-degradation*:

  * Minimal  (always emitted):  (R, t) from steps 1-3 — surface-tilt
    rotation plus centroid translation.
  * Extended (conditional):     per-axis scale ``S = diag(sx, sy, sz)``
    with a per-component ``scale_known`` mask.  Low-confidence
    components fall back to 1.0 and downstream methods are responsible
    for estimating them.

Only priors used
----------------
1. ``rotation_deg_z = 180`` — structural acquisition geometry (CZ is
   imaged upside-down relative to HCR).  Constant of the rig, not a
   population statistic.
2. Geometric prior ``CZ ⊂ HCR`` in XY — gives the
   ``sxy ≤ L_hcr / L_cz`` search upper bound (feasibility, not a prior
   on expansion).

Everything else — tilt, translation, scale — is derived from the data.

Convention
----------
Same row-vector convention as :class:`benchmark_analysis.ProcrustesFit`::

    hcr_pred = (cz - src_mean) @ R * scales + translation
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from autocoreg.io.hcr_image import depth_from_surface


PRIOR_ROTATION_DEG_Z = 180.0


# ----------------------------------------------------------------------
# Output dataclass
# ----------------------------------------------------------------------
@dataclass
class CoarseAffineV2:
    """Graceful-degradation coarse affine, row-vector convention.

    When ``scale_known[i]`` is False, ``scales[i]`` is the 1.0 placeholder
    (no scale applied on that axis) and the caller is responsible for
    estimating it downstream.
    """

    R: np.ndarray              # (3, 3) rotation, row-vector convention
    scales: np.ndarray         # (3,)  per-axis scale (x, y, z); 1.0 where unknown
    scale_known: np.ndarray    # (3,)  bool — whether each axis's scale is trustworthy
    scale_confidence: np.ndarray  # (3,) peak-to-RMS of the score surface per axis
    translation: np.ndarray    # (3,)  see apply
    src_mean: np.ndarray       # (3,)  CZ centroid (row-vec convention subtracts this)
    rotation_angle_z_deg: float
    coverage_regime: str       # "thinner" | "equal" | "thicker" | "unknown"
    minimal_translation: np.ndarray   # (3,) the always-safe centroid-based t
    diagnostics: dict = field(default_factory=dict)


def apply_coarse_affine(src_xyz_um: np.ndarray, fit: CoarseAffineV2) -> np.ndarray:
    """Apply the coarse affine: CZ µm (x, y, z) → HCR µm (x, y, z)."""
    src_c = np.asarray(src_xyz_um, dtype=float) - fit.src_mean
    return (src_c @ fit.R) * fit.scales + fit.translation


# ----------------------------------------------------------------------
# Rotation helpers (row-vector convention)
# ----------------------------------------------------------------------
def _rotation_about_z_row(deg: float) -> np.ndarray:
    """Row-vector rotation about +z.  ``atan2(R[1, 0], R[0, 0]) == deg``."""
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array(
        [
            [c,  s, 0.0],
            [-s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _rotation_between_row(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-vector rotation sending unit-vector ``a`` onto unit-vector ``b``.

    Returns an R such that ``a @ R ≈ b``.  Uses Rodrigues in
    column-vector form then transposes.
    """
    a = np.asarray(a, dtype=float); a /= np.linalg.norm(a) + 1e-12
    b = np.asarray(b, dtype=float); b /= np.linalg.norm(b) + 1e-12
    cos_angle = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if cos_angle > 1.0 - 1e-9:
        return np.eye(3)
    if cos_angle < -1.0 + 1e-9:
        # 180° rotation — pick any axis orthogonal to a
        orth = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            orth = np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, orth)
        axis /= np.linalg.norm(axis)
        angle = math.pi
    else:
        axis = np.cross(a, b)
        axis /= np.linalg.norm(axis)
        angle = math.acos(cos_angle)
    K = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    # Rodrigues: column-vec R_col s.t. R_col @ a_col = b_col
    R_col = np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)
    # Row-vec form: a @ R_row = b  <=>  R_row = R_col^T
    return R_col.T


# ----------------------------------------------------------------------
# Pia-plane normal from a surface fit
# ----------------------------------------------------------------------
def _surface_z_at(surface: dict, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Evaluate pia z at (x, y) using the existing depth_from_surface infra."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    pts = np.column_stack([x, y, np.zeros_like(x)])
    return -depth_from_surface(pts, surface)


def _plane_normal_from_surface(
    surface: dict,
    xy_points: np.ndarray,
    *,
    n_grid: int = 40,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Fit a global plane to ``surface`` over the XY envelope of ``xy_points``.

    Works for any surface type (plane, quadratic, grid, hybrid) because it
    samples ``surface_z(x, y)`` on a regular grid and does a least-squares
    plane fit.  Returns the unit normal pointing "into the tissue" (same
    half-space as positive ``depth_from_surface``) and the fitted plane
    coefficients ``(a, b, c)`` for ``z = a*x + b*y + c``.
    """
    xy = np.asarray(xy_points, dtype=float)
    x_lo, x_hi = float(xy[:, 0].min()), float(xy[:, 0].max())
    y_lo, y_hi = float(xy[:, 1].min()), float(xy[:, 1].max())
    xs = np.linspace(x_lo, x_hi, n_grid)
    ys = np.linspace(y_lo, y_hi, n_grid)
    X, Y = np.meshgrid(xs, ys)
    Z = _surface_z_at(surface, X.ravel(), Y.ravel())
    A = np.column_stack([X.ravel(), Y.ravel(), np.ones(X.size)])
    a, b, c = np.linalg.lstsq(A, Z, rcond=None)[0]
    # Surface: z - a*x - b*y - c = 0.  Gradient = (-a, -b, 1).
    # depth_from_surface returns z - z_surf, positive into tissue, so
    # the "into tissue" half-space has normal (-a, -b, 1).
    n = np.array([-float(a), -float(b), 1.0])
    n /= np.linalg.norm(n) + 1e-12
    return n, (float(a), float(b), float(c))


# ----------------------------------------------------------------------
# 1D profile tools for Z-scale + Z-offset search (step 5)
# ----------------------------------------------------------------------
def _density_profile_1d(
    depths: np.ndarray, edges: np.ndarray, smooth_sigma_bins: float = 1.0,
) -> np.ndarray:
    h, _ = np.histogram(depths, bins=edges)
    h = h.astype(float)
    if smooth_sigma_bins > 0:
        h = gaussian_filter(h, sigma=smooth_sigma_bins)
    return h


def _partial_overlap_ncc_1d(
    hcr_profile: np.ndarray,
    cz_profile: np.ndarray,
    min_overlap_frac: float,
) -> tuple[float, int, np.ndarray]:
    """Slide cz across hcr, compute Pearson NCC on the overlap region.

    Returns ``(best_score, best_shift_bins, score_per_shift)`` where
    ``best_shift_bins`` is the integer bin shift s.t. the cz template
    placed at that shift best aligns with the hcr signal.  ``shift == 0``
    means cz[0] lines up with hcr[0].  Positive shift → cz moves to
    larger depth.

    The minimum-overlap floor is ``min_overlap_frac × max(n_s, n_t)`` —
    this is a deliberate deviation from the Grand Plan wording (which
    says "25 % of the shorter profile"): keying the floor off the
    *shorter* profile lets a rescaled-tiny CZ template match any 2-3-bin
    window of HCR, since NCC is scale-invariant.  Keying off the longer
    profile (typically HCR) requires the rescaled CZ to span a
    non-trivial fraction of HCR's cortical depth before its score
    counts, which matches the intent of "prevent trivial single-peak
    alignments".
    """
    n_s = len(hcr_profile)
    n_t = len(cz_profile)
    min_overlap = max(1, int(min_overlap_frac * max(n_s, n_t)))
    shift_lo = -(n_t - min_overlap)
    shift_hi = n_s - min_overlap
    shifts = np.arange(shift_lo, shift_hi + 1)
    scores = np.full(len(shifts), np.nan)
    for k, s in enumerate(shifts):
        sig_lo = max(0, s)
        sig_hi = min(n_s, s + n_t)
        tem_lo = max(0, -s)
        tem_hi = tem_lo + (sig_hi - sig_lo)
        if (sig_hi - sig_lo) < min_overlap:
            continue
        a = hcr_profile[sig_lo:sig_hi]
        b = cz_profile[tem_lo:tem_hi]
        a_zm = a - a.mean()
        b_zm = b - b.mean()
        denom = math.sqrt(float((a_zm ** 2).sum()) * float((b_zm ** 2).sum()))
        if denom <= 0:
            continue
        scores[k] = float((a_zm * b_zm).sum() / denom)
    valid = np.isfinite(scores)
    if not valid.any():
        return float("nan"), 0, scores
    best_k = int(np.nanargmax(scores))
    return float(scores[best_k]), int(shifts[best_k]), scores


def _z_scale_offset_search(
    cz_depth_native: np.ndarray,
    hcr_depth: np.ndarray,
    *,
    sz_grid: np.ndarray,
    depth_bin_um: float,
    min_overlap_frac: float,
    depth_max_um: float,
    smooth_sigma_bins: float = 1.0,
) -> dict:
    """Grid search over (sz, tz).  For each sz, rescale CZ depth and slide
    it over the HCR depth profile on a shared 1D grid.  Score with
    partial-overlap Pearson NCC.  Return the best (sz, tz) plus the
    score surface for confidence estimation.
    """
    edges = np.arange(0.0, depth_max_um + depth_bin_um, depth_bin_um)
    # HCR profile on the full grid (same for every sz)
    hcr_prof = _density_profile_1d(hcr_depth, edges, smooth_sigma_bins)
    best_score = -np.inf
    best_sz = None
    best_shift_bins = 0
    per_sz_max_score = np.full(len(sz_grid), np.nan)
    shift_table = np.zeros(len(sz_grid), dtype=int)
    score_curves = [None] * len(sz_grid)
    for i, sz in enumerate(sz_grid):
        cz_scaled = cz_depth_native * float(sz)
        # Clip to non-negative depth (cz_depth can be slightly negative near pia)
        if cz_scaled.size == 0:
            continue
        # Build CZ template on a grid covering its own extent, starting at 0
        d_lo = max(0.0, float(cz_scaled.min()))
        d_hi = float(cz_scaled.max())
        if d_hi - d_lo < depth_bin_um:
            continue
        tem_edges = np.arange(
            d_lo, d_hi + depth_bin_um, depth_bin_um
        )
        cz_prof = _density_profile_1d(cz_scaled, tem_edges, smooth_sigma_bins)
        if cz_prof.sum() <= 0:
            continue
        # Restrict HCR profile to a matching grid (using the same bin start)
        # We just do shift-based xcorr in index space; convert shifts → µm later.
        score, shift, _ = _partial_overlap_ncc_1d(
            hcr_prof, cz_prof, min_overlap_frac,
        )
        per_sz_max_score[i] = score
        shift_table[i] = shift
        # Convert bin-shift to µm tz: if cz template starts at bin d_lo and
        # we place it at shift `s` on the HCR grid (which starts at 0),
        # the µm position where cz depth=d_lo lands is `s * depth_bin_um`.
        # So the "tz" that would turn cz_depth_native * sz into HCR depth is:
        #   hcr_depth ≈ sz * cz_depth + tz
        #   => position in µm of cz_depth=0 after scaling = 0 * sz + tz.
        # Our template profile is over cz_scaled starting at d_lo, so the
        # absolute depth (in HCR units) of the template's origin bin is:
        #   hcr_abs_depth_at_template_origin = d_lo + tz
        # And bin `shift` in the HCR grid (which starts at 0) equals
        # `shift * depth_bin_um` µm. So d_lo + tz = shift * depth_bin_um
        # => tz = shift * depth_bin_um - d_lo.
        # (We recompute this at the very end for the best sz only.)
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_sz = float(sz)
            best_shift_bins = shift
            score_curves[i] = None  # not kept to save memory
    if best_sz is None:
        return {
            "sz": float("nan"),
            "tz_um": float("nan"),
            "best_score": float("nan"),
            "score_curve": per_sz_max_score,
            "sz_grid": sz_grid,
            "ok": False,
        }
    # Reconstruct tz_um for the winning sz
    cz_scaled = cz_depth_native * best_sz
    d_lo = max(0.0, float(cz_scaled.min()))
    tz_um = float(best_shift_bins * depth_bin_um - d_lo)
    return {
        "sz": float(best_sz),
        "tz_um": tz_um,
        "best_score": float(best_score),
        "score_curve": per_sz_max_score,
        "sz_grid": sz_grid,
        "shift_table": shift_table,
        "ok": True,
    }


# ----------------------------------------------------------------------
# Step 6: XY-scale + XY-translation joint search
# ----------------------------------------------------------------------
def _density_map_2d(
    xy_um: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    sigma_px: float,
) -> np.ndarray:
    h, _, _ = np.histogram2d(xy_um[:, 0], xy_um[:, 1], bins=[x_edges, y_edges])
    if sigma_px > 0:
        h = gaussian_filter(h.astype(float), sigma=sigma_px)
    return h


def _fast_ncc_2d(
    image: np.ndarray, template: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Normalized cross-correlation of ``template`` over ``image``.

    Uses Lewis 1995's integral-image approach for O(N log N) per offset.
    Returns ``(ncc_map, best_ij)`` where ``ncc_map`` has shape
    ``image.shape - template.shape + 1`` (valid-mode), and
    ``best_ij = argmax(ncc_map)``.
    """
    # Work with floats
    I = image.astype(float)
    T = template.astype(float)
    H, W = I.shape
    h, w = T.shape
    if H < h or W < w:
        return np.array([[-np.inf]]), (0, 0)
    # Template normalization
    T_zm = T - T.mean()
    T_norm = math.sqrt(float((T_zm ** 2).sum()))
    if T_norm <= 0:
        return np.zeros((H - h + 1, W - w + 1)), (0, 0)
    # Integral images for windowed sum and sum-of-squares of I
    I_sum = np.zeros((H + 1, W + 1))
    I_sum[1:, 1:] = np.cumsum(np.cumsum(I, axis=0), axis=1)
    I2_sum = np.zeros((H + 1, W + 1))
    I2_sum[1:, 1:] = np.cumsum(np.cumsum(I * I, axis=0), axis=1)

    def win_sum(A_sum: np.ndarray) -> np.ndarray:
        return (A_sum[h:, w:] - A_sum[:-h, w:]
                - A_sum[h:, :-w] + A_sum[:-h, :-w])
    wsum = win_sum(I_sum)
    w2sum = win_sum(I2_sum)
    wmean = wsum / (h * w)
    # Variance, clipped
    wvar = np.clip(w2sum - (h * w) * wmean * wmean, 0.0, None)
    wstd = np.sqrt(wvar)
    # Numerator: xcorr(I, T_zm), valid mode
    # Use scipy fftconvolve (correlate = convolve with reversed kernel)
    from scipy.signal import fftconvolve
    num = fftconvolve(I, T_zm[::-1, ::-1], mode="valid")
    # Subtract window-mean * T_zm.sum() = 0 (since T_zm sums to 0), so num is
    # already (I - I_win_mean) · T_zm summed over the window.
    denom = wstd * T_norm
    ncc = np.zeros_like(num)
    good = denom > 1e-9
    ncc[good] = num[good] / denom[good]
    ij = np.unravel_index(int(np.argmax(ncc)), ncc.shape)
    return ncc, (int(ij[0]), int(ij[1]))


def _xy_scale_translation_search(
    cz_rot_xy: np.ndarray,
    hcr_xy: np.ndarray,
    *,
    sxy_grid: np.ndarray,
    xy_bin_um: float,
    xy_sigma_um: float,
    margin_um: float,
) -> dict:
    """Joint (sxy, tx, ty) grid search via 2D NCC.

    For each sxy, scale ``cz_rot_xy`` and rasterise its density map; then
    NCC against the HCR density map.  Track the best (sxy, tx, ty) and
    the per-sxy max-NCC curve (used for confidence).
    """
    # HCR grid — fixed across sxy
    hx_lo = float(hcr_xy[:, 0].min()) - margin_um
    hx_hi = float(hcr_xy[:, 0].max()) + margin_um
    hy_lo = float(hcr_xy[:, 1].min()) - margin_um
    hy_hi = float(hcr_xy[:, 1].max()) + margin_um
    hcr_x_edges = np.arange(hx_lo, hx_hi + xy_bin_um, xy_bin_um)
    hcr_y_edges = np.arange(hy_lo, hy_hi + xy_bin_um, xy_bin_um)
    sigma_px = xy_sigma_um / xy_bin_um
    hcr_map = _density_map_2d(hcr_xy, hcr_x_edges, hcr_y_edges, sigma_px)

    per_sxy_max = np.full(len(sxy_grid), np.nan)
    per_sxy_best_tx = np.full(len(sxy_grid), np.nan)
    per_sxy_best_ty = np.full(len(sxy_grid), np.nan)

    best_score = -np.inf
    best_sxy = None
    best_tx = None
    best_ty = None

    for i, sxy in enumerate(sxy_grid):
        cz_scaled = cz_rot_xy * float(sxy)
        cx_lo = float(cz_scaled[:, 0].min()) - margin_um
        cx_hi = float(cz_scaled[:, 0].max()) + margin_um
        cy_lo = float(cz_scaled[:, 1].min()) - margin_um
        cy_hi = float(cz_scaled[:, 1].max()) + margin_um
        cz_x_edges = np.arange(cx_lo, cx_hi + xy_bin_um, xy_bin_um)
        cz_y_edges = np.arange(cy_lo, cy_hi + xy_bin_um, xy_bin_um)
        cz_map = _density_map_2d(cz_scaled, cz_x_edges, cz_y_edges, sigma_px)

        # Ensure template fits inside image
        if cz_map.shape[0] >= hcr_map.shape[0] or cz_map.shape[1] >= hcr_map.shape[1]:
            continue
        ncc_map, (i_best, j_best) = _fast_ncc_2d(hcr_map, cz_map)
        score = float(ncc_map[i_best, j_best])
        per_sxy_max[i] = score
        # Convert (i_best, j_best) — top-left of the window — to µm origin:
        # The window covers HCR rows [i_best:i_best+h] (x), cols [j_best:j_best+w] (y)
        # The window's left-edge in HCR µm is:
        #   x_left = hcr_x_edges[i_best]; y_left = hcr_y_edges[j_best]
        # The CZ template's left edge in µm (before translation) is cx_lo.
        # So the implied translation (tx, ty) that aligns CZ->HCR is:
        #   tx = hcr_x_edges[i_best] - cx_lo
        #   ty = hcr_y_edges[j_best] - cy_lo
        tx_i = float(hcr_x_edges[i_best] - cx_lo)
        ty_i = float(hcr_y_edges[j_best] - cy_lo)
        per_sxy_best_tx[i] = tx_i
        per_sxy_best_ty[i] = ty_i

        if np.isfinite(score) and score > best_score:
            best_score = score
            best_sxy = float(sxy)
            best_tx = tx_i
            best_ty = ty_i

    if best_sxy is None:
        return {
            "sxy": float("nan"),
            "tx_um": float("nan"),
            "ty_um": float("nan"),
            "best_score": float("nan"),
            "score_curve": per_sxy_max,
            "sxy_grid": sxy_grid,
            "ok": False,
        }
    return {
        "sxy": float(best_sxy),
        "tx_um": float(best_tx),
        "ty_um": float(best_ty),
        "best_score": float(best_score),
        "score_curve": per_sxy_max,
        "sxy_grid": sxy_grid,
        "per_sxy_best_tx": per_sxy_best_tx,
        "per_sxy_best_ty": per_sxy_best_ty,
        "ok": True,
    }


# ----------------------------------------------------------------------
# Scale confidence
# ----------------------------------------------------------------------
def _peak_to_rms(curve: np.ndarray) -> float:
    """Peak prominence in robust-z units: ``(peak - median) / (1.4826·MAD)``.

    Named ``_peak_to_rms`` to match the Grand Plan's §R1 step-8 wording
    ("peak-to-RMS ratio"), but computed as a robust-z prominence so
    the metric stays meaningful on score curves with non-zero baselines
    (NCC curves typically sit at 0.6–0.8).  Raw peak/RMS on such curves
    is always near 1.0 and can't discriminate peakedness.

    Caveat validated on the 6 benchmark subjects (session 05 log): this
    metric detects "there is a peak" but NOT "the peak is at the
    correct scale".  Density-map NCC on sparse GFP+ cell clouds can
    produce a prominent but physically incorrect peak when the CZ
    template is small enough to match any local density bump in HCR.
    The default decision threshold is therefore set permissively (6.0)
    to avoid false positives — so on current benchmark data all subjects
    route to the minimal-output fallback, which is the intended
    graceful degradation.  When richer descriptors (R2 constellations,
    A-series matchers) land, the scale search can be re-enabled or the
    confidence can be cross-checked against those methods.

    NaN-safe.
    """
    c = np.asarray(curve, dtype=float)
    c = c[np.isfinite(c)]
    if c.size < 3:
        return float("nan")
    peak = float(c.max())
    median = float(np.median(c))
    mad = float(np.median(np.abs(c - median)))
    if mad <= 0 or not np.isfinite(mad):
        return float("nan")
    return (peak - median) / (1.4826 * mad)


# ----------------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------------
def coarse_align_revised(
    cz_xyz_um: np.ndarray,
    hcr_gfp_xyz_um: np.ndarray,
    cz_surface: dict,
    hcr_surface: dict,
    *,
    rotation_deg_z: float = PRIOR_ROTATION_DEG_Z,
    sz_min: float = 0.5,
    sz_max: float = 6.0,
    sz_step: float = 0.02,
    sxy_min: float = 0.5,
    sxy_step: float = 0.05,
    depth_bin_um: float = 20.0,
    z_smooth_sigma_bins: float = 1.0,
    min_overlap_frac: float = 0.25,
    xy_bin_um: float = 20.0,
    xy_sigma_um: float = 30.0,
    xy_margin_um: float = 100.0,
    scale_confidence_threshold: float = 6.0,
    aniso_refine: bool = True,
    aniso_window: float = 0.10,
    aniso_step: float = 0.02,
) -> CoarseAffineV2:
    """Revised R1 coarse alignment — no benchmark-derived priors.

    Returns a :class:`CoarseAffineV2`.  Minimal output ``(R, t)`` with
    ``scales=[1,1,1]`` and ``scale_known=[F,F,F]`` is the graceful-
    degradation fallback; when a scale component's confidence
    (peak-to-RMS of its score curve) clears ``scale_confidence_threshold``
    the extended output replaces that component.
    """
    cz = np.asarray(cz_xyz_um, dtype=float)
    hcr = np.asarray(hcr_gfp_xyz_um, dtype=float)
    if len(cz) < 10 or len(hcr) < 10:
        raise ValueError(
            f"Need ≥10 CZ and ≥10 HCR GFP+ cells; got {len(cz)} / {len(hcr)}."
        )
    if cz_surface is None or hcr_surface is None:
        raise ValueError("Both CZ and HCR pia surfaces are required.")

    # ---- Step 1: plane normal fit on each surface ----
    n_cz, cz_plane_abc = _plane_normal_from_surface(cz_surface, cz[:, :2])
    n_hcr, hcr_plane_abc = _plane_normal_from_surface(hcr_surface, hcr[:, :2])

    # ---- Step 2: tilt-aligned rotation ----
    R_180 = _rotation_about_z_row(rotation_deg_z)
    n_cz_rot_row = n_cz @ R_180
    R_tilt = _rotation_between_row(n_cz_rot_row, n_hcr)
    # Total rotation: first apply R_180, then R_tilt.  Row-vec composition:
    # v @ R_total = ((v @ R_180) @ R_tilt)  ⟹  R_total = R_180 @ R_tilt.
    R = R_180 @ R_tilt

    # ---- Step 3: centroid translation (minimal output) ----
    # "Translate so the rotated CZ centroid maps to the HCR centroid in all
    # three axes" — Grand Plan §R1 step 3.  With scales set to 1 (the
    # graceful-degradation fallback), this is just t = hcr_mean: the CZ
    # cloud centroid maps to the HCR GFP+ cloud centroid.
    #
    # The plan's parenthetical "Z from pia-alignment, XY from the roughly-
    # at-center prior" describes *why* centroid-matching is sensible in
    # each axis — because pia planes are aligned in Step 2 (their
    # centroids sit at matched z) and because CZ ⊂ HCR in XY.  A
    # literal pia-match computation (placing CZ pia onto HCR pia under
    # sz = 1) under-estimates tz by ~(sz_true − 1) × cz_depth_mean ≈
    # 200–400 µm on these subjects, because the CZ cloud without Z-scale
    # covers only 400 µm of HCR's ~1200 µm cortical range.  Centroid-
    # match is the lowest-assumption fallback.
    cz_mean = cz.mean(axis=0)
    hcr_mean = hcr.mean(axis=0)
    rot_angle_z_deg = math.degrees(math.atan2(R[1, 0], R[0, 0]))

    # ---- Step 4: depths in each native frame ----
    cz_depth = depth_from_surface(cz, cz_surface)
    hcr_depth = depth_from_surface(hcr, hcr_surface)

    # Build CZ rotated coordinates (mean-centered) — shared by steps 5/6
    cz_rot = (cz - cz_mean) @ R  # (N, 3) in HCR-orientation-compatible units

    t_minimal = hcr_mean.copy()
    # Diagnostic: residual of a pia-aligned (sz=1) fallback, for reference.
    xy_new_min = cz_rot[:, :2] + hcr_mean[:2]
    pia_at_new_min = _surface_z_at(hcr_surface, xy_new_min[:, 0], xy_new_min[:, 1])
    tz_pia_aligned = float(np.median(pia_at_new_min + cz_depth - cz_rot[:, 2]))
    tz_std_min = float(np.std(pia_at_new_min + cz_depth - cz_rot[:, 2]))

    # ---- Step 5: Z-scale + Z-offset search ----
    # Depth max: a bit beyond the greater of hcr_depth max and sz_max * cz_depth max
    depth_max_um = max(float(np.nanmax(hcr_depth)),
                        sz_max * float(np.nanmax(cz_depth))) + 200.0
    sz_grid = np.arange(sz_min, sz_max + sz_step, sz_step)
    z_res = _z_scale_offset_search(
        cz_depth, hcr_depth,
        sz_grid=sz_grid,
        depth_bin_um=depth_bin_um,
        min_overlap_frac=min_overlap_frac,
        depth_max_um=depth_max_um,
        smooth_sigma_bins=z_smooth_sigma_bins,
    )
    sz_conf = _peak_to_rms(z_res["score_curve"]) if z_res["ok"] else float("nan")
    sz_known = bool(np.isfinite(sz_conf) and sz_conf >= scale_confidence_threshold)

    # ---- Step 6: XY-scale + XY-translation joint search ----
    # Build sxy grid from the CZ⊂HCR feasibility bound
    L_cz = float(max(cz_rot[:, 0].max() - cz_rot[:, 0].min(),
                     cz_rot[:, 1].max() - cz_rot[:, 1].min()))
    L_hcr = float(max(hcr[:, 0].max() - hcr[:, 0].min(),
                      hcr[:, 1].max() - hcr[:, 1].min()))
    sxy_upper = max(sxy_min + sxy_step, L_hcr / max(L_cz, 1e-6))
    sxy_grid = np.arange(sxy_min, sxy_upper + sxy_step, sxy_step)
    xy_res = _xy_scale_translation_search(
        cz_rot[:, :2], hcr[:, :2],
        sxy_grid=sxy_grid,
        xy_bin_um=xy_bin_um,
        xy_sigma_um=xy_sigma_um,
        margin_um=xy_margin_um,
    )
    sxy_conf = _peak_to_rms(xy_res["score_curve"]) if xy_res["ok"] else float("nan")
    sxy_known = bool(np.isfinite(sxy_conf) and sxy_conf >= scale_confidence_threshold)

    # ---- Step 7: optional anisotropic refinement (sx, sy) around best sxy ----
    sx_final = sy_final = 1.0
    aniso_done = False
    if sxy_known and aniso_refine and xy_res["ok"]:
        sxy_best = xy_res["sxy"]
        ranges = np.arange(
            max(sxy_min, sxy_best * (1.0 - aniso_window)),
            min(sxy_upper, sxy_best * (1.0 + aniso_window)) + aniso_step,
            aniso_step,
        )
        best_score = -np.inf
        best_sx = best_sy = sxy_best
        best_tx_a = xy_res["tx_um"]
        best_ty_a = xy_res["ty_um"]
        for sx in ranges:
            for sy in ranges:
                cz_scaled = cz_rot[:, :2] * np.array([float(sx), float(sy)])
                cx_lo = float(cz_scaled[:, 0].min()) - xy_margin_um
                cy_lo = float(cz_scaled[:, 1].min()) - xy_margin_um
                cx_hi = float(cz_scaled[:, 0].max()) + xy_margin_um
                cy_hi = float(cz_scaled[:, 1].max()) + xy_margin_um
                cz_x_edges = np.arange(cx_lo, cx_hi + xy_bin_um, xy_bin_um)
                cz_y_edges = np.arange(cy_lo, cy_hi + xy_bin_um, xy_bin_um)
                cz_map = _density_map_2d(
                    cz_scaled, cz_x_edges, cz_y_edges, xy_sigma_um / xy_bin_um,
                )
                hx_lo = float(hcr[:, 0].min()) - xy_margin_um
                hx_hi = float(hcr[:, 0].max()) + xy_margin_um
                hy_lo = float(hcr[:, 1].min()) - xy_margin_um
                hy_hi = float(hcr[:, 1].max()) + xy_margin_um
                hcr_x_edges = np.arange(hx_lo, hx_hi + xy_bin_um, xy_bin_um)
                hcr_y_edges = np.arange(hy_lo, hy_hi + xy_bin_um, xy_bin_um)
                hcr_map = _density_map_2d(
                    hcr[:, :2], hcr_x_edges, hcr_y_edges, xy_sigma_um / xy_bin_um,
                )
                if (cz_map.shape[0] >= hcr_map.shape[0]
                        or cz_map.shape[1] >= hcr_map.shape[1]):
                    continue
                ncc_map, (i_b, j_b) = _fast_ncc_2d(hcr_map, cz_map)
                score = float(ncc_map[i_b, j_b])
                if score > best_score:
                    best_score = score
                    best_sx = float(sx)
                    best_sy = float(sy)
                    best_tx_a = float(hcr_x_edges[i_b] - cx_lo)
                    best_ty_a = float(hcr_y_edges[j_b] - cy_lo)
        sx_final, sy_final = best_sx, best_sy
        # Overwrite tx/ty in xy_res with the refined values
        xy_res["tx_um"] = best_tx_a
        xy_res["ty_um"] = best_ty_a
        xy_res["best_score_aniso"] = float(best_score)
        aniso_done = True

    # ---- Step 8: assemble extended output & coverage regime ----
    if sxy_known:
        sx_out = sx_final if aniso_done else xy_res["sxy"]
        sy_out = sy_final if aniso_done else xy_res["sxy"]
        tx_out = xy_res["tx_um"]
        ty_out = xy_res["ty_um"]
    else:
        sx_out = 1.0
        sy_out = 1.0
        tx_out = t_minimal[0]
        ty_out = t_minimal[1]

    if sz_known:
        sz_out = z_res["sz"]
    else:
        sz_out = 1.0

    # Extended Z translation: per-cell predicted z = hcr_pia(x_new, y_new) + sz*cz_depth + tz
    if sz_known:
        xy_new = cz_rot[:, :2] * np.array([sx_out, sy_out]) + np.array([tx_out, ty_out])
        pia_hcr_at_new = _surface_z_at(hcr_surface, xy_new[:, 0], xy_new[:, 1])
        z_unshifted = cz_rot[:, 2] * sz_out
        desired_z = pia_hcr_at_new + sz_out * cz_depth + z_res["tz_um"]
        tz_per_cell = desired_z - z_unshifted
        tz_out = float(np.median(tz_per_cell))
        tz_std = float(np.std(tz_per_cell))
    else:
        tz_out = t_minimal[2]
        tz_std = tz_std_min

    scales = np.array([sx_out, sy_out, sz_out], dtype=float)
    scale_known = np.array([sxy_known, sxy_known, sz_known], dtype=bool)
    scale_confidence = np.array([sxy_conf, sxy_conf, sz_conf], dtype=float)
    translation = np.array([tx_out, ty_out, tz_out], dtype=float)

    # Coverage regime from Z search
    if z_res["ok"]:
        sz_w = z_res["sz"]
        cz_depth_range_scaled = float(cz_depth.max() * sz_w)
        hcr_depth_range = float(hcr_depth.max())
        if hcr_depth_range > 1.10 * cz_depth_range_scaled:
            coverage_regime = "thicker"  # HCR thicker than mapped CZ
        elif hcr_depth_range < 0.90 * cz_depth_range_scaled:
            coverage_regime = "thinner"
        else:
            coverage_regime = "equal"
    else:
        coverage_regime = "unknown"

    return CoarseAffineV2(
        R=R,
        scales=scales,
        scale_known=scale_known,
        scale_confidence=scale_confidence,
        translation=translation,
        src_mean=cz_mean,
        rotation_angle_z_deg=rot_angle_z_deg,
        coverage_regime=coverage_regime,
        minimal_translation=t_minimal,
        diagnostics={
            "n_cz": int(len(cz)),
            "n_hcr_gfp": int(len(hcr)),
            "n_cz_depth_min": float(cz_depth.min()),
            "n_cz_depth_max": float(cz_depth.max()),
            "n_hcr_depth_min": float(hcr_depth.min()),
            "n_hcr_depth_max": float(hcr_depth.max()),
            "cz_plane_abc": cz_plane_abc,
            "hcr_plane_abc": hcr_plane_abc,
            "n_cz_normal": n_cz.tolist(),
            "n_hcr_normal": n_hcr.tolist(),
            "cz_tilt_deg": float(math.degrees(math.acos(float(np.clip(n_cz[2], -1, 1))))),
            "hcr_tilt_deg": float(math.degrees(math.acos(float(np.clip(n_hcr[2], -1, 1))))),
            "tilt_axis_cross_deg": float(
                math.degrees(
                    math.acos(float(np.clip(np.dot(n_cz_rot_row, n_hcr), -1, 1)))
                )
            ),
            "sz_grid": sz_grid,
            "sz_score_curve": z_res.get("score_curve"),
            "sz_best": z_res.get("sz"),
            "tz_best_um": z_res.get("tz_um"),
            "sz_score_best": z_res.get("best_score"),
            "sz_confidence": sz_conf,
            "sz_known": sz_known,
            "sxy_grid": sxy_grid,
            "sxy_score_curve": xy_res.get("score_curve"),
            "sxy_best": xy_res.get("sxy"),
            "xy_tx_best_um": xy_res.get("tx_um"),
            "xy_ty_best_um": xy_res.get("ty_um"),
            "sxy_score_best": xy_res.get("best_score"),
            "sxy_confidence": sxy_conf,
            "sxy_known": sxy_known,
            "aniso_done": aniso_done,
            "sx_refined": sx_final if aniso_done else None,
            "sy_refined": sy_final if aniso_done else None,
            "tz_std_um": tz_std,
            "tz_pia_aligned_fallback_um": tz_pia_aligned,
            "coverage_regime": coverage_regime,
            "L_cz_xy_um": L_cz,
            "L_hcr_xy_um": L_hcr,
            "sxy_upper_feasibility": float(sxy_upper),
            "confidence_threshold": float(scale_confidence_threshold),
            "priors_used": {"rotation_deg_z": rotation_deg_z,
                             "sxy_leq_L_hcr_over_L_cz": True},
        },
    )
