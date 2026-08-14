"""Iteration 7 compute (v3 — absolute-log-threshold on combined vol).

History:
- v1: global Gaussian LLR on log-combined → ~100-200 µm too deep.
- v2: column-local mean+k·std threshold on 488 raw → 755252 caught a
  noise-floor rise at 88 µm (~270 µm too shallow); 790322 ≈ auto
  (still too deep).

Root cause (see ``iter07_quantile_diag_*.png``):
  * ``488 raw`` contains mostly cell bodies; the L1 neuropil step is
    ≲ 0.1 log units, comparable to the noise floor.
  * ``combined`` (per-channel bg-subtracted + summed) has OOT ≡ 0 by
    construction, so OOT columns are at ``log(0+eps) = -7`` while
    tissue is ``log ~ -2 .. 0``.  The OOT→tissue step is 5-7 log units,
    completely unambiguous.

v3 approach: smooth log-combined per column and call the first z where
it rises above ``ABS_LOG_THR`` (= -4, well above OOT's -7 and well
below tissue's -1) for a sustained ``SUSTAIN_Z_UM`` the OOT→tissue
transition.  No class LLR, no per-column bg statistics, no dependence
on N22 or centroid geometry — just an absolute threshold on a signal
that has a huge, clean step.

Surface fit is a regularised thin-plate-spline via
``scipy.interpolate.RBFInterpolator`` — real pia curvature that IRLS
quadratic cannot express.

Output files are the same as v2: surfaces PNG (iter07 TPS surface vs
N22, v2.1, auto), a diag PNG (thr histogram is now degenerate so
instead show log-combined sample profiles + the abs threshold), raw
transitions NPZ, and a summary CSV.
"""
from __future__ import annotations

import importlib.util
import os
import types
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, gaussian_filter1d
from scipy.signal import find_peaks

os.environ["PYTHONUNBUFFERED"] = "1"

# 03_image_based_surface.py is a research script not vendored in this package;
# gracefully stub it so surface-fitting core functions (detect_transitions,
# fit_polysurf, sampling_grid) remain importable without it.
_img_path = Path(__file__).parent / "03_image_based_surface.py"
_spec = importlib.util.spec_from_file_location("img_surface_v21", str(_img_path)) if _img_path.exists() else None
if _spec is not None:
    _img = importlib.util.module_from_spec(_spec)
    import sys as _sys
    _sys.modules["img_surface_v21"] = _img
    _spec.loader.exec_module(_img)  # type: ignore
else:
    _img = types.SimpleNamespace(
        estimate_surface_and_l2_image_based=None,
        auto_select_surface=None,
        _surface_z=None,
    )

from autocoreg.io.hcr_image import estimate_pia_surface_image_autoselect, estimate_pia_surface_image_ceiling, estimate_pia_surface_quantile_ceiling, list_hcr_channels, load_hcr_combined, load_hcr_volume
from autocoreg.io.subjects import hcr_px_to_um, load_subject

HCR = ["755252", "767018", "767022", "782149", "788406", "790322"]
N_SIDE = 20
EDGE_FRAC = 0.15

OUT_DATA = Path("/scratch/autocoreg_outputs/dev")
OUT_FIG = Path("/scratch/autocoreg_outputs/dev")

EPS = 1e-3
SMOOTH_Z_UM = 5.0
SUSTAIN_Z_UM = 15.0
PATCH_W = 7   # half-width of xy patch averaged per grid point (voxels); 15x15 ≈ 60x60 µm at level 4
# Column-adaptive threshold: thr = max(THR_FLOOR, col_p90_log - DELTA_LOG).
# Cliff columns (755252, col_p90 ≈ -1) land at thr ≈ -4 → bottom of cliff.
# Gradient columns (790322, col_p90 ≈ -3) land at thr ≈ -6 → start of rise.
# Both end up catching the first "meaningful" tissue signal for that
# column's own dynamic range.
# Range-relative threshold:
#   thr = max(THR_FLOOR, col_p10 + TRANS_FRAC * (col_p90 - col_p10))
# Column firing only when the smoothed log passes the TRANS_FRAC point
# of its own p10→p90 dynamic range.  Distribution-driven, generalises
# across HCR combined (wide p10-to-p90 range) and CZ raw (narrow).
DELTA_LOG = 3.0      # legacy p90-offset knob, kept for logging
TRANS_FRAC = 0.5     # proportion of p10→p90 range above p10 where thr fires
THR_FLOOR = -6.3     # safety net for pure-OOT columns (log(eps)=-6.9, need margin)
NOISE_K = 3.0        # noise-anchored: thr = col_p10 + NOISE_K * sigma
NOISE_QUANTILE = 0.2 # fraction of lowest-value samples used to estimate OOT sigma
PEAK_HEIGHT = -4.0
PEAK_PROMINENCE = 1.5
POLY_DEGREE = 2            # bivariate polynomial degree (2 or 3).  Smaller = flatter.
HUBER_K = 1.5              # IRLS cutoff in units of MAD; smaller = more outlier down-weighting
IRLS_N_ITER = 8
DETECTION_SOURCE = "combined"  # label for panels


def get_n22(s, vol, xy_um, z_um, hcr_xyz):
    anchor = estimate_pia_surface_image_ceiling(
        hcr_xyz, vol, z_um, xy_um,
        relative_margin=0.005, min_signal_abs=0.05, safety_offset_um=0.0)
    if anchor is None:
        return None
    channels = list_hcr_channels(s)
    per_ch = []
    for ch in channels:
        try:
            v_ch, _, _ = load_hcr_volume(s, channel=ch, level=4)
            per_ch.append(v_ch.astype(np.float32, copy=False))
        except FileNotFoundError:
            continue
    sxy = max(1, int(round(10.0 / xy_um)))
    vols_sub = [v[:, ::sxy, ::sxy] for v in per_ch]
    _, Ys, Xs = vols_sub[0].shape
    xs_sub = np.arange(Xs) * sxy * xy_um
    ys_sub = np.arange(Ys) * sxy * xy_um
    xx_sub, yy_sub = np.meshgrid(xs_sub, ys_sub)
    return estimate_pia_surface_quantile_ceiling(
        hcr_xyz, vols_sub, z_um, sxy * xy_um, xx_sub, yy_sub, anchor)


def sampling_grid(vol_shape, xy_um):
    Z, Y, X = vol_shape
    x_pad = int(EDGE_FRAC * X); y_pad = int(EDGE_FRAC * Y)
    xs_i = np.linspace(x_pad, X - 1 - x_pad, N_SIDE).astype(int)
    ys_i = np.linspace(y_pad, Y - 1 - y_pad, N_SIDE).astype(int)
    xi, yi = np.meshgrid(xs_i, ys_i)
    return xi.ravel(), yi.ravel()


def col_detect_transition(log_col, z_um,
                          smooth_z_um=SMOOTH_Z_UM,
                          sustain_z_um=SUSTAIN_Z_UM,
                          trans_frac=TRANS_FRAC,
                          thr_floor=THR_FLOOR,
                          mode="range_relative",
                          noise_k=NOISE_K,
                          noise_quantile=NOISE_QUANTILE):
    """First z where smoothed log column crosses a column-derived threshold,
    sustained for ``sustain_z_um``.

    Two modes:

    - ``range_relative`` (HCR default) —
      ``thr = max(thr_floor, col_p10 + trans_frac*(col_p90-col_p10))``.
      Works well on HCR's cliff-like onset (OOT≡0 → log(eps)≈-6.9, tissue~0),
      where any threshold in the cliff's shoulder fires at the foot.

    - ``noise_anchored`` (for ramp-onset modalities like CZ) —
      ``thr = max(thr_floor, col_p10 + noise_k * MAD_lower)`` where
      ``MAD_lower`` is 1.4826×MAD over values ≤ median (pure-OOT estimate).
      Fires at "first sustained rise above the column's own OOT noise",
      which is what HCR's range-relative rule achieves by accident because
      HCR's p10 IS pure-OOT noise.  CZ's p10 is still within OOT (gap or
      camera-offset region) — the ramp up into tissue means p90 lands
      deep in tissue, so range_relative's halfway point is mid-tissue.
      Noise-anchored is independent of p90, so tissue depth does not
      move the threshold.
    """
    sigma_vox = max(1.0, smooth_z_um / z_um)
    smooth = gaussian_filter1d(log_col, sigma=sigma_vox)
    sustain_vox = max(1, int(sustain_z_um / z_um))

    col_p10 = float(np.percentile(smooth, 10))
    col_p90 = float(np.percentile(smooth, 90))

    if mode == "noise_anchored":
        cutoff = float(np.quantile(smooth, noise_quantile))
        lower = smooth[smooth <= cutoff]
        if lower.size == 0:
            lower = smooth
        mad = float(np.median(np.abs(lower - np.median(lower))))
        sigma = 1.4826 * mad
        thr = max(thr_floor, col_p10 + noise_k * sigma)
    elif mode == "range_relative":
        thr = max(thr_floor, col_p10 + trans_frac * (col_p90 - col_p10))
    else:
        raise ValueError(f"unknown mode {mode!r}")

    if len(smooth) < sustain_vox:
        return -1, smooth, thr
    above = smooth > thr
    for z in range(len(smooth) - sustain_vox):
        if above[z:z + sustain_vox].all():
            return z, smooth, thr
    return -1, smooth, thr


def detect_transitions(vol, z_um, grid_ix, grid_iy, patch_w=PATCH_W):
    """For each grid point, compute the log of the (2w+1)×(2w+1) xy-patch
    MAX, then detect on that column.  Taking the patch max (rather than
    mean) preserves sparse bright cells — important for subjects like
    790322 where L1 has sparse early cells that a mean washes out."""
    Z, Y, X = vol.shape
    log_vol = np.log(vol + EPS)
    zs = np.empty(len(grid_ix))
    thrs = np.empty(len(grid_ix))
    for i, (iy, ix) in enumerate(zip(grid_iy, grid_ix)):
        y0 = max(0, iy - patch_w); y1 = min(Y, iy + patch_w + 1)
        x0 = max(0, ix - patch_w); x1 = min(X, ix + patch_w + 1)
        log_col = log_vol[:, y0:y1, x0:x1].max(axis=(1, 2))
        z_vox, _, thr_col = col_detect_transition(log_col, z_um)
        zs[i]   = z_vox * z_um if z_vox >= 0 else np.nan
        thrs[i] = thr_col
    return zs, thrs


# ============================================================
# Pan-neuronal HCR GFP+ L1/L2 onset (session 23; e.g. subject 837568-01)
# ============================================================
# Companion to `cz_surface.compute_cz_pia_surface_panneuronal`: the CZ side
# finds L1/L2 as the onset of a GCaMP cell-density rise measured from a
# density-derived pia; the HCR side finds the *same* onset rule from the
# GFP+ ROI-density rise, measured as depth below the HCR image pia
# (`hcr_top_iter07`) so both sides share one depth reference. GFP+ is used
# (rather than all/ok ROIs) because CZ GCaMP cells are HCR GFP+, and
# session 23 found the superficial/pia region is filter-dependent (all-ROI
# carries a surface-artifact density spike GFP+ does not).

# Depth-below-pia histogram range spanning the full HCR cortical column
# (pia to white matter, session 23); fixed rather than data-derived
# because depth is already anchored at the image pia (depth 0).
HCR_L1L2_DEPTH_LO_UM = -150.0
HCR_L1L2_DEPTH_HI_UM = 900.0
HCR_L1L2_DEPTH_BIN_UM = 6.0
HCR_L1L2_SMOOTH_SIGMA_BINS = 1.2

# Steepest-rise search zone: excludes the pia/superficial band (a
# surface-artifact spike sits there for all/ok ROIs — session 23) and the
# deep dense-L2 plateau.
HCR_L1L2_SEARCH_LO_UM = 70.0
HCR_L1L2_SEARCH_HI_UM = 320.0

# L1/L2 boundary = ONSET (foot) of the rise: starting from the steepest-
# rise point, walk shallower until the gradient first drops below this
# fraction of its peak.
HCR_L1L2_GRADIENT_FRAC = 0.25

# Minimum GFP+ cells required before trusting the onset detector (session
# 23 ran this on 50,009 GFP+ cells for 837568-01; far below that the depth
# histogram is too noisy for a stable gradient foot).
HCR_L1L2_MIN_GFP_CELLS = 500


def hcr_l1l2_onset_from_depths(
    depth_below_pia_um,
    *,
    lo_um: float = HCR_L1L2_SEARCH_LO_UM,
    hi_um: float = HCR_L1L2_SEARCH_HI_UM,
    depth_range: tuple = (HCR_L1L2_DEPTH_LO_UM, HCR_L1L2_DEPTH_HI_UM),
    bin_um: float = HCR_L1L2_DEPTH_BIN_UM,
    smooth_sigma_bins: float = HCR_L1L2_SMOOTH_SIGMA_BINS,
    gradient_frac: float = HCR_L1L2_GRADIENT_FRAC,
) -> dict | None:
    """L1/L2 boundary = ONSET (foot) of the L1->L2 ROI-density rise, given
    per-cell depths below the HCR image pia (session 23 HCR L1/L2 notebook).

    Locates the steepest-rise depth within ``[lo_um, hi_um]``, then walks
    shallower until the gradient first drops below ``gradient_frac`` of
    its peak — that foot is the boundary (top of L2), shallower than the
    steepest-rise point. Returns ``None`` if the search zone has no cells
    to find a rise in.
    """
    depth = np.asarray(depth_below_pia_um, dtype=float)
    depth = depth[np.isfinite(depth)]
    if depth.size == 0:
        return None
    edges = np.arange(depth_range[0], depth_range[1] + bin_um, bin_um)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist, _ = np.histogram(depth, bins=edges)
    density = gaussian_filter1d(hist.astype(float), smooth_sigma_bins)
    gradient = np.gradient(density)
    zone = np.where((centers > lo_um) & (centers < hi_um))[0]
    if zone.size == 0 or not np.any(density[zone] > 0):
        return None
    gi = zone[int(np.argmax(gradient[zone]))]
    thr = gradient_frac * gradient[gi]
    i = gi
    while i - 1 >= zone[0] and gradient[i - 1] >= thr:
        i -= 1
    return dict(
        onset_um=float(centers[i]),
        steepest_rise_um=float(centers[gi]),
        peak_gradient=float(gradient[gi]),
        n_cells=int(depth.size),
    )


def compute_hcr_gfp_l1_thickness_panneuronal(
    s, hcr_top_surface: dict | None = None,
) -> dict | None:
    """GFP+ onset L1 thickness below the HCR image pia, for pan-neuronal
    subjects (session 23; the GFP+ population matches the CZ GCaMP cells).

    Loads every HCR centroid, restricts to GFP+ via
    ``autocoreg.io.inputs.strict_gfp_ids`` (the same GMM-intersection
    cutoff used elsewhere in the pipeline), computes each GFP+ cell's
    depth below ``hcr_top_surface`` (default
    ``surfaces.get_hcr_top_surface_iter07(s)``), and applies
    ``hcr_l1l2_onset_from_depths``. Returns ``None`` if the pia surface or
    the GFP+ population is unavailable / too small.
    """
    # Lazy imports: `surfaces.py` imports FROM this module at module
    # scope, so `get_hcr_top_surface_iter07` can't be imported back here
    # at module scope without a cycle; io.* is lazy simply to keep this
    # module's import-time footprint light (matches the rest of the file).
    from autocoreg.initial_registration.surfaces import get_hcr_top_surface_iter07
    from autocoreg.io.centroids import centroids_um
    from autocoreg.io.hcr_image import depth_from_surface
    from autocoreg.io.inputs import strict_gfp_ids

    if hcr_top_surface is None:
        hcr_top_surface = get_hcr_top_surface_iter07(s)
    if hcr_top_surface is None:
        return None

    hcr_zyx_um, hcr_ids = centroids_um(s, "hcr_all")
    # Always the GMM GFP+ population here — the L1/L2 tissue boundary is defined by
    # GFP+ cell density and is independent of whether the MATCHING pool is GFP-filtered
    # (AUTOCOREG_GFP_FILTER=none must not turn this into an all-cells / cutoff=None path).
    gfp_ids, gfp_meta = strict_gfp_ids(s.subject_id, force_gmm=True)
    gfp_id_arr = np.fromiter(gfp_ids, dtype=np.int64, count=len(gfp_ids))
    keep = np.isin(hcr_ids, gfp_id_arr)
    if int(keep.sum()) < HCR_L1L2_MIN_GFP_CELLS:
        return None

    hcr_xyz_um = hcr_zyx_um[keep][:, [2, 1, 0]]  # depth_from_surface wants (x, y, z)
    depth_below_pia = depth_from_surface(hcr_xyz_um, hcr_top_surface)

    onset = hcr_l1l2_onset_from_depths(depth_below_pia)
    if onset is None:
        return None
    return dict(
        l1_thickness_um=onset["onset_um"],
        steepest_rise_um=onset["steepest_rise_um"],
        n_gfp=onset["n_cells"],
        gfp_cutoff_linear=float(gfp_meta.get("cutoff_linear") or float("nan")),
    )


def _poly_basis(xn, yn, degree):
    """Design matrix columns for bivariate polynomial up to ``degree``."""
    cols = [np.ones_like(xn)]
    for d in range(1, degree + 1):
        for i in range(d + 1):
            cols.append((xn ** i) * (yn ** (d - i)))
    return np.column_stack(cols)


def fit_polysurf(xs_um, ys_um, zs_um,
                 degree=POLY_DEGREE, huber_k=HUBER_K, n_iter=IRLS_N_ITER):
    """Robust IRLS bivariate-polynomial fit.

    x, y are normalised to the data range so that `xn, yn ∈ [-1, 1]` on the
    input grid; extrapolation past the 15 %-padded interior goes to
    ≈ ±1.43, kept bounded by `degree ≤ 3`.  IRLS with Huber weights
    (`w = min(1, k / |r/s|)`, `s = 1.4826·MAD`) suppresses outlier
    transition points — typical culprits are AF-reduced columns near the
    FOV edge — so the surface is shaped by the consensus interior, not
    by a handful of deep / shallow outliers.
    """
    ok = np.isfinite(zs_um)
    n_params = (degree + 1) * (degree + 2) // 2
    if ok.sum() < n_params + 4:
        return None
    xs = xs_um[ok]; ys = ys_um[ok]; zs = zs_um[ok]
    x0 = 0.5 * (xs.max() + xs.min())
    y0 = 0.5 * (ys.max() + ys.min())
    x_scale = max(1e-6, 0.5 * (xs.max() - xs.min()))
    y_scale = max(1e-6, 0.5 * (ys.max() - ys.min()))
    xn = (xs - x0) / x_scale
    yn = (ys - y0) / y_scale
    A = _poly_basis(xn, yn, degree)
    coef, *_ = np.linalg.lstsq(A, zs, rcond=None)
    for _ in range(n_iter):
        resid = zs - A @ coef
        s = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        if s <= 0:
            break
        r = resid / s
        w = np.where(np.abs(r) <= huber_k, 1.0, huber_k / np.abs(r))
        Aw = A * w[:, None]
        bw = zs * w
        new_coef, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
        if np.max(np.abs(new_coef - coef)) < 1e-6:
            coef = new_coef
            break
        coef = new_coef
    return dict(coef=coef, degree=degree,
                x0=x0, y0=y0, x_scale=x_scale, y_scale=y_scale)


def eval_polysurf(fit, x, y):
    if fit is None:
        return np.full_like(np.asarray(x, dtype=float), np.nan)
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    shape = x.shape
    xn = (x.ravel() - fit['x0']) / fit['x_scale']
    yn = (y.ravel() - fit['y0']) / fit['y_scale']
    A = _poly_basis(xn, yn, fit['degree'])
    return (A @ fit['coef']).reshape(shape)


def render_subject(sid: str):
    print(f"=== {sid} ===", flush=True)
    s = load_subject(sid)
    vol_comb, xy_um, z_um, _ = load_hcr_combined(s, level=4)
    hcr_xyz_um = hcr_px_to_um(
        s.hcr_centroids[["z_px", "y_px", "x_px"]].values, s)[:, [2, 1, 0]]

    n22 = get_n22(s, vol_comb, xy_um, z_um, hcr_xyz_um)
    v21 = _img.estimate_surface_and_l2_image_based(
        vol_comb, z_um, xy_um, target_quantile=0.85).surface
    auto_surf, info, _ = estimate_pia_surface_image_autoselect(s, level=4)

    xi, yi = sampling_grid(vol_comb.shape, xy_um)
    xs_um = xi * xy_um; ys_um = yi * xy_um
    zs_um, thrs = detect_transitions(vol_comb, z_um, xi, yi)
    valid = np.isfinite(zs_um)
    print(f"  transitions: {valid.sum()}/{len(zs_um)} valid, "
          f"median z = {np.nanmedian(zs_um):.1f} µm "
          f"(median thr = {np.nanmedian(thrs):.2f})", flush=True)

    polyfit = fit_polysurf(xs_um, ys_um, zs_um)

    # --- Figure 1: overlays ---
    Z, Y, X = vol_comb.shape
    y_mid = Y // 2; x_mid = X // 2
    x_um = np.arange(X) * xy_um
    y_um = np.arange(Y) * xy_um
    z_um_axis = np.arange(Z) * z_um

    def z_along_x(surface, y_const_um):
        if surface is None: return None
        return _img._surface_z(surface, x_um, np.full_like(x_um, y_const_um))

    def z_along_y(surface, x_const_um):
        if surface is None: return None
        return _img._surface_z(surface, np.full_like(y_um, x_const_um), y_um)

    def iter7_along_x(y_const_um):
        return eval_polysurf(polyfit, x_um, np.full_like(x_um, y_const_um))

    def iter7_along_y(x_const_um):
        return eval_polysurf(polyfit, np.full_like(y_um, x_const_um), y_um)

    y_const = y_mid * xy_um; x_const = x_mid * xy_um

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    def draw_panel(ax, img, title, xlabel, x_axis_um, y_axis_um, overlays):
        vmax = float(np.percentile(img, 99.5))
        ax.imshow(img, aspect='auto', cmap='gray', origin='upper',
                  extent=[x_axis_um[0], x_axis_um[-1], y_axis_um[-1], y_axis_um[0]],
                  vmin=0, vmax=max(vmax, 1e-6))
        for label, arr, color in overlays:
            if arr is None: continue
            ax.plot(x_axis_um, arr, color=color, lw=1.3, label=label)
        ax.set_xlabel(xlabel); ax.set_ylabel('z (µm)')
        ax.set_title(title); ax.legend(loc='lower right', fontsize=8)

    ov_xz = [
        ('N22 (truth)',   z_along_x(n22, y_const), 'tab:green'),
        ('v2.1 default',  z_along_x(v21, y_const), 'tab:orange'),
        (f"auto: {info['selected']}", z_along_x(auto_surf, y_const), 'tab:cyan'),
        ('iter07 TPS',    iter7_along_x(y_const), 'tab:red'),
    ]
    ov_yz = [
        ('N22 (truth)',   z_along_y(n22, x_const), 'tab:green'),
        ('v2.1 default',  z_along_y(v21, x_const), 'tab:orange'),
        (f"auto: {info['selected']}", z_along_y(auto_surf, x_const), 'tab:cyan'),
        ('iter07 TPS',    iter7_along_y(x_const), 'tab:red'),
    ]

    draw_panel(axes[0, 0], vol_comb[:, y_mid, :],
               f"{sid}  combined — XZ @ y={y_const:.0f} µm",
               'x (µm)', x_um, z_um_axis, ov_xz)
    draw_panel(axes[0, 1], np.log(vol_comb[:, y_mid, :] + EPS),
               f"{sid}  log(combined+ε) — XZ @ y={y_const:.0f} µm",
               'x (µm)', x_um, z_um_axis, ov_xz)
    draw_panel(axes[1, 0], vol_comb[:, :, x_mid],
               f"{sid}  combined — YZ @ x={x_const:.0f} µm",
               'y (µm)', y_um, z_um_axis, ov_yz)
    draw_panel(axes[1, 1], np.log(vol_comb[:, :, x_mid] + EPS),
               f"{sid}  log(combined+ε) — YZ @ x={x_const:.0f} µm",
               'y (µm)', y_um, z_um_axis, ov_yz)

    fig.suptitle(
        f"{sid} — iter07 poly-deg{POLY_DEGREE} IRLS (adaptive log thr on combined):  "
        f"median_trans_z = {np.nanmedian(zs_um):.1f} µm, "
        f"median_thr = {np.nanmedian(thrs):.2f}, "
        f"valid = {valid.sum()}/{len(zs_um)}",
        fontsize=11, y=1.00)
    plt.tight_layout()
    out_png = OUT_FIG / f"iter07_surfaces_{sid}.png"
    plt.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_png}", flush=True)

    # --- Figure 2: transition-depth + per-col thr distributions +
    #                   sample smoothed column profiles ---
    fig2, ax2 = plt.subplots(1, 3, figsize=(16, 4))
    z_finite = zs_um[np.isfinite(zs_um)]
    if len(z_finite):
        ax2[0].hist(z_finite, bins=40, color='tab:red', alpha=0.7)
    ax2[0].set_xlabel('transition depth (µm)')
    ax2[0].set_ylabel('count')
    ax2[0].set_title(f"{sid} per-column transition z")

    ax2[1].hist(thrs[np.isfinite(thrs)], bins=40, color='tab:orange', alpha=0.7)
    ax2[1].set_xlabel('column-adaptive log-threshold')
    ax2[1].set_title(f"{sid} per-column thr (floor={THR_FLOOR}, δ={DELTA_LOG})")

    sample_idx = np.linspace(0, len(xs_um) - 1, 5).astype(int)
    Z, Y, X = vol_comb.shape
    log_vol = np.log(vol_comb + EPS)
    for idx in sample_idx:
        iy = int(yi[idx]); ix = int(xi[idx])
        y0 = max(0, iy - PATCH_W); y1 = min(Y, iy + PATCH_W + 1)
        x0 = max(0, ix - PATCH_W); x1 = min(X, ix + PATCH_W + 1)
        log_col = log_vol[:, y0:y1, x0:x1].max(axis=(1, 2))
        _, smooth, thr = col_detect_transition(log_col, z_um)
        ax2[2].plot(smooth, z_um_axis, lw=0.8,
                    label=f"col ({ys_um[idx]:.0f},{xs_um[idx]:.0f}) thr={thr:.1f}")
        if np.isfinite(zs_um[idx]):
            ax2[2].axhline(zs_um[idx], color='tab:red', lw=0.4, ls='-')
    ax2[2].invert_yaxis()
    ax2[2].set_xlabel('log(combined + ε)'); ax2[2].set_ylabel('z (µm)')
    ax2[2].set_title(f"{sid} sample smoothed column profiles (patch-mean)")
    ax2[2].legend(fontsize=7)

    plt.tight_layout()
    out_png2 = OUT_FIG / f"iter07_dists_{sid}.png"
    plt.savefig(out_png2, dpi=120)
    plt.close(fig2)
    print(f"  wrote {out_png2}", flush=True)

    np.savez(OUT_DATA / f"iter07_transitions_{sid}.npz",
             xs_um=xs_um, ys_um=ys_um, zs_um=zs_um, thrs=thrs)

    delta_n22 = delta_auto = np.nan
    if n22 is not None and polyfit is not None:
        z_n22_grid = _img._surface_z(n22, xs_um, ys_um)
        z_iter07 = eval_polysurf(polyfit, xs_um, ys_um)
        delta_n22 = float(np.nanmedian(z_iter07 - z_n22_grid))
    if auto_surf is not None and polyfit is not None:
        z_auto_grid = _img._surface_z(auto_surf, xs_um, ys_um)
        z_iter07 = eval_polysurf(polyfit, xs_um, ys_um)
        delta_auto = float(np.nanmedian(z_iter07 - z_auto_grid))

    return dict(
        subject=sid,
        auto_selected=info['selected'],
        detection_source=DETECTION_SOURCE,
        delta_log=DELTA_LOG,
        thr_floor=THR_FLOOR,
        poly_degree=POLY_DEGREE,
        huber_k=HUBER_K,
        median_thr=float(np.nanmedian(thrs)),
        n_valid_trans=int(valid.sum()),
        median_trans_z_um=float(np.nanmedian(zs_um)) if valid.any() else np.nan,
        delta_z_vs_n22_um=delta_n22,
        delta_z_vs_auto_um=delta_auto,
    )


def main():
    rows = [render_subject(sid) for sid in HCR]
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DATA / "iter07_summary.csv", index=False)
    print("\n=== iter07 summary ===")
    print(df.to_string(index=False, float_format=lambda x: f'{x:7.2f}'))
    print(f"\nwrote {OUT_DATA/'iter07_summary.csv'}")


if __name__ == "__main__":
    main()
