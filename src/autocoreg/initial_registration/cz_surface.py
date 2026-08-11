"""Iter 08 — CZ surface with the experimenter's 50 µm depth prior.

Motivation
----------
During CZ acquisition the experimenter manually positioned the pial
surface at z ≈ 50 µm from the first slice.  That is independent
metadata — not derived from the image — and we can exploit it to
disambiguate between candidate detector thresholds.

iter07_cz_proto (TRANS_FRAC = 0.5) matches the prior on 5/6 subjects
but lands at 102.5 µm on 767022 (AF/OOT slab above tissue → the
p10→p90 midpoint lands mid-ramp on strong tissue).  iter07_cz_lowfrac
(TRANS_FRAC = 0.02) always fires earlier but often drops below the
prior (34–46 µm), i.e. it latches on pre-pia AF / coverslip debris on
some subjects.  No single TRANS_FRAC works for every subject.

iter08 approach: sweep a small bank of TRANS_FRAC values and select
the one whose *median per-column transition depth* is closest to the
50 µm prior.  The prior acts as the selection score, so subjects with
a clean tissue cliff (low TRANS_FRAC fires early on coverslip) prefer
a larger fraction; subjects with an OOT slab (large TRANS_FRAC fires
deep) prefer a smaller fraction.

After selection, gate per-column transitions to |z - 50| ≤ GATE_UM
before the IRLS-Huber poly fit, so the surface is not bent by a
handful of AF-latched outliers.

Outputs
-------
* ``figures/iter08_cz_<sid>.png`` — 4-y-slice log(CZ) overlays with
  the selected surface (red), prior plane at z = 50 µm (yellow
  dashed), existing image_ceiling surface (cyan), and CZ centroid
  surface (green).
* ``data/iter08_cz_transitions_<sid>.npz`` — per-column zs + thrs for
  the selected candidate.
* ``data/iter08_cz_selection.csv`` — per-subject selected TRANS_FRAC,
  median_z, |median−prior|, n_in_gate.
* ``data/iter08_cz_sweep.csv`` — full bank (all subjects × all
  candidates) with median z and closeness-to-prior.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from autocoreg.io.hcr_image import analyze_subject, estimate_pia_surface
from autocoreg.io.subjects import cz_px_to_um, load_subject
from autocoreg.initial_registration.surface_detect import col_detect_transition, eval_polysurf, fit_polysurf, sampling_grid, HUBER_K, PATCH_W, POLY_DEGREE, SMOOTH_Z_UM, SUSTAIN_Z_UM

OUT_FIG = Path("/scratch/autocoreg_outputs/dev")
OUT_DATA = Path("/scratch/autocoreg_outputs/dev")

SUBJECTS = ["755252", "767018", "767022", "782149", "788406", "790322"]

EPS = 1e-3
CZ_THR_FLOOR = 4.5                  # ≈ log(90 DN); rejects pure-OOT columns
CZ_TARGET_Z_UM = 50.0               # experimenter prior: pia ≈ 50 µm
TRANS_FRAC_BANK = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50)
GATE_UM = 75.0                      # per-column outlier gate around prior
N_Y = 4
Y_INTERIOR_FRAC = 0.15


def load_cz_volume(s):
    files = (
        list(s.coreg_dir.glob("*reg-dim-swapped.ome.tif"))
        or list(s.coreg_dir.glob("*zstack.tif"))
    )
    if not files:
        raise FileNotFoundError(f"no CZ TIFF for {s.subject_id}")
    arr = tifffile.imread(str(files[0]))
    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"CZ TIFF shape={arr.shape} not ZYX")
    return arr.astype(np.float32, copy=False)


def _patch_log_columns(log_vol, grid_ix, grid_iy, patch_w=PATCH_W):
    """Pre-compute the (patch log-max) column for every grid point once,
    so we can re-threshold cheaply across the TRANS_FRAC bank."""
    Z, Y, X = log_vol.shape
    n = len(grid_ix)
    cols = np.empty((n, Z), dtype=np.float32)
    for i, (iy, ix) in enumerate(zip(grid_iy, grid_ix)):
        y0 = max(0, iy - patch_w); y1 = min(Y, iy + patch_w + 1)
        x0 = max(0, ix - patch_w); x1 = min(X, ix + patch_w + 1)
        cols[i] = log_vol[:, y0:y1, x0:x1].max(axis=(1, 2))
    return cols


def _detect_on_columns(log_cols, z_um, trans_frac, thr_floor):
    n = log_cols.shape[0]
    zs = np.empty(n)
    thrs = np.empty(n)
    for i in range(n):
        z_vox, _, thr_col = col_detect_transition(
            log_cols[i], z_um,
            smooth_z_um=SMOOTH_Z_UM,
            sustain_z_um=SUSTAIN_Z_UM,
            trans_frac=trans_frac,
            thr_floor=thr_floor,
            mode="range_relative",
        )
        zs[i] = z_vox * z_um if z_vox >= 0 else np.nan
        thrs[i] = thr_col
    return zs, thrs


def select_trans_frac(log_cols, z_um, target_z, bank=TRANS_FRAC_BANK,
                      thr_floor=CZ_THR_FLOOR):
    """Evaluate every TRANS_FRAC in the bank, return the one whose
    median column transition z is closest to the prior + the full sweep
    table for diagnostics."""
    rows = []
    best = None
    best_score = np.inf
    for tf in bank:
        zs, thrs = _detect_on_columns(log_cols, z_um, tf, thr_floor)
        valid = np.isfinite(zs)
        med = float(np.nanmedian(zs)) if valid.any() else np.nan
        score = abs(med - target_z) if np.isfinite(med) else np.inf
        rows.append(dict(
            trans_frac=tf, median_z=med,
            abs_dev_from_prior=score,
            n_valid=int(valid.sum()),
        ))
        if score < best_score:
            best_score = score
            best = (tf, zs, thrs, med)
    sweep = pd.DataFrame(rows)
    return best, sweep


def fit_gated_surface(xs_um, ys_um, zs_um, target_z=CZ_TARGET_Z_UM,
                      gate_um=GATE_UM):
    """Drop per-column transitions outside [target_z ± gate_um] before
    the IRLS-Huber poly fit; return the fit + number of in-gate points."""
    ok = np.isfinite(zs_um)
    if gate_um is not None and np.isfinite(target_z):
        in_gate = ok & (np.abs(zs_um - target_z) <= gate_um)
    else:
        in_gate = ok
    xs_in = xs_um[in_gate]; ys_in = ys_um[in_gate]; zs_in = zs_um[in_gate]
    polyfit = fit_polysurf(xs_in, ys_in, zs_in,
                           degree=POLY_DEGREE, huber_k=HUBER_K)
    return polyfit, int(in_gate.sum()), int(ok.sum())


# ============================================================
# Pan-neuronal CZ pia surface — cell-density onset rule (session 22/24)
# ============================================================
# The iter08 detector above assumes a sparse GCaMP cell cloud with a clean
# OOT->tissue image cliff. Dense pan-neuronal CZ stacks (e.g. 837568) have
# neither: the whole FOV is saturated with cell bodies near the top, so a
# per-column image threshold does not reliably land on the true pia.
#
# This alternate detector instead uses the CZ centroid *density* profile:
# cortical L1 is cell-sparse and L2 begins with a dense band, so (1) the
# tissue tilt is the plane that maximises the sharpness of that L1->L2
# density rise, (2) the L1/L2 boundary is the ONSET (foot) of the rise —
# not its steepest point, which sits mid-L2 — and (3) the pia is a
# superficial cell-density peak (the first-neuron layer sits ~10 um below
# the true membrane). Validated on 837568 against the independent
# dextran-vasculature pia (session 22 notebook, converges to ~1 um) and
# productionised in `register_l1_protocol.py` (session 24), which uses the
# resulting surface + L1 thickness to recover the true ~277 deg CZ<->HCR
# rotation via a full-circle registration search.

# Minimum CZ centroids needed before trusting the density-based tilt/onset
# search (837568 has ~1e5 cells; this just guards a near-empty stack).
PANNEURONAL_CZ_MIN_CELLS = 500

# Tissue-tilt search grid: z = a*x + b*y maximising the L1/L2 density-rise
# sharpness (session 22 sec. 2).
PANNEURONAL_TILT_GRID = np.arange(-0.25, 0.251, 0.03)

# Depth-histogram bin width / smoothing used by the tilt search + L1/L2
# onset detector.
PANNEURONAL_DENS_BIN_UM = 4.0
PANNEURONAL_DENS_SIGMA_BINS = 1.2

# Tissue onset (foot of the very first density rise — where cells start
# appearing at all) = first depth bin whose density exceeds this fraction
# of the L2+ plateau (8%: avoids sparse-L1 fluctuation without landing
# mid-tissue; plateau = median density beyond this percentile of depths).
PANNEURONAL_PLATEAU_PERCENTILE = 55.0
PANNEURONAL_TISSUE_ONSET_PLATEAU_FRAC = 0.08

# L1/L2 steepest-rise search zone, in um below the tissue onset — excludes
# the pia/superficial band and the deep L2+ plateau.
PANNEURONAL_L1L2_SEARCH_LO_UM = 45.0
PANNEURONAL_L1L2_SEARCH_HI_UM = 260.0

# L1/L2 boundary = ONSET (foot) of the rise: starting from the steepest-
# rise point, walk shallower until the gradient first drops below this
# fraction of its peak.
PANNEURONAL_L1L2_GRADIENT_FRAC = 0.25

# Superficial cell-peak finder: finer histogram than the tilt search, plus
# a peak-prominence floor (session 22 sec. 6, `scipy.signal.find_peaks`).
PANNEURONAL_PEAK_BIN_UM = 2.0
PANNEURONAL_PEAK_SIGMA_BINS = 1.0
PANNEURONAL_PEAK_PROMINENCE = 0.5
# Candidate peaks are restricted to [tissue_onset - LO, steepest_rise - HI]
# so the pia peak wins over autofluorescence debris or the L2 rise itself.
PANNEURONAL_PEAK_SEARCH_LO_MARGIN_UM = 3.0
PANNEURONAL_PEAK_SEARCH_HI_MARGIN_UM = 25.0

# Pia sits this far above the superficial cell peak — the peak is the
# first-neuron layer, ~10 um below the true membrane (calibrated against
# the independent dextran-vasculature pia on 837568, session 22 sec. 6).
PANNEURONAL_PIA_ABOVE_PEAK_UM = 10.0


def _panneuronal_density_profile(z, y, x, a, b, bin_um, sigma_bins):
    """Smoothed depth histogram of centroids after detrending the tilt
    plane ``z = a*x + b*y``. Returns ``(depth, bin_centers, density)``."""
    depth = z - (a * x + b * y)
    edges = np.arange(depth.min(), depth.max() + bin_um, bin_um)
    centers = 0.5 * (edges[:-1] + edges[1:])
    density = gaussian_filter1d(
        np.histogram(depth, bins=edges)[0].astype(float), sigma_bins)
    return depth, centers, density


def _panneuronal_l1l2_onset(z, y, x, a, b):
    """Tissue onset, L1/L2 onset (foot of the density rise), the rise's
    sharpness, and the steepest-rise reference depth, for tilt plane
    ``(a, b)``. Returns
    ``(tissue_onset_um, l1l2_onset_um, sharpness, steepest_rise_um)``."""
    depth, centers, density = _panneuronal_density_profile(
        z, y, x, a, b, PANNEURONAL_DENS_BIN_UM, PANNEURONAL_DENS_SIGMA_BINS)
    gradient = np.gradient(density)
    plateau = np.median(
        density[centers > np.percentile(depth, PANNEURONAL_PLATEAU_PERCENTILE)])
    tissue_onset = float(centers[np.argmax(
        density > PANNEURONAL_TISSUE_ONSET_PLATEAU_FRAC * plateau)])
    zone = np.where(
        (centers > tissue_onset + PANNEURONAL_L1L2_SEARCH_LO_UM)
        & (centers < tissue_onset + PANNEURONAL_L1L2_SEARCH_HI_UM))[0]
    if zone.size == 0:
        # No rise found in the search zone at this tilt — report zero
        # sharpness so the tilt grid search skips it, but still return
        # finite depths.
        return tissue_onset, tissue_onset, 0.0, tissue_onset
    gi = zone[int(np.argmax(gradient[zone]))]
    thr = PANNEURONAL_L1L2_GRADIENT_FRAC * gradient[gi]
    i = gi
    while i - 1 >= zone[0] and gradient[i - 1] >= thr:
        i -= 1
    return tissue_onset, float(centers[i]), float(gradient[gi]), float(centers[gi])


def compute_cz_pia_surface_panneuronal(s) -> dict | None:
    """Onset-rule CZ pia surface + L1 thickness for pan-neuronal (dense)
    CZ stacks (session 22/24; e.g. subject 837568).

    Reuses the existing CZ centroid loader (``cz_px_to_um``). Returns the
    canonical planar ``{a, b, c, p, q, r}`` surface dict (``p=q=r=0``, so
    it plugs directly into ``depth_from_surface`` / ``top_slab_projection``
    like every other surface in this package) plus ``l1_thickness_um`` and
    detection diagnostics — or ``None`` if there are too few CZ centroids.

    Algorithm (session 22 secs. 2 + 6; productionised in
    ``register_l1_protocol.py``, session 24):
      1. Grid-search the tissue tilt ``(a, b)`` in ``z = a*x + b*y``
         maximising the L1/L2 density-rise sharpness.
      2. L1/L2 boundary = ONSET (foot) of the L1->L2 density rise at that
         tilt (not the steepest point, which sits mid-L2).
      3. Superficial cell-density peak (``find_peaks``) between the
         tissue onset and the L1/L2 steepest-rise reference.
      4. Pia = superficial peak - ``PANNEURONAL_PIA_ABOVE_PEAK_UM``; falls
         back to the tissue onset if no peak is found (mirrors
         ``register_l1_protocol.py``).
    """
    cz_um = cz_px_to_um(
        s.cz_centroids[["z_px", "y_px", "x_px"]].to_numpy(float), s)
    if len(cz_um) < PANNEURONAL_CZ_MIN_CELLS:
        return None
    z, y, x = cz_um[:, 0], cz_um[:, 1], cz_um[:, 2]

    best_sharpness, tilt_a, tilt_b = -np.inf, 0.0, 0.0
    for a in PANNEURONAL_TILT_GRID:
        for b in PANNEURONAL_TILT_GRID:
            sharpness = _panneuronal_l1l2_onset(z, y, x, a, b)[2]
            if sharpness > best_sharpness:
                best_sharpness, tilt_a, tilt_b = sharpness, float(a), float(b)

    tissue_onset, l1l2_onset, sharpness, steepest_rise = (
        _panneuronal_l1l2_onset(z, y, x, tilt_a, tilt_b))

    # User-provided acquisition pia estimate (depth µm below the first slice) OVERRIDES the
    # superficial-peak detection below — that step is corrupted by junk ROIs above the pia, which
    # look like real cells and can't be classified out. The tilt (a,b) + L1/L2 onset above are
    # detected from the density and kept (junk-immune); this only pins the pia intercept. 0 is
    # allowed (stack starts at the pia). Env: AUTOCOREG_CZ_PIA_UM (legacy alias MFISH_CZ_PIA_UM).
    pia_in = os.environ.get("AUTOCOREG_CZ_PIA_UM", os.environ.get("MFISH_CZ_PIA_UM", "")).strip()
    if pia_in != "":
        pia = float(pia_in)
        return dict(
            a=float(tilt_a), b=float(tilt_b), c=float(pia), p=0.0, q=0.0, r=0.0,
            l1_thickness_um=float(l1l2_onset - pia),
            method="panneuronal_user_pia",
            subject_id=str(s.subject_id), n_cz_cells=int(len(cz_um)),
            tissue_onset_um=float(tissue_onset), l1l2_onset_um=float(l1l2_onset),
            l1l2_steepest_rise_um=float(steepest_rise), l1l2_sharpness=float(sharpness),
            superficial_peak_um=None, pia_source="acquisition_user_input",
        )

    _, centers, density = _panneuronal_density_profile(
        z, y, x, tilt_a, tilt_b,
        PANNEURONAL_PEAK_BIN_UM, PANNEURONAL_PEAK_SIGMA_BINS)
    peaks, props = find_peaks(density, prominence=PANNEURONAL_PEAK_PROMINENCE)
    peak_zone = (
        (centers[peaks] > tissue_onset - PANNEURONAL_PEAK_SEARCH_LO_MARGIN_UM)
        & (centers[peaks] < steepest_rise - PANNEURONAL_PEAK_SEARCH_HI_MARGIN_UM))
    if peak_zone.any():
        k = int(np.argmax(props["prominences"][peak_zone]))
        superficial_peak = float(centers[peaks[peak_zone][k]])
        pia = superficial_peak - PANNEURONAL_PIA_ABOVE_PEAK_UM
    else:
        superficial_peak = None
        pia = tissue_onset

    return dict(
        a=float(tilt_a), b=float(tilt_b), c=float(pia), p=0.0, q=0.0, r=0.0,
        l1_thickness_um=float(l1l2_onset - pia),
        method="panneuronal_cell_density_onset",
        subject_id=str(s.subject_id),
        n_cz_cells=int(len(cz_um)),
        tissue_onset_um=float(tissue_onset),
        l1l2_onset_um=float(l1l2_onset),
        l1l2_steepest_rise_um=float(steepest_rise),
        l1l2_sharpness=float(sharpness),
        superficial_peak_um=superficial_peak,
    )


# ── Dextran-plexus leading-edge CZ pia (ALTERNATIVE pan-neuronal pia) ──────────────
# Pan-neuronal (dense GCaMP) subjects that ALSO acquired a dextran vasculature channel can
# derive the CZ pia from the dextran-plexus leading edge instead of the cell-density onset.
# Only the pia/tilt differ; the L1 slab thickness is kept from the density-onset method
# (session 25). Gated by the env var MFISH_CZ_DEXTRAN_TIF (path to the registered dextran
# channel, e.g. channel_1_ref_1/*.tif) — unset ⇒ density-onset (default). Dextran only applies
# to pan-neuronal samples (pan-inhibitory stacks have no dextran channel). Constants + logic
# validated in session 22/25 (the dextran run reproduced the density-onset coreg to 99.1%).
DEX_GRID_MARGIN_PX = 60      # keep sample columns clear of FOV-edge / tile-seam artifacts
DEX_GRID_N = 30              # 30x30 sampling grid across the FOV
DEX_P99_PCTL = 99.0          # per-column AND global-profile "vessel present" percentile
DEX_GATE_MARGIN_UM = 25.0    # drop columns whose leading-edge crossing is >25um deeper than the
                             # per-column median (folds/debris, not the true plexus) before tilt fit
DEX_PEAK_SEARCH_Z_PX = 120   # plexus peak must be shallow (first 120 z-slices)
DEX_LEADING_EDGE_FRAC = 0.2  # global leading edge = first z where whole-FOV p99 reaches 20% of
                             # (peak - deep-parenchyma baseline)
DEX_BASELINE_Z_LO_UM = 150.0
DEX_BASELINE_Z_HI_UM = 400.0  # deep parenchyma window for the baseline dextran level
DEX_MIN_GRID_POINTS = 20      # below this the plane fit is not trustworthy -- fail loudly


def compute_dextran_cz_pia_surface(s, dextran_tif_path: str,
                                   l1_thickness_um: float) -> dict:
    """Dextran-plexus leading-edge CZ pia surface, in the SAME canonical
    ``{a,b,c,p,q,r}`` dict + coordinate convention as
    ``compute_cz_pia_surface_panneuronal`` (so it plugs into
    ``depth_from_surface`` / ``cz_binary_top_mip`` identically). x/y in CZ µm via
    ``s.cz_xy_um``, z via ``s.cz_z_um``. ``l1_thickness_um`` is passed in (borrowed from
    the density-onset surface), not recomputed — only pia/tilt vary vs the density-onset
    pia. Raises (fail-loud) if the dextran TIF is unreadable or too few grid columns cross
    the plexus threshold. Session 22/25; gated by ``MFISH_CZ_DEXTRAN_TIF``."""
    xy_um = float(s.cz_xy_um)
    z_um = float(s.cz_z_um)

    # FOV-center anchor in the dextran/CZ pixel coordinate system.
    # Use the registered dextran volume dimensions (not centroid extents) so the anchor is stable.
    dex = tifffile.imread(dextran_tif_path).astype(np.float32)  # (Z, Y, X)
    Zn, Yn, Xn = dex.shape
    print(f"[dextran-surface] loaded {dextran_tif_path} shape={dex.shape}", flush=True)
    xc, yc = (Xn - 1) * xy_um / 2.0, (Yn - 1) * xy_um / 2.0
    zz = np.arange(Zn) * z_um

    def fit_plane_irls(P, n_iter=5):
        X = np.c_[P[:, 0], P[:, 1], np.ones(len(P))]
        yv = P[:, 2]
        w = np.ones(len(P))
        for _ in range(n_iter):
            W = np.sqrt(w)[:, None]
            cf = np.linalg.lstsq(X * W, yv * W[:, 0], rcond=None)[0]
            r = yv - X @ cf
            mad = np.median(np.abs(r)) + 1e-6
            w = 1.0 / (1.0 + (r / (2.5 * mad)) ** 2)
        return float(cf[0]), float(cf[1]), float(cf[2])

    gyv = np.linspace(DEX_GRID_MARGIN_PX, Yn - DEX_GRID_MARGIN_PX, DEX_GRID_N).astype(int)
    gxv = np.linspace(DEX_GRID_MARGIN_PX, Xn - DEX_GRID_MARGIN_PX, DEX_GRID_N).astype(int)
    thr99 = float(np.percentile(dex, DEX_P99_PCTL))
    P_list = []
    for iy in gyv:
        for ix in gxv:
            col = gaussian_filter1d(dex[:, iy, ix], 2)
            above = np.where(col > thr99)[0]
            if above.size:
                P_list.append((ix * xy_um, iy * xy_um, float(zz[above[0]])))
    Pdex = np.asarray(P_list)
    if len(Pdex) < DEX_MIN_GRID_POINTS:
        raise RuntimeError(
            f"dextran surface: only {len(Pdex)} of {DEX_GRID_N * DEX_GRID_N} grid columns "
            f"crossed the p99 threshold -- too few for a robust plane fit")
    a0, b0, c0_ungated = fit_plane_irls(Pdex)  # diagnostic only

    med_z = float(np.median(Pdex[:, 2]))
    gate = Pdex[:, 2] < med_z + DEX_GATE_MARGIN_UM
    aD, bD, _ = fit_plane_irls(Pdex[gate])

    dpx = np.percentile(dex.reshape(Zn, -1), DEX_P99_PCTL, axis=1)  # whole-FOV p99 per z
    baseline_mask = (zz > DEX_BASELINE_Z_LO_UM) & (zz < DEX_BASELINE_Z_HI_UM)
    if not baseline_mask.any():
        raise RuntimeError("dextran surface: no z-slices in the deep-parenchyma baseline window "
                           f"[{DEX_BASELINE_Z_LO_UM},{DEX_BASELINE_Z_HI_UM}]um -- stack too shallow")
    dbase = float(np.median(dpx[baseline_mask]))
    peak_window = min(DEX_PEAK_SEARCH_Z_PX, Zn)
    pk_i = int(np.argmax(dpx[:peak_window]))
    pk_z, pk_v = float(zz[pk_i]), float(dpx[:peak_window].max())
    edge_thr = dbase + DEX_LEADING_EDGE_FRAC * (pk_v - dbase)
    edge_candidates = np.where((zz <= pk_z) & (dpx >= edge_thr))[0]
    if edge_candidates.size == 0:
        raise RuntimeError("dextran surface: no leading-edge crossing found shallower than the "
                           "plexus peak -- global profile too flat / threshold too high")
    edge_z = float(zz[edge_candidates[0]])

    # Anchor the gated tilt's intercept so the plane equals the global leading-edge depth AT the
    # FOV center (spatially-resolved tilt + spatially-averaged, robust absolute depth reference).
    c_edge = edge_z - (aD * xc + bD * yc)
    pia_at_center = aD * xc + bD * yc + c_edge

    surface = dict(
        a=float(aD), b=float(bD), c=float(c_edge), p=0.0, q=0.0, r=0.0,
        l1_thickness_um=float(l1_thickness_um),
        method="panneuronal_dextran_plexus_leading_edge",
        subject_id=str(s.subject_id),
        n_grid_columns=int(len(Pdex)), n_grid_columns_gated=int(gate.sum()),
        ungated_tilt_ab=[float(a0), float(b0)], ungated_intercept_um=float(c0_ungated),
        plexus_peak_z_um=pk_z, plexus_peak_p99=pk_v, parenchyma_baseline_p99=dbase,
        leading_edge_z_um=edge_z, pia_at_fov_center_um=float(pia_at_center),
        fov_center_xy_um=[xc, yc], cz_xy_um=xy_um, cz_z_um=z_um,
        dextran_tif=str(dextran_tif_path),
    )
    print(f"[dextran-surface] tilt=({aD:+.4f},{bD:+.4f}) pia@center={pia_at_center:.1f}um "
          f"leading_edge={edge_z:.1f}um peak=({pk_z:.1f}um,{pk_v:.1f}) baseline={dbase:.1f} "
          f"n_grid={len(Pdex)} n_gated={int(gate.sum())} l1_thickness_um={l1_thickness_um:.1f}",
          flush=True)
    return surface


def get_cz_pia_surface_panneuronal(
    s, *, use_cache: bool = True, write_cache: bool = True,
) -> dict | None:
    """Cache-aware accessor for ``compute_cz_pia_surface_panneuronal``.

    Cached alongside the other promoted surfaces
    (``autocoreg.config.SURFACES_CACHE_DIR``) under a distinct filename so
    ``surface_registration.py`` and ``locked_prior.py`` resolve the exact
    same surface without recomputing it. Kept self-contained here (rather
    than routed through ``surfaces.py``'s cache helpers) to avoid a module
    import cycle — ``surfaces.py`` imports helpers FROM this module.
    """
    from autocoreg import config as _config
    # Dextran gate: MFISH_CZ_DEXTRAN_TIF (path to the registered dextran channel) ⇒ use the
    # dextran-plexus pia instead of the density-onset one. Only pia/tilt differ; L1 thickness is
    # borrowed from the density-onset surface. Distinct cache filename so it never clobbers the
    # density-onset cache. Unset ⇒ density-onset (default, unchanged). Pan-neuronal only.
    dextran_tif = os.environ.get("MFISH_CZ_DEXTRAN_TIF", "").strip()
    if dextran_tif:
        dex_cache = (
            _config.SURFACES_CACHE_DIR / f"{s.subject_id}_cz_dextran_panneuronal.json"
        )
        if use_cache and dex_cache.exists():
            return json.loads(dex_cache.read_text())
        onset = compute_cz_pia_surface_panneuronal(s)   # for the L1 thickness only
        if onset is None:
            return None
        surface = compute_dextran_cz_pia_surface(s, dextran_tif, onset["l1_thickness_um"])
        if write_cache:
            dex_cache.parent.mkdir(parents=True, exist_ok=True)
            dex_cache.write_text(json.dumps(surface, indent=2))
        return surface
    cache_path = (
        _config.SURFACES_CACHE_DIR / f"{s.subject_id}_cz_onset_panneuronal.json"
    )
    if use_cache and cache_path.exists():
        return json.loads(cache_path.read_text())
    surface = compute_cz_pia_surface_panneuronal(s)
    if write_cache and surface is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(surface, indent=2))
    return surface


def _cz_existing_surface(s):
    """Existing CZ image-based ceiling surface from analyze_subject."""
    try:
        info = analyze_subject(s)
    except Exception as exc:  # pragma: no cover
        print(f"  analyze_subject failed for {s.subject_id}: {exc}")
        return None
    return info.get("cz_surface")


def _cz_centroid_surface(s):
    cz_um = cz_px_to_um(
        s.cz_centroids[["z_px", "y_px", "x_px"]].values, s)
    cz_xyz = cz_um[:, [2, 1, 0]]
    try:
        return estimate_pia_surface(cz_xyz)
    except Exception as exc:  # pragma: no cover
        print(f"  centroid surface failed for {s.subject_id}: {exc}")
        return None


def _surface_z(surface, xs, y0):
    if surface is None:
        return np.full_like(np.asarray(xs, dtype=float), np.nan)
    a, b, c = surface["a"], surface["b"], surface["c"]
    p = surface.get("p", 0.0); q = surface.get("q", 0.0)
    r = surface.get("r", 0.0)
    xs = np.asarray(xs, dtype=float)
    return (a * xs + b * y0 + c + p * xs * xs
            + q * xs * y0 + r * y0 * y0)


def render_subject(sid):
    print(f"=== {sid} ===", flush=True)
    s = load_subject(sid)
    vol = load_cz_volume(s)
    z_um, xy_um = s.cz_z_um, s.cz_xy_um
    Z, Y, X = vol.shape
    log_vol = np.log(vol + EPS)

    xi, yi = sampling_grid(vol.shape, xy_um)
    xs_um = xi * xy_um; ys_um = yi * xy_um

    log_cols = _patch_log_columns(log_vol, xi, yi)
    (sel_tf, sel_zs, sel_thrs, sel_med), sweep = select_trans_frac(
        log_cols, z_um, CZ_TARGET_Z_UM)
    sweep["subject"] = sid
    print(f"  TRANS_FRAC sweep (target = {CZ_TARGET_Z_UM:.0f} µm):")
    for _, r in sweep.iterrows():
        mark = " *" if r["trans_frac"] == sel_tf else "  "
        print(f"  {mark} tf={r['trans_frac']:.2f}  median_z={r['median_z']:6.1f}  "
              f"|dev|={r['abs_dev_from_prior']:6.2f}  n_valid={r['n_valid']}")
    print(f"  selected TRANS_FRAC = {sel_tf}  median_z = {sel_med:.1f} µm")

    polyfit, n_in, n_valid = fit_gated_surface(xs_um, ys_um, sel_zs)
    if polyfit is None:
        print("  polyfit failed")
        return None, sweep
    print(f"  in-gate: {n_in}/{n_valid} (valid), gate ±{GATE_UM:.0f} µm")

    cz_centroid = _cz_centroid_surface(s)
    cz_existing = _cz_existing_surface(s)

    x_um = np.arange(X) * xy_um
    z_axis = np.arange(Z) * z_um
    y_lo = int(Y_INTERIOR_FRAC * Y); y_hi = Y - 1 - y_lo
    y_idx = np.linspace(y_lo, y_hi, N_Y).astype(int)

    fig, axes = plt.subplots(1, N_Y, figsize=(5 * N_Y, 4.8), sharey=True)
    for ax, iy in zip(axes, y_idx):
        y_const = iy * xy_um
        img = np.log(vol[:, iy, :] + EPS)
        vmin = float(np.percentile(img, 5))
        vmax = float(np.percentile(img, 99.5))
        ax.imshow(img, aspect="auto", cmap="gray", origin="upper",
                  extent=[x_um[0], x_um[-1], z_axis[-1], z_axis[0]],
                  vmin=vmin, vmax=vmax)
        ax.axhline(CZ_TARGET_Z_UM, color="gold", lw=1.0, ls="--",
                   label=f"prior (z = {CZ_TARGET_Z_UM:.0f} µm)")
        if cz_centroid is not None:
            ax.plot(x_um, _surface_z(cz_centroid, x_um, y_const),
                    color="tab:green", lw=1.3, label="CZ centroid")
        if cz_existing is not None:
            ax.plot(x_um, _surface_z(cz_existing, x_um, y_const),
                    color="tab:cyan", lw=1.3, label="CZ image_ceiling")
        z_poly = eval_polysurf(polyfit, x_um, np.full_like(x_um, y_const))
        ax.plot(x_um, z_poly, color="tab:red", lw=1.8,
                label=f"iter08 (tf={sel_tf})")
        ax.set_xlabel("x (µm)")
        ax.set_title(f"y = {y_const:.0f} µm")
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("z (µm)")
    fig.suptitle(
        f"{sid} — iter08 CZ (prior-selected TRANS_FRAC={sel_tf}, "
        f"poly-deg{POLY_DEGREE}, gate ±{GATE_UM:.0f} µm): "
        f"median_z = {sel_med:.1f} µm",
        fontsize=12, y=1.02)
    plt.tight_layout()
    out = OUT_FIG / f"iter08_cz_{sid}.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")

    np.savez(
        OUT_DATA / f"iter08_cz_transitions_{sid}.npz",
        xs_um=xs_um, ys_um=ys_um, zs_um=sel_zs, thrs=sel_thrs,
        selected_trans_frac=np.float32(sel_tf),
        prior_z_um=np.float32(CZ_TARGET_Z_UM),
        gate_um=np.float32(GATE_UM),
    )

    # Evaluate poly at a dense grid to report median surface z
    z_surf = eval_polysurf(polyfit, xs_um, ys_um)
    surf_med = float(np.nanmedian(z_surf))

    row = dict(
        subject=sid,
        cz_shape=f"({Z}, {Y}, {X})",
        prior_z_um=CZ_TARGET_Z_UM,
        selected_trans_frac=sel_tf,
        median_col_trans_z=sel_med,
        median_surface_z=surf_med,
        abs_dev_from_prior=abs(surf_med - CZ_TARGET_Z_UM),
        n_in_gate=n_in, n_valid=n_valid,
        gate_um=GATE_UM,
    )
    return row, sweep


def main():
    rows, sweeps = [], []
    for sid in SUBJECTS:
        row, sweep = render_subject(sid)
        sweeps.append(sweep)
        if row is not None:
            rows.append(row)

    df = pd.DataFrame(rows)
    out = OUT_DATA / "iter08_cz_selection.csv"
    df.to_csv(out, index=False)
    print("\n=== iter08 CZ selection ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:7.2f}"))
    print(f"wrote {out}")

    sweep_df = pd.concat(sweeps, ignore_index=True)
    out2 = OUT_DATA / "iter08_cz_sweep.csv"
    sweep_df.to_csv(out2, index=False)
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
