"""Promoted v2 surface registration — main-pipeline entry point.

Session 08 (``08_surface_vascular_match``) compared four 2-D top-slab
registration strategies (rigid; rigid+affine; rigid+affine+PWR 3×3;
rigid+affine+PWR 4×4) against two HCR targets (raw 488 MIP intensity,
zarr-derived GFP+ ROI MIP).  All four were run with overlap settings
optimised in ``pwr_overlap.py`` (3×3 → 0.30 overlap, 4×4 → 0.0 overlap)
and the best-of-four was selected per subject.

This module promotes that recipe to the main pipeline:

* **CZ source**: binary CZ-ROI top-slab MIP (0–80 µm beneath CZ pia,
  fill-then-MIP of the level-0 segmentation outline) on the HCR-level-4
  pixel grid (warm-started by 180° rotation + ``sxy`` rescale).
* **HCR target**: raw 488 MIP, depth slab 0–150 µm beneath HCR pia
  (no binarisation in the comparison metric).

  MIP slab thickness was promoted 2026-06-04 from 50/100 to **80/150 µm** (a
  denser registration MIP lands for thin-HCR subjects like 782149).  ``sxy``
  base now comes from the min-rule 2× ¼-FOV estimator
  (``roi_area_sxy.estimate_sxy_min_rule``), also promoted 2026-06-04.
* **Stage 1 rigid bootstrap**: binarised (watershed) 488 MIP, used only
  to seed the rigid θ + tx/ty sweep so the optimisation is stable when
  the initial overlap is poor.
* **Methods scored** (Pearson NCC on the M1-bbox + 15 % crop,
  warped-CZ vs raw-488):
    M1 rigid                  — translation + rotation only
    M2 rig + affine           — 4-DOF similarity, scale ±15 %, θ ±15°
    M4a rig + aff + PWR 3×3   — Suite2p-style PWR seeded from affine,
                                30 % block overlap
    M4b rig + aff + PWR 4×4   — same, 4×4 grid, 0 % overlap
* **Selection**: the method with highest Pearson NCC.

Public API
----------
* :func:`compute_surface_registration`        — run from scratch
* :func:`get_surface_registration`            — cache-aware accessor
* :func:`apply_registration`                  — re-render warped CZ
                                                 binary at any time
* :func:`build_main_registration_store`       — batch all 6 benchmark
                                                 subjects, write CSV

Cache
-----
JSON parameters under
``/root/capsule/code/dev_code/cached_surface_registration/<sid>.json``.
Warp arrays themselves are NOT cached — they re-render in <1 s from the
parameters by calling :func:`apply_registration`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import affine_transform

# Helpers vendored from session 08 — now in-package.
from .compare_binarization import variant_watershed
from .register_binary import (
    cz_binary_top_mip,
    hcr488_top_mip,
    stage1_rigid,
    stage2_affine,
    warm_start_cz_binary,
)
# NOTE: get_sxy_with_fallback (from register_binary) is intentionally NOT imported
# here. compute_surface_registration now uses estimate_sxy_min_rule from
# roi_area_sxy (PROMOTED 2026-06-04) — GT-free, min-rule 2× ¼-FOV; it recovers
# thin-HCR 782149 where the old full-span/slab-auto estimators collapsed.  The
# old fallback had a landmark_gt_fallback that was a GT-leak for 782149 (see
# project_782149_sxy_gt_leak memory).
from .register_nonrigid_variants import (
    nonrigid_piecewise_rigid,
)

from .surfaces_iter08 import (
    get_cz_surface_iter08,
    get_hcr_top_surface_iter07,
)

from . import config as _config
CACHE_DIR = _config.SURFACE_REGISTRATION_CACHE_DIR
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Default protocol constants.
# Registration MIP slab thickness (PROMOTED 2026-06-04): CZ 0–80 µm, HCR 0–150 µm.
# A denser (thicker) MIP packs more 488 signal into the registration target, so
# the rigid/affine/PWR fit lands for thin-HCR subjects whose top slab is sparse
# (e.g. 782149: at the old 50/100 the 488-MIP NCC was ~0.115, lowest of the panel,
# and the matcher collapsed to ~0 matches). 80/150 keeps the same 1:~1.9 CZ:HCR
# axial ratio while giving the optimiser enough structure to converge.
# Supersedes the prior 50/100 MIP. See docs/08 Automated surface registration.md.
CZ_SLAB = (0.0, 80.0)
HCR_SLAB = (0.0, 150.0)
HCR_LEVEL = 4
SCALE_BOUNDS = (0.85, 1.15)
THETA_BOUND_DEG = 15.0
PWR_MAX_SHIFT_PX = 16
PWR_SMOOTH_SIGMA_PX = 8.0
CROP_MARGIN_FRAC = 0.15
WARP_BIN_THR = 0.5

METHOD_KEYS = ("rigid", "affine", "pwr3x3", "pwr4x4")
METHOD_LABELS = {
    "rigid":  "M1 rigid",
    "affine": "M2 rig+aff",
    "pwr3x3": "M4a aff+PWR 3×3",
    "pwr4x4": "M4b aff+PWR 4×4",
}


# --------------------------------------------------------------
# NCC + bbox helpers
# --------------------------------------------------------------
def ncc_pearson(warped: np.ndarray, hcr_target: np.ndarray,
                mask: np.ndarray | None = None) -> float:
    a = np.asarray(warped, dtype=np.float64)
    b = np.asarray(hcr_target, dtype=np.float64)
    if mask is None:
        mask = np.ones(b.shape, dtype=bool)
    if mask.sum() < 50:
        return -1.0
    av = a[mask]; bv = b[mask]
    av = av - av.mean(); bv = bv - bv.mean()
    den = av.std() * bv.std()
    if den <= 1e-12:
        return -1.0
    return float((av * bv).mean() / den)


def _rigid_to_M_off(rigid: dict, template_shape, template_shape_rot):
    th = np.deg2rad(rigid["theta"])
    c, s = np.cos(th), np.sin(th)
    M0 = np.array([[c, s], [-s, c]], dtype=np.float64)
    H_t, W_t = template_shape
    rh, rw = template_shape_rot
    cy_out = rigid["oy"] + rh / 2.0
    cx_out = rigid["ox"] + rw / 2.0
    cy_in = H_t / 2.0; cx_in = W_t / 2.0
    off0 = np.array([cy_in, cx_in]) - M0 @ np.array([cy_out, cx_out])
    return M0, off0


def _bbox_with_margin(mask_bool: np.ndarray, margin_frac: float, shape):
    H, W = shape
    if not mask_bool.any():
        return 0, H, 0, W
    rows = np.any(mask_bool, axis=1)
    cols = np.any(mask_bool, axis=0)
    y0, y1 = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
    x0, x1 = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])
    h = y1 - y0 + 1; w = x1 - x0 + 1
    my = int(round(h * margin_frac)); mx = int(round(w * margin_frac))
    return (max(0, y0 - my), min(H, y1 + 1 + my),
            max(0, x0 - mx), min(W, x1 + 1 + mx))


# --------------------------------------------------------------
# Top-level registration entry
# --------------------------------------------------------------
def compute_surface_registration(
    s,
    *,
    hcr_top_surface: dict | None = None,
    cz_surface: dict | None = None,
    sxy: float | None = None,
    sxy_source: str = "",
    return_warps: bool = False,
    verbose: bool = False,
) -> dict:
    """Run M1/M2/M4a/M4b on ``s`` and return the result, including the
    selected best method.  Surfaces are loaded via the iter07/08
    promoted accessors when not supplied.

    Parameters
    ----------
    s
        ``SubjectData`` from ``benchmark_data_loader.load_subject``.
    hcr_top_surface, cz_surface
        Optional pre-fit surfaces.  Defaults call
        :func:`get_hcr_top_surface_iter07` / :func:`get_cz_surface_iter08`.
    sxy
        Anisotropy scale (CZ-µm → HCR-µm).  When ``None``,
        :func:`roi_area_sxy.estimate_sxy_min_rule` is called (PRODUCTION,
        promoted 2026-06-04): the min-rule 2× ¼-FOV estimator (HCR slab =
        min(p99 HCR GFP+∩ok∩¼-FOV depth, 2·p99 CZ depth); CZ slab = half that;
        sxy = sqrt(median HCR max-xsection / median CZ max-xsection)).  GT-free;
        recovers thin-HCR 782149.  Grid-search fallback at
        ``SXY_GRID_SEARCH_OFFSETS`` for new subjects that roam.  Never falls
        back to GT.
    sxy_source
        Provenance label propagated into the result dict.
    return_warps
        If True, include the rendered warped-CZ arrays for every
        method.  Defaults to False to keep memory bounded.

    Returns
    -------
    dict
        Includes ``best_method``, per-method NCCs and parameters,
        crop bbox, and the rigid-bootstrap target metadata.
    """
    sid = s.subject_id
    if hcr_top_surface is None:
        hcr_top_surface = get_hcr_top_surface_iter07(s, level=HCR_LEVEL)
    if cz_surface is None:
        cz_surface = get_cz_surface_iter08(s)
    if sxy is None:
        # PRODUCTION base sxy (PROMOTED 2026-06-04): min-rule, 2× heuristic, ¼-FOV.
        # GT-free. Rule (see roi_area_sxy.estimate_sxy_min_rule for the full
        # derivation): zstack_thickness = p99(CZ depth); hcr_slab = min(p99(HCR
        # GFP+∩ok∩¼-FOV depth), 2·zstack_thickness); cz_slab = hcr_slab/2 (CZ slab
        # is HALF the HCR slab — axial ~2× expansion, capping CZ shallower raises
        # sxy); sxy = sqrt(median max_xsection HCR[0,hcr_slab] / median CZ[0,cz_slab]).
        # The 2× is a heuristic, NOT the measured sz (sz needs a pose → circular).
        # Recovers thin-HCR 782149 (→1.7336) where the prior full-span/slab-auto
        # estimator collapsed.  Never falls back to GT.
        #
        # FALLBACK for a NEW subject whose registration roams / matcher collapses
        # at this sxy: grid-search the base sxy at roi_area_sxy.SXY_GRID_SEARCH_OFFSETS
        # and pick the pose that lands (highest soma-print mutual-best / lowest
        # rigid off-centre). GT-free.
        from .roi_area_sxy import estimate_sxy_min_rule
        sxy = float(estimate_sxy_min_rule(sid)["sxy_median"])
        sxy_source = sxy_source or "min_rule_2x_quarterfov"

    # Targets: raw 488 MIP for *comparison*, watershed binary for
    # *rigid bootstrap* (more robust when initial overlap is small).
    raw488_full, hcr_xy_um = hcr488_top_mip(
        s, hcr_top_surface, *HCR_SLAB)
    raw488_full = np.nan_to_num(raw488_full.astype(np.float32), nan=0.0)
    rigid_target = variant_watershed(raw488_full, hcr_xy_um)

    cz_bin = cz_binary_top_mip(s, cz_surface, *CZ_SLAB)
    cz_bin_warm = warm_start_cz_binary(
        cz_bin, cz_xy_um=float(s.cz_xy_um), hcr_xy_um=hcr_xy_um, sxy=sxy,
    )

    # ---------- Stage 1: rigid (binary bootstrap target) -----------
    rigid = stage1_rigid(cz_bin_warm, rigid_target)
    M0, off0 = _rigid_to_M_off(
        rigid, template_shape=cz_bin_warm.shape,
        template_shape_rot=rigid["template_shape"],
    )
    rigid_warp_full = affine_transform(
        cz_bin_warm.astype(np.float32), M0, offset=off0,
        output_shape=raw488_full.shape, order=1, mode="constant", cval=0.0,
    )
    fg = rigid_warp_full > WARP_BIN_THR
    y0, y1, x0, x1 = _bbox_with_margin(
        fg, CROP_MARGIN_FRAC, raw488_full.shape)
    hcr_crop = raw488_full[y0:y1, x0:x1].astype(np.float32)
    rigid_warp = rigid_warp_full[y0:y1, x0:x1]
    off0_c = off0 + M0 @ np.array([y0, x0], dtype=np.float64)
    rigid_ncc = ncc_pearson(rigid_warp, hcr_crop)

    if verbose:
        print(f"  {sid}: rigid θ={rigid['theta']:.2f}°  NCC={rigid_ncc:.3f}",
              flush=True)

    # ---------- Stage 2: 4-DOF similarity --------------------------
    affine = stage2_affine(
        cz_bin_warm, hcr_crop, M0, off0_c,
        scale_bounds=SCALE_BOUNDS, theta_bounds_deg=THETA_BOUND_DEG,
    )
    affine_warp = affine_transform(
        cz_bin_warm.astype(np.float32), affine["M"], offset=affine["offset"],
        output_shape=hcr_crop.shape, order=1, mode="constant", cval=0.0,
    )
    affine_ncc = ncc_pearson(affine_warp, hcr_crop)
    if verbose:
        print(f"  {sid}: affine NCC={affine_ncc:.3f}", flush=True)

    # ---------- Stage 3: PWR seeded from AFFINE warp ---------------
    pwr3 = nonrigid_piecewise_rigid(
        affine_warp, hcr_crop, n_blocks=3,
        max_shift_px=PWR_MAX_SHIFT_PX,
        interp_smooth_sigma_px=PWR_SMOOTH_SIGMA_PX,
        overlap_frac=0.30,
    )
    pwr3_ncc = ncc_pearson(pwr3["warped"], hcr_crop)

    pwr4 = nonrigid_piecewise_rigid(
        affine_warp, hcr_crop, n_blocks=4,
        max_shift_px=PWR_MAX_SHIFT_PX,
        interp_smooth_sigma_px=PWR_SMOOTH_SIGMA_PX,
        overlap_frac=0.0,
    )
    pwr4_ncc = ncc_pearson(pwr4["warped"], hcr_crop)
    if verbose:
        print(f"  {sid}: PWR3 NCC={pwr3_ncc:.3f}  PWR4 NCC={pwr4_ncc:.3f}",
              flush=True)

    # ---------- Pack -----------------------------------------------
    methods = {
        "rigid": dict(
            ncc=float(rigid_ncc),
            theta_deg=float(rigid["theta"]),
            M=np.asarray(M0, dtype=float).tolist(),
            offset=np.asarray(off0_c, dtype=float).tolist(),
        ),
        "affine": dict(
            ncc=float(affine_ncc),
            params=[float(v) for v in affine["params"]],
            M=np.asarray(affine["M"], dtype=float).tolist(),
            offset=np.asarray(affine["offset"], dtype=float).tolist(),
        ),
        "pwr3x3": dict(
            ncc=float(pwr3_ncc),
            n_blocks=3, overlap_frac=0.30,
            max_shift_px=int(PWR_MAX_SHIFT_PX),
            interp_smooth_sigma_px=float(PWR_SMOOTH_SIGMA_PX),
            shifts=np.asarray(pwr3["shifts"], dtype=float).tolist(),
            block_nccs=np.asarray(pwr3["block_nccs"], dtype=float).tolist(),
            accepted=np.asarray(pwr3["accepted"]).astype(int).tolist(),
            n_accepted=int(pwr3.get("n_accepted", 0)),
            df_max_dy_px=float(pwr3.get("df_max_dy_px", 0.0)),
            df_max_dx_px=float(pwr3.get("df_max_dx_px", 0.0)),
            seed_method="affine",
        ),
        "pwr4x4": dict(
            ncc=float(pwr4_ncc),
            n_blocks=4, overlap_frac=0.0,
            max_shift_px=int(PWR_MAX_SHIFT_PX),
            interp_smooth_sigma_px=float(PWR_SMOOTH_SIGMA_PX),
            shifts=np.asarray(pwr4["shifts"], dtype=float).tolist(),
            block_nccs=np.asarray(pwr4["block_nccs"], dtype=float).tolist(),
            accepted=np.asarray(pwr4["accepted"]).astype(int).tolist(),
            n_accepted=int(pwr4.get("n_accepted", 0)),
            df_max_dy_px=float(pwr4.get("df_max_dy_px", 0.0)),
            df_max_dx_px=float(pwr4.get("df_max_dx_px", 0.0)),
            seed_method="affine",
        ),
    }
    best = max(METHOD_KEYS, key=lambda k: methods[k]["ncc"])

    out = dict(
        subject_id=sid,
        method="surface_registration_v2",
        best_method=best,
        best_ncc=float(methods[best]["ncc"]),
        sxy=float(sxy),
        sxy_source=sxy_source,
        hcr_xy_um=float(hcr_xy_um),
        cz_xy_um=float(s.cz_xy_um),
        cz_z_um=float(s.cz_z_um),
        hcr_z_um=float(s.hcr_z_um),
        cz_slab_um=list(CZ_SLAB),
        hcr_slab_um=list(HCR_SLAB),
        hcr_full_shape=list(raw488_full.shape),
        crop_bbox=[int(y0), int(y1), int(x0), int(x1)],
        crop_shape=list(hcr_crop.shape),
        cz_warm_shape=list(cz_bin_warm.shape),
        warp_bin_thr=float(WARP_BIN_THR),
        scale_bounds=list(SCALE_BOUNDS),
        theta_bound_deg=float(THETA_BOUND_DEG),
        crop_margin_frac=float(CROP_MARGIN_FRAC),
        methods=methods,
    )
    if return_warps:
        out["warps"] = dict(
            rigid=rigid_warp.astype(np.float32),
            affine=affine_warp.astype(np.float32),
            pwr3x3=pwr3["warped"].astype(np.float32),
            pwr4x4=pwr4["warped"].astype(np.float32),
            hcr_target_crop=hcr_crop.astype(np.float32),
        )
    return out


# --------------------------------------------------------------
# Apply (re-render) a cached registration
# --------------------------------------------------------------
def apply_registration(
    s,
    reg: dict,
    *,
    cz_surface: dict | None = None,
    method: str | None = None,
) -> dict:
    """Re-render the warped CZ binary on the cropped HCR-level-4 grid
    using the parameters in ``reg``.  ``method`` defaults to
    ``reg['best_method']``.  Returns
    ``{"warped": ..., "hcr_target_crop": ..., "method": ...}`` —
    suitable for plotting or downstream metric checks.
    """
    if cz_surface is None:
        cz_surface = get_cz_surface_iter08(s)
    method = method or reg["best_method"]

    raw488_full, _ = hcr488_top_mip(s, get_hcr_top_surface_iter07(s, level=HCR_LEVEL),
                                    *HCR_SLAB)
    raw488_full = np.nan_to_num(raw488_full.astype(np.float32), nan=0.0)
    y0, y1, x0, x1 = reg["crop_bbox"]
    hcr_crop = raw488_full[y0:y1, x0:x1].astype(np.float32)

    cz_bin = cz_binary_top_mip(s, cz_surface, *CZ_SLAB)
    cz_bin_warm = warm_start_cz_binary(
        cz_bin, cz_xy_um=float(s.cz_xy_um),
        hcr_xy_um=float(reg["hcr_xy_um"]), sxy=float(reg["sxy"]),
    )

    if method in ("rigid", "affine"):
        m = reg["methods"][method]
        M = np.asarray(m["M"], dtype=float)
        off = np.asarray(m["offset"], dtype=float)
        warped = affine_transform(
            cz_bin_warm.astype(np.float32), M, offset=off,
            output_shape=hcr_crop.shape, order=1, mode="constant", cval=0.0,
        )
    else:
        # PWR — re-run from the affine warp to re-derive the displacement
        # field deterministically (block params live in the cached dict).
        m_aff = reg["methods"]["affine"]
        M_aff = np.asarray(m_aff["M"], dtype=float)
        off_aff = np.asarray(m_aff["offset"], dtype=float)
        affine_warp = affine_transform(
            cz_bin_warm.astype(np.float32), M_aff, offset=off_aff,
            output_shape=hcr_crop.shape, order=1, mode="constant", cval=0.0,
        )
        m = reg["methods"][method]
        pwr = nonrigid_piecewise_rigid(
            affine_warp, hcr_crop,
            n_blocks=int(m["n_blocks"]),
            max_shift_px=int(m["max_shift_px"]),
            interp_smooth_sigma_px=float(m["interp_smooth_sigma_px"]),
            overlap_frac=float(m["overlap_frac"]),
        )
        warped = pwr["warped"]
    return dict(warped=warped.astype(np.float32),
                hcr_target_crop=hcr_crop, method=method)


# --------------------------------------------------------------
# JSON cache I/O
# --------------------------------------------------------------
def _cache_path(subject_id: str) -> Path:
    return CACHE_DIR / f"{subject_id}.json"


def save_cached_registration(subject_id: str, reg: dict | None):
    if reg is None:
        return
    path = _cache_path(subject_id)
    keep = {k: v for k, v in reg.items() if k != "warps"}
    path.write_text(json.dumps(keep, indent=2, default=float))


def load_cached_registration(subject_id: str) -> dict | None:
    path = _cache_path(subject_id)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def get_surface_registration(
    s,
    *,
    use_cache: bool = True,
    write_cache: bool = True,
    verbose: bool = False,
) -> dict:
    """Cache-aware entry point.  Returns the v2 registration dict for
    subject ``s``, computing it if not cached.
    """
    sid = s.subject_id
    if use_cache:
        cached = load_cached_registration(sid)
        if cached is not None:
            return cached
    reg = compute_surface_registration(s, verbose=verbose)
    if write_cache:
        save_cached_registration(sid, reg)
    return reg


# --------------------------------------------------------------
# Batch entry — build the main pipeline registration store
# --------------------------------------------------------------
SUBJECTS_DEFAULT = ("755252", "767018", "767022", "782149", "788406", "790322")


def build_main_registration_store(subjects=SUBJECTS_DEFAULT):
    from .benchmark_data_loader import load_subject  # type: ignore
    rows = []
    for sid in subjects:
        print(f"=== {sid} ===", flush=True)
        s = load_subject(sid)
        reg = compute_surface_registration(s, verbose=True)
        save_cached_registration(sid, reg)
        rows.append({
            "subject": sid,
            "best_method": reg["best_method"],
            "best_ncc": reg["best_ncc"],
            "rigid_ncc": reg["methods"]["rigid"]["ncc"],
            "affine_ncc": reg["methods"]["affine"]["ncc"],
            "pwr3x3_ncc": reg["methods"]["pwr3x3"]["ncc"],
            "pwr4x4_ncc": reg["methods"]["pwr4x4"]["ncc"],
            "sxy": reg["sxy"],
            "sxy_source": reg["sxy_source"],
            "hcr_xy_um": reg["hcr_xy_um"],
            "crop_h": reg["crop_shape"][0],
            "crop_w": reg["crop_shape"][1],
        })
    df = pd.DataFrame(rows)
    csv_path = CACHE_DIR / "main_registration_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nwrote {csv_path}")
    return df


if __name__ == "__main__":
    build_main_registration_store()
