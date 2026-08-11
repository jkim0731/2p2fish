"""Stage B — sz estimator via iter-7 slab-wise side-view FFT-NCC sweep.

Algorithm (canonical iter-7, promoted 2026-05-02):
  For each candidate sz on a grid [1.5, 4.0] at 0.10 step, warp the
  binarized CZ ROI segmentation into the HCR crop using the locked-prior
  frame.  Take 5 × 100 µm x-slabs centred on lp.translation[2], compute
  x-MIP side views (z×y), find the best rigid translation via FFT
  cross-correlation, then compute Pearson NCC of the shifted CZ binary
  against raw HCR-488.  Score = mean NCC across 5 slabs; pick the peak.

Public API
----------
* :func:`get_sz`   — cache-aware entry point; returns dict with ``sz_best``
* HCR_LEVEL        — HCR pyramid level used internally (also imported by
                     run_iter7_slab_rigid_ncc.py via ``from sz_estimator import …``)
* _warp_cz_into_hcr_crop — private warp helper (also imported by the same script)

Cache schema (version 2):
    sz_best       — chosen sz in µm (float)
    sz_lp         — Stage A locked-prior sz
    ncc_peak      — mean NCC at the best sz
    ncc_median    — median NCC across the sweep
    peak_ratio    — ncc_peak / ncc_median
    slab_rows     — list of per-(sz, slab) NCC dicts, for diagnostics
    version       — 2
"""
from __future__ import annotations

import glob
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from scipy.ndimage import affine_transform, binary_dilation
from scipy.ndimage import shift as ndi_shift
from scipy.signal import fftconvolve
import tifffile

from autocoreg.io.hcr_image import depth_from_surface, load_hcr_volume
from autocoreg.initial_registration.locked_prior import LockedPriorWarmStart, compute_locked_prior_warm_start
from autocoreg.initial_registration.surface_registration import get_surface_registration
from autocoreg.initial_registration.surfaces import get_cz_surface_iter08
from autocoreg import config as _config

# ----------------------------------------------------------------------
# Cache
# ----------------------------------------------------------------------
CACHE_DIR = _config.SZ_CACHE_DIR
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_CACHE_VERSION = 2


def _cache_path(subject_id: str) -> Path:
    return CACHE_DIR / f"{subject_id}.json"


def _load_cached_sz(subject_id: str) -> dict | None:
    p = _cache_path(subject_id)
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    if d.get("version") != _CACHE_VERSION:
        return None
    return d


def _save_cached_sz(subject_id: str, payload: dict) -> None:
    """Atomic write so a crash mid-write doesn't corrupt the cache."""
    p = _cache_path(subject_id)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(p)


# ----------------------------------------------------------------------
# Sweep constants  (match the iter-7 research script exactly)
# ----------------------------------------------------------------------
SZ_GRID = np.arange(1.5, 4.01, 0.10)

HCR_LEVEL = 4               # pyramid level for the HCR 488 volume in the sweep
                             # (level-4 ≈ 4 µm/px xy, manageable in RAM)

SLAB_THICKNESS_UM = 100.0   # x-slab thickness
N_SLABS = 5                 # number of x-slabs
TOTAL_X_UM = N_SLABS * SLAB_THICKNESS_UM  # 500 µm

SHIFT_HALF_UM_Z = 60.0      # rigid-translation search half-window (z)
SHIFT_HALF_UM_Y = 40.0      # rigid-translation search half-window (y)
MASK_DILATION_PX = 5        # ring added to CZ binary when it is constant
                             # (constant binary → no variance for Pearson)


# ----------------------------------------------------------------------
# Private geometry helpers (also imported by run_iter7_slab_rigid_ncc.py)
# ----------------------------------------------------------------------

def _build_affine(
    lp: LockedPriorWarmStart, sz: float, tz_offset_um: float,
    cz_xy_um: float, cz_z_um: float,
    hcr_xy_um: float, hcr_z_um: float,
):
    """Inverse map from HCR-output-pixel-index → CZ-input-pixel-index.

    Forward LP (zyx-µm): pred = ((cz - src_mean) * scales) @ R.T + translation
    With sz overriding lp.scales[0] and a tz offset on lp.translation[0]:

        A = D_cz_inv · diag(1/scales') · R.T · D_hcr
        b = D_cz_inv · (src_mean - diag(1/scales') · R.T · translation')
    """
    R = lp.R
    src_mean = lp.src_mean
    scales = np.array([sz, lp.scales[1], lp.scales[2]], dtype=float)
    translation = lp.translation.copy()
    translation[0] = translation[0] + tz_offset_um

    D_hcr = np.diag([hcr_z_um, hcr_xy_um, hcr_xy_um])
    D_cz_inv = np.diag([1.0 / cz_z_um, 1.0 / cz_xy_um, 1.0 / cz_xy_um])
    inv_scales = 1.0 / scales

    A = D_cz_inv @ np.diag(inv_scales) @ R.T @ D_hcr
    b_um = src_mean - np.diag(inv_scales) @ R.T @ translation
    b = D_cz_inv @ b_um
    return A, b


def _warp_cz_into_hcr_crop(
    cz_vol: np.ndarray,
    lp: LockedPriorWarmStart, sz: float, tz_offset_um: float,
    cz_xy_um: float, cz_z_um: float,
    hcr_xy_um: float, hcr_z_um: float,
    crop_bbox_px: tuple[int, int, int, int],
    z_lo_um: float, z_hi_um: float,
) -> np.ndarray:
    """Warp CZ volume onto the cropped HCR slab grid for given (sz, tz_offset).

    crop_bbox_px = (y0, y1, x0, x1) in level-HCR_LEVEL HCR pixels.
    Returns shape (Z_out, Y_out, X_out) in HCR-level-HCR_LEVEL pixel grid.
    """
    A, b_root = _build_affine(
        lp, sz, tz_offset_um, cz_xy_um, cz_z_um, hcr_xy_um, hcr_z_um
    )
    y0, y1, x0, x1 = crop_bbox_px
    z0_idx = int(np.floor(z_lo_um / hcr_z_um))
    z1_idx = int(np.ceil(z_hi_um / hcr_z_um))
    out_shape = (z1_idx - z0_idx, y1 - y0, x1 - x0)

    # Output index (z_local, y_local, x_local) maps to absolute HCR pixel
    # (z0_idx + z_local, y0 + y_local, x0 + x_local); shift b accordingly.
    crop_origin = np.array([z0_idx, y0, x0], dtype=float)
    b = b_root + A @ crop_origin

    return affine_transform(
        cz_vol, A, offset=b, output_shape=out_shape,
        order=1, mode="constant", cval=0.0,
    )


# ----------------------------------------------------------------------
# CZ ROI segmentation loader
# ----------------------------------------------------------------------

def _find_cz_seg_tiff(sid: str) -> str:
    # Direct override: point at the exact segmentation_masks.tif (naming/structure-agnostic).
    override = os.environ.get("MFISH_CZ_SEG_TIF", "").strip()
    if override:
        if not Path(override).exists():
            raise FileNotFoundError(f"MFISH_CZ_SEG_TIF set but not found: {override}")
        return override
    # Else glob under DATA_ROOT (was hardcoded /data), first the channel_0_ref_0 layout then a
    # structure-agnostic recursive fallback (e.g. ophys-z-stacks seg has the tif at the stack root).
    root = str(_config.DATA_ROOT)
    for pat in (
        f"{root}/multiplane-ophys_{sid}_*-segmentation_*/channel_0_ref_0/segmentation_masks.tif",
        f"{root}/multiplane-ophys_{sid}_*-segmentation_*/**/segmentation_masks.tif",
    ):
        paths = sorted(glob.glob(pat, recursive=True))
        if paths:
            return paths[-1]
    raise FileNotFoundError(
        f"No CZ seg TIFF for {sid} under {root} (set MFISH_CZ_SEG_TIF to point at it directly)")


def _load_cz_seg_binary(sid: str) -> np.ndarray:
    """Load CZ ROI segmentation TIFF and binarize to float32."""
    seg = tifffile.imread(_find_cz_seg_tiff(sid))
    return (seg > 0).astype(np.float32)


# ----------------------------------------------------------------------
# FFT rigid-translation + NCC helpers
# ----------------------------------------------------------------------

def _fft_translation(
    cz_side: np.ndarray, hcr_side: np.ndarray,
    half_z: int, half_y: int,
) -> tuple[int, int]:
    """Integer (dz, dy) that maximises matched-filter score via FFT."""
    corr = fftconvolve(hcr_side, cz_side[::-1, ::-1], mode="same")
    cz, cy = corr.shape[0] // 2, corr.shape[1] // 2
    z0 = max(0, cz - half_z); z1 = min(corr.shape[0], cz + half_z + 1)
    y0 = max(0, cy - half_y); y1 = min(corr.shape[1], cy + half_y + 1)
    sub = corr[z0:z1, y0:y1]
    j = np.unravel_index(np.argmax(sub), sub.shape)
    dz = (z0 + j[0]) - cz
    dy = (y0 + j[1]) - cy
    return int(dz), int(dy)


def _ncc_under_mask(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Pearson NCC over mask pixels.  Dilates mask if CZ binary is constant."""
    if not mask.any():
        return float("nan")
    a_in = a[mask]; b_in = b[mask]
    if np.std(a_in) < 1e-12:
        mask = binary_dilation(mask, iterations=MASK_DILATION_PX)
        a_in = a[mask]; b_in = b[mask]
    if int(mask.sum()) < 100 or np.std(a_in) < 1e-12 or np.std(b_in) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a_in.astype(np.float64), b_in.astype(np.float64))[0, 1])


# ----------------------------------------------------------------------
# Core sweep
# ----------------------------------------------------------------------

def _run_slab_sweep(s, sz_grid: np.ndarray) -> tuple[list[dict], float, float]:
    """Run the iter-7 slab sweep; return (rows, sz_lp, cz_mean_depth_um).

    rows = list of per-(sz, slab) dicts with keys: sz, slab_idx, ncc.
    """
    sid = s.subject_id
    lp = compute_locked_prior_warm_start(s)
    reg = get_surface_registration(s)

    cz_bin = _load_cz_seg_binary(sid)
    hcr_vol, hcr_xy_um, hcr_z_um = load_hcr_volume(s, channel="488", level=HCR_LEVEL)
    hcr_vol = np.asarray(hcr_vol, dtype=np.float32)

    cz_xy_um = float(s.cz_xy_um)
    cz_z_um = float(s.cz_z_um)
    crop_bbox_px = tuple(reg["crop_bbox"])  # (y0, y1, x0, x1) level-HCR_LEVEL px
    y0, y1, x0, x1 = crop_bbox_px

    cz_surface = get_cz_surface_iter08(s)
    cz_mean_xyz_um = lp.src_mean[[2, 1, 0]]  # zyx → xyz for depth_from_surface
    cz_mean_depth_um = float(depth_from_surface(cz_mean_xyz_um[None, :], cz_surface)[0])
    sz_lp = float(lp.scales[0])

    # 5 × 100 µm x-slabs centred on lp.translation[2]
    x_center_um = float(lp.translation[2])
    slab_starts_um = x_center_um - TOTAL_X_UM / 2 + np.arange(N_SLABS) * SLAB_THICKNESS_UM
    slab_ranges_um = [(float(s0), float(s0 + SLAB_THICKNESS_UM)) for s0 in slab_starts_um]

    # z-extent covers all sz candidates in the sweep
    cz_z_extent_um = cz_bin.shape[0] * cz_z_um
    half_warped_z = float(sz_grid.max()) * cz_z_extent_um / 2
    z_lo_um = max(
        0.0,
        lp.translation[0] + (float(sz_grid.min()) - sz_lp) * cz_mean_depth_um
        - half_warped_z - 200.0,
    )
    z_hi_um = min(
        hcr_vol.shape[0] * hcr_z_um,
        lp.translation[0] + (float(sz_grid.max()) - sz_lp) * cz_mean_depth_um
        + half_warped_z + 200.0,
    )
    z0_idx = int(np.floor(z_lo_um / hcr_z_um))
    z1_idx = int(np.ceil(z_hi_um / hcr_z_um))
    hcr_target = hcr_vol[z0_idx:z1_idx, y0:y1, x0:x1].astype(np.float32)

    # x-slab indices in the cropped HCR frame
    slab_idx_ranges = []
    for xa_um, xb_um in slab_ranges_um:
        xa_local = xa_um - x0 * hcr_xy_um
        xb_local = xb_um - x0 * hcr_xy_um
        xa_idx = max(0, int(np.floor(xa_local / hcr_xy_um)))
        xb_idx = min(hcr_target.shape[2], int(np.ceil(xb_local / hcr_xy_um)))
        slab_idx_ranges.append((xa_idx, xb_idx))

    print(
        f"  {sid}: HCR target {hcr_target.shape}  "
        f"sz_lp={sz_lp:.3f}  cz_depth={cz_mean_depth_um:.1f}µm  "
        f"slabs x∈[{slab_ranges_um[0][0]:.0f},{slab_ranges_um[-1][1]:.0f}]µm",
        flush=True,
    )

    half_z_steps = int(round(SHIFT_HALF_UM_Z / hcr_z_um))
    half_y_steps = int(round(SHIFT_HALF_UM_Y / hcr_xy_um))

    rows = []
    for sz in sz_grid:
        tz_offset = (float(sz) - sz_lp) * cz_mean_depth_um
        warped_cz_bin = _warp_cz_into_hcr_crop(
            cz_bin, lp, float(sz), tz_offset_um=tz_offset,
            cz_xy_um=cz_xy_um, cz_z_um=cz_z_um,
            hcr_xy_um=hcr_xy_um, hcr_z_um=hcr_z_um,
            crop_bbox_px=crop_bbox_px,
            z_lo_um=z_lo_um, z_hi_um=z_hi_um,
        )

        for slab_idx, (xa, xb) in enumerate(slab_idx_ranges):
            if xb <= xa:
                rows.append({"sz": float(sz), "slab_idx": slab_idx, "ncc": float("nan")})
                continue

            cz_side_bin = (warped_cz_bin[:, :, xa:xb].max(axis=2) > 0.5).astype(np.float32)
            hcr_side = hcr_target[:, :, xa:xb].max(axis=2)

            if cz_side_bin.sum() < 50:
                rows.append({"sz": float(sz), "slab_idx": slab_idx, "ncc": float("nan")})
                continue

            dz_pix, dy_pix = _fft_translation(cz_side_bin, hcr_side, half_z_steps, half_y_steps)
            cz_shifted = ndi_shift(cz_side_bin, (dz_pix, dy_pix), order=0, cval=0.0)
            mask = cz_shifted > 0.5
            ncc = _ncc_under_mask(cz_shifted, hcr_side, mask)
            rows.append({"sz": float(sz), "slab_idx": slab_idx, "ncc": ncc})

    return rows, sz_lp, cz_mean_depth_um


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def get_sz(s, *, use_cache: bool = True, write_cache: bool = True) -> dict:
    """Cache-aware iter-7 slab-NCC sz estimation.

    Returns a dict with keys:
        ``sz_best``    — chosen sz in µm (float)
        ``sz_lp``      — Stage A locked-prior sz
        ``ncc_peak``   — mean NCC at sz_best
        ``ncc_median`` — median mean-NCC across the sweep
        ``peak_ratio`` — ncc_peak / ncc_median
        ``slab_rows``  — per-(sz, slab) NCC list (for diagnostics)
        ``version``    — cache version (2)
    """
    sid = s.subject_id
    if use_cache:
        cached = _load_cached_sz(sid)
        if cached is not None:
            return cached

    rows, sz_lp, cz_mean_depth_um = _run_slab_sweep(s, SZ_GRID)

    # Aggregate: mean NCC per sz (ignoring nan slabs)
    sz_vals = np.array(sorted(set(r["sz"] for r in rows)))
    mean_ncc = np.array([
        np.nanmean([r["ncc"] for r in rows if r["sz"] == sz]) for sz in sz_vals
    ])

    finite = np.isfinite(mean_ncc)
    print(
        f"  {sid}: {int(finite.sum())}/{len(sz_vals)} sz candidates have finite NCC  "
        f"ncc range [{np.nanmin(mean_ncc):.3f}, {np.nanmax(mean_ncc):.3f}]",
        flush=True,
    )

    if finite.sum() < 3:
        raise RuntimeError(
            f"[sz_estimator] {sid}: fewer than 3 finite NCC values — sweep failed"
        )

    peak_i = int(np.nanargmax(mean_ncc))
    sz_best = float(sz_vals[peak_i])
    ncc_peak = float(mean_ncc[peak_i])
    ncc_median = float(np.nanmedian(mean_ncc))
    peak_ratio = ncc_peak / max(abs(ncc_median), 1e-9)

    print(
        f"  {sid}: sz_best={sz_best:.2f}  ncc_peak={ncc_peak:.4f}  "
        f"ncc_median={ncc_median:.4f}  peak_ratio={peak_ratio:.3f}",
        flush=True,
    )

    payload = {
        "version": _CACHE_VERSION,
        "subject_id": sid,
        "sz_best": sz_best,
        "sz_lp": sz_lp,
        "ncc_peak": ncc_peak,
        "ncc_median": ncc_median,
        "peak_ratio": peak_ratio,
        "slab_rows": rows,
    }
    if write_cache:
        _save_cached_sz(sid, payload)
    return payload
