"""Candidate-pool preparation for the soma-print matcher.

Builds the CZ / HCR candidate pools in the locked-prior frame: GFP+∩ok HCR
filtering, Z-density-intersection band (pia-anchored), CZ in-band trim, and the
adaptive candidate radius.  Extracted from the original refined protocol (now
archived at autocoreg.archive.refined_benchmark).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from autocoreg.io.inputs import subject_inputs
from autocoreg.initial_registration.surfaces import get_hcr_top_surface_iter07

Z_DENSITY_PEAK_FRAC = 0.10
XY_MARGIN_UM = 30.0


def z_density_intersection_bounds(
    cz_lp_um: np.ndarray, hcr_um: np.ndarray, hcr_keep: np.ndarray,
    z_bin_um: float = 30.0,
    peak_frac: float = Z_DENSITY_PEAK_FRAC,
) -> tuple[float, float, dict]:
    """Return (z_lo, z_hi) where both CZ and HCR-OK densities exceed
    ``peak_frac × their peak``.

    Falls back to cz_lp.min/max if no overlap region is found.
    """
    cz_z = cz_lp_um[:, 0]
    hcr_z = hcr_um[hcr_keep, 0]
    # Build common bins spanning union of ranges
    z_lo = min(float(cz_z.min()), float(hcr_z.min()))
    z_hi = max(float(cz_z.max()), float(hcr_z.max()))
    bins = np.arange(z_lo - z_bin_um, z_hi + 2 * z_bin_um, z_bin_um)
    cz_hist, _ = np.histogram(cz_z, bins=bins)
    hcr_hist, _ = np.histogram(hcr_z, bins=bins)
    cz_thresh = max(1, cz_hist.max() * peak_frac)
    hcr_thresh = max(1, hcr_hist.max() * peak_frac)
    both = (cz_hist >= cz_thresh) & (hcr_hist >= hcr_thresh)
    if both.any():
        idxs = np.flatnonzero(both)
        z_lo_int = float(bins[idxs[0]])
        z_hi_int = float(bins[idxs[-1] + 1])
    else:
        z_lo_int = float(cz_z.min())
        z_hi_int = float(cz_z.max())
    return z_lo_int, z_hi_int, dict(
        cz_peak=int(cz_hist.max()), hcr_peak=int(hcr_hist.max()),
        z_bin_um=z_bin_um, peak_frac=peak_frac,
    )


def adaptive_r_cand(
    cz_lp_um: np.ndarray, hcr_pool_zyx: np.ndarray,
    floor_um: float = 50.0, mult: float = 2.0,
) -> tuple[float, dict]:
    if hcr_pool_zyx.shape[0] == 0:
        return floor_um, dict(nn_p50=float("nan"), nn_p90=float("nan"),
                              nn_p99=float("nan"))
    tree = cKDTree(hcr_pool_zyx)
    nn_d, _ = tree.query(cz_lp_um, k=1)
    p50, p90, p99 = np.percentile(nn_d, [50, 90, 99])
    R_cand = max(floor_um, float(mult * p90))
    return R_cand, dict(nn_p50=float(p50), nn_p90=float(p90),
                        nn_p99=float(p99))


def hcr_pia_z_over_region(s, xy_pts: np.ndarray) -> float:
    """Min HCR-pia z over the given xy points (HCR µm coords).
    Returns the smallest pia z — guaranteeing all points lie at or below pia.
    """
    surf = get_hcr_top_surface_iter07(s)
    if surf is None:
        return float("nan")
    a, b, c = surf["a"], surf["b"], surf["c"]
    p, q, r = surf.get("p", 0.0), surf.get("q", 0.0), surf.get("r", 0.0)
    x = xy_pts[:, 1]; y = xy_pts[:, 0]
    pia_z = a * x + b * y + c + p * x * x + q * x * y + r * y * y
    return float(np.min(pia_z))


def prepare_subject(sid: str, sz_pins: dict) -> dict:
    inp = subject_inputs(sid, sz_pins=sz_pins)
    cz_lp = inp.cz_lp_um.copy()
    ok_set = inp.gfp_ids & inp.ok_ids
    in_ok = np.array([int(h) in ok_set for h in inp.hcr_ids])

    # Refinement 2: Z-density intersection
    z_lo_density, z_hi_density, z_meta = z_density_intersection_bounds(
        cz_lp, inp.hcr_um, in_ok,
    )
    # Refinement 2b: anchor z_lo at the pia surface (per-xy minimum over
    # the working in-plane region).  Ensures the pia — where surface
    # registration is the warm-up anchor — is always inside the bounds.
    pia_z_min = hcr_pia_z_over_region(inp.s, cz_lp[:, 1:])
    z_lo = min(z_lo_density, pia_z_min) if np.isfinite(pia_z_min) else z_lo_density
    z_hi = z_hi_density
    z_meta = dict(z_meta, pia_z_min=float(pia_z_min),
                  z_lo_density=float(z_lo_density), z_lo_final=float(z_lo))
    # XY bbox from CZ
    lo_xy = cz_lp.min(axis=0)[1:] - XY_MARGIN_UM
    hi_xy = cz_lp.max(axis=0)[1:] + XY_MARGIN_UM
    in_bbox = (
        (inp.hcr_um[:, 0] >= z_lo) & (inp.hcr_um[:, 0] <= z_hi)
        & (inp.hcr_um[:, 1] >= lo_xy[0]) & (inp.hcr_um[:, 1] <= hi_xy[0])
        & (inp.hcr_um[:, 2] >= lo_xy[1]) & (inp.hcr_um[:, 2] <= hi_xy[1])
    )
    # Also drop CZ cells outside the intersection Z band
    cz_in_band = (cz_lp[:, 0] >= z_lo) & (cz_lp[:, 0] <= z_hi)

    keep = in_ok & in_bbox
    hcr_pool_zyx = inp.hcr_um[keep]
    hcr_pool_ids = inp.hcr_ids[keep].astype(int)
    hcr_id_to_row = {int(h): r for r, h in enumerate(hcr_pool_ids)}

    cz_keep_ids = inp.cz_ids[cz_in_band]
    cz_pool_zyx = cz_lp[cz_in_band]
    cz_id_to_row = {int(c): r for r, c in enumerate(cz_keep_ids)}

    # IN-POOL GT only (used for per-match is_gt logging and descriptor evaluation).
    # These are NOT the scoring denominator — for headline recall/precision use
    # scoring_gt(inp) from _data.py, which is pose-independent (HCR GFP+∩ok,
    # no spatial/pool filter).
    gt_pairs = []
    for c, h in zip(inp.coreg["cz_id"].astype(int), inp.coreg["hcr_id"].astype(int)):
        if int(c) in cz_id_to_row and int(h) in hcr_id_to_row:
            gt_pairs.append((int(c), int(h)))
    gt_cz_rows = np.array([cz_id_to_row[c] for c, _ in gt_pairs])
    gt_hcr_rows = np.array([hcr_id_to_row[h] for _, h in gt_pairs])

    # Refinement 1: adaptive R_cand
    R_cand, nn_meta = adaptive_r_cand(cz_pool_zyx, hcr_pool_zyx)

    return dict(
        sid=sid, inp=inp,
        z_bounds=(z_lo, z_hi),
        z_meta=z_meta,
        cz_pool_zyx=cz_pool_zyx,
        cz_pool_ids=cz_keep_ids,
        hcr_pool_zyx=hcr_pool_zyx,
        hcr_pool_ids=hcr_pool_ids,
        cz_id_to_row=cz_id_to_row,
        hcr_id_to_row=hcr_id_to_row,
        gt_pairs=gt_pairs,
        gt_cz_rows=gt_cz_rows,
        gt_hcr_rows=gt_hcr_rows,
        R_cand_um=R_cand,
        nn_meta=nn_meta,
    )
