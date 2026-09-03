"""Adaptive sz estimation for pan-neuronal coregistration.

Two-pass protocol (replaces the NCC sweep for pan-neuronal / pan-excitatory):

  1. L1 thickness ratio seed  →  sz_seed = hcr_l1_um / cz_l1_um
  2. Pass 1 matcher at sz_seed
  3. OLS empirical sz from pass-1 matched pairs  →  sz_ols
  4. Pass 2 matcher at sz_ols  (caller's responsibility)

The NCC sweep (axial_scale.get_sz) is unreliable on pan-neuronal data because
the HCR-488 channel is spatially uniform — every depth is equally lit, so the
NCC curve is flat across the sz sweep and argmax lands at the boundary.

Public API
----------
estimate_sz_l1_ratio(s) -> dict
    Compute the sz seed from L1 cortical-layer thickness in both modalities.

estimate_sz_ols_from_pairs(matches_csv, s, *, min_cz_depth, cz_surf, hcr_surf) -> dict
    OLS slope of hcr_depth ~ cz_depth (forced through origin) from matched pairs.

get_sz_adaptive_panneuronal(s, run_matcher_fn, *, work_dir, min_cz_depth) -> dict
    Full two-pass: seed → Pass 1 → OLS → return sz for Pass 2.
    ``run_matcher_fn(sz_pin, pass_work_dir) -> matches_csv_path``

Validated on 837568 (sz=2.279, 12,938 pairs) and 839909 (sz=2.687, 10,929 pairs).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from autocoreg.io.centroids import centroids_um
from autocoreg.io.hcr_image import depth_from_surface
from autocoreg.initial_registration.cz_surface import get_cz_pia_surface_panneuronal
from autocoreg.initial_registration.surface_detect import (
    compute_hcr_gfp_l1_thickness_panneuronal,
)
from autocoreg.initial_registration.surfaces import get_hcr_top_surface_iter07


# ---------------------------------------------------------------------------
# L1 seed
# ---------------------------------------------------------------------------

def estimate_sz_l1_ratio(s) -> dict:
    """Compute sz seed = hcr_l1_um / cz_l1_um from L1 cortical thickness.

    CZ L1 thickness: ``l1_thickness_um`` stored in the CZ pia surface JSON.
    HCR L1 thickness: GFP+ cell-density onset below the HCR pia surface.

    Both modalities use the same onset rule — depth below pia at which cell /
    image density rises most steeply — so the ratio is a robust sz seed even
    when the NCC sweep would fail.

    Returns
    -------
    dict with keys:
        cz_l1_um          CZ L1 thickness in CZ µm
        hcr_l1_um         HCR L1 thickness in HCR µm
        l1_ratio          hcr_l1_um / cz_l1_um  (use as sz_seed)
        cz_surf_method    method tag from the CZ surface JSON
        hcr_l1_result     raw dict from compute_hcr_gfp_l1_thickness_panneuronal
    """
    cz_surf = get_cz_pia_surface_panneuronal(s)
    cz_l1   = float(cz_surf.get("l1_thickness_um", float("nan")))
    if not np.isfinite(cz_l1) or cz_l1 <= 0:
        raise ValueError(f"CZ L1 thickness invalid ({cz_l1}); cannot compute sz seed")

    hcr_top = get_hcr_top_surface_iter07(s)
    hcr_l1_result = compute_hcr_gfp_l1_thickness_panneuronal(s, hcr_top_surface=hcr_top)
    if hcr_l1_result is None:
        raise ValueError("HCR L1 thickness returned None — too few GFP+ cells?")
    hcr_l1 = float(hcr_l1_result.get("l1_thickness_um", float("nan")))
    if not np.isfinite(hcr_l1) or hcr_l1 <= 0:
        raise ValueError(f"HCR L1 thickness invalid ({hcr_l1}); cannot compute sz seed")

    ratio = hcr_l1 / cz_l1
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError(f"sz_seed (L1 ratio) invalid ({ratio}); cannot proceed")
    return dict(
        cz_l1_um      = cz_l1,
        hcr_l1_um     = hcr_l1,
        l1_ratio      = ratio,
        cz_surf_method = cz_surf.get("method", "unknown"),
        hcr_l1_result  = hcr_l1_result,
    )


# ---------------------------------------------------------------------------
# OLS empirical sz
# ---------------------------------------------------------------------------

def estimate_sz_ols_from_pairs(
    matches_csv: str | Path,
    s,
    *,
    min_cz_depth: float = 30.0,
    cz_surf: dict | None = None,
    hcr_surf: dict | None = None,
) -> dict:
    """OLS slope of hcr_depth ~ cz_depth (forced through origin) from matched pairs.

    Works on any match CSV that has ``cz_id`` and ``hcr_id`` columns.
    CZ and HCR coordinates are resolved from SubjectData centroids by ID.

    Parameters
    ----------
    matches_csv:
        Path to a match CSV.  Typically the final-round CSV from Pass 1.
    s:
        SubjectData object (provides CZ + HCR centroid arrays in native µm).
    min_cz_depth:
        Exclude pairs shallower than this depth (µm) below CZ pia.
        Per-pair sz diverges near pia where depth ≈ 0.
    cz_surf:
        CZ pia surface dict.  If None, loaded via get_cz_pia_surface_panneuronal(s).
    hcr_surf:
        HCR top surface dict.  If None, loaded via get_hcr_top_surface_iter07(s).

    Returns
    -------
    dict with keys:
        sz_ols       OLS slope (the empirical sz estimate)
        n_pairs      total pairs in the CSV
        n_used       pairs passing depth filter
        median_ratio median per-pair hcr_depth / cz_depth
        iqr_ratio    IQR of per-pair ratio
        cz_depths_um list[float]  (for diagnostic plots)
        hcr_depths_um list[float]
    """
    if cz_surf is None:
        cz_surf = get_cz_pia_surface_panneuronal(s)
    if hcr_surf is None:
        hcr_surf = get_hcr_top_surface_iter07(s)

    try:
        df = pd.read_csv(matches_csv, usecols=["cz_id", "hcr_id"]).dropna()
    except ValueError as exc:
        raise ValueError(f"{matches_csv}: expected columns 'cz_id' and 'hcr_id'") from exc
    if df.empty:
        return dict(sz_ols=float("nan"), n_pairs=0, n_used=0,
                    median_ratio=float("nan"), iqr_ratio=float("nan"),
                    cz_depths_um=[], hcr_depths_um=[])

    cz_zyx_um, cz_ids   = centroids_um(s, "cz")
    hcr_zyx_um, hcr_ids = centroids_um(s, "hcr_all")
    cz_id_map  = {int(cid): i for i, cid in enumerate(cz_ids)}
    hcr_id_map = {int(hid): i for i, hid in enumerate(hcr_ids)}

    cz_depths: list[float] = []
    hcr_depths: list[float] = []

    # Precompute depths once (depth_from_surface is vectorized)
    cz_depth_all = depth_from_surface(cz_zyx_um[:, [2, 1, 0]], cz_surf).astype(float)
    hcr_depth_all = depth_from_surface(hcr_zyx_um[:, [2, 1, 0]], hcr_surf).astype(float)

    for cz_id, hcr_id in zip(df["cz_id"].astype(int), df["hcr_id"].astype(int)):
        ci = cz_id_map.get(int(cz_id))
        hi = hcr_id_map.get(int(hcr_id))
        if ci is None or hi is None:
            continue

        cz_d = float(cz_depth_all[ci])
        hcr_d = float(hcr_depth_all[hi])

        if cz_d >= min_cz_depth and hcr_d > 0:
            cz_depths.append(cz_d)
            hcr_depths.append(hcr_d)
    n_used  = len(cz_depths)
    n_total = len(df)
    if n_used < 10:
        return dict(sz_ols=float("nan"), n_pairs=n_total, n_used=n_used,
                    median_ratio=float("nan"), iqr_ratio=float("nan"),
                    cz_depths_um=cz_depths, hcr_depths_um=hcr_depths)

    cz_arr  = np.array(cz_depths)
    hcr_arr = np.array(hcr_depths)

    sz_ols  = float(np.dot(cz_arr, hcr_arr) / np.dot(cz_arr, cz_arr))
    ratios  = hcr_arr / cz_arr
    median  = float(np.median(ratios))
    iqr     = float(np.percentile(ratios, 75) - np.percentile(ratios, 25))

    return dict(
        sz_ols        = sz_ols,
        n_pairs       = n_total,
        n_used        = n_used,
        median_ratio  = median,
        iqr_ratio     = iqr,
        cz_depths_um  = cz_arr.tolist(),
        hcr_depths_um = hcr_arr.tolist(),
    )


# ---------------------------------------------------------------------------
# High-level two-pass orchestrator
# ---------------------------------------------------------------------------

def get_sz_adaptive_panneuronal(
    s,
    run_matcher_fn: Callable[[float, Path], Path],
    *,
    work_dir: str | Path,
    min_cz_depth: float = 30.0,
) -> dict:
    """Adaptive two-pass sz estimation for pan-neuronal coregistration.

    Pass 1 at the L1 ratio seed; OLS on the resulting pairs → sz_ols for Pass 2.
    Caller is responsible for running Pass 2 with the returned ``sz_ols``.

    Parameters
    ----------
    s:
        SubjectData object.
    run_matcher_fn:
        Callable that runs the matcher for one pass.
        Signature: ``run_matcher_fn(sz_pin: float, pass_work_dir: Path) -> Path``
        Must return the path to the final-round match CSV from that pass.
    work_dir:
        Directory for intermediate Pass 1 output.
    min_cz_depth:
        Depth filter for OLS (see ``estimate_sz_ols_from_pairs``).

    Returns
    -------
    dict with keys:
        sz_seed    L1 ratio seed used for Pass 1
        sz_ols     Empirical OLS sz estimated from Pass 1 pairs (use for Pass 2)
        l1_result  Full dict from estimate_sz_l1_ratio
        ols_result Full dict from estimate_sz_ols_from_pairs
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    l1_result  = estimate_sz_l1_ratio(s)
    sz_seed    = l1_result["l1_ratio"]
    print(f"  [adaptive_sz] L1 seed: cz={l1_result['cz_l1_um']:.1f}µm  "
          f"hcr={l1_result['hcr_l1_um']:.1f}µm  seed={sz_seed:.4f}", flush=True)

    pass1_dir = work_dir / "pass1"
    pass1_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [adaptive_sz] Pass 1 at sz_seed={sz_seed:.4f} ...", flush=True)
    pass1_csv = run_matcher_fn(sz_seed, pass1_dir)

    cz_surf  = get_cz_pia_surface_panneuronal(s)
    hcr_surf = get_hcr_top_surface_iter07(s)
    ols_result = estimate_sz_ols_from_pairs(
        pass1_csv, s, min_cz_depth=min_cz_depth,
        cz_surf=cz_surf, hcr_surf=hcr_surf,
    )
    sz_ols = ols_result["sz_ols"]
    if not np.isfinite(sz_ols):
        raise RuntimeError(
            f"OLS sz is NaN — only {ols_result['n_used']}/{ols_result['n_pairs']} "
            f"pairs passed cz_depth >= {min_cz_depth}µm filter.  "
            "Pass 1 may have matched too few cells."
        )
    print(f"  [adaptive_sz] OLS sz={sz_ols:.4f}  "
          f"(n={ols_result['n_used']}/{ols_result['n_pairs']}  "
          f"median={ols_result['median_ratio']:.4f}  "
          f"iqr={ols_result['iqr_ratio']:.4f})", flush=True)

    # Save intermediates
    _save_json(work_dir / "sz_seed_l1.json", {
        k: v for k, v in l1_result.items() if k != "hcr_l1_result"
    })
    _save_json(work_dir / "sz_estimated_ols.json", {
        k: v for k, v in ols_result.items()
        if k not in ("cz_depths_um", "hcr_depths_um")
    })

    return dict(
        sz_seed    = sz_seed,
        sz_ols     = sz_ols,
        l1_result  = l1_result,
        ols_result = ols_result,
    )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)
