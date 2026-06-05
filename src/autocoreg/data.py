"""Data-loading helpers for Session 15.

Provides:
* ``subject_inputs(sid)`` — bundles centroids, GFP+ ids, argmax-ok ids,
  ground-truth pairs, locked-frame mapped CZ, and a per-subject dict.
* ``strict_gfp_ids(sid)`` — strict-GFP+ ids per the cached
  GMM-intersection threshold (S12-compatible).
* ``argmax_ok_ids(sid)`` — HCR ids whose v5d 4-class argmax falls in
  ``{p_good, p_bad_ok}`` (per user feedback; supersedes ``binary≥0.5``).
* ``scoring_gt(inp)`` — AUTHORITATIVE pose-independent GT for
  recall/precision scoring.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as _config
from .benchmark_data_loader import load_subject
from .centroid_helpers import centroids_um
from .locked_prior_warm import apply_to_cz_um, compute_locked_prior_warm_start
from .overlap_crop import get_overlap_crop
from .surfaces_iter08 import get_hcr_top_surface_iter07

BENCHMARK_SUBJECTS = ("755252", "767018", "767022", "782149", "788406", "790322")


# --------------------------------------------------------------------------- #
# GFP+ filter (S12-style — cached GMM-intersection threshold)
# --------------------------------------------------------------------------- #
def _load_gmm_threshold(sid: str) -> dict:
    data = json.loads(Path(_config.GMM_THRESH_JSON).read_text())
    return next(r for r in data if r["subject"] == sid)


def strict_gfp_ids(sid: str) -> tuple[set[int], dict]:
    """Strict-GFP+ HCR ids using the cached 07b GMM-intersection cutoff."""
    rec = _load_gmm_threshold(sid)
    cutoff = float(rec["cutoff_linear"])
    feature = rec["feature_name"]
    s = load_subject(sid)

    if feature == "density":
        coreg_dir = Path(s.coreg_dir)
        files = list(coreg_dir.glob("*spot_488_counts.csv"))
        if files:
            df = pd.read_csv(files[0])
            if "density" not in df.columns and {"counts", "volume"}.issubset(df.columns):
                df["density"] = df["counts"] / df["volume"]
            tbl = df[["hcr_id", "density"]].dropna()
        else:
            from .benchmark_data_loader import _aggregate_spots_from_hcr
            agg = _aggregate_spots_from_hcr(Path(s.hcr_dir))
            tbl = agg[["hcr_id", "density"]].dropna()
        keep = tbl[tbl["density"].astype(float) >= cutoff]
    elif feature == "mean_minus_bg":
        csv = _config.DATA_LEVEL / f"cell_data_mean_{sid}_R1.csv"
        df = pd.read_csv(csv)
        if "channel" in df.columns:
            df = df[df["channel"] == 488]
        for k in ("cell_id", "id"):
            if k in df.columns:
                df = df.rename(columns={k: "hcr_id"})
        df = df.copy()
        df["mean_minus_bg"] = df["mean"].astype(float) - df["background"].astype(float)
        keep = df[df["mean_minus_bg"].astype(float) >= cutoff]
    else:
        raise RuntimeError(f"unknown gmm feature {feature}")

    ids = set(int(x) for x in keep["hcr_id"].values)
    meta = dict(
        cutoff_linear=cutoff,
        feature=feature,
        n_strict=len(ids),
        coreg_total=int(rec["coreg_total"]),
        coreg_kept_strict=int(rec["coreg_kept_strict"]),
        coreg_coverage_strict=float(rec["coreg_coverage_strict"]),
    )
    return ids, meta


# --------------------------------------------------------------------------- #
# ROI quality argmax-ok (per user feedback)
# --------------------------------------------------------------------------- #
def argmax_ok_ids(sid: str) -> set[int]:
    p = _config.ROI_QUALITY_DIR / f"{sid}_stage2_4class_proba_v5d_um.parquet"
    df = pd.read_parquet(p)
    cls = ["p_bad", "p_bad_ok", "p_good", "p_merged"]
    am = df[cls].idxmax(axis=1)
    return set(df.loc[am.isin(["p_good", "p_bad_ok"]), "hcr_id"].astype(int).tolist())


# --------------------------------------------------------------------------- #
# Locked-frame sz pin
# --------------------------------------------------------------------------- #
def load_sz_pins() -> dict[str, float]:
    df = pd.read_csv(_config.SZ_TABLE_CSV)
    df["subject_id"] = df["subject_id"].astype(str)
    return dict(zip(df["subject_id"], df["sz_peak"].astype(float)))


# --------------------------------------------------------------------------- #
# Bundle
# --------------------------------------------------------------------------- #
@dataclass
class SubjectInputs:
    sid: str
    s: object  # SubjectData
    cz_um: np.ndarray  # (N_cz, 3) zyx µm
    cz_ids: np.ndarray  # (N_cz,) int
    hcr_um: np.ndarray  # (N_hcr_all, 3) zyx µm
    hcr_ids: np.ndarray  # (N_hcr_all,) int
    cz_lp_um: np.ndarray  # (N_cz, 3) zyx µm — CZ under locked-prior warm-start
    crop_bbox_zyx_um: tuple  # (z0,z1,y0,y1,x0,x1)
    gfp_ids: set[int]
    ok_ids: set[int]
    coreg: pd.DataFrame  # cz_id, hcr_id (int, both present)
    gmm_meta: dict
    sz_pin: float


def subject_inputs(sid: str, sz_pins: dict[str, float] | None = None) -> SubjectInputs:
    s = load_subject(sid)
    cz_um, cz_ids = centroids_um(s, "cz")
    hcr_um, hcr_ids = centroids_um(s, "hcr_all")
    if sz_pins is not None:
        sz_pin = float(sz_pins[sid])
    else:
        # GT-free path: use sz_estimator to compute sz from image data.
        from .sz_estimator import get_sz
        sz_pin = float(get_sz(s))
    lp = compute_locked_prior_warm_start(s, sz_init=sz_pin)
    cz_lp = apply_to_cz_um(lp, cz_um)
    crop = get_overlap_crop(s, margin_frac=0.10)
    bbox = tuple(crop["bbox_hcr_um"])
    gfp_ids, gmm_meta = strict_gfp_ids(sid)
    ok_ids = argmax_ok_ids(sid)
    coreg = s.coreg_table[["cz_id", "hcr_id"]].dropna().astype(int).reset_index(drop=True)
    return SubjectInputs(
        sid=sid, s=s,
        cz_um=cz_um, cz_ids=cz_ids,
        hcr_um=hcr_um, hcr_ids=hcr_ids,
        cz_lp_um=cz_lp, crop_bbox_zyx_um=bbox,
        gfp_ids=gfp_ids, ok_ids=ok_ids,
        coreg=coreg, gmm_meta=gmm_meta, sz_pin=sz_pin,
    )


# --------------------------------------------------------------------------- #
# Authoritative scoring GT (POSE-INDEPENDENT)
# --------------------------------------------------------------------------- #
def scoring_gt(inp: "SubjectInputs") -> "dict[int, int]":
    """Authoritative recall/precision GT — POSE-INDEPENDENT.

    G = coreg pairs whose HCR cell is GFP+ ∩ classifier-ok.  CZ side unfiltered.

    DO NOT filter by spatial pool/bbox/z-band/hcr_pool_ids/cz_pool_ids — those are
    pose-derived, so filtering GT by them is CIRCULAR: it scores only pose-covered
    pairs, inflates recall for bad poses, and gives |G|≈0 for a broken pose instead
    of recall≈0.  A GT pair missed because the loaded region was too small is a FN,
    not removed from the denominator.

    Returns {cz_id (int): hcr_id (int)} for every qualifying coreg pair.
    """
    gfp_ok = inp.gfp_ids & inp.ok_ids
    G: dict[int, int] = {}
    for c, h in zip(inp.coreg["cz_id"].astype(int), inp.coreg["hcr_id"].astype(int)):
        if int(h) in gfp_ok:
            G[int(c)] = int(h)
    return G


# --------------------------------------------------------------------------- #
# Inside-crop test
# --------------------------------------------------------------------------- #
def inside_crop_mask(pts_zyx_um: np.ndarray, bbox: tuple) -> np.ndarray:
    z0, z1, y0, y1, x0, x1 = bbox
    return (
        (pts_zyx_um[:, 0] >= z0) & (pts_zyx_um[:, 0] <= z1)
        & (pts_zyx_um[:, 1] >= y0) & (pts_zyx_um[:, 1] <= y1)
        & (pts_zyx_um[:, 2] >= x0) & (pts_zyx_um[:, 2] <= x1)
    )
