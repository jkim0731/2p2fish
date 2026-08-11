"""Per-cell ROI cross-sectional-area sxy estimator — promoted main-pipeline entry.

Under isotropic lateral tissue expansion `sxy`, the xy footprint of a cell
body scales as `sxy^2`:

    area_HCR ≈ sxy^2 · area_CZ  →  sxy = sqrt(median(area_HCR) / median(area_CZ))

Two area definitions are supported via the ``area_mode`` parameter:

``"bbox"`` (default, production):
    Per-cell area = tight xy bounding-box area (y_extent × x_extent in µm²).
    This is the promoted main-pipeline estimator.

``"max_xsection"`` (experimental):
    Per-cell area = maximum over z-slices of the in-plane mask pixel count
    × pixel area in µm².  Needs the actual label mask, not just the bbox —
    see :func:`compute_max_xsection_hcr` and :func:`cz_cell_max_xsection`.

CZ areas come from `segmentation_masks.tif` (regionprops). HCR areas come
from `cell_body_segmentation/segmentation_mask.zarr`, masked by the
per-cell label id inside the `metrics.pickle` tile bbox, to recover the
true tight xy footprint.

Scope
-----
Spot subjects (788406, 790322, 767018, 782149) ship
`cell_body_segmentation/metrics.pickle` and use it directly. Intensity
subjects (755252, 767022) lack the pickle (they are the oldest HCR
processings) but DO have `segmentation_mask.zarr` + `cell_centroids.npy`;
for them `_load_hcr_metrics` reconstructs the per-cell tile-bbox index
from the centroids (level-2 frame, label == hcr_id), so ROI-area sxy is
computed the same way and a GT fallback is no longer required.

On-disk caches
--------------
Per-cell tight HCR bboxes:
  ``/root/capsule/code/dev_code/cached_hcr_cell_tight_bbox/
    {sid}_hcr_cell_tight_bbox_v1.parquet``

Per-cell max-xsection HCR pixel counts:
  ``/root/capsule/code/dev_code/cached_hcr_cell_max_xsection/
    {sid}_hcr_cell_max_xsection_v1.parquet``
  Columns: hcr_id, max_xsection_pix (raw pixel count at the widest z-slice).

Both caches are additive: re-running with a new ``hcr_ids`` set extends them.

Scale note — this was a subtle bug before promotion
---------------------------------------------------
``segmentation_mask.zarr`` is at LEVEL-0 in xy (0.247 µm/vox on spot
subjects); the sibling ``segmentation_mask_orig_res.zarr`` is the
level-2 version — the name "orig_res" is misleading. The standard
``s.hcr_xy_um`` (~0.988 µm) is the level-2 scale used by
``hcr_centroids``; applying it to level-0 voxel extents inflates each
xy span 4× and area 16×, so sxy comes out 4× too high. This module
uses ``s.hcr_seg_xy_um`` (= ``s.hcr_xy_um / HCR_SEG_XY_DOWNSAMPLE``)
when converting bbox extents from ``segmentation_mask.zarr``, and
rescales voxel centroids to the level-2 pixel frame before feeding
them to ``hcr_px_to_um``.

Validation (2026-04-23, bbox mode)
-----------------------------------
Median-area sxy vs landmark-Procrustes GT:

    788406   est 1.749   GT 1.778   −1.6 %
    790322   est 1.785   GT 1.763   +1.2 %
    767018   est 1.812   GT 1.702   +6.5 %
    782149   est 1.631   GT 1.924   −15.2 %  (HCR_span=566 µm;
             anomalous surface / FOV truncation, not a scale bug)

Public API
----------
* :func:`compute_tight_hcr_bboxes` — cache-backed per-cell tight bboxes
  from ``segmentation_mask.zarr``.
* :func:`compute_max_xsection_hcr` — cache-backed per-cell max z-slice
  pixel count from ``segmentation_mask.zarr``.
* :func:`cz_cell_tight_bboxes` — per-CZ-cell bbox + depth, from
  ``segmentation_masks.tif``.
* :func:`hcr_cell_tight_bboxes` — per-HCR-cell bbox + depth, level-0
  correctly applied, for a caller-supplied hcr_id set.
* :func:`estimate_sxy_roi_area` — full driver for one subject.
* :func:`_argmax_ok_ids` — HCR ids whose 4-class argmax ∈ {p_good, p_bad_ok}.
* :func:`estimate_sxy_roi_area_slab` — slab∩ok estimator for thin-HCR subjects.
* :func:`estimate_sxy_auto` — auto-detects thin vs thick and returns sxy_median.

PRODUCTION sxy entry point (promoted 2026-06-04)
------------------------------------------------
* :func:`estimate_sxy_min_rule` — the **production** sxy estimator: min-rule 2×
  ¼-FOV.  ``surface_registration_v2`` routes its ``sxy=None`` path through this.
  HCR slab = ``min(p99(HCR GFP+∩ok∩¼-FOV depth), 2·p99(CZ depth))``, CZ slab =
  half that (HCR is axially ~2× expanded; capping CZ shallower raises sxy);
  ``sxy = sqrt(median HCR max-xsection / median CZ max-xsection)``.  GT-free.
  Reference ``782149 → 1.7336``.  See :data:`MIN_RULE_AXIAL_FACTOR`,
  :data:`SXY_GRID_SEARCH_OFFSETS` (GT-free grid-search fallback for roaming
  new subjects).  Supersedes ``estimate_sxy_roi_area`` / ``estimate_sxy_auto``
  as the production scale source.
"""
from __future__ import annotations

import glob
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import zarr
from skimage.measure import regionprops

from autocoreg import config as _config
from autocoreg.io.hcr_image import analyze_subject, depth_from_surface, fit_anisotropic_similarity
from autocoreg.io.subjects import DATA_DIR, HCR_SEG_XY_DOWNSAMPLE, SubjectData, cz_px_to_um, hcr_px_to_um, landmark_pairs_um, load_subject
from autocoreg.initial_registration.coarse_align import coarse_align_revised

SPOT_SUBJECTS = frozenset({"788406", "790322", "767018", "782149"})
INTENSITY_SUBJECTS = frozenset({"755252", "767022"})


def _gt_aniso_fit(s: SubjectData):
    """Anisotropic-similarity fit to the QCed GT landmarks, or ``None`` when GT
    is unavailable.

    Benchmark subjects ship manually-QCed landmarks; new / pipeline-monitor
    subjects have none (``landmark_pairs_um`` returns empty), so the fit cannot
    be computed.  Callers treat ``None`` as "GT unavailable" and report ``NaN``
    for the diagnostic ``*_gt`` / ``err_pct`` fields — these never feed the pose,
    so they are pure validation read-outs.
    """
    try:
        return fit_anisotropic_similarity(*landmark_pairs_um(s, active_only=True))
    except (ValueError, IndexError):
        return None

TIGHT_BBOX_CACHE = _config.HCR_TIGHT_BBOX_CACHE_DIR
TIGHT_BBOX_CACHE.mkdir(parents=True, exist_ok=True)
TIGHT_BBOX_VERSION = "v1"

MAX_XSECTION_CACHE = _config.HCR_MAX_XSECTION_CACHE_DIR
MAX_XSECTION_CACHE.mkdir(parents=True, exist_ok=True)
MAX_XSECTION_CACHE_VERSION = "v1"

D_SKIN_UM = 100.0  # cortex surface → first cortical cells

# ----------------------------------------------------------------
# Thin-HCR auto-detection thresholds (slab∩ok branch)
# ----------------------------------------------------------------
# When the strict-GFP+ depth span is below this, the HCR FOV is
# truncated (e.g. 782149 ~566 µm vs thick subjects 900–1200 µm).
# The full-span estimator under-estimates sxy in that case because
# the full-depth band is not available; we switch to the slab recipe.
HCR_TRUNCATION_SPAN_UM = 700.0

# Depth windows for the DEPRECATED slab∩ok estimator (estimate_sxy_roi_area_slab,
# do not use in production).  These are the OLD 50/100 MIP bounds; they are no
# longer kept in sync with surface_registration_v2.CZ_SLAB / HCR_SLAB, which were
# promoted to 80/150 on 2026-06-04 alongside the min-rule sxy estimator
# (estimate_sxy_min_rule — the production sxy).  Left as-is purely so the
# deprecated slab estimator stays runnable as a research artefact.
SLAB_CZ_UM = 50.0
SLAB_HCR_UM = 100.0

# Path to the v5d ROI-quality 4-class probability parquets (repo B output).
ROI_QUALITY_DIR = _config.ROI_QUALITY_DIR

# ----------------------------------------------------------------
# Min-rule sxy estimator constants (PRODUCTION, promoted 2026-06-04)
# ----------------------------------------------------------------
# The "heuristic 2×" axial-expansion factor: HCR tissue is expanded ~2× in
# the axial (z) direction relative to the CZ z-stack, so for a given physical
# depth band the HCR slab is ~twice as thick as the matching CZ slab.  We use
# a fixed heuristic factor of 2 rather than the measured sz because sz itself
# requires a correct registration to estimate (circular): the sxy estimator
# must run BEFORE the pose exists.
MIN_RULE_AXIAL_FACTOR = 2.0

# FALLBACK grid for a NEW subject whose registration roams / matcher collapses
# at the min-rule sxy.  Grid-search the base sxy at these fractional offsets
# around the min-rule value and pick the pose that lands (highest soma-print
# mutual-best count / lowest rigid off-centre).  GT-free — driven purely by
# which candidate pose produces a self-consistent matching, never by GT.
SXY_GRID_SEARCH_OFFSETS = (-0.10, -0.05, -0.02, 0.02, 0.05, 0.10)


# -----------------------------------------------------------
# path discovery
# -----------------------------------------------------------
def _cz_seg_mask_path(sid: str) -> Path:
    override = os.environ.get("MFISH_CZ_SEG_TIF", "").strip()
    if override:
        p = Path(override)
        if not p.exists():
            raise FileNotFoundError(f"MFISH_CZ_SEG_TIF set but not found: {override}")
        return p
    for d in DATA_DIR.glob(f"multiplane-ophys_{sid}_*cortical-zstack-segmentation*"):
        tif = d / "channel_0_ref_0" / "segmentation_masks.tif"
        if tif.exists():
            return tif
        alt = sorted(d.rglob("segmentation_masks.tif"))   # structure-agnostic fallback
        if alt:
            return alt[-1]
    raise FileNotFoundError(
        f"CZ segmentation_masks.tif for {sid} (set MFISH_CZ_SEG_TIF to point at it directly)")


def _hcr_metrics_path(sid: str) -> Path:
    hits = sorted(glob.glob(
        str(_config.DATA_ROOT / f"HCR_{sid}_*/cell_body_segmentation/metrics.pickle")
    ))
    if not hits:
        hcr_dir = os.environ.get("MFISH_HCR_DIR")
        if hcr_dir:
            p = Path(hcr_dir) / "cell_body_segmentation" / "metrics.pickle"
            if p.exists():
                return p
        raise FileNotFoundError(f"HCR cell_body_segmentation/metrics.pickle for {sid}")
    return Path(hits[-1])


def _hcr_seg_zarr_path(sid: str) -> Path:
    hits = sorted(glob.glob(
        str(_config.DATA_ROOT / f"HCR_{sid}_*/cell_body_segmentation/segmentation_mask.zarr")
    ))
    if not hits:
        hcr_dir = os.environ.get("MFISH_HCR_DIR")
        if hcr_dir:
            p = Path(hcr_dir) / "cell_body_segmentation" / "segmentation_mask.zarr"
            if p.exists():
                return p
        raise FileNotFoundError(f"HCR cell_body_segmentation/segmentation_mask.zarr for {sid}")
    return Path(hits[-1])


def _hcr_centroids_npy_path(sid: str) -> Path:
    hits = sorted(glob.glob(
        str(_config.DATA_ROOT / f"HCR_{sid}_*/cell_body_segmentation/cell_centroids.npy")
    ))
    if not hits:
        hcr_dir = os.environ.get("MFISH_HCR_DIR")
        if hcr_dir:
            p = Path(hcr_dir) / "cell_body_segmentation" / "cell_centroids.npy"
            if p.exists():
                return p
        raise FileNotFoundError(f"HCR cell_body_segmentation/cell_centroids.npy for {sid}")
    return Path(hits[-1])


# Padding (level-0 voxels) around each centroid when reconstructing a
# tile bbox from cell_centroids.npy.  Half-widths: ~18 µm xy / ~? µm z at
# 0.247 µm/vox level-0 lateral; generous enough to contain a soma, while
# compute_tight_hcr_bboxes still recovers the TIGHT footprint by masking
# == hcr_id inside the crop.  Clipping a rare large cell only affects its
# own bbox, not the population median.
_SYNTH_BBOX_HALF_XY_VOX = 96
_SYNTH_BBOX_HALF_Z_VOX = 48


def _synthetic_metrics_from_centroids(
    sid: str,
    half_xy_vox: int = _SYNTH_BBOX_HALF_XY_VOX,
    half_z_vox: int = _SYNTH_BBOX_HALF_Z_VOX,
) -> dict:
    """Reconstruct a ``metrics.pickle``-equivalent index from
    ``cell_centroids.npy`` for subjects whose (older) HCR processing did
    not ship ``metrics.pickle`` (755252, 767022).

    ``cell_centroids.npy`` rows are ``[z, y, x, label]`` in the LEVEL-2
    (``segmentation_mask_orig_res``) frame: z is NOT downsampled, xy are
    downsampled by ``HCR_SEG_XY_DOWNSAMPLE``; ``label`` == ``hcr_id``.
    We scale xy to level-0 and pad a fixed window so the existing
    level-0 crop machinery in :func:`compute_tight_hcr_bboxes` can recover
    each cell's tight footprint exactly as for the spot subjects.

    Returns ``{hcr_id: {"global_bbox": [zlo, ylo, xlo, zhi, yhi, xhi]}}``
    in level-0 inclusive-endpoint voxel coordinates.
    """
    cents = np.load(_hcr_centroids_npy_path(sid))
    if cents.ndim != 2 or cents.shape[1] < 4:
        raise ValueError(f"{sid}: unexpected cell_centroids.npy shape {cents.shape}")
    _, _, Z, Y, X = zarr.open(str(_hcr_seg_zarr_path(sid)))["0"].shape
    f = HCR_SEG_XY_DOWNSAMPLE
    metrics: dict[int, dict] = {}
    for row in cents:
        zc, yc, xc, lab = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        zc0, yc0, xc0 = zc, yc * f, xc * f  # level-2 → level-0 (z not downsampled)
        metrics[lab] = {"global_bbox": [
            max(0, zc0 - half_z_vox), max(0, yc0 - half_xy_vox), max(0, xc0 - half_xy_vox),
            min(Z - 1, zc0 + half_z_vox), min(Y - 1, yc0 + half_xy_vox), min(X - 1, xc0 + half_xy_vox),
        ]}
    return metrics


def _load_hcr_metrics(sid: str) -> dict:
    """Per-cell ``{hcr_id: {"global_bbox": [...level-0 incl...]}}``.

    Uses ``cell_body_segmentation/metrics.pickle`` when present; otherwise
    (755252, 767022 — the oldest HCR processings, which lack the pickle but
    DO have ``segmentation_mask.zarr`` + ``cell_centroids.npy``) it
    reconstructs an equivalent index from the centroids, so ROI-area sxy
    works without falling back to GT.  Consumers use only ``["global_bbox"]``
    and ``.keys()``, which the reconstruction supplies.
    """
    try:
        with open(_hcr_metrics_path(sid), "rb") as fh:
            return pickle.load(fh)
    except FileNotFoundError:
        return _synthetic_metrics_from_centroids(sid)


def _tight_bbox_cache_path(sid: str) -> Path:
    return TIGHT_BBOX_CACHE / f"{sid}_hcr_cell_tight_bbox_{TIGHT_BBOX_VERSION}.parquet"


# -----------------------------------------------------------
# HCR tight-bbox cache
# -----------------------------------------------------------
def compute_tight_hcr_bboxes(
    sid: str,
    hcr_ids,
    force: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Cache-backed per-cell tight HCR bboxes in level-0 voxel indices.

    Parameters
    ----------
    sid : str
        Subject id (must be a spot subject with `metrics.pickle`).
    hcr_ids : iterable of int
        Cell label ids of interest. Cached entries are reused; missing
        ids are computed by reading the zarr crop covering the pickle
        tile bbox and masking `segmentation_mask == hcr_id`.
    force : bool
        Re-compute every id even if cached.

    Returns
    -------
    pd.DataFrame
        Columns: ``hcr_id, zmin_vox, zmax_vox, ymin_vox, ymax_vox,
        xmin_vox, xmax_vox, volume_vox, zc_vox, yc_vox, xc_vox``
        restricted to the requested `hcr_ids` (see README in the cache
        directory for the schema).
    """
    want = {int(x) for x in hcr_ids}
    cache_path = _tight_bbox_cache_path(sid)
    if cache_path.exists() and not force:
        cached = pd.read_parquet(cache_path)
    else:
        cached = pd.DataFrame(columns=[
            "hcr_id", "zmin_vox", "ymin_vox", "xmin_vox",
            "zmax_vox", "ymax_vox", "xmax_vox",
            "volume_vox", "zc_vox", "yc_vox", "xc_vox",
        ])

    have = set(int(x) for x in cached["hcr_id"].values)
    missing = sorted(want - have)

    if missing:
        if verbose:
            print(f"  [{sid}] tight-bbox: {len(have & want)} cached, "
                  f"{len(missing)} to compute")
        metrics = _load_hcr_metrics(sid)
        z = zarr.open(str(_hcr_seg_zarr_path(sid)))["0"]
        new_rows = []
        t0 = time.time()
        for i, hid in enumerate(missing, 1):
            m = metrics.get(hid)
            if m is None:
                continue
            bb = np.asarray(m["global_bbox"], dtype=int)
            zlo, ylo, xlo, zhi, yhi, xhi = bb
            # pickle bbox endpoints are inclusive (verified with hid=10317);
            # read +1 to include the last slice
            crop = np.asarray(z[0, 0, zlo:zhi + 1, ylo:yhi + 1, xlo:xhi + 1])
            mask = (crop == hid)
            if not mask.any():
                continue
            zs, ys, xs = np.where(mask)
            new_rows.append({
                "hcr_id": int(hid),
                "zmin_vox": int(zs.min() + zlo),
                "zmax_vox": int(zs.max() + zlo + 1),
                "ymin_vox": int(ys.min() + ylo),
                "ymax_vox": int(ys.max() + ylo + 1),
                "xmin_vox": int(xs.min() + xlo),
                "xmax_vox": int(xs.max() + xlo + 1),
                "volume_vox": int(mask.sum()),
                "zc_vox": float(zs.mean() + zlo),
                "yc_vox": float(ys.mean() + ylo),
                "xc_vox": float(xs.mean() + xlo),
            })
            if verbose and (i % 200 == 0 or i == len(missing)):
                dt = time.time() - t0
                print(f"    {i}/{len(missing)}  ({i/max(dt,1e-3):.1f} cells/s)")
        new_df = pd.DataFrame(new_rows)
        if not cached.empty:
            combined = pd.concat([cached, new_df], ignore_index=True)
            combined = combined.drop_duplicates("hcr_id", keep="last")
        else:
            combined = new_df
        combined = combined.sort_values("hcr_id").reset_index(drop=True)
        combined.to_parquet(cache_path, index=False)
        cached = combined

    return cached[cached["hcr_id"].isin(want)].reset_index(drop=True)


def _max_xsection_cache_path(sid: str) -> Path:
    return MAX_XSECTION_CACHE / f"{sid}_hcr_cell_max_xsection_{MAX_XSECTION_CACHE_VERSION}.parquet"


def compute_max_xsection_hcr(
    sid: str,
    hcr_ids,
    force: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Cache-backed per-cell maximum in-plane pixel count across z-slices.

    For each cell, reads the same zarr crop as :func:`compute_tight_hcr_bboxes`
    (using the metrics.pickle tile bbox as a search window), masks by label id,
    and returns ``max_z( sum_yx(mask[z]) )``.  This is the pixel count at the
    widest z-slice — multiply by ``seg_xy_um**2`` to get µm².

    Coordinate frame: level-0 (segmentation_mask.zarr), raw pixel counts —
    no µm conversion here.  Callers apply ``s.hcr_seg_xy_um**2``.

    Parameters
    ----------
    sid : str
    hcr_ids : iterable of int
    force : bool
        Re-compute all ids even if cached.

    Returns
    -------
    pd.DataFrame
        Columns: ``hcr_id, max_xsection_pix``
    """
    want = {int(x) for x in hcr_ids}
    cache_path = _max_xsection_cache_path(sid)
    if cache_path.exists() and not force:
        cached = pd.read_parquet(cache_path)
    else:
        cached = pd.DataFrame(columns=["hcr_id", "max_xsection_pix"])

    have = set(int(x) for x in cached["hcr_id"].values)
    missing = sorted(want - have)

    if missing:
        if verbose:
            print(f"  [{sid}] max-xsection: {len(have & want)} cached, "
                  f"{len(missing)} to compute")
        metrics = _load_hcr_metrics(sid)
        z = zarr.open(str(_hcr_seg_zarr_path(sid)))["0"]
        new_rows = []
        t0 = time.time()
        for i, hid in enumerate(missing, 1):
            m = metrics.get(hid)
            if m is None:
                continue
            bb = np.asarray(m["global_bbox"], dtype=int)
            zlo, ylo, xlo, zhi, yhi, xhi = bb
            crop = np.asarray(z[0, 0, zlo:zhi + 1, ylo:yhi + 1, xlo:xhi + 1])
            mask = (crop == hid)  # shape (dZ, dY, dX)
            if not mask.any():
                continue
            # count in-plane pixels per z-slice; take the maximum
            per_z = mask.sum(axis=(1, 2))  # shape (dZ,)
            new_rows.append({
                "hcr_id": int(hid),
                "max_xsection_pix": int(per_z.max()),
            })
            if verbose and (i % 200 == 0 or i == len(missing)):
                dt = time.time() - t0
                print(f"    {i}/{len(missing)}  ({i/max(dt,1e-3):.1f} cells/s)")
        new_df = pd.DataFrame(new_rows)
        if not cached.empty:
            combined = pd.concat([cached, new_df], ignore_index=True)
            combined = combined.drop_duplicates("hcr_id", keep="last")
        else:
            combined = new_df
        combined = combined.sort_values("hcr_id").reset_index(drop=True)
        combined.to_parquet(cache_path, index=False)
        cached = combined

    return cached[cached["hcr_id"].isin(want)].reset_index(drop=True)


def cz_cell_max_xsection(sid: str, s: SubjectData, cz_surf: dict) -> pd.DataFrame:
    """Per-CZ-cell max in-plane mask area (µm²) and pia depth.

    Reads ``segmentation_masks.tif`` once, then for each label counts
    pixels per z-slice and takes the maximum.  Area = max_pix × cz_xy_um².
    The ``depth_um`` and centroid columns match :func:`cz_cell_tight_bboxes`.

    Coordinate frame: CZ tif is (Z, Y, X) at ``s.cz_xy_um`` µm/pix (xy)
    and ``s.cz_z_um`` µm/pix (z).
    """
    mask_vol = tifffile.imread(_cz_seg_mask_path(sid))  # (Z, Y, X) uint32
    rp = regionprops(mask_vol)
    dz_um = s.cz_z_um
    dx_um = s.cz_xy_um  # same in x and y
    pixel_area_um2 = dx_um * dx_um
    rows = []
    for r in rp:
        zlo, ylo, xlo, zhi, yhi, xhi = r.bbox
        # regionprops slice already set in r.image (bounding-box sub-volume)
        # r.image is True where mask==label inside the bbox
        per_z = r.image.sum(axis=(1, 2))  # count per z-plane, shape (zhi-zlo,)
        max_pix = int(per_z.max())
        max_xsection_um2 = max_pix * pixel_area_um2
        zc, yc, xc = r.centroid
        xyz_um = cz_px_to_um(np.array([[zc, yc, xc]]), s)[0]
        x_um, y_um, z_um_c = xyz_um[2], xyz_um[1], xyz_um[0]
        depth = float(depth_from_surface(np.array([[x_um, y_um, z_um_c]]), cz_surf)[0])
        rows.append({
            "cz_id": r.label,
            "xy_area_um2": max_xsection_um2,
            "max_xsection_pix": max_pix,
            "volume_vox": r.area,
            "x_um": x_um, "y_um": y_um, "z_um": z_um_c,
            "depth_um": depth,
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------
# per-cell bbox tables
# -----------------------------------------------------------
def cz_cell_tight_bboxes(sid: str, s: SubjectData, cz_surf: dict) -> pd.DataFrame:
    """Per-CZ-cell tight xy-bbox area and pia depth."""
    mask = tifffile.imread(_cz_seg_mask_path(sid))
    rp = regionprops(mask)
    dz_um, dy_um, dx_um = s.cz_z_um, s.cz_xy_um, s.cz_xy_um
    rows = []
    for r in rp:
        zlo, ylo, xlo, zhi, yhi, xhi = r.bbox
        dz = (zhi - zlo) * dz_um
        dy = (yhi - ylo) * dy_um
        dx = (xhi - xlo) * dx_um
        zc, yc, xc = r.centroid
        xyz_um = cz_px_to_um(np.array([[zc, yc, xc]]), s)[0]
        x_um, y_um, z_um_c = xyz_um[2], xyz_um[1], xyz_um[0]
        depth = float(depth_from_surface(np.array([[x_um, y_um, z_um_c]]), cz_surf)[0])
        rows.append({
            "cz_id": r.label,
            "xy_area_um2": dx * dy,
            "bbox_dx_um": dx, "bbox_dy_um": dy, "bbox_dz_um": dz,
            "volume_vox": r.area,
            "volume_um3": r.area * dz_um * dy_um * dx_um,
            "x_um": x_um, "y_um": y_um, "z_um": z_um_c,
            "depth_um": depth,
        })
    return pd.DataFrame(rows)


def hcr_cell_tight_bboxes(
    sid: str,
    s: SubjectData,
    hcr_surf: dict,
    hcr_ids,
    area_mode: str = "max_xsection",
) -> pd.DataFrame:
    """Per-HCR-cell area and pia depth for the given ids.

    Parameters
    ----------
    area_mode : ``"bbox"`` | ``"max_xsection"``
        ``"bbox"`` (default): xy_area_um2 = tight bbox extents² in µm².
        ``"max_xsection"``: xy_area_um2 = max over z-slices of in-plane
        pixel count × seg_xy_um².  Uses :func:`compute_max_xsection_hcr`.

    Uses `segmentation_mask.zarr` (level-0) via the appropriate cache.
    Applies `s.hcr_seg_xy_um` for µm conversion and rescales voxel
    centroids to the level-2 frame before `hcr_px_to_um`.
    """
    if area_mode not in ("bbox", "max_xsection"):
        raise ValueError(f"area_mode must be 'bbox' or 'max_xsection', got {area_mode!r}")
    metrics_keys = set(_load_hcr_metrics(sid).keys())
    want = {int(i) for i in hcr_ids if int(i) in metrics_keys}
    if not want:
        return pd.DataFrame()
    tb = compute_tight_hcr_bboxes(sid, want)
    if tb.empty:
        return pd.DataFrame()

    seg_xy_um = s.hcr_seg_xy_um
    seg_z_um = s.hcr_seg_z_um
    dx_vox = (tb["xmax_vox"] - tb["xmin_vox"]).to_numpy(float)
    dy_vox = (tb["ymax_vox"] - tb["ymin_vox"]).to_numpy(float)
    dz_vox = (tb["zmax_vox"] - tb["zmin_vox"]).to_numpy(float)

    if area_mode == "bbox":
        # tight xy bounding-box: dx_um × dy_um (level-0 voxels × seg_xy_um)
        xy_area_um2 = (dx_vox * seg_xy_um) * (dy_vox * seg_xy_um)
    else:
        # max z-slice in-plane pixel count × pixel area (µm²)
        mx = compute_max_xsection_hcr(sid, want)
        # align to tb order by hcr_id join
        mx_map = dict(zip(mx["hcr_id"].to_numpy(), mx["max_xsection_pix"].to_numpy()))
        max_pix = np.array([mx_map.get(int(hid), 0) for hid in tb["hcr_id"].to_numpy()], dtype=float)
        pixel_area_um2 = seg_xy_um * seg_xy_um  # level-0 pixel area
        xy_area_um2 = max_pix * pixel_area_um2

    zyx_px = tb[["zc_vox", "yc_vox", "xc_vox"]].to_numpy(float).copy()
    zyx_px[:, 1] /= HCR_SEG_XY_DOWNSAMPLE
    zyx_px[:, 2] /= HCR_SEG_XY_DOWNSAMPLE
    xyz_um = hcr_px_to_um(zyx_px, s)[:, [2, 1, 0]]
    depths = depth_from_surface(xyz_um, hcr_surf)
    return pd.DataFrame({
        "hcr_id": tb["hcr_id"].to_numpy(),
        "xy_area_um2": xy_area_um2,
        "bbox_dx_um": dx_vox * seg_xy_um,
        "bbox_dy_um": dy_vox * seg_xy_um,
        "bbox_dz_um": dz_vox * seg_z_um,
        "volume_vox": tb["volume_vox"].to_numpy(),
        "volume_um3": tb["volume_vox"].to_numpy() * seg_z_um * seg_xy_um * seg_xy_um,
        "x_um": xyz_um[:, 0], "y_um": xyz_um[:, 1], "z_um": xyz_um[:, 2],
        "depth_um": depths,
    })


# -----------------------------------------------------------
# driver
# -----------------------------------------------------------
def _prefilter_center_fov(
    sid: str, s: SubjectData, hcr_ids
) -> set[int]:
    """Keep only cells whose approximate centroid lies in the HCR center
    ¼ FOV (linear half-width), using the cheap pickle-tile-bbox center
    so we avoid paying zarr I/O for cells we'd drop anyway."""
    metrics = _load_hcr_metrics(sid)
    ids = np.array(sorted(int(i) for i in hcr_ids if int(i) in metrics))
    if ids.size == 0:
        return set()
    bb = np.stack([np.asarray(metrics[int(i)]["global_bbox"], dtype=float)
                   for i in ids])
    zc = 0.5 * (bb[:, 0] + bb[:, 3])
    yc = 0.5 * (bb[:, 1] + bb[:, 4])
    xc = 0.5 * (bb[:, 2] + bb[:, 5])
    zyx_px = np.column_stack([zc, yc, xc]).copy()
    zyx_px[:, 1] /= HCR_SEG_XY_DOWNSAMPLE
    zyx_px[:, 2] /= HCR_SEG_XY_DOWNSAMPLE
    xyz_um = hcr_px_to_um(zyx_px, s)[:, [2, 1, 0]]
    x_um, y_um = xyz_um[:, 0], xyz_um[:, 1]
    cx, cy = 0.5 * (x_um.min() + x_um.max()), 0.5 * (y_um.min() + y_um.max())
    qx = (x_um.max() - x_um.min()) / 4
    qy = (y_um.max() - y_um.min()) / 4
    mask = ((x_um >= cx - qx) & (x_um <= cx + qx)
            & (y_um >= cy - qy) & (y_um <= cy + qy))
    return set(int(i) for i in ids[mask])


def estimate_sxy_roi_area(
    sid: str,
    strict_hcr_ids=None,
    center_fov_quarter: bool = True,
    area_mode: str = "max_xsection",
) -> dict:
    """Estimate sxy from CZ↔HCR per-cell ROI area ratio.

    Parameters
    ----------
    sid : str
        Subject id (spot or intensity).
    strict_hcr_ids : iterable of int, optional
        HCR cell ids to use. Defaults to the BIC-GMM strict-GFP+ set
        from ``07b_gfp_intersection_threshold.strict_gfp_df``.
    center_fov_quarter : bool
        If True (default), restrict HCR cells to the center ¼ FOV so
        the CZ and HCR footprints match in aspect.
    area_mode : ``"bbox"`` | ``"max_xsection"``
        How to compute per-cell xy area. ``"max_xsection"`` (DEFAULT since
        2026-06-02) uses the maximum over z-slices of the in-plane mask pixel
        count — the truer footprint; nets closer to GT (mean |err| 4.1% vs
        bbox 4.6%, see sxy_maxxsection_vs_bbox_2026-06-01.md). ``"bbox"`` uses
        the tight xy bounding-box extents (legacy; over-estimates for elongated
        somata).

    Returns
    -------
    dict
        Includes ``sxy_median``, ``sxy_mean``, ``sxy_gt``,
        per-side medians / means, cell counts, depth spans, and the
        strict-cutoff metadata.  Also includes ``area_mode`` for
        downstream bookkeeping.
    """
    if area_mode not in ("bbox", "max_xsection"):
        raise ValueError(f"area_mode must be 'bbox' or 'max_xsection', got {area_mode!r}")
    # No hardcoded subject-set gate: the GFP class (spot/intensity) is detected
    # from the attached data downstream (gfp_threshold.detect_gfp_class), so new
    # subjects are supported.  Missing feature data raises a clear error there.
    # 755252 / 767022 (INTENSITY_SUBJECTS) lack metrics.pickle but DO have
    # segmentation_mask.zarr + cell_centroids.npy, so _load_hcr_metrics
    # reconstructs the per-cell index — they are supported here.
    s = load_subject(sid)
    info = analyze_subject(s)
    cz_surf, hcr_surf = info["cz_surface"], info["hcr_surface"]

    # Strict-GFP+ set — BIC-GMM intersection cutoff (07b), matching production.
    # Do NOT substitute s.hcr_gfp_df (peakgauss3_density_p0.1); that gives a
    # different, less-strict cell set and different sxy.
    strict_cutoff = None
    n_components = None
    if strict_hcr_ids is None:
        from autocoreg.io import gfp_threshold as _gfp
        gi = _gfp.analyze_subject(sid)
        strict_cutoff = float(gi.cutoff_linear)
        n_components = int(gi.n_components)
        strict_df = _gfp.strict_gfp_df(sid, strict_cutoff)
        strict_hcr_ids = set(int(x) for x in strict_df["hcr_id"].values)
    else:
        strict_hcr_ids = set(int(x) for x in strict_hcr_ids)

    if center_fov_quarter:
        fov_ids = _prefilter_center_fov(sid, s, strict_hcr_ids)
    else:
        fov_ids = strict_hcr_ids

    if area_mode == "bbox":
        cz_df_all = cz_cell_tight_bboxes(sid, s, cz_surf)
    else:
        cz_df_all = cz_cell_max_xsection(sid, s, cz_surf)
    hcr_df_all = hcr_cell_tight_bboxes(sid, s, hcr_surf, fov_ids, area_mode=area_mode)

    cz_span = float(np.nanpercentile(cz_df_all.depth_um, 99))
    hcr_span = float(np.nanpercentile(hcr_df_all.depth_um, 99))

    cz_mask = (cz_df_all["depth_um"] >= D_SKIN_UM) & (cz_df_all["depth_um"] <= cz_span)
    hcr_mask = (hcr_df_all["depth_um"] >= D_SKIN_UM) & (hcr_df_all["depth_um"] <= hcr_span)
    cz_df = cz_df_all[cz_mask].copy()
    hcr_df = hcr_df_all[hcr_mask].copy()

    med_cz = float(cz_df.xy_area_um2.median())
    med_hcr = float(hcr_df.xy_area_um2.median())
    mean_cz = float(cz_df.xy_area_um2.mean())
    mean_hcr = float(hcr_df.xy_area_um2.mean())
    sxy_med = float(np.sqrt(med_hcr / med_cz))
    sxy_mean = float(np.sqrt(mean_hcr / mean_cz))

    _gt = _gt_aniso_fit(s)
    sxy_gt = float(np.sqrt(_gt.scales[0] * _gt.scales[1])) if _gt is not None else float("nan")

    return {
        "sid": sid,
        "area_mode": area_mode,
        "sxy_median": sxy_med,
        "sxy_mean": sxy_mean,
        "sxy_gt": sxy_gt,
        "err_pct_median": 100 * (sxy_med - sxy_gt) / sxy_gt,
        "err_pct_mean": 100 * (sxy_mean - sxy_gt) / sxy_gt,
        "n_cz": int(len(cz_df)),
        "n_hcr_strict": int(len(hcr_df)),
        "cz_area_median": med_cz,
        "hcr_area_median": med_hcr,
        "cz_area_mean": mean_cz,
        "hcr_area_mean": mean_hcr,
        "cz_span_um": cz_span,
        "hcr_span_um": hcr_span,
        "center_fov_quarter": bool(center_fov_quarter),
        "strict_cutoff_linear": strict_cutoff,
        "n_components_bic": n_components,
        "n_hcr_input": len(strict_hcr_ids),
        "n_hcr_center_fov": len(fov_ids),
    }


# -----------------------------------------------------------
# Slab∩ok estimator — thin-HCR branch
# -----------------------------------------------------------
def _argmax_ok_ids(sid: str) -> set[int] | None:
    """Return HCR ids whose 4-class argmax ∈ {p_good, p_bad_ok}.

    Reads ``cached_roi_quality/{sid}_roi_quality_proba.parquet``
    (relative to this module's directory).  Returns None when the file is
    missing so callers can fall back gracefully.

    Coordinate frame: hcr_id labels in the HCR segmentation (dimensionless
    integer cell identifiers, same frame as hcr_centroids and metrics.pickle).
    """
    p = ROI_QUALITY_DIR / f"{sid}_roi_quality_proba.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    cls = ["p_bad", "p_bad_ok", "p_good", "p_merged"]
    am = df[cls].idxmax(axis=1)
    return set(df.loc[am.isin(["p_good", "p_bad_ok"]), "hcr_id"].astype(int).tolist())


def estimate_sxy_roi_area_slab(
    sid: str,
    strict_hcr_ids=None,
) -> dict:
    """Slab∩ok sxy estimator for thin-HCR subjects.

    Unlike :func:`estimate_sxy_roi_area` (which uses full-depth cells and
    the center-¼-FOV filter), this estimator:

    - HCR pool: strict-GFP+ ∩ argmax-ok, further restricted to the CZ
      FOV xy bbox (from ``overlap_crop``) to match the same tissue column
      as the surface-registration top-slab MIP.  NO _prefilter_center_fov.
    - HCR depth: [0, SLAB_HCR_UM] = [0, 100] µm below HCR pia.
    - CZ depth:  [0, SLAB_CZ_UM]  = [0, 50]  µm below CZ  pia.
    - Area: max_xsection (same as full-span production default).
    - sxy_median = sqrt(median(area_HCR_slab) / median(area_CZ_slab)).

    The slab bounds match surface_registration_v2.HCR_SLAB / CZ_SLAB
    (0–100 µm HCR, 0–50 µm CZ), so both sides sample the same physical
    tissue band used to build the top-slab MIP for surface registration.
    Excluding the full-depth band avoids the truncation bias that makes
    the full-span estimator systematically underestimate sxy when the HCR
    depth is <700 µm (as for 782149 ~566 µm).

    The xy bbox filter is crucial for thin-HCR subjects like 782149 where
    the GFP+ cells span the full HCR FOV (much larger than the CZ FOV),
    so using all xy positions introduces FOV-mismatch cells that inflate
    or deflate the area ratio.  The bbox is read from the currently cached
    surface registration via ``overlap_crop`` — this is by design: the
    estimator is called once from ``compute_surface_registration`` before
    the cache is written, so it uses whatever bbox the current (warm-start)
    registration provides.

    Returns the same dict schema as :func:`estimate_sxy_roi_area` with
    additional keys ``method="slab_ok"`` and ``truncated=True``.

    Parameters
    ----------
    sid : str
    strict_hcr_ids : iterable of int, optional
        Override the BIC-GMM strict-GFP+ set (mainly for testing).

    Raises
    ------
    RuntimeError
        If fewer than 5 HCR or CZ cells survive the slab filter (the
        median would be unreliable).
    """
    # No hardcoded subject-set gate: the GFP class (spot/intensity) is detected
    # from the attached data downstream (gfp_threshold.detect_gfp_class), so new
    # subjects are supported.  Missing feature data raises a clear error there.
    s = load_subject(sid)
    info = analyze_subject(s)
    cz_surf, hcr_surf = info["cz_surface"], info["hcr_surface"]

    # Strict-GFP+ set — BIC-GMM intersection cutoff (07b), matching production.
    strict_cutoff = None
    n_components = None
    if strict_hcr_ids is None:
        from autocoreg.io import gfp_threshold as _gfp
        gi = _gfp.analyze_subject(sid)
        strict_cutoff = float(gi.cutoff_linear)
        n_components = int(gi.n_components)
        strict_df = _gfp.strict_gfp_df(sid, strict_cutoff)
        strict_hcr_ids = set(int(x) for x in strict_df["hcr_id"].values)
    else:
        strict_hcr_ids = set(int(x) for x in strict_hcr_ids)

    print(f"  [{sid}] estimate_sxy_roi_area_slab: n_gfp={len(strict_hcr_ids)}")

    # Argmax-ok filter — missing parquet → fall back but warn
    ok_ids = _argmax_ok_ids(sid)
    if ok_ids is None:
        print(f"  [{sid}] WARNING: no roi-quality parquet; slab estimator "
              "will use GFP+ only (argmax-ok filter skipped)")
        pool = strict_hcr_ids
    else:
        pool = strict_hcr_ids & ok_ids
    print(f"  [{sid}] n_pool (gfp+ok, pre-spatial-filter) = {len(pool)}")

    # Spatial filter via locked-prior warm-start (same filter as prepare_subject).
    # The prepare_subject pool restricts HCR cells to the CZ FOV + z-density
    # intersection, which is much tighter than the full GFP+∩ok set.  We
    # replicate this by importing run_step2p5_refined.prepare_subject and
    # intersecting with the GFP+∩ok pool.  prepare_subject reads the currently
    # cached surface registration (called BEFORE compute_surface_registration
    # writes the new cache), so it uses the warm-start bbox.
    try:
        from autocoreg.finetune_soma_print.pool_prep import prepare_subject
        from autocoreg.io.inputs import load_sz_pins
        _sz_pins = load_sz_pins()
        _subj = prepare_subject(sid, sz_pins=_sz_pins)
        _spatial_pool = set(int(h) for h in _subj["hcr_pool_ids"]) & pool
        print(f"  [{sid}] spatial pool (prepare_subject∩gfp+ok): "
              f"{len(_spatial_pool)}  (from {len(_subj['hcr_pool_ids'])} pool total)")
        hcr_input_ids = _spatial_pool
    except Exception as _sp_err:
        print(f"  [{sid}] WARNING: prepare_subject unavailable ({_sp_err}); "
              "using full GFP+∩ok without spatial filter")
        hcr_input_ids = pool

    # HCR side: compute tight bboxes for the spatially filtered pool, then
    # filter to slab depth [0, SLAB_HCR_UM].  NO _prefilter_center_fov.
    hcr_df_all = hcr_cell_tight_bboxes(
        sid, s, hcr_surf, hcr_input_ids, area_mode="max_xsection"
    )
    hcr_span_um = float(np.nanpercentile(hcr_df_all.depth_um, 99))

    hcr_slab_mask = (
        (hcr_df_all["depth_um"] >= 0.0) & (hcr_df_all["depth_um"] <= SLAB_HCR_UM)
    )
    hcr_df = hcr_df_all[hcr_slab_mask].copy()
    n_hcr = int(len(hcr_df))
    print(f"  [{sid}] HCR slab [0,{SLAB_HCR_UM:.0f}µm]: n={n_hcr}  "
          f"(spatial_pool={len(hcr_df_all)}, hcr_span_p99={hcr_span_um:.0f}µm)")

    # CZ side: all cells, max_xsection, slab depth [0, SLAB_CZ_UM]
    cz_df_all = cz_cell_max_xsection(sid, s, cz_surf)
    cz_slab_mask = (
        (cz_df_all["depth_um"] >= 0.0) & (cz_df_all["depth_um"] <= SLAB_CZ_UM)
    )
    cz_df = cz_df_all[cz_slab_mask].copy()
    n_cz = int(len(cz_df))
    print(f"  [{sid}] CZ slab [0,{SLAB_CZ_UM:.0f}µm]: n={n_cz}")

    if n_hcr < 5 or n_cz < 5:
        raise RuntimeError(
            f"{sid}: slab filter too aggressive — n_hcr={n_hcr}, n_cz={n_cz}; "
            "cannot compute reliable median"
        )

    med_hcr = float(hcr_df.xy_area_um2.median())
    med_cz = float(cz_df.xy_area_um2.median())
    sxy_med = float(np.sqrt(med_hcr / med_cz))
    print(f"  [{sid}] med_hcr={med_hcr:.1f}µm² med_cz={med_cz:.1f}µm² "
          f"sxy_slab_ok={sxy_med:.4f}")

    _gt = _gt_aniso_fit(s)
    sxy_gt = float(np.sqrt(_gt.scales[0] * _gt.scales[1])) if _gt is not None else float("nan")

    return {
        "sid": sid,
        "method": "slab_ok",
        "area_mode": "max_xsection",
        "sxy_median": sxy_med,
        "sxy_gt": sxy_gt,
        "err_pct_median": 100.0 * (sxy_med - sxy_gt) / sxy_gt,
        "n_cz": n_cz,
        "n_hcr": n_hcr,
        "hcr_area_median": med_hcr,
        "cz_area_median": med_cz,
        "hcr_span_um": hcr_span_um,
        "truncated": True,
        "strict_cutoff_linear": strict_cutoff,
        "n_components_bic": n_components,
        "n_hcr_input": len(strict_hcr_ids),
        "n_hcr_ok": len(pool),
    }


def estimate_sxy_min_rule(sid: str, strict_hcr_ids=None) -> dict:
    """PRODUCTION sxy estimator (promoted 2026-06-04): min-rule, 2× heuristic, ¼-FOV.

    Under isotropic lateral tissue expansion ``sxy`` the in-plane footprint of a
    soma scales as ``sxy²``, so ``sxy = sqrt(median(area_HCR) / median(area_CZ))``
    (max-xsection footprints, the truer per-cell area; see :func:`estimate_sxy_roi_area`).
    This estimator computes that ratio over a **matched depth band** chosen by the
    min-rule, instead of the full HCR/CZ depth span.

    The rule (NO sz — see below for why)
    ------------------------------------
    ::

        zstack_thickness = p99( CZ max_xsection depth_um )                 # ~390 µm
        hcr_thickness    = p99( HCR (GFP+∩ok∩¼-FOV) depth_um )
        hcr_slab = min( hcr_thickness, MIN_RULE_AXIAL_FACTOR · zstack_thickness )
        cz_slab  = hcr_slab / MIN_RULE_AXIAL_FACTOR                        # CZ slab is HALF the HCR slab
        med_HCR  = median max_xsection of (strict_gfp ∩ argmax_ok ∩ ¼-FOV) with depth ∈ [0, hcr_slab]
        med_CZ   = median max_xsection of ALL CZ cells              with depth ∈ [0, cz_slab]
        sxy      = sqrt( med_HCR / med_CZ )

    Why the CZ slab is HALF the HCR slab (the CZ-cap rationale)
    -----------------------------------------------------------
    HCR tissue is axially expanded ~2× relative to the CZ z-stack, so the same
    physical cortical column occupies ~twice the depth in HCR.  To compare the
    *same physical tissue band* on both sides we cap the CZ slab at half the HCR
    slab.  Capping CZ shallower keeps it in the upper cortical layers (smaller,
    more uniform somata) → it lowers the CZ median area → it RAISES the sxy
    estimate, which is the correction the full-span estimator was missing for
    thin-HCR subjects (it under-estimated sxy by averaging over too-different
    depth ranges).

    Why a heuristic 2×, not the measured sz (circularity)
    -----------------------------------------------------
    The axial-expansion factor is fixed at ``MIN_RULE_AXIAL_FACTOR = 2.0`` rather
    than read from ``sz_estimator.get_sz``.  ``sz`` itself needs a correct
    registration to estimate (the FFT-NCC sweep is over an already-posed volume),
    and this sxy value is consumed to *build* that pose — using sz here would be
    circular.  The 2× heuristic is the documented mean axial expansion and is
    pose-free.

    ¼-FOV is HCR-side only
    ----------------------
    The center-¼-FOV filter (:func:`_prefilter_center_fov`) is applied to the HCR
    cells only — it matches the HCR footprint aspect to the (smaller) CZ FOV.  The
    CZ side uses ALL cells (its FOV is already the matched column).

    HCR-limited flag
    ----------------
    ``hcr_limited`` is True when ``hcr_thickness < 2·zstack_thickness`` (the HCR
    volume itself is thinner than 2× the CZ depth, so ``hcr_slab`` is clamped to
    the HCR thickness rather than to 2× the CZ depth).  This is the thin-HCR case
    (e.g. 782149 ~566 µm).

    Reference value: ``estimate_sxy_min_rule("782149")["sxy_median"] == 1.7336``.

    Parameters
    ----------
    sid : str
        Subject id (spot or intensity — same support as
        :func:`estimate_sxy_roi_area`).
    strict_hcr_ids : iterable of int, optional
        Override the BIC-GMM strict-GFP+ set (mainly for testing).

    Returns
    -------
    dict
        ``sxy_median, zstack_thickness, hcr_thickness, hcr_slab, cz_slab,
        med_hcr, med_cz, n_hcr, n_cz, hcr_limited (bool), sxy_gt,
        err_pct_median, method="min_rule_2x_quarterfov"``.
    """
    # No hardcoded subject-set gate: the GFP class (spot/intensity) is detected
    # from the attached data downstream (gfp_threshold.detect_gfp_class), so new
    # subjects are supported.  Missing feature data raises a clear error there.
    s = load_subject(sid)
    info = analyze_subject(s)
    cz_surf, hcr_surf = info["cz_surface"], info["hcr_surface"]

    # strict-GFP+ set — BIC-GMM intersection cutoff (07b), seeded random_state=0
    # for determinism.  This is the production source: it uses a stricter cutoff
    # than the peakgauss3_density_p0.1 loader default and gives different (correct)
    # n_strict values (e.g. 790322 9675 vs loader ~13k).  Do NOT substitute
    # s.hcr_gfp_df here — that gives a different cell set and a different sxy.
    if strict_hcr_ids is None:
        from autocoreg.io import gfp_threshold as _gfp
        gi = _gfp.analyze_subject(sid)
        strict_df = _gfp.strict_gfp_df(sid, float(gi.cutoff_linear))
        strict_hcr_ids = set(int(x) for x in strict_df["hcr_id"].values)
    else:
        strict_hcr_ids = set(int(x) for x in strict_hcr_ids)

    # HCR pool: strict-GFP+ ∩ argmax-ok ∩ center-¼-FOV (HCR side only)
    ok = _argmax_ok_ids(sid)
    pool = strict_hcr_ids & (ok if ok is not None else strict_hcr_ids)
    fov = _prefilter_center_fov(sid, s, pool)
    hdf = hcr_cell_tight_bboxes(sid, s, hcr_surf, fov, area_mode="max_xsection")
    hdf = hdf[(hdf.depth_um >= 0) & (hdf.xy_area_um2 > 0)]

    # CZ side: ALL cells, max_xsection footprint
    cdf = cz_cell_max_xsection(sid, s, cz_surf)
    cdf = cdf[(cdf.depth_um >= 0) & (cdf.xy_area_um2 > 0)]

    zstack_thickness = float(np.nanpercentile(cdf.depth_um, 99))
    hcr_thickness = float(np.nanpercentile(hdf.depth_um, 99))
    hcr_slab = min(hcr_thickness, MIN_RULE_AXIAL_FACTOR * zstack_thickness)
    cz_slab = hcr_slab / MIN_RULE_AXIAL_FACTOR
    hcr_limited = bool(hcr_thickness < MIN_RULE_AXIAL_FACTOR * zstack_thickness)

    h = hdf[hdf.depth_um <= hcr_slab]
    c = cdf[cdf.depth_um <= cz_slab]
    # Guard against an empty/degenerate slab (a new subject with a very thin
    # HCR span, aggressive ¼-FOV∩ok∩GFP+, or no shallow CZ cells) — fail loudly
    # rather than let a NaN sxy propagate silently into the pose. If this fires
    # for new data, the fallback is the SXY_GRID_SEARCH_OFFSETS seed-scan.
    if len(h) < 5 or len(c) < 5:
        raise RuntimeError(
            f"{sid}: min-rule sxy slab too sparse (n_hcr={len(h)}, n_cz={len(c)}); "
            f"hcr_slab={hcr_slab:.0f} cz_slab={cz_slab:.0f}. Fall back to "
            f"SXY_GRID_SEARCH_OFFSETS seed-scan.")
    med_hcr = float(h.xy_area_um2.median())
    med_cz = float(c.xy_area_um2.median())
    if not (med_cz > 0) or not np.isfinite(med_hcr):
        raise RuntimeError(
            f"{sid}: min-rule sxy degenerate median (med_hcr={med_hcr}, "
            f"med_cz={med_cz}). Fall back to SXY_GRID_SEARCH_OFFSETS seed-scan.")
    sxy = float(np.sqrt(med_hcr / med_cz))

    _gt = _gt_aniso_fit(s)
    sxy_gt = float(np.sqrt(_gt.scales[0] * _gt.scales[1])) if _gt is not None else float("nan")

    return {
        "sid": sid,
        "method": "min_rule_2x_quarterfov",
        "sxy_median": sxy,
        "zstack_thickness": zstack_thickness,
        "hcr_thickness": hcr_thickness,
        "hcr_slab": hcr_slab,
        "cz_slab": cz_slab,
        "med_hcr": med_hcr,
        "med_cz": med_cz,
        "n_hcr": int(len(h)),
        "n_cz": int(len(c)),
        "hcr_limited": hcr_limited,
        "sxy_gt": sxy_gt,
        "err_pct_median": 100.0 * (sxy - sxy_gt) / sxy_gt,
    }


def estimate_sxy_auto(
    sid: str,
    span_threshold_um: float = HCR_TRUNCATION_SPAN_UM,
) -> dict:
    """DEPRECATED (2026-06-04) — DO NOT use in production. Auto-detect thin vs
    thick HCR and return the appropriate sxy estimate.

    ⚠ The thin-HCR slab∩ok branch was found to be a NON-REPRODUCIBLE artifact
    (its overlap-bbox pool is circular w.r.t. the pose) and, on the corrected
    pose-independent GT, it does NOT recover 782149 (collapses at the honest
    full-span sxy regardless of slab/MIP thickness). `surface_registration_v2`
    no longer calls this — it uses plain `estimate_sxy_roi_area` (full-span).
    Kept only as a research record. See project_782149_sxy_gt_leak memory.

    Strategy
    --------
    Cheaply obtains the strict-GFP+ depth span (p99) via
    :func:`estimate_sxy_roi_area` (which caches its zarr I/O via the
    tight-bbox cache).  If span < ``span_threshold_um`` (default 700 µm),
    the HCR volume is truncated; the full-span estimator under-samples the
    cortex and underestimates sxy.  In that case the slab∩ok recipe
    (:func:`estimate_sxy_roi_area_slab`) is used instead.

    NEVER falls back to GT.

    Parameters
    ----------
    sid : str
    span_threshold_um : float
        GFP+ depth p99 (µm) below which we switch to the slab recipe.
        Default ``HCR_TRUNCATION_SPAN_UM`` = 700 µm.

    Returns
    -------
    dict
        Always contains: ``sxy_median``, ``hcr_span_um``, ``truncated``
        (bool), ``span_threshold_um``, ``method`` (``"full_span"`` or
        ``"slab_ok"``), ``sxy_gt``, ``err_pct_median``.  Additional keys
        from the underlying estimator are passed through.
    """
    # No hardcoded subject-set gate: the GFP class (spot/intensity) is detected
    # from the attached data downstream (gfp_threshold.detect_gfp_class), so new
    # subjects are supported.  Missing feature data raises a clear error there.

    # Always run full_span first — it's cheap after the first call
    # because it uses cached tight bboxes and max-xsection.
    full = estimate_sxy_roi_area(sid, area_mode="max_xsection")
    hcr_span_um = float(full["hcr_span_um"])
    truncated = hcr_span_um < span_threshold_um

    print(f"  [{sid}] hcr_span_um={hcr_span_um:.0f}  threshold={span_threshold_um:.0f}  "
          f"truncated={truncated}")

    if not truncated:
        return dict(
            full,
            method="full_span",
            truncated=False,
            span_threshold_um=span_threshold_um,
        )

    # Thin-HCR branch: slab∩ok estimator
    slab = estimate_sxy_roi_area_slab(sid)
    return dict(
        slab,
        truncated=True,
        span_threshold_um=span_threshold_um,
    )


def _cz_xyz_um(s: SubjectData) -> np.ndarray:
    arr = s.cz_centroids[["z_px", "y_px", "x_px"]].to_numpy(float)
    return cz_px_to_um(arr, s)[:, [2, 1, 0]]


def _hcr_gfp_xyz_um(s: SubjectData, strict_hcr_ids) -> np.ndarray:
    """Return HCR strict-GFP+ centroids as (x, y, z) µm via hcr_id join."""
    want = set(int(x) for x in strict_hcr_ids)
    if not want or s.hcr_centroids.empty:
        return np.zeros((0, 3))
    px = s.hcr_centroids.copy()
    keep = px["hcr_id"].astype(int).isin(want)
    arr = px.loc[keep, ["z_px", "y_px", "x_px"]].to_numpy(float)
    if arr.size == 0:
        return np.zeros((0, 3))
    return hcr_px_to_um(arr, s)[:, [2, 1, 0]]


def estimate_sz_roi_density(
    sid: str,
    sxy: float | None = None,
    strict_hcr_ids=None,
) -> dict:
    """Estimate sz by ROI-density conservation across matched CZ ↔ HCR boxes.

    Under isotropic xy stretch ``sxy`` and axial stretch ``sz``, the same
    physical tissue occupies a volume that is ``sxy² · sz`` larger in
    HCR than in CZ. With the same cell count N in both, the native
    densities satisfy ``ρ_CZ / ρ_HCR = sxy² · sz``.

    Recipe
    ------
    1. ``sxy`` from :func:`estimate_sxy_roi_area` (tight-bbox area ratio)
       unless supplied.
    2. R1 (``coarse_align_revised``) projects CZ centroid mean into the
       HCR µm frame — the HCR box centre. The box xy extent is
       ``sxy × (CZ xy extent)`` so that after undoing the stretch the
       box represents the same physical column as the CZ FOV.
    3. In CZ native: keep centroids with pia depth ∈ [d_skin, p99];
       ``V_CZ = A_CZ · (d_cz_span − d_skin)``, ``N_CZ`` = surviving count.
    4. In HCR native: strict-GFP+ centroids whose xy lies in the R1-
       centred matched box and whose pia depth ∈ [d_skin, p99];
       ``V_HCR = (sxy² · A_CZ) · (d_hcr_span − d_skin)``.
    5. ``sz = (ρ_CZ / ρ_HCR) / sxy² = (N_CZ · T_HCR) / (N_HCR · T_CZ)``.

    Notes
    -----
    - ``A_CZ`` is the xy AABB of CZ centroids that survive the cortex
      filter (not the full FOV, so skin/outside-FOV volume isn't
      over-counted).
    - Accuracy depends on (i) sxy (which propagates as 1/sxy²), and
      (ii) strict-GFP+ vs CZ-active cell-count matching — 07c showed
      integrated GFP+/truth ≈ 1.0, so bulk density should be unbiased.
    """
    from autocoreg.io.gfp_threshold import detect_gfp_class
    if detect_gfp_class(sid) != "spot":
        raise ValueError(
            f"{sid}: intensity/unsupported subject — no HCR "
            "cell_body_segmentation/metrics.pickle"
        )
    s = load_subject(sid)
    info = analyze_subject(s)
    cz_surf, hcr_surf = info["cz_surface"], info["hcr_surface"]

    n_components = None
    strict_cutoff = None
    if sxy is None:
        roi = estimate_sxy_roi_area(sid)
        sxy = float(roi["sxy_median"])
        n_components = roi.get("n_components_bic")
        strict_cutoff = roi.get("strict_cutoff_linear")

    if strict_hcr_ids is None:
        strict_hcr_ids = set(int(x) for x in s.hcr_gfp_df["hcr_id"].values)

    cz_um = _cz_xyz_um(s)
    gfp_um = _hcr_gfp_xyz_um(s, strict_hcr_ids)
    if len(gfp_um) == 0:
        raise RuntimeError(f"{sid}: no strict GFP+ centroids")

    # CZ side — cortex band + AABB
    cz_depth = depth_from_surface(cz_um, cz_surf)
    cz_span = float(np.nanpercentile(cz_depth, 99))
    cz_mask = (cz_depth >= D_SKIN_UM) & (cz_depth <= cz_span)
    n_cz = int(cz_mask.sum())
    if n_cz == 0:
        raise RuntimeError(f"{sid}: no CZ cortex cells")
    cz_xy = cz_um[cz_mask, :2]
    cz_x_span = float(cz_xy[:, 0].max() - cz_xy[:, 0].min())
    cz_y_span = float(cz_xy[:, 1].max() - cz_xy[:, 1].min())
    cz_A = cz_x_span * cz_y_span
    cz_T = cz_span - D_SKIN_UM
    cz_V = cz_A * cz_T

    # R1 to locate CZ centroid mean in HCR frame (rigid only; ignore
    # R1's own scales — sxy is trusted from the area estimator).
    fit = coarse_align_revised(
        cz_um, gfp_um, cz_surf, hcr_surf, aniso_refine=False
    )
    hcr_box_center = fit.minimal_translation  # (x, y, z) µm in HCR

    # HCR matched box — xy at the R1-projected CZ centre, half-widths
    # = sxy × (CZ half-spans), so A_HCR = sxy² · A_CZ
    half_x = 0.5 * sxy * cz_x_span
    half_y = 0.5 * sxy * cz_y_span
    box_x0 = hcr_box_center[0] - half_x
    box_x1 = hcr_box_center[0] + half_x
    box_y0 = hcr_box_center[1] - half_y
    box_y1 = hcr_box_center[1] + half_y
    hcr_A = (box_x1 - box_x0) * (box_y1 - box_y0)  # = sxy² · cz_A

    # HCR side — depth band on GFP+ cells inside the matched xy box
    gfp_depth = depth_from_surface(gfp_um, hcr_surf)
    in_box_xy = (
        (gfp_um[:, 0] >= box_x0) & (gfp_um[:, 0] <= box_x1)
        & (gfp_um[:, 1] >= box_y0) & (gfp_um[:, 1] <= box_y1)
    )
    gfp_in_box_depth = gfp_depth[in_box_xy]
    if gfp_in_box_depth.size == 0:
        raise RuntimeError(f"{sid}: no HCR strict-GFP+ cells in matched box")
    hcr_span = float(np.nanpercentile(gfp_in_box_depth, 99))
    hcr_mask = in_box_xy & (gfp_depth >= D_SKIN_UM) & (gfp_depth <= hcr_span)
    n_hcr = int(hcr_mask.sum())
    hcr_T = hcr_span - D_SKIN_UM
    hcr_V = hcr_A * hcr_T

    rho_cz = n_cz / cz_V
    rho_hcr = n_hcr / hcr_V if hcr_V > 0 else float("nan")
    if rho_hcr == 0 or not np.isfinite(rho_hcr):
        raise RuntimeError(f"{sid}: degenerate HCR density")
    sz_est = (rho_cz / rho_hcr) / (sxy ** 2)

    _gt = _gt_aniso_fit(s)
    sxy_gt = float(np.sqrt(_gt.scales[0] * _gt.scales[1])) if _gt is not None else float("nan")
    sz_gt = float(_gt.scales[2]) if _gt is not None else float("nan")

    return {
        "sid": sid,
        "sz": sz_est,
        "sz_gt": sz_gt,
        "err_pct": 100 * (sz_est - sz_gt) / sz_gt,
        "sxy_used": float(sxy),
        "sxy_gt": sxy_gt,
        "n_cz": n_cz,
        "n_hcr": n_hcr,
        "cz_T_um": cz_T,
        "hcr_T_um": hcr_T,
        "cz_A_um2": cz_A,
        "hcr_A_um2": hcr_A,
        "rho_cz_per_um3": rho_cz,
        "rho_hcr_per_um3": rho_hcr,
        "hcr_box_xy": [box_x0, box_x1, box_y0, box_y1],
        "r1_translation": hcr_box_center.tolist(),
        "strict_cutoff_linear": strict_cutoff,
        "n_components_bic": n_components,
    }


if __name__ == "__main__":
    import json
    cmd = sys.argv[1] if len(sys.argv) > 1 else "both"
    if cmd == "sxy":
        sids, fn = sys.argv[2:] or sorted(SPOT_SUBJECTS), estimate_sxy_roi_area
    elif cmd == "sz":
        sids, fn = sys.argv[2:] or sorted(SPOT_SUBJECTS), estimate_sz_roi_density
    else:
        sids = sys.argv[1:] or sorted(SPOT_SUBJECTS)
        fn = None

    if fn is None:
        print(f"{'sid':<8} {'sxy':>6} {'sxy_GT':>7} {'sxy_err':>8}  "
              f"{'sz':>6} {'sz_GT':>7} {'sz_err':>8}")
        for sid in sids:
            try:
                rsxy = estimate_sxy_roi_area(sid)
                rsz = estimate_sz_roi_density(sid, sxy=rsxy["sxy_median"])
            except Exception as e:
                print(f"{sid}: ERROR {e}")
                continue
            print(f"{sid:<8} {rsxy['sxy_median']:>6.3f} "
                  f"{rsxy['sxy_gt']:>7.3f} {rsxy['err_pct_median']:>+7.1f}%  "
                  f"{rsz['sz']:>6.3f} {rsz['sz_gt']:>7.3f} "
                  f"{rsz['err_pct']:>+7.1f}%")
    else:
        out = [fn(sid) for sid in sids]
        print(json.dumps(out, indent=2, default=float))
