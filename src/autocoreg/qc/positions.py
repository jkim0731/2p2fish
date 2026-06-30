"""Qt-free, single-source-of-truth helpers for per-pair position tables.

Both the PyQt5 QC app (autocoreg.qc.app) and headless callers (the autocoreg
capsule) import from here so GUI-export and headless-export are byte-identical.

Coordinate conventions
----------------------
All positions are in µm unless the column name ends in ``_px``.
``cz_native_*`` : CZ native stack frame (from centroids_um(s, "cz")).
``cz_in_hcr_*`` : CZ warped into HCR µm frame (baseline = cz_world centroid
                  from the warped-seg volume; overridden by TPS in the GUI).
``hcr_*``       : HCR native frame (from centroids_um(s, "hcr_all")).

The ``derived`` dict passed around here must have the following keys:
    cz_native_by_id : dict[int, ndarray(3,)]  CZ native µm zyx per cz_id
    cz_in_hcr_by_id : dict[int, ndarray(3,)]  CZ-in-HCR µm zyx per cz_id
    hcr_by_id       : dict[int, ndarray(3,)]  HCR µm zyx per hcr_id
    cz_xy_um        : float
    cz_z_um         : float
    hcr_xy_um       : float
    hcr_z_um        : float
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

POS_COLS: list[str] = [
    "cz_id", "hcr_id", "soma_score",
    "cz_native_z_um", "cz_native_y_um", "cz_native_x_um",
    "cz_native_z_px", "cz_native_y_px", "cz_native_x_px",
    "cz_in_hcr_z_um", "cz_in_hcr_y_um", "cz_in_hcr_x_um",
    "hcr_z_um", "hcr_y_um", "hcr_x_um",
    "hcr_z_px", "hcr_y_px", "hcr_x_px",
]


def fmt_val(v) -> str:
    """Format one cell for the positions CSV.

    Floats → 3 decimal places; NaN → empty string; everything else → str.
    This matches the app's original ``_fmt_val`` exactly.
    """
    if isinstance(v, float):
        return "" if math.isnan(v) else f"{v:.3f}"
    return str(v)


# ---------------------------------------------------------------------------
# Core array helpers
# ---------------------------------------------------------------------------

def compute_centroids(label_arr: np.ndarray, ids: list[int]) -> dict[int, np.ndarray]:
    """Pixel-index centroids for each label id via bincount.

    Returns ``{id: ndarray([z, y, x])}`` in voxel coordinates (float).
    Ids absent from the array or covering zero voxels are omitted.
    """
    flat = label_arr.ravel()
    mask = np.isin(flat, ids)
    if not mask.any():
        return {}
    labels = flat[mask]
    z_idx, y_idx, x_idx = np.unravel_index(np.flatnonzero(mask), label_arr.shape)
    ml = int(labels.max()) + 1
    sums_z = np.bincount(labels, weights=z_idx, minlength=ml)
    sums_y = np.bincount(labels, weights=y_idx, minlength=ml)
    sums_x = np.bincount(labels, weights=x_idx, minlength=ml)
    counts = np.bincount(labels, minlength=ml)
    out: dict[int, np.ndarray] = {}
    for v in set(ids):
        if v < len(counts) and counts[v] > 0:
            out[int(v)] = np.array(
                [sums_z[v] / counts[v],
                 sums_y[v] / counts[v],
                 sums_x[v] / counts[v]], dtype=float
            )
    return out


def cz_world_from_seg(
    cz_matched_arr: np.ndarray,
    cz_unmatched_arr: np.ndarray,
    cz_bb: dict,
    cz_vox: float,
) -> dict[int, np.ndarray]:
    """Baseline CZ centroids warped into HCR µm from the warped-seg volumes.

    ``cz_bb`` must have keys ``z_lo``, ``y_lo``, ``x_lo`` (µm origins of the
    warped-CZ seg bbox).  ``cz_vox`` is the isotropic voxel size in µm.

    The matched array takes priority over unmatched where both are nonzero.
    Returns ``{cz_id: ndarray([z_um, y_um, x_um])}``.
    """
    combo = np.where(cz_matched_arr > 0, cz_matched_arr, cz_unmatched_arr)
    all_cz = sorted(
        set(int(v) for v in np.unique(cz_matched_arr) if v != 0)
        | set(int(v) for v in np.unique(cz_unmatched_arr) if v != 0)
    )
    cz_centroids_vox = compute_centroids(combo, all_cz)
    return {
        int(v): np.array([
            cz_bb["z_lo"] + c[0] * cz_vox,
            cz_bb["y_lo"] + c[1] * cz_vox,
            cz_bb["x_lo"] + c[2] * cz_vox,
        ])
        for v, c in cz_centroids_vox.items()
    }


# ---------------------------------------------------------------------------
# Per-row and per-table builders
# ---------------------------------------------------------------------------

def position_row(
    cz_id: int,
    hcr_id: int | None,
    soma: float,
    derived: dict,
) -> dict:
    """Build one POS_COLS dict for a single CZ ROI.

    ``hcr_id`` may be -1 or None (→ HCR fields omitted from dict).
    ``soma`` may be NaN (recorded as NaN).
    ``derived`` keys: cz_native_by_id, cz_in_hcr_by_id, hcr_by_id,
                       cz_xy_um, cz_z_um, hcr_xy_um, hcr_z_um.
    """
    cz_id = int(cz_id)
    hcr_matched = hcr_id is not None and int(hcr_id) != -1
    d: dict = {
        "cz_id": cz_id,
        "hcr_id": int(hcr_id) if hcr_id is not None else -1,
        "soma_score": float(soma),
    }

    cn = derived["cz_native_by_id"].get(cz_id)
    if cn is not None:
        d.update(
            cz_native_z_um=float(cn[0]),
            cz_native_y_um=float(cn[1]),
            cz_native_x_um=float(cn[2]),
            cz_native_z_px=cn[0] / derived["cz_z_um"],
            cz_native_y_px=cn[1] / derived["cz_xy_um"],
            cz_native_x_px=cn[2] / derived["cz_xy_um"],
        )

    cw = derived["cz_in_hcr_by_id"].get(cz_id)
    if cw is not None:
        d.update(
            cz_in_hcr_z_um=float(cw[0]),
            cz_in_hcr_y_um=float(cw[1]),
            cz_in_hcr_x_um=float(cw[2]),
        )

    if hcr_matched:
        hp = derived["hcr_by_id"].get(int(hcr_id))
        if hp is not None:
            d.update(
                hcr_z_um=float(hp[0]),
                hcr_y_um=float(hp[1]),
                hcr_x_um=float(hp[2]),
                hcr_z_px=hp[0] / derived["hcr_z_um"],
                hcr_y_px=hp[1] / derived["hcr_xy_um"],
                hcr_x_px=hp[2] / derived["hcr_xy_um"],
            )
    return d


def compute_pair_positions(
    cz_to_hcr: dict[int, int],
    cz_to_soma: dict[int, float],
    derived: dict,
    cz_ids: Sequence[int] | None = None,
) -> "pd.DataFrame":
    """Build a DataFrame with columns exactly POS_COLS for all cz_ids.

    ``cz_ids`` defaults to ``sorted(derived["cz_in_hcr_by_id"])``.
    Missing values (no native centroid, no HCR match) appear as NaN.
    """
    import pandas as pd  # lazy — keeps import-time cheap

    if cz_ids is None:
        cz_ids = sorted(derived["cz_in_hcr_by_id"])

    rows = [
        position_row(
            cz_id=c,
            hcr_id=cz_to_hcr.get(int(c)),
            soma=cz_to_soma.get(int(c), float("nan")),
            derived=derived,
        )
        for c in cz_ids
    ]
    df = pd.DataFrame(rows, columns=POS_COLS)
    return df


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def write_positions_csv(df: "pd.DataFrame", path) -> None:
    """Write a positions DataFrame to CSV using fmt_val per cell.

    Output is byte-identical to the app's hand-written CSV: 3-decimal floats,
    NaN → empty string, POS_COLS column order.
    """
    path = Path(path)
    with open(path, "w") as f:
        f.write(",".join(POS_COLS) + "\n")
        for _, row in df.iterrows():
            f.write(",".join(fmt_val(row.get(c, float("nan"))) for c in POS_COLS) + "\n")


# ---------------------------------------------------------------------------
# Qt-free artifact loader
# ---------------------------------------------------------------------------

def load_derived_from_artifacts(
    sid: str,
    artifact_dir,
    *,
    s=None,
) -> dict:
    """Load all ``derived`` fields from an artifact directory without PyQt5.

    Reads ``seg_volumes_meta.json``, ``cz_matched_seg.tif``,
    ``cz_unmatched_seg.tif`` to build ``cz_world``; then calls
    ``load_subject`` / ``centroids_um`` for native centroids and resolutions.

    ``s`` can be a pre-loaded SubjectData to skip the ``load_subject`` call.

    Returns a dict with keys:
        cz_native_by_id, cz_in_hcr_by_id, hcr_by_id,
        cz_xy_um, cz_z_um, hcr_xy_um, hcr_z_um
    """
    import json as _json
    import tifffile as _tifffile  # lazy
    from autocoreg.io.subjects import load_subject as _load_subject  # lazy
    from autocoreg.io.centroids import centroids_um as _centroids_um  # lazy

    adir = Path(artifact_dir)
    meta_path = adir / "seg_volumes_meta.json"
    seg_meta = _json.loads(meta_path.read_text())

    cz_bb = seg_meta["bbox_cz_warped"]
    cz_vox = float(seg_meta["voxel_um_cz_warped"])

    print(f"[positions] loading seg volumes from {adir}")
    cz_matched_arr = _tifffile.imread(str(adir / "cz_matched_seg.tif"))
    cz_unmatched_arr = _tifffile.imread(str(adir / "cz_unmatched_seg.tif"))
    print(f"[positions] cz_matched_arr shape={cz_matched_arr.shape} "
          f"dtype={cz_matched_arr.dtype} "
          f"max={int(cz_matched_arr.max())}")
    print(f"[positions] cz_unmatched_arr shape={cz_unmatched_arr.shape} "
          f"dtype={cz_unmatched_arr.dtype} "
          f"max={int(cz_unmatched_arr.max())}")

    cz_world = cz_world_from_seg(cz_matched_arr, cz_unmatched_arr, cz_bb, cz_vox)
    print(f"[positions] cz_world: {len(cz_world)} CZ centroids in HCR µm")

    if s is None:
        s = _load_subject(sid)

    cz_um, cz_ids = _centroids_um(s, "cz")
    hcr_um, hcr_ids = _centroids_um(s, "hcr_all")
    print(f"[positions] cz native centroids: {len(cz_ids)}  "
          f"HCR centroids: {len(hcr_ids)}")

    cz_native_by_id = {int(i): np.asarray(p, float) for i, p in zip(cz_ids, cz_um)}
    hcr_by_id = {int(i): np.asarray(p, float) for i, p in zip(hcr_ids, hcr_um)}

    return {
        "cz_native_by_id": cz_native_by_id,
        "cz_in_hcr_by_id": cz_world,   # baseline; GUI overrides with TPS
        "hcr_by_id": hcr_by_id,
        "cz_xy_um": float(s.cz_xy_um),
        "cz_z_um": float(s.cz_z_um),
        "hcr_xy_um": float(s.hcr_xy_um),
        "hcr_z_um": float(s.hcr_z_um),
    }


# ---------------------------------------------------------------------------
# Convenience entry point for the headless capsule
# ---------------------------------------------------------------------------

def positions_from_artifacts(
    sid: str,
    artifact_dir,
    matches_csv,
    final_pairs_csv=None,
    *,
    s=None,
) -> "pd.DataFrame":
    """End-to-end headless positions table for a single subject.

    ``matches_csv``   : path to matches CSV with columns cz_id, hcr_id.
    ``final_pairs_csv``: optional path to final_pairs.csv (cz_id, soma_score).
    ``s``             : optional pre-loaded SubjectData (skips load_subject).

    Returns a DataFrame with columns POS_COLS, one row per CZ ROI
    (matched + unmatched), sorted by cz_id.
    """
    import pandas as _pd  # lazy

    derived = load_derived_from_artifacts(sid, artifact_dir, s=s)

    df_m = _pd.read_csv(str(matches_csv))
    df_m["cz_id"] = df_m["cz_id"].astype(int)
    df_m["hcr_id"] = df_m["hcr_id"].astype(int)
    cz_to_hcr = dict(zip(df_m["cz_id"], df_m["hcr_id"]))
    print(f"[positions] matches: {len(cz_to_hcr)} pairs from {matches_csv}")

    cz_to_soma: dict[int, float] = {}
    if final_pairs_csv is not None:
        fp_path = Path(str(final_pairs_csv))
        if fp_path.exists():
            fp = _pd.read_csv(str(fp_path))
            cz_to_soma = {int(c): float(s_) for c, s_ in zip(fp["cz_id"], fp["soma_score"])}
            print(f"[positions] soma scores from {fp_path}: {len(cz_to_soma)} entries")

    return compute_pair_positions(cz_to_hcr, cz_to_soma, derived)
