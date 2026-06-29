"""Session 08 — surface-vascular match.

Compute tilt-corrected top-of-tissue slab projections for HCR (all
channels) and CZ, for the two primary subjects 788406 and 790322. The
surface fits flatten the pia so that per-(y,x) we sample voxels at
depth `d` below the pia; mean / max projections across small depth
slabs (0-30 / 30-60 / 60-100 µm) then expose in-plane vascular
features that are common to 2P (GCaMP) and HCR (especially 405).

Outputs per subject go to ``figures/`` as PNGs + a NumPy archive of
every projection array keyed by ``(modality, channel, slab, mode)``.
A notebook is built separately in ``_build_notebook.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile

from autocoreg.io.hcr_image import list_hcr_channels, load_hcr_volume
from autocoreg.io.subjects import load_subject
from autocoreg.initial_registration.surfaces import get_cz_surface_iter08, get_hcr_top_surface_iter07

HERE = Path(__file__).resolve().parent
FIG = Path("/scratch/autocoreg_outputs/dev")  # figures go to /scratch in package context (was HERE/figures)

SUBJECTS = ["788406", "790322"]
HCR_LEVEL = 4  # ~4 µm XY / 4 µm Z — plenty for top-surface vessels
SLABS_UM = [(0.0, 30.0), (30.0, 60.0), (60.0, 100.0)]


# --------------------------------------------------------------
# Core helpers
# --------------------------------------------------------------
def evaluate_surface(surf: dict, X: int, Y: int, xy_um: float) -> np.ndarray:
    """Evaluate the quadratic pia surface on the (Y, X) grid of the
    volume. Returns z_pia(y, x) in µm — the pia depth at each column.
    """
    xs = np.arange(X, dtype=np.float32) * xy_um
    ys = np.arange(Y, dtype=np.float32) * xy_um
    xx, yy = np.meshgrid(xs, ys)  # (Y, X)
    a, b, c = surf["a"], surf["b"], surf["c"]
    p = surf.get("p", 0.0); q = surf.get("q", 0.0); r = surf.get("r", 0.0)
    z_pia = a * xx + b * yy + c + p * xx * xx + q * xx * yy + r * yy * yy
    return z_pia.astype(np.float32)


def top_slab_projection(
    vol: np.ndarray,
    surf: dict,
    xy_um: float,
    z_um: float,
    d_lo: float,
    d_hi: float,
    mode: str = "max",
) -> np.ndarray:
    """For every (y, x) column, gather voxels at depth d ∈ [d_lo, d_hi]
    below the pia and project with ``mode`` ('max' or 'mean').

    Columns whose depth window falls outside the volume are returned
    as NaN. Output shape = (Y, X).
    """
    Z, Y, X = vol.shape
    z_pia = evaluate_surface(surf, X=X, Y=Y, xy_um=xy_um)  # (Y, X) in µm
    depths = np.arange(d_lo, d_hi + z_um / 2, z_um, dtype=np.float32)

    iy = np.arange(Y)[:, None]
    ix = np.arange(X)[None, :]
    acc = np.full((Y, X), np.nan, dtype=np.float32)
    counts = np.zeros((Y, X), dtype=np.int32)
    running_sum = np.zeros((Y, X), dtype=np.float64)

    for d in depths:
        iz = np.round((z_pia + d) / z_um).astype(np.int64)
        valid = (iz >= 0) & (iz < Z)
        iz_c = np.clip(iz, 0, Z - 1)
        sample = vol[iz_c, iy, ix].astype(np.float32)
        sample_valid = np.where(valid, sample, np.nan)

        if mode == "max":
            prev = np.where(np.isnan(acc), -np.inf, acc)
            cur = np.where(np.isnan(sample_valid), -np.inf, sample_valid)
            new = np.maximum(prev, cur)
            any_valid = valid | np.isfinite(acc)
            acc = np.where(any_valid, new, np.nan).astype(np.float32)
        elif mode == "mean":
            running_sum += np.where(valid, sample, 0.0)
            counts += valid.astype(np.int32)
        else:
            raise ValueError(f"unknown mode {mode!r}")

    if mode == "mean":
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(counts > 0, running_sum / np.maximum(counts, 1), np.nan)
        return out.astype(np.float32)
    return acc


def load_cz_zstack(s) -> np.ndarray:
    files = list(s.coreg_dir.glob("*reg-dim-swapped.ome.tif"))
    if not files:
        files = list(s.coreg_dir.glob("*zstack.tif"))
    with tifffile.TiffFile(files[0]) as tf:
        arr = tf.asarray()
    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]
    return np.asarray(arr, dtype=np.float32)


# --------------------------------------------------------------
# Driver
# --------------------------------------------------------------
def process_subject(sid: str) -> dict:
    print(f"\n=== {sid} ===", flush=True)
    s = load_subject(sid)
    out: dict = {"subject": sid, "projections": {}, "meta": {}}

    # --- HCR ---
    hcr_surf = get_hcr_top_surface_iter07(s, level=HCR_LEVEL)
    print(
        f"  hcr_top surface: median_col_z={hcr_surf['median_col_z_um']:.1f} µm  "
        f"tilt a={hcr_surf['a']:.4f}  b={hcr_surf['b']:.4f}  "
        f"quad p={hcr_surf['p']:.2e} q={hcr_surf['q']:.2e} r={hcr_surf['r']:.2e}"
    )
    out["meta"]["hcr_top_surface"] = {k: hcr_surf[k] for k in ("a", "b", "c", "p", "q", "r")}
    out["meta"]["hcr_top_median_z_um"] = float(hcr_surf["median_col_z_um"])

    channels = list_hcr_channels(s)
    out["meta"]["hcr_channels"] = channels

    for ch in channels:
        vol, xy_um, z_um = load_hcr_volume(s, channel=ch, level=HCR_LEVEL)
        vol = vol.astype(np.float32)
        print(f"  HCR ch{ch}: shape={vol.shape}  xy={xy_um:.3f} z={z_um:.3f}  "
              f"vol_p50={np.percentile(vol, 50):.1f}  p99={np.percentile(vol, 99):.1f}")
        for (d_lo, d_hi) in SLABS_UM:
            for mode in ("max", "mean"):
                proj = top_slab_projection(
                    vol, hcr_surf, xy_um=xy_um, z_um=z_um,
                    d_lo=d_lo, d_hi=d_hi, mode=mode,
                )
                key = f"HCR_ch{ch}_{int(d_lo)}-{int(d_hi)}_{mode}"
                out["projections"][key] = proj
                out["meta"].setdefault("hcr_xy_um", float(xy_um))
                out["meta"].setdefault("hcr_z_um", float(z_um))
        # Also keep a Y-slab MIP for sanity (y at mid, ±20 px wide)
        Y = vol.shape[1]
        y0 = max(0, Y // 2 - 20)
        y1 = min(Y, Y // 2 + 20)
        mip_zx = vol[:, y0:y1, :].max(axis=1)
        out["projections"][f"HCR_ch{ch}_yMIP"] = mip_zx.astype(np.float32)
        out["meta"].setdefault("hcr_vol_shape", list(vol.shape))
        out["meta"].setdefault(
            "hcr_y_center_um", float(((y0 + y1) / 2) * xy_um)
        )
        del vol  # free memory before next channel

    # --- CZ ---
    cz_surf = get_cz_surface_iter08(s)
    print(
        f"  cz surface: median_col_z={cz_surf['median_col_z_um']:.1f} µm  "
        f"tilt a={cz_surf['a']:.4f}  b={cz_surf['b']:.4f}  "
        f"quad p={cz_surf['p']:.2e} q={cz_surf['q']:.2e} r={cz_surf['r']:.2e}"
    )
    out["meta"]["cz_surface"] = {k: cz_surf[k] for k in ("a", "b", "c", "p", "q", "r")}
    out["meta"]["cz_median_z_um"] = float(cz_surf["median_col_z_um"])
    out["meta"]["cz_xy_um"] = float(s.cz_xy_um)
    out["meta"]["cz_z_um"] = float(s.cz_z_um)

    cz_vol = load_cz_zstack(s)
    print(
        f"  CZ:     shape={cz_vol.shape}  xy={s.cz_xy_um:.3f} z={s.cz_z_um:.3f}  "
        f"vol_p50={np.percentile(cz_vol, 50):.1f}  p99={np.percentile(cz_vol, 99):.1f}"
    )
    for (d_lo, d_hi) in SLABS_UM:
        for mode in ("max", "mean"):
            proj = top_slab_projection(
                cz_vol, cz_surf, xy_um=float(s.cz_xy_um), z_um=float(s.cz_z_um),
                d_lo=d_lo, d_hi=d_hi, mode=mode,
            )
            key = f"CZ_{int(d_lo)}-{int(d_hi)}_{mode}"
            out["projections"][key] = proj

    Y = cz_vol.shape[1]
    y0 = max(0, Y // 2 - 20); y1 = min(Y, Y // 2 + 20)
    out["projections"]["CZ_yMIP"] = cz_vol[:, y0:y1, :].max(axis=1).astype(np.float32)
    out["meta"]["cz_vol_shape"] = list(cz_vol.shape)
    out["meta"]["cz_y_center_um"] = float(((y0 + y1) / 2) * float(s.cz_xy_um))

    # Save arrays
    npz_path = HERE / f"projections_{sid}.npz"
    meta_path = HERE / f"meta_{sid}.json"
    np.savez_compressed(npz_path, **{k: v for k, v in out["projections"].items()})
    meta_path.write_text(json.dumps(out["meta"], indent=2, default=float))
    print(f"  wrote {npz_path.name} ({sum(p.nbytes for p in out['projections'].values())/1e6:.1f} MB)")
    return out


def main():
    for sid in SUBJECTS:
        process_subject(sid)


if __name__ == "__main__":
    main()
