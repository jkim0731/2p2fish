"""Build the QC app's input artifacts for one subject, end-to-end.

Given a matcher-output matches CSV (columns ``cz_id, hcr_id`` at minimum, e.g.
``run_step3_v3`` ``matches_*round*.csv``), this writes, into ``out_dir``:

  cz_matched_seg.tif / cz_unmatched_seg.tif      CZ ROIs warped into HCR µm,
                                                 split by whether they matched
  hcr_matched_seg.tif / hcr_unmatched_seg.tif    HCR GFP+∩ok ROIs (in the pool),
                                                 split by matched/unmatched
  hcr_failed_gfp_seg.tif                         HCR ROIs that failed GFP+
  hcr_failed_classifier_seg.tif                  HCR ROIs that failed the v5d ok-gate
  cz_warped_in_hcr_um.tif + cz_warped_meta.json  CZ 488 image warped into HCR µm
  seg_volumes_meta.json                          bboxes, voxel sizes, cell counts

These are exactly the files ``qc/app.py`` reads.  Ported + merged from the
session-15 ``build_seg_volumes.py`` + ``build_warped_cz_volume.py``; the CZ
label volume (order-0) and the CZ image volume (order-1) are warped in a single
inverse-TPS pass so they share the output grid exactly.
"""
from __future__ import annotations

import glob
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.interpolate import Rbf
from scipy.ndimage import map_coordinates
from scipy.spatial import ConvexHull

from autocoreg import config as _config
from autocoreg.io.cz_volume import load_cz_volume
from autocoreg.io.inputs import load_sz_pins, subject_inputs

# Default output voxel size (µm) for the warped CZ image + CZ/HCR seg overlays.
# 2.0 µm gives crisp ROI boundaries + CZ image (4.0 was blocky); env-overridable.
DEFAULT_VOXEL_UM = float(os.environ.get("MFISH_QC_VOXEL_UM", "2.0"))

# Cap on the number of TPS anchors used for the inverse warp. scipy.Rbf is O(N^3) to fit and
# materializes an (N_query x N_anchor) kernel matrix per evaluation, so dense subjects (~8k
# matched pairs) blow up both fit time (~20 min) and peak memory (an OOM here can dump a core
# that fills the tiny root overlay and locks the env). Above this cap we subsample anchors but
# ALWAYS keep the convex-hull vertices, so the bounded-extrapolation hull is exact and the field
# stays faithful. 0 disables the cap. Subjects with <= this many anchors are not subsampled, so
# the fitted RBF / inv_affine / inv_hull are unchanged (including every GT benchmark subject).
MFISH_QC_TPS_MAX_ANCHORS = int(os.environ.get("MFISH_QC_TPS_MAX_ANCHORS", "3000"))

# Shared read-only context for the parallel inverse-TPS warp (fork-inherited).
_WARP_CTX: dict = {}


def _rbf_eval_chunked(rbf_axis, Z, Y, X, chunk=250_000):
    """Evaluate a scipy Rbf in query-point chunks so the full (N_query x N_anchor) kernel matrix
    never materializes at once (bounds peak memory on dense subjects). Numerically identical to
    ``rbf_axis(Z, Y, X)`` — each query point is evaluated independently; only BLAS's reduction
    tiling over the anchor axis shifts the last bits (~1e-11 µm, 11 orders below the voxel)."""
    n = int(Z.shape[0])
    if n <= chunk:
        return rbf_axis(Z, Y, X)
    out = np.empty(n, dtype=float)
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        out[i:j] = rbf_axis(Z[i:j], Y[i:j], X[i:j])
    return out


def _available_memory_bytes() -> int:
    """Best-effort available RAM in bytes, honouring a CONTAINER's cgroup limit (a capsule's
    /proc/meminfo reports the whole HOST, which overshoots and is what OOM-killed the 700 warp).
    Returns min(host MemAvailable, cgroup limit − current usage). ``MFISH_QC_MEM_BUDGET_GB``
    overrides everything; conservative fallback if nothing is readable."""
    ov = os.environ.get("MFISH_QC_MEM_BUDGET_GB", "").strip()
    if ov:
        try:
            return max(1, int(float(ov) * (1024 ** 3)))
        except ValueError:
            pass
    cands = []
    try:                                            # host reclaimable-aware available
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    cands.append(int(line.split()[1]) * 1024)
                    break
    except OSError:
        pass
    for lim_p, use_p in (                           # cgroup ceiling INSIDE the container
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),               # v2
        ("/sys/fs/cgroup/memory/memory.limit_in_bytes",
         "/sys/fs/cgroup/memory/memory.usage_in_bytes"),                              # v1
    ):
        try:
            with open(lim_p) as f:
                raw = f.read().strip()
            if raw and raw != "max":
                lim = int(raw)
                if 0 < lim < (1 << 62):
                    try:
                        used = int(open(use_p).read().strip())
                    except OSError:
                        used = 0
                    cands.append(max(lim - used, 1))
        except OSError:
            pass
    if cands:
        return max(1, min(cands))
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) // 2
    except (ValueError, OSError):
        return 8 * (1024 ** 3)


def _plan_warp_resources(n_anchors, slice_npix, shared_bytes, nz, requested_workers=None):
    """Pick (n_workers, rbf_chunk) so the parallel inverse-TPS warp fits in RAM at ANY z-stack /
    anchor-set size — no hard-coded 700 special case.

    The peak-memory driver is one Rbf eval's transient ``(chunk × n_anchors)`` distance+kernel
    matrix, which scipy materialises twice; the fork-shared CZ volumes are counted ONCE (COW). So
    total peak ≈ ``n_workers × (FIXED + chunk × n_anchors × 16 B)`` on top of the shared volumes.
    We read the live (cgroup-aware) budget, subtract the shared volumes + a parent cushion, then
    (a) take the most workers that still afford a minimal chunk each, and (b) grow the chunk to fill
    each worker's share. Larger volume / more anchors / less RAM ⇒ fewer workers and/or smaller
    chunk, automatically. Env overrides: MFISH_QC_WARP_WORKERS, MFISH_QC_RBF_CHUNK, MFISH_QC_MEM_*."""
    avail = _available_memory_bytes()
    safety = float(os.environ.get("MFISH_QC_MEM_SAFETY", "0.6"))
    reserve = int(shared_bytes) + 512 * 1024 * 1024          # shared CZ vols (once) + parent cushion
    usable = max(int(avail * safety) - reserve, 256 * 1024 * 1024)
    kpq = max(1, int(n_anchors)) * 8 * 2                      # bytes/query in one eval (dist + kernel)
    FIXED = 160 * 1024 * 1024                                # per-worker interp + slice/coord buffers
    # Cap the chunk: past ~64k query points BLAS is already saturated, so a larger transient
    # matrix buys no speed and only widens the blast radius of a memory-estimate miss.
    MIN_CHUNK, MAX_CHUNK = 4096, min(65536, max(4096, int(slice_npix)))
    cpu = max(1, (os.cpu_count() or 2) - 2)
    env_w = os.environ.get("MFISH_QC_WARP_WORKERS", "").strip()
    if env_w:
        try:
            req = int(env_w)
        except ValueError:
            req = requested_workers or cpu
    else:
        req = requested_workers or cpu
    req = max(1, min(int(req), int(nz)))
    afford = max(1, int(usable // (FIXED + MIN_CHUNK * kpq)))  # most workers affording a min chunk
    n_workers = max(1, min(req, afford))
    chunk = int((usable // n_workers - FIXED) // kpq)          # grow chunk to each worker's share
    chunk = max(MIN_CHUNK, min(chunk, MAX_CHUNK))
    env_c = os.environ.get("MFISH_QC_RBF_CHUNK", "").strip()
    if env_c:
        try:
            chunk = max(1, int(env_c))
        except ValueError:
            pass
    peak_gb = n_workers * (FIXED + chunk * kpq) / (1024 ** 3)
    return n_workers, chunk, avail, usable, peak_gb


def _warp_zslice_chunk(k_range):
    """Warp a contiguous block of output z-slices (CZ seg order-0 + CZ image
    order-1) using the fork-inherited ``_WARP_CTX``.  Returns (k0, cz_block,
    img_block)."""
    c = _WARP_CTX
    rbf, Y_flat, X_flat = c["rbf"], c["Y_flat"], c["X_flat"]
    z_lo, vox = c["z_lo"], c["vox"]
    cz_z_um, cz_xy_um = c["cz_z_um"], c["cz_xy_um"]
    cz_seg, cz_vol, ny, nx = c["cz_seg"], c["cz_vol"], c["ny"], c["nx"]
    inv_affine = c.get("inv_affine"); inv_hull = c.get("inv_hull")
    tau = c.get("tau", 40.0); margin = c.get("margin", 0.0)
    rbf_chunk = int(c.get("rbf_chunk", 250_000))   # memory-planned query-chunk (see _plan_warp_resources)
    ks = list(k_range)
    cz_block = np.zeros((len(ks), ny, nx), dtype=np.int32)
    img_block = np.zeros((len(ks), ny, nx), dtype=np.float32)
    for idx, k in enumerate(ks):
        z = z_lo + (k + 0.5) * vox
        Z_flat = np.full_like(Y_flat, z)
        # thin-plate prediction (CZ µm)
        cz_z_um_p = _rbf_eval_chunked(rbf[0], Z_flat, Y_flat, X_flat, chunk=rbf_chunk)
        cz_y_um_p = _rbf_eval_chunked(rbf[1], Z_flat, Y_flat, X_flat, chunk=rbf_chunk)
        cz_x_um_p = _rbf_eval_chunked(rbf[2], Z_flat, Y_flat, X_flat, chunk=rbf_chunk)
        if inv_affine is not None and inv_hull is not None:
            # blend to bounded affine OUTSIDE the HCR-anchor hull (w=1 inside).
            P = np.column_stack([Z_flat, Y_flat, X_flat, np.ones_like(Z_flat)])
            aff = P @ inv_affine                      # (N,3) CZ µm
            sd = (np.column_stack([Z_flat, Y_flat, X_flat]) @ inv_hull[:, :3].T
                  + inv_hull[:, 3]).max(axis=1)        # >0 outside hull
            w = np.exp(-(np.clip(sd + margin, 0.0, None) / tau) ** 2)
            cz_z_um_p = w * cz_z_um_p + (1 - w) * aff[:, 0]
            cz_y_um_p = w * cz_y_um_p + (1 - w) * aff[:, 1]
            cz_x_um_p = w * cz_x_um_p + (1 - w) * aff[:, 2]
        cz_z_vox = cz_z_um_p / cz_z_um
        cz_y_vox = cz_y_um_p / cz_xy_um
        cz_x_vox = cz_x_um_p / cz_xy_um
        coords = np.stack([cz_z_vox, cz_y_vox, cz_x_vox])
        cz_block[idx] = map_coordinates(
            cz_seg, coords, order=0, mode="constant", cval=0).reshape(ny, nx)
        img_block[idx] = map_coordinates(
            cz_vol, coords, order=1, mode="constant", cval=0.0).reshape(ny, nx)
    return ks[0], cz_block, img_block


def final_round_csv(sid_out_dir) -> Path:
    """Final-round matcher CSV in a matcher ``<out_dir>/<sid>/`` dir: last
    ``matches_anchor_restricted_round*.csv`` (Stage-2) if present, else last
    ``matches_round*.csv`` (Stage-1).  ``matches_wang_round*`` accepted as legacy."""
    import re
    d = Path(sid_out_dir)
    for pat in ("matches_anchor_restricted_round*.csv", "matches_wang_round*.csv"):
        cands = sorted(d.glob(pat), key=lambda p: int(re.findall(r"\d+", p.stem)[-1]))
        if cands:
            return cands[-1]
    rounds = sorted(d.glob("matches_round*.csv"),
                    key=lambda p: int(re.findall(r"\d+", p.stem)[-1]))
    if not rounds:
        raise FileNotFoundError(f"No matches CSVs under {d}")
    return rounds[-1]


def find_cz_seg_tiff(sid: str) -> Path:
    """CZ segmentation_masks.tif (label volume, ZYX, int). MFISH_CZ_SEG_TIF pins it directly;
    else glob under DATA_ROOT (channel_0_ref_0 layout, then a structure-agnostic recursive fallback)."""
    override = os.environ.get("MFISH_CZ_SEG_TIF", "").strip()
    if override:
        if not Path(override).exists():
            raise FileNotFoundError(f"MFISH_CZ_SEG_TIF set but not found: {override}")
        return Path(override)
    root = str(_config.DATA_ROOT)
    pats = [
        f"{root}/multiplane-ophys_{sid}_*-segmentation_*/channel_0_ref_0/segmentation_masks.tif",
        f"{root}/multiplane-ophys_{sid}_*-seg_*/channel_0_ref_0/segmentation_masks.tif",
        f"{root}/multiplane-ophys_{sid}_*-segmentation_*/**/segmentation_masks.tif",
    ]
    for pat in pats:
        paths = sorted(glob.glob(pat, recursive=True))
        if paths:
            return Path(paths[-1])
    raise FileNotFoundError(
        f"No CZ seg TIFF for {sid} under {root} (set MFISH_CZ_SEG_TIF to point at it directly)")


def open_hcr_seg_zarr_array(s):
    """Open HCR segmentation_mask.zarr lazily; return (zyx_slicer, zyx_shape,
    xy_um, z_um).  The whole volume can be hundreds of GB — callers must slice
    before materialising."""
    import zarr
    p = Path(s.hcr_dir) / "cell_body_segmentation" / "segmentation_mask.zarr"
    z = zarr.open(str(p), mode="r")
    try:
        node = z["0"]
    except (KeyError, TypeError):
        node = z
    arr = node
    if arr.ndim == 5:        # (T, C, Z, Y, X)
        zyx_slicer = lambda zs, ys, xs: arr[0, 0, zs, ys, xs]
        zyx_shape = arr.shape[2:]
    elif arr.ndim == 4:      # (C, Z, Y, X)
        zyx_slicer = lambda zs, ys, xs: arr[0, zs, ys, xs]
        zyx_shape = arr.shape[1:]
    elif arr.ndim == 3:
        zyx_slicer = lambda zs, ys, xs: arr[zs, ys, xs]
        zyx_shape = arr.shape
    else:
        raise RuntimeError(f"unexpected HCR seg ndim {arr.ndim}, shape {arr.shape}")
    return zyx_slicer, zyx_shape, float(s.hcr_seg_xy_um), float(s.hcr_seg_z_um)


def _bbox_from_lp(inp, margin_um: float):
    """Output bbox in HCR µm = LP-warped CZ extent + uniform margin."""
    cz_lp = inp.cz_lp_um
    m = float(margin_um)
    return (
        float(cz_lp[:, 0].min()) - m, float(cz_lp[:, 0].max()) + m,
        float(cz_lp[:, 1].min()) - m, float(cz_lp[:, 1].max()) + m,
        float(cz_lp[:, 2].min()) - m, float(cz_lp[:, 2].max()) + m,
    )


def _fit_inverse_tps(inp, df):
    """Per-axis thin-plate Rbf mapping HCR µm -> CZ-native µm, from matches."""
    cz_id_to_row = {int(c): r for r, c in enumerate(inp.cz_ids)}
    hcr_id_to_row = {int(h): r for r, h in enumerate(inp.hcr_ids)}
    src_pts, dst_pts = [], []
    for cz_id, hcr_id in zip(df["cz_id"].astype(int), df["hcr_id"].astype(int)):
        cr = cz_id_to_row.get(int(cz_id))
        hr = hcr_id_to_row.get(int(hcr_id))
        if cr is None or hr is None:
            continue
        src_pts.append(inp.cz_um[cr])   # CZ-native µm (zyx)
        dst_pts.append(inp.hcr_um[hr])  # HCR µm (zyx)
    if len(src_pts) < 4:
        raise ValueError(
            f"only {len(src_pts)} usable anchors in matches CSV — need >= 4 for TPS"
        )
    src = np.asarray(src_pts, dtype=float)   # CZ µm
    dst = np.asarray(dst_pts, dtype=float)   # HCR µm (the query frame)
    # Cap anchors (keeping the convex-hull vertices) so dense subjects don't blow up the Rbf
    # fit/eval. A subject with <= MFISH_QC_TPS_MAX_ANCHORS pairs skips this entirely, so its
    # rbf / inv_affine / inv_hull are bit-identical to before. Deterministic (fixed seed).
    n_full = len(src)
    if MFISH_QC_TPS_MAX_ANCHORS and n_full > MFISH_QC_TPS_MAX_ANCHORS:
        try:
            hull_v = np.unique(ConvexHull(dst).vertices)
        except Exception:
            hull_v = np.empty(0, dtype=int)
        keep = np.zeros(n_full, dtype=bool)
        keep[hull_v] = True                                  # hull kept -> inv_hull unchanged
        n_extra = MFISH_QC_TPS_MAX_ANCHORS - int(keep.sum())
        rest = np.flatnonzero(~keep)
        if n_extra > 0 and rest.size:
            rng = np.random.RandomState(0)
            keep[rng.choice(rest, size=min(n_extra, rest.size), replace=False)] = True
        src, dst = src[keep], dst[keep]
        print(
            f"[build_qc]   TPS anchors capped {n_full} -> {len(src)} "
            f"({len(hull_v)} hull vertices kept; MFISH_QC_TPS_MAX_ANCHORS={MFISH_QC_TPS_MAX_ANCHORS})",
            flush=True,
        )
    rbf = [
        Rbf(dst[:, 0], dst[:, 1], dst[:, 2], src[:, a], function="thin_plate")
        for a in range(3)
    ]
    if os.environ.get("MFISH_TPS_BOUND", "1") == "0":
        return rbf, None, None, len(src)
    # Bounded extrapolation (same fix as the matcher's forward TPS, tps.py): the
    # inverse thin-plate warp is trustworthy only INSIDE the HCR-anchor hull; the
    # deep-edge corner of the output grid lies OUTSIDE it, where the unbounded
    # r^2·log(r) kernel smears the sampled CZ image (the visible over-warp
    # streak). Fit a robust global affine HCR→CZ and the HCR-anchor hull so the
    # warp blends to the bounded affine outside the hull. w=1 strictly inside →
    # the sampled image is BIT-IDENTICAL to the pure-TPS render there.
    from autocoreg.finetune_soma_print.tps import fit_robust_affine
    try:
        inv_affine = fit_robust_affine(dst, src)   # HCR µm → CZ µm (4×3)
    except Exception:
        inv_affine = None
    try:
        inv_hull = ConvexHull(dst).equations         # hull of HCR anchor points
    except Exception:
        inv_hull = None
    return rbf, inv_affine, inv_hull, len(src)


def build_qc_artifacts(
    sid: str,
    matches_csv,
    out_dir,
    *,
    voxel_um: float = DEFAULT_VOXEL_UM,
    margin_um: float = 30.0,
    sz_pins: dict | None = None,
    write_failed: bool = True,
) -> dict:
    """Build all QC artifacts for ``sid`` into ``out_dir``.  Returns the meta dict.

    ``matches_csv`` — matcher output (needs ``cz_id``, ``hcr_id``).
    ``sz_pins``     — optional {sid: sz}; if None the GT-free sz estimator runs.
    """
    matches_csv = Path(matches_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"[build_qc] {sid}: artifacts -> {out_dir}", flush=True)

    if sz_pins is None:
        try:
            sz_pins = load_sz_pins()
        except Exception:
            sz_pins = None  # subject_inputs falls back to GT-free get_sz
    inp = subject_inputs(sid, sz_pins=sz_pins)

    df = pd.read_csv(matches_csv)
    df["cz_id"] = df["cz_id"].astype(int)
    df["hcr_id"] = df["hcr_id"].astype(int)
    matched_cz_ids = set(df["cz_id"].tolist())
    matched_hcr_ids = set(df["hcr_id"].tolist())
    n_gt = int(df["is_gt"].sum()) if "is_gt" in df.columns else -1
    print(f"[build_qc]   matches: {len(df)} pairs "
          f"({len(matched_cz_ids)} CZ, {len(matched_hcr_ids)} HCR"
          + (f", {n_gt} GT-agree" if n_gt >= 0 else "") + ")", flush=True)

    # Shared output bbox (HCR µm) + grid.
    z_lo, z_hi, y_lo, y_hi, x_lo, x_hi = _bbox_from_lp(inp, margin_um)
    vox = float(voxel_um)
    nz = int(np.ceil((z_hi - z_lo) / vox))
    ny = int(np.ceil((y_hi - y_lo) / vox))
    nx = int(np.ceil((x_hi - x_lo) / vox))
    print(f"[build_qc]   output grid ({nz},{ny},{nx}) @ {vox}µm", flush=True)

    # ----- inverse TPS (HCR µm -> CZ-native µm), bounded-extrapolation -----
    rbf, inv_affine, inv_hull, n_anchors = _fit_inverse_tps(inp, df)
    print(f"[build_qc]   inverse TPS fit on {n_anchors} anchors "
          f"(bounded-extrap: {'on' if inv_hull is not None else 'off'}) "
          f"[{time.time()-t0:.1f}s]", flush=True)

    # ----- load CZ seg (labels) + CZ 488 image (both CZ-native ZYX) -----
    cz_seg = tifffile.imread(str(find_cz_seg_tiff(sid))).astype(np.int32, copy=False)
    while cz_seg.ndim > 3 and cz_seg.shape[0] == 1:
        cz_seg = cz_seg[0]
    cz_vol = load_cz_volume(inp.s)
    cz_xy_um = float(inp.s.cz_xy_um)
    cz_z_um = float(inp.s.cz_z_um)
    print(f"[build_qc]   CZ seg {cz_seg.shape}  CZ img {cz_vol.shape} "
          f"(vox z={cz_z_um:.3f} xy={cz_xy_um:.3f}µm)", flush=True)

    # ----- single inverse-TPS pass: warp CZ seg (order 0) + CZ image (order 1) -----
    warped_cz = np.zeros((nz, ny, nx), dtype=np.int32)
    warped_img = np.zeros((nz, ny, nx), dtype=np.float32)
    y_centers = y_lo + (np.arange(ny) + 0.5) * vox
    x_centers = x_lo + (np.arange(nx) + 0.5) * vox
    Y, X = np.meshgrid(y_centers, x_centers, indexing="ij")
    Y_flat, X_flat = Y.ravel(), X.ravel()
    t = time.time()
    from autocoreg.finetune_soma_print.tps import TPS_EXTRAP_TAU_UM, TPS_EXTRAP_MARGIN_UM
    # Memory-adaptive parallelism: size worker count + per-eval Rbf chunk to the live,
    # cgroup-aware RAM budget so the warp scales to ANY z-stack size without OOM. (The 700×700
    # run was SIGKILLed here: a fixed 250k chunk × 3000 anchors × 14 workers ≈ 84 GB.)
    shared_bytes = int(getattr(cz_seg, "nbytes", 0)) + int(getattr(cz_vol, "nbytes", 0))
    n_workers, rbf_chunk, _avail, _usable, _peak = _plan_warp_resources(
        n_anchors, ny * nx, shared_bytes, nz)
    _WARP_CTX.update(dict(
        rbf=rbf, Y_flat=Y_flat, X_flat=X_flat, z_lo=z_lo, vox=vox,
        cz_z_um=cz_z_um, cz_xy_um=cz_xy_um, cz_seg=cz_seg, cz_vol=cz_vol,
        ny=ny, nx=nx, rbf_chunk=rbf_chunk,
        inv_affine=inv_affine, inv_hull=inv_hull,
        tau=TPS_EXTRAP_TAU_UM, margin=TPS_EXTRAP_MARGIN_UM))
    print(f"[build_qc]   warp mem plan: avail={_avail/2**30:.1f}GB usable={_usable/2**30:.1f}GB "
          f"shared(CZ vols)={shared_bytes/2**30:.2f}GB -> {n_workers} workers × "
          f"rbf_chunk={rbf_chunk} (~{_peak:.1f}GB peak, {n_anchors} anchors)", flush=True)
    if n_workers == 1:
        k0, cz_b, img_b = _warp_zslice_chunk(range(nz))
        warped_cz[:] = cz_b
        warped_img[:] = img_b
    else:
        # Parallelize over z-slice chunks (each independent; cz_seg/cz_vol are
        # fork-shared read-only — the Rbf evaluation is the cost, scales ~linearly).
        chunk = max(1, nz // (n_workers * 3))
        chunks = [range(i, min(i + chunk, nz)) for i in range(0, nz, chunk)]
        print(f"[build_qc]   CZ warp: {nz} z @ {vox}µm, {len(chunks)} chunks, "
              f"{n_workers} workers", flush=True)
        done = 0
        with mp.get_context("fork").Pool(n_workers) as pool:
            for k0, cz_b, img_b in pool.imap_unordered(_warp_zslice_chunk, chunks):
                warped_cz[k0:k0 + cz_b.shape[0]] = cz_b
                warped_img[k0:k0 + img_b.shape[0]] = img_b
                done += cz_b.shape[0]
                print(f"[build_qc]     warped {done}/{nz} z [{time.time()-t:.1f}s]",
                      flush=True)
    _WARP_CTX.clear()
    print(f"[build_qc]   CZ warp done [{time.time()-t:.1f}s]", flush=True)

    # ----- split CZ warped seg into matched / unmatched -----
    cz_matched_mask = np.isin(warped_cz, list(matched_cz_ids))
    cz_matched_vol = np.where(cz_matched_mask, warped_cz, 0).astype(np.int32)
    cz_unmatched_vol = np.where(
        ~cz_matched_mask & (warped_cz > 0), warped_cz, 0).astype(np.int32)
    tifffile.imwrite(out_dir / "cz_matched_seg.tif", cz_matched_vol)
    tifffile.imwrite(out_dir / "cz_unmatched_seg.tif", cz_unmatched_vol)
    n_cz_matched = len(np.unique(cz_matched_vol)) - 1
    n_cz_unmatched = len(np.unique(cz_unmatched_vol)) - 1

    # ----- warped CZ image -----
    tifffile.imwrite(
        out_dir / "cz_warped_in_hcr_um.tif", warped_img,
        imagej=True, resolution=(1.0 / vox, 1.0 / vox),
        metadata={"spacing": vox, "unit": "um", "axes": "ZYX"},
    )
    (out_dir / "cz_warped_meta.json").write_text(json.dumps({
        "sid": sid, "voxel_um": vox,
        "bbox_um": {"z_lo": z_lo, "z_hi": z_hi, "y_lo": y_lo, "y_hi": y_hi,
                    "x_lo": x_lo, "x_hi": x_hi},
        "shape": [nz, ny, nx], "n_anchors": int(n_anchors),
    }, indent=2))

    # ----- HCR seg: crop bbox from the (huge) zarr, stride-downsample -----
    print("[build_qc]   opening HCR seg zarr ...", flush=True)
    zyx_slicer, (Zsz, Ysz, Xsz), hseg_xy, hseg_z = open_hcr_seg_zarr_array(inp.s)
    z0 = max(0, int(z_lo / hseg_z));   z1 = min(Zsz, int(z_hi / hseg_z) + 1)
    y0 = max(0, int(y_lo / hseg_xy));  y1 = min(Ysz, int(y_hi / hseg_xy) + 1)
    x0 = max(0, int(x_lo / hseg_xy));  x1 = min(Xsz, int(x_hi / hseg_xy) + 1)
    factor_z = max(1, int(round(vox / hseg_z)))
    factor_xy = max(1, int(round(vox / hseg_xy)))
    hcr_seg_crop = np.asarray(
        zyx_slicer(slice(z0, z1, factor_z), slice(y0, y1, factor_xy),
                   slice(x0, x1, factor_xy)),
        dtype=np.int32,
    )
    out_xy = hseg_xy * factor_xy
    out_z = hseg_z * factor_z
    print(f"[build_qc]   HCR seg crop {hcr_seg_crop.shape} "
          f"(vox xy={out_xy:.3f} z={out_z:.3f}µm) [{time.time()-t0:.1f}s]", flush=True)

    pool_arr = np.fromiter(inp.gfp_ids & inp.ok_ids, dtype=np.int32)
    matched_arr = np.fromiter(matched_hcr_ids, dtype=np.int32)
    in_pool = np.isin(hcr_seg_crop, pool_arr)
    is_matched = np.isin(hcr_seg_crop, matched_arr)
    hcr_matched_vol = np.where(is_matched & in_pool, hcr_seg_crop, 0).astype(np.int32)
    hcr_unmatched_vol = np.where(in_pool & ~is_matched, hcr_seg_crop, 0).astype(np.int32)
    tifffile.imwrite(out_dir / "hcr_matched_seg.tif", hcr_matched_vol)
    tifffile.imwrite(out_dir / "hcr_unmatched_seg.tif", hcr_unmatched_vol)
    n_hcr_matched = len(np.unique(hcr_matched_vol)) - 1
    n_hcr_unmatched = len(np.unique(hcr_unmatched_vol)) - 1

    n_failed_gfp = n_failed_cls = 0
    if write_failed:
        uniq = np.unique(hcr_seg_crop)
        uniq = uniq[uniq != 0]
        failed_gfp = np.fromiter(
            (int(h) for h in uniq if int(h) not in inp.gfp_ids), dtype=np.int32)
        failed_cls = np.fromiter(
            (int(h) for h in uniq if int(h) not in inp.ok_ids), dtype=np.int32)
        hcr_failed_gfp_vol = np.where(
            np.isin(hcr_seg_crop, failed_gfp), hcr_seg_crop, 0).astype(np.int32)
        hcr_failed_cls_vol = np.where(
            np.isin(hcr_seg_crop, failed_cls), hcr_seg_crop, 0).astype(np.int32)
        tifffile.imwrite(out_dir / "hcr_failed_gfp_seg.tif", hcr_failed_gfp_vol)
        tifffile.imwrite(out_dir / "hcr_failed_classifier_seg.tif", hcr_failed_cls_vol)
        n_failed_gfp = len(np.unique(hcr_failed_gfp_vol)) - 1
        n_failed_cls = len(np.unique(hcr_failed_cls_vol)) - 1

    meta = dict(
        sid=sid,
        voxel_um_cz_warped=vox,
        voxel_um_hcr_seg_xy=out_xy,
        voxel_um_hcr_seg_z=out_z,
        bbox_cz_warped=dict(z_lo=z_lo, z_hi=z_hi, y_lo=y_lo, y_hi=y_hi,
                            x_lo=x_lo, x_hi=x_hi),
        bbox_hcr_seg=dict(z_lo=z0 * hseg_z, y_lo=y0 * hseg_xy, x_lo=x0 * hseg_xy,
                          shape=list(hcr_seg_crop.shape)),
        counts=dict(
            cz_matched=n_cz_matched, cz_unmatched=n_cz_unmatched,
            hcr_matched=n_hcr_matched, hcr_unmatched=n_hcr_unmatched,
            hcr_failed_gfp=n_failed_gfp, hcr_failed_classifier=n_failed_cls,
        ),
    )
    (out_dir / "seg_volumes_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[build_qc]   done [{time.time()-t0:.1f}s]  counts={meta['counts']}",
          flush=True)
    return meta
