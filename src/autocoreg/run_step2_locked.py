"""Step 2 — Locked-frame competency benchmark.

Same algorithmic machinery as run_step1_oracle.py, but the CZ mapping is
the locked-prior warm-start (no GT-TPS).  Quantifies how pose error
degrades the descriptors.

Differences from Step 1:
* CZ centroids enter the HCR µm frame via ``locked_prior_warm.apply_to_cz_um``.
* HCR pool bbox is around locked-frame CZ + 30 µm.
* Soma ``R_cand`` grid is widened to cover Step-0 locked-frame p95
  residuals (range ~60 µm for 790322 to ~900 µm for 755252/767022).

Output: outputs/step2_locked.parquet (long-form one row per
(subject, descriptor, sweep_config, near-miss radius)).
"""
from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .data import BENCHMARK_SUBJECTS, load_sz_pins, subject_inputs
from .soma_print import cell_vectors
from .shape_context import build_histograms, score_dense_gt_to_pool
from .surfaces_iter08 import get_hcr_top_surface_iter07
from .run_step1_oracle import (
    EXPAND_UM, NEAR_MISS_RADII, RECALL_K, metrics_from_D,
)

OUT_DIR = Path("/tmp/autocoreg_outputs")


# --------------------------------------------------------------------------- #
# Sweep grids
# --------------------------------------------------------------------------- #
SC_SWEEP = dict(
    R_outer_um=[80.0, 100.0, 150.0],
    r_inner_um=[8.0, 16.0],
    r_bins=[4, 6],
    theta_bins=[3, 5],
    phi_bins=[6, 10],
    az_shift_max_bins=[0, 1],
)


def sc_configs():
    keys = list(SC_SWEEP.keys())
    for vals in itertools.product(*(SC_SWEEP[k] for k in keys)):
        yield dict(zip(keys, vals))


SOMA_SWEEP = dict(
    m_cz=[10, 15, 20],
    m_hcr=[15, 20, 30],
    n=[5, 10],
    R_cand_um=[50.0, 100.0, 300.0, 1000.0],
)


def soma_configs():
    keys = list(SOMA_SWEEP.keys())
    for vals in itertools.product(*(SOMA_SWEEP[k] for k in keys)):
        d = dict(zip(keys, vals))
        if d["n"] > d["m_cz"]:
            continue
        yield d


# --------------------------------------------------------------------------- #
# Per-subject loader: locked-frame CZ + HCR filter
# --------------------------------------------------------------------------- #
def prepare_subject_locked(sid: str, sz_pins: dict):
    inp = subject_inputs(sid, sz_pins=sz_pins)
    # CZ → HCR µm via locked frame (NOT GT-TPS)
    cz_lp = inp.cz_lp_um.copy()
    # HCR filter
    ok_set = inp.gfp_ids & inp.ok_ids
    in_ok = np.array([int(h) in ok_set for h in inp.hcr_ids])
    lo = cz_lp.min(axis=0) - EXPAND_UM
    hi = cz_lp.max(axis=0) + EXPAND_UM
    in_bbox = ((inp.hcr_um >= lo) & (inp.hcr_um <= hi)).all(axis=1)
    keep = in_ok & in_bbox
    hcr_pool_zyx = inp.hcr_um[keep]
    hcr_pool_ids = inp.hcr_ids[keep].astype(int)
    hcr_id_to_row = {int(h): r for r, h in enumerate(hcr_pool_ids)}
    cz_id_to_row = {int(c): r for r, c in enumerate(inp.cz_ids)}
    # GT survivors after filter
    gt_pairs = []
    for c, h in zip(inp.coreg["cz_id"].astype(int), inp.coreg["hcr_id"].astype(int)):
        if int(c) in cz_id_to_row and int(h) in hcr_id_to_row:
            gt_pairs.append((int(c), int(h)))
    gt_cz_rows = np.array([cz_id_to_row[c] for c, _ in gt_pairs])
    gt_hcr_rows = np.array([hcr_id_to_row[h] for _, h in gt_pairs])
    surface = get_hcr_top_surface_iter07(inp.s)
    # Per-cell locked-frame residual at GT
    if len(gt_pairs):
        resid = np.linalg.norm(
            cz_lp[gt_cz_rows] - hcr_pool_zyx[gt_hcr_rows], axis=1,
        )
        p50 = float(np.percentile(resid, 50))
        p95 = float(np.percentile(resid, 95))
    else:
        p50 = p95 = float("nan")
    return dict(
        sid=sid, inp=inp,
        cz_locked_zyx=cz_lp,
        hcr_pool_zyx=hcr_pool_zyx,
        hcr_pool_ids=hcr_pool_ids,
        cz_id_to_row=cz_id_to_row,
        hcr_id_to_row=hcr_id_to_row,
        gt_pairs=gt_pairs,
        gt_cz_rows=gt_cz_rows,
        gt_hcr_rows=gt_hcr_rows,
        surface=surface,
        residual_p50=p50,
        residual_p95=p95,
    )


# --------------------------------------------------------------------------- #
# Centroid_LP baseline (here it's the same definition as Step 1, except
# the pool is now locked-frame-bbox)
# --------------------------------------------------------------------------- #
def score_centroid_lp(subj):
    sid = subj["sid"]
    cz_lp = subj["cz_locked_zyx"]
    cz_rows = subj["gt_cz_rows"]
    cz_lp_rows = cz_lp[cz_rows]
    diff = cz_lp_rows[:, None, :] - subj["hcr_pool_zyx"][None, :, :]
    D = np.linalg.norm(diff, axis=2)
    m = metrics_from_D(D, subj["gt_hcr_rows"], subj["hcr_pool_zyx"])
    return dict(
        subject_id=sid, descriptor="centroid_LP", sweep_config="-",
        n_eligible=len(subj["gt_pairs"]), **m,
    )


# --------------------------------------------------------------------------- #
# Shape context (same as Step 1 — geometry is intrinsic to each cell;
# only matching is affected by pose, which is handled in the metric)
# --------------------------------------------------------------------------- #
def run_sc_for_subject(subj, *, log_prefix: str) -> list[dict]:
    sid = subj["sid"]
    rows = []
    cz_zyx = subj["cz_locked_zyx"]
    hcr_zyx = subj["hcr_pool_zyx"]
    surface = subj["surface"]
    n_gt = len(subj["gt_pairs"])

    configs = list(sc_configs())
    total = len(configs)
    hist_cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
    t_subj = time.time()
    for k, cfg in enumerate(configs, start=1):
        t0 = time.time()
        key = (cfg["r_inner_um"], cfg["R_outer_um"], cfg["r_bins"],
               cfg["theta_bins"], cfg["phi_bins"])
        if key not in hist_cache:
            H_cz = build_histograms(
                cz_zyx, surface=surface,
                r_inner=cfg["r_inner_um"], R_outer=cfg["R_outer_um"],
                r_bins=cfg["r_bins"], theta_bins=cfg["theta_bins"],
                phi_bins=cfg["phi_bins"],
            )
            H_hcr = build_histograms(
                hcr_zyx, surface=surface,
                r_inner=cfg["r_inner_um"], R_outer=cfg["R_outer_um"],
                r_bins=cfg["r_bins"], theta_bins=cfg["theta_bins"],
                phi_bins=cfg["phi_bins"],
            )
            hist_cache[key] = (H_cz, H_hcr)
        else:
            H_cz, H_hcr = hist_cache[key]
        H_cz_gt = H_cz[subj["gt_cz_rows"]]
        D = score_dense_gt_to_pool(
            H_cz_gt, H_hcr,
            r_bins=cfg["r_bins"], theta_bins=cfg["theta_bins"],
            phi_bins=cfg["phi_bins"],
            az_shift_max_bins=cfg["az_shift_max_bins"],
        )
        m = metrics_from_D(D, subj["gt_hcr_rows"], hcr_zyx)
        cfg_str = ",".join(f"{kk}={vv}" for kk, vv in cfg.items())
        rows.append(dict(
            subject_id=sid, descriptor="shape_context",
            sweep_config=cfg_str,
            n_eligible=n_gt, **m,
            elapsed_s=round(time.time() - t0, 1),
        ))
        if k % 16 == 0 or k == total:
            print(
                f"  {log_prefix} SC {k}/{total} "
                f"cfg={cfg_str} "
                f"r@5={m['recall@5']:.3f} AUC50={m['auc_R50um']:.2f} "
                f"[this {time.time()-t0:.1f}s, cum {time.time()-t_subj:.1f}s]",
                flush=True,
            )
    return rows


# --------------------------------------------------------------------------- #
# Soma-print (locked-frame, no penalty per plan)
# --------------------------------------------------------------------------- #
def _score_soma_per_gt(
    cz_vec: np.ndarray, hcr_vecs_stack: np.ndarray, n: int,
) -> np.ndarray:
    """Vectorised soma score for one GT-CZ vs many HCR candidates.

    cz_vec: (m_cz, 3)
    hcr_vecs_stack: (n_cands, m_hcr, 3)
    Returns: (n_cands,) — mean of n-smallest distances over m_cz × m_hcr per cand.
    """
    diff = cz_vec[None, :, None, :] - hcr_vecs_stack[:, None, :, :]
    d = np.linalg.norm(diff, axis=-1)  # (n_cands, m_cz, m_hcr)
    flat = d.reshape(d.shape[0], -1)  # (n_cands, m_cz*m_hcr)
    if flat.shape[1] < n:
        return np.full(flat.shape[0], np.inf, dtype=np.float32)
    order = np.argpartition(flat, n - 1, axis=1)[:, :n]
    rows = np.arange(flat.shape[0])[:, None]
    return flat[rows, order].mean(axis=1).astype(np.float32)


def run_soma_for_subject(subj, *, log_prefix: str) -> list[dict]:
    sid = subj["sid"]
    rows = []
    cz_zyx = subj["cz_locked_zyx"]
    hcr_zyx = subj["hcr_pool_zyx"]
    n_gt = len(subj["gt_pairs"])
    n_hcr = len(hcr_zyx)

    cache_cz: dict[int, list[np.ndarray]] = {}
    cache_hcr_stack: dict[int, np.ndarray] = {}

    configs = list(soma_configs())
    total = len(configs)
    t_subj = time.time()
    hcr_tree = cKDTree(hcr_zyx)
    for k, cfg in enumerate(configs, start=1):
        t0 = time.time()
        m_cz = cfg["m_cz"]; m_hcr = cfg["m_hcr"]; n = cfg["n"]
        R_cand = cfg["R_cand_um"]
        if m_cz not in cache_cz:
            cache_cz[m_cz] = cell_vectors(cz_zyx, m=m_cz)
        cz_vecs = cache_cz[m_cz]
        if m_hcr not in cache_hcr_stack:
            hv_list = cell_vectors(hcr_zyx, m=m_hcr)
            # Pad to (n_hcr, m_hcr, 3) by repeating last row if a cell has fewer
            # than m_hcr neighbours (rare but possible at the periphery).
            stack = np.zeros((n_hcr, m_hcr, 3), dtype=np.float32)
            valid = np.zeros(n_hcr, dtype=bool)
            for r, v in enumerate(hv_list):
                if v.shape[0] >= 1:
                    valid[r] = v.shape[0] == m_hcr
                    if v.shape[0] >= m_hcr:
                        stack[r] = v[:m_hcr]
                    else:
                        stack[r, :v.shape[0]] = v
                        # pad with the last neighbour repeated (distance penalty)
                        stack[r, v.shape[0]:] = v[-1]
            cache_hcr_stack[m_hcr] = (stack, valid)
        hcr_stack, hcr_valid = cache_hcr_stack[m_hcr]

        D = np.full((n_gt, n_hcr), fill_value=np.inf, dtype=np.float32)
        for i in range(n_gt):
            cz_row = subj["gt_cz_rows"][i]
            cand = hcr_tree.query_ball_point(cz_zyx[cz_row], r=R_cand)
            cand_arr = np.fromiter(set(cand) | {int(subj["gt_hcr_rows"][i])}, dtype=int)
            cand_arr = cand_arr[hcr_valid[cand_arr]]
            if cand_arr.size == 0:
                continue
            ci = cz_vecs[cz_row]
            if ci.shape[0] < n:
                continue
            scores = _score_soma_per_gt(ci, hcr_stack[cand_arr], n)
            D[i, cand_arr] = scores

        m = metrics_from_D(D, subj["gt_hcr_rows"], hcr_zyx)
        cfg_str = ",".join(f"{kk}={vv}" for kk, vv in cfg.items())
        rows.append(dict(
            subject_id=sid, descriptor="soma_print",
            sweep_config=cfg_str,
            n_eligible=n_gt, **m,
            elapsed_s=round(time.time() - t0, 1),
        ))
        if k % 16 == 0 or k == total:
            print(
                f"  {log_prefix} soma {k}/{total} cfg={cfg_str} "
                f"r@5={m.get('recall@5', float('nan')):.3f} "
                f"AUC50={m.get('auc_R50um', float('nan')):.3f} "
                f"[this {time.time()-t0:.1f}s, cum {time.time()-t_subj:.1f}s]",
                flush=True,
            )
    return rows


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(subjects: tuple = BENCHMARK_SUBJECTS) -> int:
    sz_pins = load_sz_pins()
    all_rows = []
    for sid in subjects:
        t0 = time.time()
        print(f"\n=== {sid} ===")
        subj = prepare_subject_locked(sid, sz_pins)
        print(
            f"  cz_locked n={len(subj['cz_locked_zyx'])}  "
            f"hcr_pool n={len(subj['hcr_pool_zyx'])}  "
            f"GT n={len(subj['gt_pairs'])}  "
            f"residual p50={subj['residual_p50']:.1f} µm  "
            f"p95={subj['residual_p95']:.1f} µm"
        )
        baseline = score_centroid_lp(subj)
        baseline["elapsed_s"] = 0.0
        all_rows.append(baseline)
        print(f"  centroid_LP r@5={baseline['recall@5']:.3f}  "
              f"AUC50={baseline['auc_R50um']:.2f}")

        sc_rows = run_sc_for_subject(subj, log_prefix=sid)
        all_rows.extend(sc_rows)
        print(f"  SC done ({len(sc_rows)} configs, [{time.time()-t0:.1f}s subj-cum])")

        soma_rows = run_soma_for_subject(subj, log_prefix=sid)
        all_rows.extend(soma_rows)
        print(f"  soma done ({len(soma_rows)} configs, [{time.time()-t0:.1f}s subj-cum])")

    df = pd.DataFrame(all_rows)
    out = OUT_DIR / "step2_locked.parquet"
    df.to_parquet(out, index=False)
    print(f"\nwrote {out}  ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
