"""Step 1 — Oracle competency benchmark for shape context + soma-print.

Mirrors S12's setup so results are directly comparable:
* GT-TPS warp of full CZ centroid set into HCR µm.
* HCR pool = strict-GFP+ (07b GMM-intersection cutoff) ∩ argmax-ok (v5d
  4-class) ∩ bbox(warped-CZ) + 30 µm.
* Metrics per (subject, descriptor, sweep config):
  - AUC vs hard near-miss at R = 30, 50, 100 µm.
  - recall@K (K ∈ {1, 5, 20}) over the full filtered HCR pool.
* Also runs ``centroid_LP`` (LP-warmed CZ centroid distance) as the S12
  sanity baseline (target recall@5 ≈ 0.80).

Output: outputs/step1_oracle.parquet (long-form one row per
(subject, descriptor, sweep_config, near-miss radius)).

Sweep grid (kept conservative for the first pass; expand later if
results merit):

Option A (shape context):
  R_outer ∈ {80, 100, 150} µm
  r_inner ∈ {8, 16} µm
  r_bins ∈ {4, 6}
  θ_bins ∈ {3, 5}
  φ_bins ∈ {6, 10}
  az_shift_max_bins ∈ {0, 1}                  → 48 configs

Option B (soma-print, round 0 → no penalty):
  m_cz ∈ {10, 15, 20}
  m_hcr ∈ {15, 20, 30}
  n ∈ {5, 10, 15}
  R_cand ∈ {30, 50, 100} µm                   → 81 configs

centroid_LP is added once per subject (no sweep).
"""
from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import Rbf
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score

from autocoreg.io.inputs import BENCHMARK_SUBJECTS, argmax_ok_ids, load_sz_pins, strict_gfp_ids, subject_inputs
from autocoreg.finetune_soma_print.descriptor import cell_vectors, score_many_to_many as soma_score_many
from autocoreg.archive.shape_context import build_histograms, score_dense_gt_to_pool
from autocoreg.initial_registration.surfaces import get_hcr_top_surface_iter07

OUT_DIR = Path("/scratch/autocoreg_outputs")

EXPAND_UM = 30.0
NEAR_MISS_RADII = (30.0, 50.0, 100.0)
RECALL_K = (1, 5, 20)


# --------------------------------------------------------------------------- #
# GT-TPS warp (same as S12 fit_gt_tps)
# --------------------------------------------------------------------------- #
def fit_gt_tps(
    cz_um_zyx: np.ndarray, cz_ids: np.ndarray,
    hcr_um_zyx: np.ndarray, hcr_ids: np.ndarray,
    coreg: pd.DataFrame,
) -> tuple[np.ndarray, int]:
    cz_lookup = {int(i): pt for i, pt in zip(cz_ids, cz_um_zyx)}
    hcr_lookup = {int(i): pt for i, pt in zip(hcr_ids, hcr_um_zyx)}
    src, dst = [], []
    for c, h in zip(coreg["cz_id"].astype(int), coreg["hcr_id"].astype(int)):
        if c in cz_lookup and h in hcr_lookup:
            src.append(cz_lookup[c])
            dst.append(hcr_lookup[h])
    src = np.asarray(src)
    dst = np.asarray(dst)
    rbf_z = Rbf(src[:, 0], src[:, 1], src[:, 2], dst[:, 0], function="thin_plate")
    rbf_y = Rbf(src[:, 0], src[:, 1], src[:, 2], dst[:, 1], function="thin_plate")
    rbf_x = Rbf(src[:, 0], src[:, 1], src[:, 2], dst[:, 2], function="thin_plate")
    z = rbf_z(cz_um_zyx[:, 0], cz_um_zyx[:, 1], cz_um_zyx[:, 2])
    y = rbf_y(cz_um_zyx[:, 0], cz_um_zyx[:, 1], cz_um_zyx[:, 2])
    x = rbf_x(cz_um_zyx[:, 0], cz_um_zyx[:, 1], cz_um_zyx[:, 2])
    return np.column_stack([z, y, x]), len(src)


# --------------------------------------------------------------------------- #
# Score helper — given a distance matrix CZ_GT × HCR_pool, compute AUC and
# recall@K (S12-style).
# --------------------------------------------------------------------------- #
def metrics_from_D(
    D: np.ndarray,
    gt_hcr_rows: np.ndarray,
    hcr_zyx: np.ndarray,
    near_miss_radii: tuple = NEAR_MISS_RADII,
    recall_k: tuple = RECALL_K,
) -> dict:
    """`D` has shape (n_gt, n_hcr_pool); D[i, j] is the descriptor distance
    from GT-CZ_i to filtered HCR_j.  Lower = better."""
    n_gt = D.shape[0]
    # Replace inf with a large finite value (worst possible score) so AUC /
    # rank tools can ingest the matrix.  Different inf cells must remain
    # ordered after this replacement → use max-finite + 1 (NaN-safe).
    finite_max = np.nanmax(np.where(np.isfinite(D), D, np.nan))
    if not np.isfinite(finite_max):
        finite_max = 1.0
    D = np.where(np.isfinite(D), D, finite_max + 1.0)
    ranks = np.argsort(D, axis=1)
    pos = np.array([np.where(ranks[i] == gt_hcr_rows[i])[0][0] for i in range(n_gt)])
    recall = {f"recall@{k}": float((pos < k).mean()) for k in recall_k}
    hcr_tree = cKDTree(hcr_zyx)
    aucs = {}
    for R in near_miss_radii:
        scores, labels = [], []
        for i in range(n_gt):
            j = gt_hcr_rows[i]
            nbrs = hcr_tree.query_ball_point(hcr_zyx[j], r=R)
            nbrs = [k for k in nbrs if k != j]
            if not nbrs:
                continue
            scores.append(-D[i, j])
            labels.append(1)
            scores.extend((-D[i, nbrs]).tolist())
            labels.extend([0] * len(nbrs))
        if len(set(labels)) < 2:
            aucs[f"auc_R{int(R)}um"] = float("nan")
            aucs[f"n_nm_R{int(R)}um"] = 0
        else:
            aucs[f"auc_R{int(R)}um"] = float(roc_auc_score(labels, scores))
            aucs[f"n_nm_R{int(R)}um"] = int(np.sum(np.array(labels) == 0))
    return {**recall, **aucs}


# --------------------------------------------------------------------------- #
# Per-subject loader: GT-TPS warp + HCR filter
# --------------------------------------------------------------------------- #
def prepare_subject(sid: str, sz_pins: dict):
    inp = subject_inputs(sid, sz_pins=sz_pins)
    cz_warped, n_ctrl = fit_gt_tps(
        inp.cz_um, inp.cz_ids, inp.hcr_um, inp.hcr_ids, inp.coreg,
    )
    # HCR filter
    ok_set = inp.gfp_ids & inp.ok_ids
    in_ok = np.array([int(h) in ok_set for h in inp.hcr_ids])
    lo = cz_warped.min(axis=0) - EXPAND_UM
    hi = cz_warped.max(axis=0) + EXPAND_UM
    in_bbox = ((inp.hcr_um >= lo) & (inp.hcr_um <= hi)).all(axis=1)
    keep = in_ok & in_bbox
    hcr_pool_zyx = inp.hcr_um[keep]
    hcr_pool_ids = inp.hcr_ids[keep].astype(int)
    hcr_id_to_row = {int(h): r for r, h in enumerate(hcr_pool_ids)}
    cz_id_to_row = {int(c): r for r, c in enumerate(inp.cz_ids)}
    # GT survivors
    gt_pairs = []
    for c, h in zip(inp.coreg["cz_id"].astype(int), inp.coreg["hcr_id"].astype(int)):
        if int(c) in cz_id_to_row and int(h) in hcr_id_to_row:
            gt_pairs.append((int(c), int(h)))
    gt_cz_rows = np.array([cz_id_to_row[c] for c, _ in gt_pairs])
    gt_hcr_rows = np.array([hcr_id_to_row[h] for _, h in gt_pairs])
    surface = get_hcr_top_surface_iter07(inp.s)
    return dict(
        sid=sid, inp=inp,
        cz_warped_zyx=cz_warped,
        hcr_pool_zyx=hcr_pool_zyx,
        hcr_pool_ids=hcr_pool_ids,
        cz_id_to_row=cz_id_to_row,
        hcr_id_to_row=hcr_id_to_row,
        gt_pairs=gt_pairs,
        gt_cz_rows=gt_cz_rows,
        gt_hcr_rows=gt_hcr_rows,
        surface=surface,
        n_ctrl=n_ctrl,
    )


# --------------------------------------------------------------------------- #
# Centroid_LP baseline (S12 sanity)
# --------------------------------------------------------------------------- #
def score_centroid_lp(subj):
    sid = subj["sid"]
    inp = subj["inp"]
    cz_lp = inp.cz_lp_um
    # For S12 comparison, GT-CZ rows live in inp.cz_ids; LP-warped centroid
    # is used as the "descriptor coordinate"; distance is Euclidean to the
    # filtered HCR pool.
    cz_rows = subj["gt_cz_rows"]
    cz_lp_rows = cz_lp[cz_rows]
    # D shape (n_gt, n_hcr_pool)
    diff = cz_lp_rows[:, None, :] - subj["hcr_pool_zyx"][None, :, :]
    D = np.linalg.norm(diff, axis=2)
    m = metrics_from_D(D, subj["gt_hcr_rows"], subj["hcr_pool_zyx"])
    return dict(
        subject_id=sid, descriptor="centroid_LP", sweep_config="-",
        n_eligible=len(subj["gt_pairs"]), **m,
    )


# --------------------------------------------------------------------------- #
# Sweep — Option A (shape context)
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


def run_sc_for_subject(subj, *, log_prefix: str) -> list[dict]:
    """Shape-context sweep. Caches histograms by (r_inner, R_outer,
    r_bins, theta_bins, phi_bins) so the az_shift sub-sweep is free.
    Uses dense vectorised χ² (score_dense_gt_to_pool).
    """
    sid = subj["sid"]
    rows = []
    cz_zyx = subj["cz_warped_zyx"]
    hcr_zyx = subj["hcr_pool_zyx"]
    surface = subj["surface"]
    n_gt = len(subj["gt_pairs"])
    n_hcr = len(hcr_zyx)

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
        cfg_str = ",".join(f"{k}={v}" for k, v in cfg.items())
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
# Sweep — Option B (soma-print, round 0: no penalty)
# --------------------------------------------------------------------------- #
SOMA_SWEEP = dict(
    m_cz=[10, 15, 20],
    m_hcr=[15, 20, 30],
    n=[5, 10, 15],
    R_cand_um=[30.0, 50.0, 100.0],
)


def soma_configs():
    keys = list(SOMA_SWEEP.keys())
    for vals in itertools.product(*(SOMA_SWEEP[k] for k in keys)):
        d = dict(zip(keys, vals))
        if d["n"] > d["m_cz"] * d["m_hcr"]:
            continue
        if d["n"] > d["m_cz"]:
            # n-best-from-(m_cz × m_hcr) is fine technically, but a useful
            # constraint to keep the cost interpretable is n ≤ min(m_cz, m_hcr).
            continue
        yield d


def run_soma_for_subject(subj, *, log_prefix: str) -> list[dict]:
    sid = subj["sid"]
    rows = []
    cz_zyx = subj["cz_warped_zyx"]
    hcr_zyx = subj["hcr_pool_zyx"]
    n_gt = len(subj["gt_pairs"])
    n_hcr = len(hcr_zyx)

    # Cache vectors by m
    cache_cz: dict[int, list[np.ndarray]] = {}
    cache_hcr: dict[int, list[np.ndarray]] = {}

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
        if m_hcr not in cache_hcr:
            cache_hcr[m_hcr] = cell_vectors(hcr_zyx, m=m_hcr)
        cz_vecs = cache_cz[m_cz]
        hcr_vecs = cache_hcr[m_hcr]

        # Candidate set per GT-CZ: HCR cells within R_cand of cz_warped pos
        # (GT-TPS frame) plus GT partner.
        D = np.full((n_gt, n_hcr), fill_value=np.inf, dtype=np.float32)
        for i in range(n_gt):
            cz_row = subj["gt_cz_rows"][i]
            cand = hcr_tree.query_ball_point(cz_zyx[cz_row], r=R_cand)
            cand_set = set(cand) | {int(subj["gt_hcr_rows"][i])}
            for j in cand_set:
                ci = cz_vecs[cz_row]
                hj = hcr_vecs[j]
                if ci.size == 0 or hj.size == 0:
                    continue
                # m_cz × m_hcr distances
                diff = ci[:, None, :] - hj[None, :, :]
                d = np.linalg.norm(diff, axis=2)
                flat = d.ravel()
                if flat.size < n:
                    continue
                order = np.argpartition(flat, n - 1)[:n]
                D[i, j] = float(flat[order].mean())

        m = metrics_from_D(D, subj["gt_hcr_rows"], hcr_zyx)
        cfg_str = ",".join(f"{k}={v}" for k, v in cfg.items())
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
        subj = prepare_subject(sid, sz_pins)
        print(f"  cz_warped n={len(subj['cz_warped_zyx'])}  "
              f"hcr_pool n={len(subj['hcr_pool_zyx'])}  "
              f"GT n={len(subj['gt_pairs'])} (ctrl={subj['n_ctrl']})")
        # baseline
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
    out = OUT_DIR / "step1_oracle.parquet"
    df.to_parquet(out, index=False)
    print(f"\nwrote {out}  ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
