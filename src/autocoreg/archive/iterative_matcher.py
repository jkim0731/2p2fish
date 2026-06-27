"""Step 3 — Iterative TPS + soma-print expansion.

GT-FREE protocol (per user, 2026-05-18):
* No metrics or thresholds checked against ``coreg_table.csv`` during
  development.  GT is consulted ONLY at the very end as a reporting
  metric.
* Per-pair confidence combines four GT-free signals:
    (1) soma_score      — descriptor distance, lower = better
    (2) local_ncc       — per-pair CZ-patch ↔ HCR-patch Pearson NCC
    (3) anchor_vote     — fraction of soma's n-best neighbour pairs whose
                          implied cell-pair is in the current active set
    (4) tps_residual    — after the current TPS warp, |cz_tps - hcr|
* Stop criteria are intrinsic: acceptance rate < eps OR no NCC delta OR
  max rounds.
* No use of shape-context (per user — single-descriptor design).

Outputs per subject (under outputs/step3/<sid>/):
  matches_final.csv  — accepted matches with all five signals + round
  rejected.csv       — candidates considered but not accepted
  round_log.csv      — per-round bookkeeping (intrinsic signals)
  tps_warp.pkl       — final per-axis Rbf objects + control points
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import Rbf
from scipy.spatial import cKDTree

from autocoreg.io.inputs import BENCHMARK_SUBJECTS, load_sz_pins, subject_inputs
from autocoreg.io.hcr_image import load_hcr_volume
from autocoreg.io.cz_volume import load_cz_volume
from autocoreg.finetune_soma_print.local_ncc import per_pair_local_ncc
from autocoreg.archive.locked_benchmark import _score_soma_per_gt
from autocoreg.archive.refined_benchmark import adaptive_r_cand, hcr_pia_z_over_region, z_density_intersection_bounds
from autocoreg.finetune_soma_print.descriptor import cell_vectors

OUT_ROOT = Path("/scratch/autocoreg_outputs/step3")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# ----- Soma-print config (from Step 2.5 winning region) -----
SOMA_M_CZ = 15
SOMA_M_HCR = 30
SOMA_N = 5

# ----- Loop parameters (GT-free) -----
MAX_ROUNDS = 6
MAX_STALL_ROUNDS = 1  # stop after this many consecutive rounds with no new accepts
LOCAL_NCC_PATCH_UM = (20.0, 20.0, 20.0)
LOCAL_NCC_TARGET_UM = 2.0  # patch grid spacing in µm — level-4 friendly
HCR_LEVEL_FOR_NCC = 4  # ~4 µm/vox, ~450 MB (level 2 would be 7 GB → OOM)

# Per-pair confidence weights (fixed by reasoning, NOT tuned to GT)
# Each component is normalised to [0, 1] per subject before combining.
W_SOMA = 0.35
W_NCC = 0.35
W_ANCHOR = 0.20
W_TPS = 0.10
# Acceptance threshold: per-round adaptive quantile of confidence
ACCEPT_QUANTILE_ROUND0 = 0.50  # round 0 accepts top 50 % by confidence
ACCEPT_QUANTILE_GROW = 0.40    # later rounds: top 40 %


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def stack_vec_list(vec_list, m: int):
    n = len(vec_list)
    stack = np.zeros((n, m, 3), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    for r, v in enumerate(vec_list):
        if v.shape[0] == 0:
            continue
        if v.shape[0] >= m:
            stack[r] = v[:m]; valid[r] = True
        else:
            stack[r, :v.shape[0]] = v
            stack[r, v.shape[0]:] = v[-1]
    return stack, valid


def fit_tps(src_zyx: np.ndarray, dst_zyx: np.ndarray) -> dict | None:
    """Per-axis thin-plate Rbf src → dst."""
    if len(src_zyx) < 4:
        return None
    try:
        rz = Rbf(src_zyx[:, 0], src_zyx[:, 1], src_zyx[:, 2],
                  dst_zyx[:, 0], function="thin_plate")
        ry = Rbf(src_zyx[:, 0], src_zyx[:, 1], src_zyx[:, 2],
                  dst_zyx[:, 1], function="thin_plate")
        rx = Rbf(src_zyx[:, 0], src_zyx[:, 1], src_zyx[:, 2],
                  dst_zyx[:, 2], function="thin_plate")
    except Exception:
        return None
    return dict(rbf_z=rz, rbf_y=ry, rbf_x=rx, src=src_zyx, dst=dst_zyx)


def apply_tps(tps: dict, pts_zyx: np.ndarray) -> np.ndarray:
    z = tps["rbf_z"](pts_zyx[:, 0], pts_zyx[:, 1], pts_zyx[:, 2])
    y = tps["rbf_y"](pts_zyx[:, 0], pts_zyx[:, 1], pts_zyx[:, 2])
    x = tps["rbf_x"](pts_zyx[:, 0], pts_zyx[:, 1], pts_zyx[:, 2])
    return np.column_stack([z, y, x])


def soma_score_with_neighbour_indices(
    cz_zyx: np.ndarray, hcr_zyx: np.ndarray,
    cand_pairs: list[tuple[int, int]],
    *, m_cz: int, m_hcr: int, n: int,
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Score each candidate pair AND return the n best (cz_nbr, hcr_nbr)
    index pairs in the original CZ / HCR neighbour lists so we can run
    the anchor-vote later.
    """
    cz_vecs = cell_vectors(cz_zyx, m=m_cz)
    hcr_vecs = cell_vectors(hcr_zyx, m=m_hcr)
    cz_tree = cKDTree(cz_zyx)
    hcr_tree = cKDTree(hcr_zyx)
    _, cz_nbr_idx = cz_tree.query(cz_zyx, k=min(m_cz + 1, len(cz_zyx)))
    _, hcr_nbr_idx = hcr_tree.query(hcr_zyx, k=min(m_hcr + 1, len(hcr_zyx)))
    cz_nbr_idx = cz_nbr_idx[:, 1:]
    hcr_nbr_idx = hcr_nbr_idx[:, 1:]
    scores = np.full(len(cand_pairs), np.inf, dtype=np.float32)
    best_pairs_per_cand: list[tuple[np.ndarray, np.ndarray]] = []
    for k, (i, j) in enumerate(cand_pairs):
        ci = cz_vecs[i]
        hj = hcr_vecs[j]
        if ci.shape[0] == 0 or hj.shape[0] == 0:
            best_pairs_per_cand.append((np.empty(0, int), np.empty(0, int)))
            continue
        diff = ci[:, None, :] - hj[None, :, :]
        d = np.linalg.norm(diff, axis=-1)
        flat = d.ravel()
        if flat.size < n:
            best_pairs_per_cand.append((np.empty(0, int), np.empty(0, int)))
            continue
        order = np.argpartition(flat, n - 1)[:n]
        scores[k] = float(flat[order].mean())
        rows = order // d.shape[1]
        cols = order % d.shape[1]
        rows = np.clip(rows, 0, cz_nbr_idx.shape[1] - 1)
        cols = np.clip(cols, 0, hcr_nbr_idx.shape[1] - 1)
        best_pairs_per_cand.append(
            (cz_nbr_idx[i, rows], hcr_nbr_idx[j, cols])
        )
    return scores, best_pairs_per_cand


def normalise_lower_better(arr: np.ndarray) -> np.ndarray:
    """Map a lower-is-better metric to [0, 1] confidence (higher = better).
    Robust to NaN/inf — median-normalised, clipped."""
    a = np.asarray(arr, dtype=float)
    finite = np.isfinite(a)
    if not finite.any():
        return np.zeros_like(a)
    med = np.median(a[finite])
    p95 = np.quantile(a[finite], 0.95)
    if not np.isfinite(p95) or p95 <= med:
        p95 = med + 1.0
    conf = 1.0 - (a - med) / (p95 - med)
    conf = np.clip(conf, 0.0, 1.0)
    conf[~np.isfinite(a)] = 0.0
    return conf


def normalise_higher_better(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    finite = np.isfinite(a)
    if not finite.any():
        return np.zeros_like(a)
    med = np.median(a[finite])
    p95 = np.quantile(a[finite], 0.95)
    if not np.isfinite(p95) or p95 <= med:
        return np.where(finite, (a >= med).astype(float), 0.0)
    conf = (a - med) / (p95 - med)
    conf = np.clip(conf, 0.0, 1.0)
    conf[~np.isfinite(a)] = 0.0
    return conf


# --------------------------------------------------------------------------- #
# Subject driver
# --------------------------------------------------------------------------- #
def prepare_subject(sid: str, sz_pins: dict):
    """Match Step 2.5 prep — pia-anchored Z, GT-free R_cand, GFP+ ∩ ok."""
    inp = subject_inputs(sid, sz_pins=sz_pins)
    cz_lp = inp.cz_lp_um.copy()
    ok_set = inp.gfp_ids & inp.ok_ids
    in_ok = np.array([int(h) in ok_set for h in inp.hcr_ids])
    z_lo_density, z_hi_density, z_meta = z_density_intersection_bounds(
        cz_lp, inp.hcr_um, in_ok,
    )
    pia_z_min = hcr_pia_z_over_region(inp.s, cz_lp[:, 1:])
    z_lo = min(z_lo_density, pia_z_min) if np.isfinite(pia_z_min) else z_lo_density
    z_hi = z_hi_density
    lo_xy = cz_lp.min(axis=0)[1:] - 30.0
    hi_xy = cz_lp.max(axis=0)[1:] + 30.0
    in_bbox = (
        (inp.hcr_um[:, 0] >= z_lo) & (inp.hcr_um[:, 0] <= z_hi)
        & (inp.hcr_um[:, 1] >= lo_xy[0]) & (inp.hcr_um[:, 1] <= hi_xy[0])
        & (inp.hcr_um[:, 2] >= lo_xy[1]) & (inp.hcr_um[:, 2] <= hi_xy[1])
    )
    cz_in_band = (cz_lp[:, 0] >= z_lo) & (cz_lp[:, 0] <= z_hi)
    keep_h = in_ok & in_bbox
    hcr_pool_zyx = inp.hcr_um[keep_h]
    hcr_pool_ids = inp.hcr_ids[keep_h].astype(int)
    cz_pool_zyx_lp = cz_lp[cz_in_band]
    cz_pool_ids = inp.cz_ids[cz_in_band]
    cz_pool_zyx_native = inp.cz_um[cz_in_band]
    R_cand, nn_meta = adaptive_r_cand(cz_pool_zyx_lp, hcr_pool_zyx)
    return dict(
        sid=sid, inp=inp,
        cz_pool_zyx_lp=cz_pool_zyx_lp,
        cz_pool_zyx_native=cz_pool_zyx_native,
        cz_pool_ids=cz_pool_ids,
        hcr_pool_zyx=hcr_pool_zyx,
        hcr_pool_ids=hcr_pool_ids,
        R_cand_um=R_cand,
        z_bounds=(z_lo, z_hi),
        nn_meta=nn_meta,
    )


def round_score(
    cz_cur_zyx: np.ndarray, hcr_zyx: np.ndarray,
    R_cand: float, m_cz: int, m_hcr: int, n: int,
):
    """Score every CZ × candidate-HCR (within R_cand).  Returns (D, cands_per_cz)."""
    hcr_stack, hcr_valid = stack_vec_list(cell_vectors(hcr_zyx, m=m_hcr), m_hcr)
    cz_vecs = cell_vectors(cz_cur_zyx, m=m_cz)
    hcr_tree = cKDTree(hcr_zyx)
    n_cz = len(cz_cur_zyx); n_hcr = len(hcr_zyx)
    D = np.full((n_cz, n_hcr), np.inf, dtype=np.float32)
    cands_per_cz: list[np.ndarray] = []
    for i in range(n_cz):
        ci = cz_vecs[i]
        if ci.shape[0] < n:
            cands_per_cz.append(np.empty(0, dtype=int))
            continue
        cand = np.fromiter(hcr_tree.query_ball_point(cz_cur_zyx[i], r=R_cand), dtype=int)
        cand = cand[hcr_valid[cand]] if cand.size > 0 else cand
        cands_per_cz.append(cand)
        if cand.size == 0:
            continue
        scores = _score_soma_per_gt(ci, hcr_stack[cand], n)
        D[i, cand] = scores
    return D, cands_per_cz


def mutual_best_pairs(D: np.ndarray) -> list[tuple[int, int]]:
    cz_best = np.full(D.shape[0], -1, dtype=int)
    hcr_best = np.full(D.shape[1], -1, dtype=int)
    for i in range(D.shape[0]):
        row = D[i]
        if np.isfinite(row).any():
            cz_best[i] = int(np.argmin(np.where(np.isfinite(row), row, np.inf)))
    for j in range(D.shape[1]):
        col = D[:, j]
        if np.isfinite(col).any():
            hcr_best[j] = int(np.argmin(np.where(np.isfinite(col), col, np.inf)))
    pairs = []
    for i in range(D.shape[0]):
        j = cz_best[i]
        if j >= 0 and hcr_best[j] == i:
            pairs.append((i, j))
    return pairs


def anchor_vote(
    best_pair_indices: list[tuple[np.ndarray, np.ndarray]],
    cand_row_pairs: list[tuple[int, int]],
    active_row_pairs: set[tuple[int, int]],
) -> np.ndarray:
    """Fraction of (cz_nbr, hcr_nbr) cells implied by the soma score's
    n-best vector pairs that are themselves in the active match set."""
    out = np.zeros(len(cand_row_pairs), dtype=np.float32)
    for k, (cz_idx, hcr_idx) in enumerate(best_pair_indices):
        if cz_idx.size == 0:
            continue
        hits = sum(
            int((int(a), int(b)) in active_row_pairs)
            for a, b in zip(cz_idx, hcr_idx)
        )
        out[k] = hits / cz_idx.size
    return out


def run_subject(sid: str, sz_pins: dict) -> dict:
    print(f"\n=== {sid} ===")
    t_subj = time.time()
    subj = prepare_subject(sid, sz_pins)
    inp = subj["inp"]
    cz_zyx_native = subj["cz_pool_zyx_native"]   # CZ µm
    cz_zyx_lp = subj["cz_pool_zyx_lp"]            # HCR µm under locked frame
    hcr_zyx = subj["hcr_pool_zyx"]
    n_cz = len(cz_zyx_native); n_hcr = len(hcr_zyx)
    R_cand = subj["R_cand_um"]
    print(
        f"  CZ pool {n_cz}  HCR pool {n_hcr}  R_cand={R_cand:.0f}µm  "
        f"Z={subj['z_bounds']}"
    )

    # Load image volumes for per-pair local NCC
    cz_vol = load_cz_volume(inp.s)
    hcr_vol, hcr_xy_um, hcr_z_um = load_hcr_volume(
        inp.s, channel="488", level=HCR_LEVEL_FOR_NCC,
    )
    cz_xy_um = float(inp.s.cz_xy_um); cz_z_um = float(inp.s.cz_z_um)
    print(
        f"  CZ vol {cz_vol.shape}  HCR vol {hcr_vol.shape}  "
        f"HCR vox={hcr_xy_um:.2f}xy {hcr_z_um:.2f}z µm"
    )

    cz_cur_zyx = cz_zyx_lp.copy()   # mutated by TPS each round
    accepted_row_pairs: set[tuple[int, int]] = set()
    accepted_meta: dict[tuple[int, int], dict] = {}
    tps = None
    round_log = []
    stall = 0

    for rd in range(MAX_ROUNDS):
        t_rd = time.time()
        # --- 1. Score every CZ × candidate-HCR with current pose ---
        D, _ = round_score(
            cz_cur_zyx, hcr_zyx, R_cand=R_cand,
            m_cz=SOMA_M_CZ, m_hcr=SOMA_M_HCR, n=SOMA_N,
        )
        mb_pairs = mutual_best_pairs(D)
        if len(mb_pairs) == 0:
            print(f"  rd{rd}: no mutual-best pairs"); break

        # --- 2. Recompute soma score + best-neighbour indices for MB pairs ---
        scores, best_pair_indices = soma_score_with_neighbour_indices(
            cz_cur_zyx, hcr_zyx, mb_pairs,
            m_cz=SOMA_M_CZ, m_hcr=SOMA_M_HCR, n=SOMA_N,
        )

        # --- 3. Local image NCC for each MB pair ---
        cz_center_um_native = cz_zyx_native[np.array([p[0] for p in mb_pairs])]
        hcr_center_um = hcr_zyx[np.array([p[1] for p in mb_pairs])]
        ncc = per_pair_local_ncc(
            cz_vol, hcr_vol,
            cz_center_um_native, hcr_center_um,
            cz_voxel_um_z=cz_z_um, cz_voxel_um_xy=cz_xy_um,
            hcr_voxel_um_z=hcr_z_um, hcr_voxel_um_xy=hcr_xy_um,
            patch_um=LOCAL_NCC_PATCH_UM, target_um=LOCAL_NCC_TARGET_UM,
        )

        # --- 4. Anchor-vote (using accepted set from previous rounds) ---
        ancv = anchor_vote(best_pair_indices, mb_pairs, accepted_row_pairs)

        # --- 5. TPS residual (only meaningful after round 0) ---
        if tps is not None:
            cz_warped = apply_tps(tps, cz_zyx_lp)
            tps_resid = np.array([
                np.linalg.norm(cz_warped[i] - hcr_zyx[j])
                for (i, j) in mb_pairs
            ], dtype=float)
        else:
            tps_resid = np.full(len(mb_pairs), np.nan, dtype=float)

        # --- 6. Combine into per-pair confidence (GT-free) ---
        c_soma = normalise_lower_better(scores)
        c_ncc = normalise_higher_better(ncc)
        c_anchor = ancv  # already in [0, 1]
        if np.all(np.isnan(tps_resid)):
            c_tps = np.zeros_like(tps_resid, dtype=float)
            w_tps_eff = 0.0  # no signal yet
        else:
            c_tps = normalise_lower_better(tps_resid)
            w_tps_eff = W_TPS
        # Rebalance weights given missing components
        w_sum = W_SOMA + W_NCC + W_ANCHOR + w_tps_eff
        confidence = (
            W_SOMA * c_soma + W_NCC * c_ncc
            + W_ANCHOR * c_anchor + w_tps_eff * c_tps
        ) / w_sum

        # --- 7. Promote pairs above adaptive quantile ---
        q = ACCEPT_QUANTILE_ROUND0 if rd == 0 else ACCEPT_QUANTILE_GROW
        thr = float(np.quantile(confidence, 1.0 - q))
        new_count = 0
        for k, p in enumerate(mb_pairs):
            if confidence[k] >= thr and p not in accepted_row_pairs:
                accepted_row_pairs.add(p)
                accepted_meta[p] = dict(
                    soma_score=float(scores[k]),
                    local_ncc=float(ncc[k]) if np.isfinite(ncc[k]) else float("nan"),
                    anchor_vote=float(c_anchor[k]),
                    tps_residual=float(tps_resid[k]) if np.isfinite(tps_resid[k]) else float("nan"),
                    confidence=float(confidence[k]),
                    accepted_round=rd,
                )
                new_count += 1

        # --- 8. Fit TPS on the active set; shrink R_cand by residual quantile ---
        if len(accepted_row_pairs) >= 4:
            src = np.array([cz_zyx_lp[i] for (i, _) in accepted_row_pairs])
            dst = np.array([hcr_zyx[j] for (_, j) in accepted_row_pairs])
            tps = fit_tps(src, dst)
            if tps is not None:
                cz_warped = apply_tps(tps, cz_zyx_lp)
                resid = np.array([
                    np.linalg.norm(cz_warped[i] - hcr_zyx[j])
                    for (i, j) in accepted_row_pairs
                ])
                R_cand = max(15.0, float(np.quantile(resid, 0.95) * 1.5))
                cz_cur_zyx = cz_warped

        round_log.append(dict(
            round=rd, n_mb=len(mb_pairs),
            n_new=new_count, n_active=len(accepted_row_pairs),
            threshold=thr, R_cand_um=R_cand,
            confidence_p10=float(np.quantile(confidence, 0.1)),
            confidence_p50=float(np.quantile(confidence, 0.5)),
            confidence_p90=float(np.quantile(confidence, 0.9)),
            elapsed_s=round(time.time() - t_rd, 1),
        ))
        print(
            f"  rd{rd}: |MB|={len(mb_pairs):>4}  "
            f"thr={thr:.3f}  new={new_count:>3}  "
            f"|active|={len(accepted_row_pairs):>4}  "
            f"R_cand_next={R_cand:.0f}µm  "
            f"[{time.time()-t_rd:.1f}s]"
        )
        if new_count == 0:
            stall += 1
            if stall > MAX_STALL_ROUNDS:
                print(f"  saturated after round {rd}")
                break
        else:
            stall = 0

    # Materialise outputs
    out_dir = OUT_ROOT / sid
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    cz_pool_ids = subj["cz_pool_ids"]
    hcr_pool_ids = subj["hcr_pool_ids"]
    coreg_lookup = {
        int(c): int(h)
        for c, h in zip(inp.coreg["cz_id"].astype(int),
                          inp.coreg["hcr_id"].astype(int))
    }
    for (i, j), meta in accepted_meta.items():
        cz_id = int(cz_pool_ids[i]); hcr_id = int(hcr_pool_ids[j])
        is_gt = int(coreg_lookup.get(cz_id) == hcr_id)
        rows.append(dict(
            sid=sid, cz_id=cz_id, hcr_id=hcr_id,
            cz_lp_z=float(cz_zyx_lp[i, 0]),
            cz_lp_y=float(cz_zyx_lp[i, 1]),
            cz_lp_x=float(cz_zyx_lp[i, 2]),
            hcr_z=float(hcr_zyx[j, 0]),
            hcr_y=float(hcr_zyx[j, 1]),
            hcr_x=float(hcr_zyx[j, 2]),
            **meta,
            is_gt=is_gt,
        ))
    matches_df = pd.DataFrame(rows)
    matches_df.to_csv(out_dir / "matches_final.csv", index=False)
    round_df = pd.DataFrame(round_log)
    round_df.to_csv(out_dir / "round_log.csv", index=False)
    if tps is not None:
        with open(out_dir / "tps_warp.pkl", "wb") as f:
            pickle.dump(tps, f)
    summary = dict(
        sid=sid,
        n_active_final=len(accepted_row_pairs),
        n_rounds=len(round_log),
        wall_s=round(time.time() - t_subj, 1),
    )
    print(f"  done: |active|={summary['n_active_final']} in "
          f"{summary['n_rounds']} rounds  [{summary['wall_s']}s]")
    return summary


def main():
    sz_pins = load_sz_pins()
    summary = []
    for sid in BENCHMARK_SUBJECTS:
        try:
            s = run_subject(sid, sz_pins)
            summary.append(s)
        except Exception as exc:
            print(f"[{sid}] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            raise
    df = pd.DataFrame(summary)
    df.to_csv(OUT_ROOT / "step3_summary.csv", index=False)
    print("\n=== Summary ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
