"""Step 2.5 — Locked-frame benchmark with three refinements.

Same machinery as run_step2_locked.py, but:

1. **Per-subject R_cand derived from data (no GT, no benchmark).**
   ``R_cand = max(50, 2 × NN_p90)`` per subject — derived from cell
   positions only.

2. **Z boundary from ROI density intersection, anchored at the pia.**
   The HCR pia surface is the registration anchor (surface_registration_v2
   was fit there), so it must always be inside the Z range.  We compute
   ``z_lo`` as the *minimum* of (density-intersection lower bound,
   per-xy pia z over the working region) — guaranteeing the pia is
   included.  ``z_hi`` still comes from the density intersection so
   we trim the deep end where one modality runs out.

3. Save per-subject matched-pair table to outputs/step2p5_matches/<sid>.csv
   for downstream visualization.

GFP+ filter is retained — Step 1 showed the descriptor isn't hurt by
it, only the pool count.

Output: outputs/step2p5_refined_summary.csv +
        outputs/step2p5_matches/<sid>.csv (matched pairs per subject)
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from autocoreg.io.inputs import BENCHMARK_SUBJECTS, load_sz_pins, subject_inputs
from autocoreg.archive.oracle_benchmark import metrics_from_D
from autocoreg.archive.locked_benchmark import _score_soma_per_gt
from autocoreg.finetune_soma_print.descriptor import cell_vectors
from autocoreg.initial_registration.surfaces import get_hcr_top_surface_iter07

MATCH_DIR = Path("/scratch/autocoreg_outputs/step2p5_matches")
MATCH_DIR.mkdir(parents=True, exist_ok=True)

OUT_DIR = Path("/scratch/autocoreg_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Same soma config family as Step 2 winning region
SOMA_M_CZ = 15
SOMA_M_HCR = 30
SOMA_N = 5

# Z-density threshold (fraction of peak) to define "where both modalities have cells"
Z_DENSITY_PEAK_FRAC = 0.10
# XY margin around bbox (µm) — keep the original Step 2 margin
XY_MARGIN_UM = 30.0


def stack_vecs(vec_list, m: int):
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


# --------------------------------------------------------------------------- #
# Refinement 2: Z-density intersection
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Refinement 1: per-subject R_cand from CZ→HCR NN distribution
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Per-subject prep with the two refinements
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def score_subject(subj) -> dict:
    cz_zyx = subj["cz_pool_zyx"]
    hcr_zyx = subj["hcr_pool_zyx"]
    n_gt = len(subj["gt_pairs"])
    n_hcr = len(hcr_zyx)
    n_cz = len(cz_zyx)
    if n_gt == 0:
        return dict(empty=True)
    cz_vecs_list = cell_vectors(cz_zyx, m=SOMA_M_CZ)
    hcr_stack, hcr_valid = stack_vecs(
        cell_vectors(hcr_zyx, m=SOMA_M_HCR), SOMA_M_HCR,
    )
    hcr_tree = cKDTree(hcr_zyx)
    cz_tree = cKDTree(cz_zyx)
    R_cand = subj["R_cand_um"]

    # Full per-CZ × candidate-HCR scoring
    D_full = np.full((n_cz, n_hcr), fill_value=np.inf, dtype=np.float32)
    for i in range(n_cz):
        ci = cz_vecs_list[i]
        if ci.shape[0] < SOMA_N:
            continue
        cand = np.fromiter(hcr_tree.query_ball_point(cz_zyx[i], r=R_cand), dtype=int)
        cand = cand[hcr_valid[cand]] if cand.size > 0 else cand
        if cand.size == 0:
            continue
        scores = _score_soma_per_gt(ci, hcr_stack[cand], SOMA_N)
        D_full[i, cand] = scores

    # Metrics on GT-CZ rows
    D_gt = D_full[subj["gt_cz_rows"]]
    m = metrics_from_D(D_gt, subj["gt_hcr_rows"], hcr_zyx)

    # Mutual-best fraction on GT pairs
    cz_best = np.full(n_cz, -1, dtype=int)
    hcr_best = np.full(n_hcr, -1, dtype=int)
    for i in range(n_cz):
        row = D_full[i]
        if np.isfinite(row).any():
            cz_best[i] = int(np.argmin(np.where(np.isfinite(row), row, np.inf)))
    for j in range(n_hcr):
        col = D_full[:, j]
        if np.isfinite(col).any():
            hcr_best[j] = int(np.argmin(np.where(np.isfinite(col), col, np.inf)))
    mb = 0
    for i, hcr_r in zip(subj["gt_cz_rows"], subj["gt_hcr_rows"]):
        if cz_best[i] == hcr_r and hcr_best[hcr_r] == i:
            mb += 1
    mb_frac = mb / max(n_gt, 1)
    system_mb = sum(
        1 for i in range(n_cz)
        if cz_best[i] >= 0 and hcr_best[cz_best[i]] == i
    )

    # Build matched-pair table (all mutual-best pairs) and save
    matches = []
    cz_ids = subj["cz_pool_ids"]
    hcr_ids = subj["hcr_pool_ids"]
    gt_lookup = {int(c): int(h) for c, h in subj["gt_pairs"]}
    for i in range(n_cz):
        if cz_best[i] < 0:
            continue
        j = cz_best[i]
        if hcr_best[j] != i:
            continue
        is_gt = int(gt_lookup.get(int(cz_ids[i])) == int(hcr_ids[j]))
        matches.append(dict(
            sid=subj["sid"],
            cz_id=int(cz_ids[i]),
            hcr_id=int(hcr_ids[j]),
            cz_lp_z=float(cz_zyx[i, 0]),
            cz_lp_y=float(cz_zyx[i, 1]),
            cz_lp_x=float(cz_zyx[i, 2]),
            hcr_z=float(hcr_zyx[j, 0]),
            hcr_y=float(hcr_zyx[j, 1]),
            hcr_x=float(hcr_zyx[j, 2]),
            soma_score=float(D_full[i, j]),
            is_gt=is_gt,
        ))
    matches_df = pd.DataFrame(matches)
    matches_df.to_csv(MATCH_DIR / f"{subj['sid']}.csv", index=False)

    return dict(
        n_gt=n_gt, n_cz_pool=n_cz, n_hcr_pool=n_hcr,
        R_cand_used=float(R_cand), **m,
        mutual_best_gt=int(mb), mutual_best_gt_frac=float(mb_frac),
        mutual_best_system=int(system_mb),
        n_matches_saved=len(matches),
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    sz_pins = load_sz_pins()
    rows = []
    for sid in BENCHMARK_SUBJECTS:
        t0 = time.time()
        print(f"\n=== {sid} ===")
        subj = prepare_subject(sid, sz_pins)
        z_lo, z_hi = subj["z_bounds"]
        zm = subj["z_meta"]
        print(
            f"  Z [{z_lo:.0f}, {z_hi:.0f}] µm  "
            f"(pia_z={zm.get('pia_z_min', float('nan')):.0f}, "
            f"density_z_lo={zm.get('z_lo_density', float('nan')):.0f})  "
            f"(CZ kept {len(subj['cz_pool_zyx'])} / HCR kept {len(subj['hcr_pool_zyx'])})  "
            f"GT in pool {len(subj['gt_pairs'])}  "
            f"R_cand={subj['R_cand_um']:.0f} µm "
            f"(p90={subj['nn_meta']['nn_p90']:.0f} µm)"
        )
        m = score_subject(subj)
        if m.get("empty"):
            continue
        print(
            f"  r@1={m['recall@1']:.3f}  r@5={m['recall@5']:.3f}  "
            f"r@20={m['recall@20']:.3f}  "
            f"AUC50={m['auc_R50um']:.3f}  "
            f"mutual_best_GT={m['mutual_best_gt']}/{m['n_gt']} "
            f"({m['mutual_best_gt_frac']:.3f})  "
            f"system_mb={m['mutual_best_system']}  "
            f"[{time.time()-t0:.1f}s]"
        )
        rows.append(dict(
            subject_id=sid,
            z_lo=z_lo, z_hi=z_hi,
            nn_p50=subj["nn_meta"]["nn_p50"],
            nn_p90=subj["nn_meta"]["nn_p90"],
            **m,
        ))
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "step2p5_refined_summary.csv", index=False)
    print("\n=== Summary ===")
    print(df.to_string(index=False))
    print(f"\nwrote {OUT_DIR / 'step2p5_refined_summary.csv'}")


if __name__ == "__main__":
    main()
