"""Step 3 v3 — Decluttered geometric matcher.

Three primary gates compared one at a time: likelihood_ratio / anchor_vote / ncc.

Design decisions from step3_v3_spec_2026-06-01.md:
  • Fixed R_CAND_UM=150 every round (radius re-centres on TPS-warped CZ).
  • One primary gate per run, applied identically every round.
  • local_flow is round-0-ONLY pre-filter, toggled by --local_flow_rd0 (default ON).
  • Mutual-best ON every round.
  • No Stage 2 / anchor_restricted by default.
  • Re-evaluation each round (no locking).

Optional Stage-2 anchor-restricted addendum (--anchor_restricted):
  • After Stage-1 (the chosen --gate) converges, if |accepted| >= ANCHOR_RESTRICTED_N_ANCHORS,
    run a Stage-2 that rebuilds each CZ cell's descriptor from the n nearest accepted
    anchor pairs (anchor-restricted approach).
  • Stage-2 uses the same R_CAND_UM=150 radius for candidates (not mutual KNN)
    and the same anchor_vote gate (ANCHOR_VOTE_FRAC=3/5).
  • Stage-2 writes to outputs/<gate>[_noLF]_anchor_restricted/<sid>/.
  • Production defaults are UNCHANGED; --anchor_restricted is opt-in only.

Usage:
  python python -m autocoreg.finetune_soma_print.matcher --gate {likelihood_ratio,anchor_vote,ncc} [--no_local_flow_rd0] [--sids SID ...]
  python python -m autocoreg.finetune_soma_print.matcher --gate anchor_vote --anchor_restricted --sids 755252 767022 788406 790322

Outputs (Stage-1 only):
  outputs/<gate>[_noLF]/<sid>/
    matches_round{0..T}.csv
    rounds_log.csv
    tps_warp_final.pkl
    local_flow_rd0_dropped.csv
    mb_rejection_log.csv
    pair_trajectory.csv
  outputs/<gate>[_noLF]/summary.csv

Outputs (--anchor_restricted, Stage-2):
  outputs/<gate>[_noLF]_anchor_restricted/<sid>/
    (same files as Stage-1, plus anchor_restricted_rounds_log.csv)
  outputs/<gate>[_noLF]_anchor_restricted/summary.csv
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import Rbf
from scipy.spatial import cKDTree, ConvexHull
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

from autocoreg.io.inputs import BENCHMARK_SUBJECTS, load_sz_pins, scoring_gt
from autocoreg.finetune_soma_print.scoring import _score_soma_per_gt
from autocoreg.finetune_soma_print.pool_prep import prepare_subject
from autocoreg.finetune_soma_print.tps import anchor_vote, fit_tps, apply_tps, soma_score_with_neighbour_indices
from autocoreg.finetune_soma_print.descriptor import cell_vectors, score_pair

# Matcher output root. Env-overridable; NEVER defaults to /tmp or / (storage rule).
OUT_BASE = Path(os.environ.get("MFISH_MATCHER_OUT_BASE", "/scratch/autocoreg_outputs/matches"))

# ── Constants (single source of truth per spec §8) ────────────────────────────

# Descriptor (m-NN soma-print)
SOMA_M_CZ = 15
SOMA_M_HCR = 30
SOMA_N = 5

# Candidate search: single fixed radius, applied every round.
# The TPS re-centres cz_cur each round; the radius stays constant.
# 150 µm: GT-coverage diagnostic (2026-06-01) showed the warm-start positional
# error is large — 50 µm covered only 1–83 % of GT matches per subject (755252 1 %,
# 782149 2 %). R150 covers ≥98 % on all 6; R100 still misses 755252/782149.
R_CAND_UM = 150.0

# Round control
MAX_ROUNDS = 5
CONVERGE_REL_DELTA = 0.02  # converge if (added+removed)/|prev| < 2 %
MIN_LANDMARKS_FOR_ABORT = 20

# Round-0-only pre-filter defaults (overridden by --no_local_flow_rd0)
LOCAL_FLOW_K = 15
LOCAL_FLOW_QUANTILE = 0.90

# Anchor-support gate (every round, both stages): a candidate may only be
# promoted to an ANCHOR if its CZ source position lies within the convex hull of
# the PREVIOUS round's accepted-anchor source points, plus this margin. This is
# a hard, subtractive-only guarantee that a cell dragged OUTSIDE the trustworthy
# (interpolated) support region — e.g. the deepest CZ slab whose true HCR
# partner is beyond the overlap — can never anchor the TPS and lock in a fold.
# Keyed off the PREVIOUS round's accepted set, so it is non-circular (last
SUPPORT_HULL_MARGIN_UM = R_CAND_UM
# Primary gate: LR
LIKELIHOOD_RATIO_THRESHOLD = 0.05

# Primary gate: anchor_vote
# 0.6 = 3/5 — relaxed from the cross-round 5/5 (1.0) per spec §3 note.
# The within-round match-set has no bootstrapping advantage, so 100 % is too
# strict; 60 % passes a majority of the soma-score's supporting neighbours.
ANCHOR_VOTE_FRAC = 0.6

# Primary gate: NCC (LOO image δ)
NCC_K_NEIGHBOURS = 20       # K nearest anchors (after radius skip) for local TPS
NCC_SKIP_RADIUS_UM = 50.0   # physical-radius skip replacing count-based k_skip_nearest
NCC_PATCH_XY_UM = 160.0
NCC_PATCH_Z_UM = 24.0
NCC_STEP_UM = 4.0
NCC_MIP_2D = True           # only used in the single-slab (n_slabs=1) path
NCC_N_SLABS = 3             # 3-slab mean: z→xy, y→xz, x→yz (86.4% pre-check accuracy)
NCC_DELTA_THRESHOLD = -0.05
# NCC LOO-δ is the cost driver; parallelize the per-pair loop across processes.
# fork() shares the (read-only) image volumes copy-on-write, so workers don't
# copy them. Default leaves headroom if several subjects run concurrently.
NCC_WORKERS = min(8, max(1, (os.cpu_count() or 4) - 2))

# Stage-2 anchor-restricted descriptor (--anchor_restricted only)
# Matches v2's ANCHOR_RESTRICTED_N_ANCHORS=10; centroid-only, no images.
ANCHOR_RESTRICTED_N_ANCHORS = 10


# ── Scoring helpers ────────────────────────────────────────────────────────────

def _stack_vec_list(vec_list, m: int):
    """Stack cell_vectors output into a (N, m, 3) array + valid mask."""
    n = len(vec_list)
    stack = np.zeros((n, m, 3), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    for r, v in enumerate(vec_list):
        if v.shape[0] == 0:
            continue
        if v.shape[0] >= m:
            stack[r] = v[:m]
            valid[r] = True
        else:
            stack[r, :v.shape[0]] = v
            stack[r, v.shape[0]:] = v[-1]
    return stack, valid


def soma_score_radius(
    cz_zyx: np.ndarray, hcr_zyx: np.ndarray, R_cand: float,
    m_cz: int, m_hcr: int, n: int,
) -> np.ndarray:
    """Score every CZ cell against HCR candidates within R_cand µm.

    Used every round (not just round 0): cz_zyx is already TPS-warped to the
    current best estimate of the HCR frame.
    """
    cz_vecs = cell_vectors(cz_zyx, m=m_cz)
    hcr_stack, hcr_valid = _stack_vec_list(cell_vectors(hcr_zyx, m=m_hcr), m_hcr)
    hcr_tree = cKDTree(hcr_zyx)
    n_cz = len(cz_zyx)
    n_hcr = len(hcr_zyx)
    D = np.full((n_cz, n_hcr), np.inf, dtype=np.float32)
    for i in range(n_cz):
        ci = cz_vecs[i]
        if ci.shape[0] < n:
            continue
        cand = np.fromiter(hcr_tree.query_ball_point(cz_zyx[i], r=R_cand), dtype=int)
        if cand.size == 0:
            continue
        cand = cand[hcr_valid[cand]]
        if cand.size == 0:
            continue
        scores = _score_soma_per_gt(ci, hcr_stack[cand], n)
        D[i, cand] = scores
    return D


def anchor_restricted_score_radius(
    cz_cur: np.ndarray, hcr_zyx: np.ndarray,
    accepted: set[tuple[int, int]],
    R_cand: float,
    n_anchors: int = ANCHOR_RESTRICTED_N_ANCHORS,
) -> np.ndarray:
    """anchor-restricted descriptor, radius-candidate version for v3.

    For each CZ cell i:
      • Find the n_anchors nearest CZ cells that are in `accepted` (Euclidean in
        the current TPS frame); their paired HCR cells are the local HCR anchors.
      • CZ descriptor d_cz[k] = cz_cur[anchor_cz[k]] - cz_cur[i]
    For each HCR candidate j within R_cand µm of cz_cur[i]:
      • d_hcr[k] = hcr_zyx[anchor_hcr[k]] - hcr_zyx[j]
      • score(i, j) = mean_k ||d_cz[k] - d_hcr[k]||   (mean Euclidean, µm)

    Radius candidates (R_CAND_UM=150) instead of mutual KNN keeps candidate
    selection identical to the v3 Stage-1 `soma_score_radius`.
    """
    n_cz, n_hcr = cz_cur.shape[0], hcr_zyx.shape[0]
    D = np.full((n_cz, n_hcr), np.inf, dtype=np.float32)
    if len(accepted) < n_anchors:
        return D
    acc = np.array(list(accepted), dtype=int)
    cz_anchor_idx = acc[:, 0]
    hcr_anchor_idx = acc[:, 1]
    cz_anchor_pos = cz_cur[cz_anchor_idx]
    cz_anchor_tree = cKDTree(cz_anchor_pos)
    hcr_tree = cKDTree(hcr_zyx)

    for i in range(n_cz):
        # n_anchors nearest accepted CZ cells (excluding self if accepted).
        k_ask = min(n_anchors + 1, len(cz_anchor_pos))
        _, anchor_local = cz_anchor_tree.query(cz_cur[i], k=k_ask)
        anchor_local = np.atleast_1d(anchor_local)
        # Drop self if cell i is itself an anchor (avoid zero-vector).
        keep = cz_anchor_idx[anchor_local] != i
        anchor_local = anchor_local[keep][:n_anchors]
        if len(anchor_local) < n_anchors:
            continue
        cz_pair = cz_anchor_idx[anchor_local]
        hcr_pair = hcr_anchor_idx[anchor_local]
        d_cz = cz_cur[cz_pair] - cz_cur[i]       # (n_anchors, 3)
        hcr_anchor_pos = hcr_zyx[hcr_pair]        # (n_anchors, 3)

        # Radius candidates in HCR µm (same as v3 Stage-1).
        cand = np.fromiter(hcr_tree.query_ball_point(cz_cur[i], r=R_cand), dtype=int)
        if cand.size == 0:
            continue
        for j in cand:
            d_hcr = hcr_anchor_pos - hcr_zyx[j]  # (n_anchors, 3)
            D[i, j] = float(np.linalg.norm(d_cz - d_hcr, axis=1).mean())
    return D


def mutual_best_pairs_from_D(D: np.ndarray):
    """Return (set of (i,j) mutual-best pairs, cz_best array, hcr_best array)."""
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
    pairs: set[tuple[int, int]] = set()
    for i in range(D.shape[0]):
        j = cz_best[i]
        if j >= 0 and hcr_best[j] == i:
            pairs.add((i, j))
    return pairs, cz_best, hcr_best


# ── Primary gate implementations ───────────────────────────────────────────────

def local_flow_filter(
    mb_pairs: list[tuple[int, int]],
    cz_zyx: np.ndarray, hcr_zyx: np.ndarray,
    k: int = LOCAL_FLOW_K, quantile: float = LOCAL_FLOW_QUANTILE,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Keep pairs whose displacement is within p_quantile of the subject-level
    deviation distribution.  Round-0-ONLY pre-filter."""
    if len(mb_pairs) == 0:
        return [], np.empty(0)
    pair_cz = np.array([p[0] for p in mb_pairs], dtype=int)
    pair_hcr = np.array([p[1] for p in mb_pairs], dtype=int)
    cz_pos = cz_zyx[pair_cz]
    hcr_pos = hcr_zyx[pair_hcr]
    disp = hcr_pos - cz_pos
    tree = cKDTree(cz_pos)
    k_query = min(k + 1, len(mb_pairs))
    _, idx = tree.query(cz_pos, k=k_query)
    devs = np.empty(len(mb_pairs))
    for n_idx in range(len(mb_pairs)):
        nbr = idx[n_idx][1:]  # exclude self
        v_local = np.median(disp[nbr], axis=0)
        devs[n_idx] = np.linalg.norm(disp[n_idx] - v_local)
    thr = float(np.quantile(devs, quantile))
    kept_mask = devs <= thr
    kept = [mb_pairs[k_] for k_, keep in enumerate(kept_mask) if keep]
    return kept, devs


def support_gate(
    pairs: list[tuple[int, int]],
    cz_src: np.ndarray,
    ref_pairs,
    margin: float = SUPPORT_HULL_MARGIN_UM,
) -> list[tuple[int, int]]:
    """Keep only ``pairs`` whose CZ source point (``cz_src[i]``) lies within the
    convex hull of ``ref_pairs``' CZ source points, expanded by ``margin`` µm.

    ``ref_pairs`` = the PREVIOUS round's accepted anchor set. Prevents a
    candidate whose CZ cell was warped OUTSIDE the trustworthy interpolation
    region from being promoted to an anchor (which would let it — via the
    unbounded thin-plate extrapolation — lock in a local fold). Inert when the
    reference set is too small/degenerate or when every candidate is in-hull.
    """
    if len(pairs) == 0 or len(ref_pairs) < 4 or os.environ.get("MFISH_TPS_BOUND", "1") == "0":
        return pairs
    ref = np.asarray([cz_src[i] for (i, _) in ref_pairs], dtype=float)
    try:
        eqs = ConvexHull(ref).equations           # (n_faces, 4)
    except Exception:
        return pairs
    pts = np.asarray([cz_src[i] for (i, _) in pairs], dtype=float)
    sd = (pts @ eqs[:, :3].T + eqs[:, 3]).max(axis=1)   # >0 outside the hull
    return [p for p, d in zip(pairs, sd) if d <= margin]


def likelihood_ratio_filter(
    mb_pairs: list[tuple[int, int]],
    D: np.ndarray,
    threshold: float = LIKELIHOOD_RATIO_THRESHOLD,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Fit 2-component GMM on per-CZ best scores; 1-component on 2nd-best.
    Keep pairs with likelihood ratio LR(incorrect/correct) < threshold."""
    if len(mb_pairs) == 0:
        return [], np.empty(0)
    best, second = [], []
    for i in range(D.shape[0]):
        row = D[i]
        finite = row[np.isfinite(row)]
        if finite.size < 2:
            continue
        s = np.sort(finite)
        best.append(s[0])
        second.append(s[1])
    best = np.array(best)
    second = np.array(second)
    if len(best) < 30 or len(second) < 30:
        # Too few samples to fit reliably — let all pairs through.
        return list(mb_pairs), np.zeros(len(mb_pairs))
    gmm_best = GaussianMixture(n_components=2, random_state=0).fit(best.reshape(-1, 1))
    gmm_2nd = GaussianMixture(n_components=1, random_state=0).fit(second.reshape(-1, 1))
    mu_best = gmm_best.means_.ravel()
    sd_best = np.sqrt(gmm_best.covariances_.ravel())
    correct_idx = int(np.argmin(mu_best))
    mu_corr = float(mu_best[correct_idx])
    sd_corr = float(sd_best[correct_idx])
    mu_inc = float(gmm_2nd.means_.ravel()[0])
    sd_inc = float(np.sqrt(gmm_2nd.covariances_.ravel()[0]))
    lratio_vals = np.empty(len(mb_pairs))
    for k_, (i, j) in enumerate(mb_pairs):
        s = float(D[i, j])
        p_inc = norm.pdf(s, loc=mu_inc, scale=sd_inc)
        p_cor = norm.pdf(s, loc=mu_corr, scale=sd_corr)
        lratio_vals[k_] = p_inc / (p_cor + 1e-30)
    kept = [mb_pairs[k_] for k_, ratio in enumerate(lratio_vals) if ratio < threshold]
    return kept, lratio_vals


def anchor_vote_gate(
    mb_pairs: list[tuple[int, int]],
    cz_cur: np.ndarray, hcr_zyx: np.ndarray,
    frac_threshold: float = ANCHOR_VOTE_FRAC,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Within-round anchor-vote gate.

    For each mutual-best candidate pair (i, j):
      • Retrieve the SOMA_N best (cz_nbr, hcr_nbr) neighbour-index pairs that
        contributed to the soma-score.
      • Count how many of those neighbour pairs are themselves in the CURRENT
        ROUND'S mutual-best set (NOT accepted_{t-1}).
      • Keep (i, j) if vote_fraction >= frac_threshold.

    Using the within-round MB set avoids bootstrap circularity: round 0 can
    seed itself without any prior accepted set.
    """
    if len(mb_pairs) == 0:
        return [], np.empty(0)
    # Re-score to get neighbour indices (soma_score_with_neighbour_indices
    # computes the same score used in soma_score_radius, just also returns
    # the contributing neighbour pairs per candidate).
    _, best_pair_indices = soma_score_with_neighbour_indices(
        cz_cur, hcr_zyx, mb_pairs,
        m_cz=SOMA_M_CZ, m_hcr=SOMA_M_HCR, n=SOMA_N,
    )
    # match_set = current round's mutual-best pairs (within-round, not t-1).
    match_set: set[tuple[int, int]] = set(mb_pairs)
    av = anchor_vote(best_pair_indices, mb_pairs, match_set)
    kept = [mb_pairs[k_] for k_, frac in enumerate(av)
            if frac >= frac_threshold - 1e-9]
    return kept, av


# Parallel NCC: the read-only context is stashed in a module global and the
# pool is created AFTER it is set, so fork() inherits the (large) image volumes
# copy-on-write — they are never pickled per task. The worker takes only the
# small (i, j) pair. _NCC_CTX is reset between gate calls (anchor_pool changes
# each round).
_NCC_CTX = None


def _ncc_one_pair(pair):
    """Worker: compute LOO-δ for one pair using the forked-in _NCC_CTX."""
    from autocoreg.finetune_soma_print.loo_image_ncc import loo_delta_ncc
    g = _NCC_CTX
    return loo_delta_ncc(
        pair, g["anchor_pool"],
        cz_zyx_native_um=g["cz_native"], hcr_zyx_um=g["hcr_zyx"],
        hcr488_vol=g["hcr488"], hcr488_origin=g["hcr488_origin"],
        hcr488_voxel=g["hcr488_voxel"], cz_vol=g["cz_vol"], cz_voxel=g["cz_voxel"],
        k_neighbours=NCC_K_NEIGHBOURS, skip_radius_um=NCC_SKIP_RADIUS_UM,
        patch_xy_um=NCC_PATCH_XY_UM, patch_z_um=NCC_PATCH_Z_UM,
        step_um=NCC_STEP_UM, mip_2d=NCC_MIP_2D,
        n_slabs=NCC_N_SLABS, pia_surf=g.get("pia_surf"),
    )


def ncc_gate(
    mb_pairs: list[tuple[int, int]],
    cz_zyx_native_um: np.ndarray,
    hcr_zyx: np.ndarray,
    pathb_ctx: dict,
    delta_threshold: float = NCC_DELTA_THRESHOLD,
    n_workers: int | None = None,
) -> tuple[list[tuple[int, int]], list]:
    """LOO image-NCC δ gate.  Physical-radius anchor skip (NCC_SKIP_RADIUS_UM).

    Parallelized over candidate pairs (each ``loo_delta_ncc`` is independent
    given the shared read-only volumes). ``n_workers`` defaults to NCC_WORKERS;
    set to 1 for the sequential path (identical results, used as fallback).
    """
    from autocoreg.finetune_soma_print.loo_image_ncc import loo_delta_ncc

    if len(mb_pairs) == 0:
        return [], []
    if n_workers is None:
        n_workers = NCC_WORKERS
    anchor_pool: set[tuple[int, int]] = set(mb_pairs)  # LOO anchors = current MB set
    t_start = time.time()
    print(f"    NCC LOO-δ: scoring {len(mb_pairs)} pairs ({n_workers} workers) ...",
          flush=True)

    deltas: list = [None] * len(mb_pairs)
    if n_workers > 1 and len(mb_pairs) > 1:
        global _NCC_CTX
        _NCC_CTX = dict(
            anchor_pool=anchor_pool, cz_native=cz_zyx_native_um, hcr_zyx=hcr_zyx,
            hcr488=pathb_ctx["hcr488"], hcr488_origin=pathb_ctx["hcr488_origin"],
            hcr488_voxel=pathb_ctx["hcr488_voxel"], cz_vol=pathb_ctx["cz_vol"],
            cz_voxel=pathb_ctx["cz_voxel"],
            pia_surf=pathb_ctx.get("pia_surf"),  # None-safe; zeroes above-pia voxels
        )
        try:
            ctx = mp.get_context("fork")
            with ctx.Pool(processes=n_workers) as pool:
                for k_, d in enumerate(pool.imap(_ncc_one_pair, mb_pairs, chunksize=4)):
                    deltas[k_] = d
                    if (k_ + 1) % 100 == 0:
                        el = time.time() - t_start
                        rate = (k_ + 1) / max(el, 1e-6)
                        eta = (len(mb_pairs) - k_ - 1) / max(rate, 1e-6)
                        print(f"      {k_+1}/{len(mb_pairs)} ({rate:.1f}/s  eta {eta:.0f}s)",
                              flush=True)
        finally:
            _NCC_CTX = None
    else:
        for k_, (ii, jj) in enumerate(mb_pairs):
            deltas[k_] = loo_delta_ncc(
                (ii, jj), anchor_pool,
                cz_zyx_native_um=cz_zyx_native_um, hcr_zyx_um=hcr_zyx,
                hcr488_vol=pathb_ctx["hcr488"], hcr488_origin=pathb_ctx["hcr488_origin"],
                hcr488_voxel=pathb_ctx["hcr488_voxel"], cz_vol=pathb_ctx["cz_vol"],
                cz_voxel=pathb_ctx["cz_voxel"], k_neighbours=NCC_K_NEIGHBOURS,
                skip_radius_um=NCC_SKIP_RADIUS_UM, patch_xy_um=NCC_PATCH_XY_UM,
                patch_z_um=NCC_PATCH_Z_UM, step_um=NCC_STEP_UM, mip_2d=NCC_MIP_2D,
                n_slabs=NCC_N_SLABS, pia_surf=pathb_ctx.get("pia_surf"),
            )

    kept = [(ii, jj) for d, (ii, jj) in zip(deltas, mb_pairs)
            if d is None or d >= delta_threshold]
    fin = [d for d in deltas if d is not None]
    med_d = float(np.nanmedian(fin)) if fin else 0.0
    print(f"    NCC LOO-δ: kept {len(kept)}/{len(mb_pairs)}  "
          f"median δ={med_d:.3f}  [{time.time()-t_start:.0f}s]", flush=True)
    return kept, deltas


# ── Per-subject runner ─────────────────────────────────────────────────────────

def _build_pathb_ctx(inp, cz_pool_ids: np.ndarray) -> dict:
    """Load HCR 488, CZ z-stack, and pia surface for the NCC gate."""
    from autocoreg.io.hcr_image import load_hcr_volume
    from autocoreg.io.cz_volume import load_cz_volume
    from autocoreg.initial_registration.surfaces import get_hcr_top_surface_iter07
    t0 = time.time()
    print("  loading HCR 488 (level 3) for NCC gate ...", flush=True)
    vol, xy_um, z_um = load_hcr_volume(inp.s, channel="488", level=3)
    print(f"  HCR 488 {vol.shape}  xy={xy_um:.2f} z={z_um:.2f} µm  "
          f"[{time.time()-t0:.1f}s]", flush=True)
    t1 = time.time()
    print("  loading CZ z-stack ...", flush=True)
    cz_vol = load_cz_volume(inp.s).astype(np.float32, copy=False)
    cz_xy = float(inp.s.cz_xy_um)
    cz_z = float(inp.s.cz_z_um)
    print(f"  CZ z-stack {cz_vol.shape}  vox z={cz_z:.2f} xy={cz_xy:.2f} µm  "
          f"[{time.time()-t1:.1f}s]", flush=True)
    # CZ native µm coordinates indexed by pool row (not cell id).
    cz_native_um = np.array([inp.cz_um[r] for r, _ in enumerate(inp.cz_ids)
                              if True], dtype=float)
    # Build mapping from pool id back to native µm per pool row.
    cz_id_to_native = {int(c): inp.cz_um[r] for r, c in enumerate(inp.cz_ids)}
    cz_native_um_arr = np.array(
        [cz_id_to_native[int(c)] for c in cz_pool_ids], dtype=float
    )
    # Pia surface: used to zero above-pia voxels in each slab before MIP.
    # Keys: a, b, c, (p, q, r optional quadratic).  None if unavailable.
    pia_surf = get_hcr_top_surface_iter07(inp.s)
    if pia_surf is None:
        print(f"  WARNING: pia surface not available for {inp.s.subject_id}"
              f" — NCC gate will run without pia masking", flush=True)
    else:
        print(f"  Pia surface loaded (keys: {list(pia_surf.keys())})", flush=True)
    return dict(
        hcr488=vol,
        hcr488_origin=(0.0, 0.0, 0.0),
        hcr488_voxel=(float(z_um), float(xy_um), float(xy_um)),
        cz_vol=cz_vol,
        cz_voxel=(cz_z, cz_xy, cz_xy),
        cz_native_um=cz_native_um_arr,
        pia_surf=pia_surf,
    )


def run_subject(
    sid: str,
    sz_pins: dict,
    gate: str,
    use_local_flow_rd0: bool,
    out_dir: Path,
    anchor_vote_frac: float = ANCHOR_VOTE_FRAC,
    anchor_restricted: bool = False,
) -> dict:
    label = gate + ("" if use_local_flow_rd0 else "_noLF") + ("+anchor_restricted" if anchor_restricted else "")
    print(f"\n=== {sid} [gate={gate}  local_flow_rd0={use_local_flow_rd0}] ===",
          flush=True)
    t_subj = time.time()

    subj = prepare_subject(sid, sz_pins=sz_pins)
    inp = subj["inp"]
    # cz_pool_zyx: CZ centroids in the locked (LP/HCR) µm frame.
    # This is the frame in which soma_score_radius operates.
    cz_zyx_lp: np.ndarray = subj["cz_pool_zyx"]
    hcr_zyx: np.ndarray = subj["hcr_pool_zyx"]
    cz_pool_ids: np.ndarray = subj["cz_pool_ids"]
    hcr_pool_ids: np.ndarray = subj["hcr_pool_ids"]
    coreg_lookup = {
        int(c): int(h)
        for c, h in zip(inp.coreg["cz_id"].astype(int),
                        inp.coreg["hcr_id"].astype(int))
    }
    print(f"  CZ pool {len(cz_zyx_lp)}  HCR pool {len(hcr_zyx)}  "
          f"R_cand={R_CAND_UM:.0f}µm  (adaptive R was {subj['R_cand_um']:.0f}µm)")

    sid_out = out_dir / sid
    sid_out.mkdir(parents=True, exist_ok=True)

    # Locked-prior-frame soma descriptors, computed once.  The per-pair score written
    # to every matches CSV (`soma_score`) is this LP-frame value (radius-free
    # score_pair) — a stable per-pair confidence (lower = better) that the QC app reads
    # directly.  NOT the per-round warped D, which collapses to ~0 as the TPS converges.
    _cz_vecs_lp = cell_vectors(cz_zyx_lp, m=SOMA_M_CZ)
    _hcr_vecs_all = cell_vectors(hcr_zyx, m=SOMA_M_HCR)

    def _lp_soma(i: int, j: int) -> float:
        return float(score_pair(_cz_vecs_lp[i], _hcr_vecs_all[j], n=SOMA_N))

    # NCC gate needs image volumes — load once before the round loop.
    pathb_ctx = None
    if gate == "ncc":
        pathb_ctx = _build_pathb_ctx(inp, cz_pool_ids)
        # cz_native_um inside ctx is used by loo_delta_ncc for inverse-TPS dst
        # (CZ native µm, not LP µm).  Already built by _build_pathb_ctx.

    # ── Round loop ─────────────────────────────────────────────────────────────
    cz_cur = cz_zyx_lp.copy()  # starts as LP frame; updated by TPS each round
    accepted: set[tuple[int, int]] = set()
    tps = None

    rounds_log: list[dict] = []
    # Trajectory accumulator: pair → list of round-presence booleans
    pair_history: dict[tuple[int, int], list[bool]] = {}
    # Track first and latest partner seen for partner-switch detection
    pair_partner: dict[int, int] = {}  # cz_row → last-seen hcr_row

    # Log 1: local_flow round-0 drops
    lf_dropped_rows: list[dict] = []
    # Log 2: MB rejection of GT pairs (rounds ≥ 1)
    mb_rejection_rows: list[dict] = []

    for rd in range(MAX_ROUNDS):
        t_rd = time.time()

        # Re-centre CZ on the improving TPS warp before scoring.
        if tps is not None:
            cz_cur = apply_tps(tps, cz_zyx_lp)

        # ── Descriptor scoring (fixed R_CAND_UM every round) ─────────────────
        D = soma_score_radius(
            cz_cur, hcr_zyx, R_CAND_UM,
            SOMA_M_CZ, SOMA_M_HCR, SOMA_N,
        )

        # ── Mutual-best ───────────────────────────────────────────────────────
        mb_set, cz_best, hcr_best = mutual_best_pairs_from_D(D)
        base_pairs = list(mb_set)

        # Log MB rejection of GT pairs for round ≥ 1.
        if rd >= 1:
            # per-CZ argmin (one-sided best) minus mutual-best: pairs MB rejected
            mb_rejected_gt = 0
            for i in range(D.shape[0]):
                j = cz_best[i]
                if j < 0:
                    continue
                # pair i→j was CZ-best but not mutual-best
                if (i, j) not in mb_set:
                    cz_id = int(cz_pool_ids[i])
                    if coreg_lookup.get(cz_id) == int(hcr_pool_ids[j]):
                        mb_rejected_gt += 1
            mb_rejection_rows.append(dict(sid=sid, round=rd,
                                          n_mb_rejected_gt=mb_rejected_gt))

        # ── Round-0 local_flow pre-filter (before primary gate) ───────────────
        n_before_lf = len(base_pairs)
        if rd == 0 and use_local_flow_rd0:
            kept_lf, _ = local_flow_filter(base_pairs, cz_cur, hcr_zyx)
            dropped_set = set(base_pairs) - set(kept_lf)
            for (i, j) in dropped_set:
                cz_id = int(cz_pool_ids[i])
                hcr_id = int(hcr_pool_ids[j])
                lf_dropped_rows.append(dict(
                    sid=sid, cz_id=cz_id, hcr_id=hcr_id,
                    is_gt=int(coreg_lookup.get(cz_id) == hcr_id),
                ))
            base_pairs = kept_lf
        n_filtered_lf = n_before_lf - len(base_pairs)

        # ── Primary gate ──────────────────────────────────────────────────────
        n_before_gate = len(base_pairs)
        if gate == "likelihood_ratio":
            kept, _ = likelihood_ratio_filter(base_pairs, D)
        elif gate == "anchor_vote":
            kept, _ = anchor_vote_gate(base_pairs, cz_cur, hcr_zyx,
                                       frac_threshold=anchor_vote_frac)
        elif gate == "ncc":
            assert pathb_ctx is not None
            kept, _ = ncc_gate(base_pairs, pathb_ctx["cz_native_um"],
                               hcr_zyx, pathb_ctx)
        else:
            raise ValueError(f"unknown gate: {gate!r}")
        n_filtered_gate = n_before_gate - len(kept)

        # ── Anchor-support gate (rounds ≥1): reject candidates dragged outside
        #    the previous round's trustworthy (interpolated) support region.
        #    `accepted` here still holds the PREVIOUS round's set (reassigned
        #    below); round 0 has an empty ref set so nothing is dropped. ───────
        n_before_support = len(kept)
        kept = support_gate(kept, cz_zyx_lp, accepted, SUPPORT_HULL_MARGIN_UM)
        n_filtered_support = n_before_support - len(kept)

        # ── Re-evaluate accepted set (no locking) ─────────────────────────────
        prev_accepted = accepted
        accepted = set(kept)
        n_added = len(accepted - prev_accepted)
        n_removed = len(prev_accepted - accepted)
        rel_delta = (n_added + n_removed) / max(len(prev_accepted), 1)

        # n_gt here is in-pool GT count (per-round logging only).
        # The scoring denominator is scoring_gt(inp) — computed at the end of run_subject.
        n_gt = sum(
            1 for (i, j) in accepted
            if coreg_lookup.get(int(cz_pool_ids[i])) == int(hcr_pool_ids[j])
        )
        n_gt_removed = sum(
            1 for (i, j) in (prev_accepted - accepted)
            if coreg_lookup.get(int(cz_pool_ids[i])) == int(hcr_pool_ids[j])
        )

        # ── TPS refit ─────────────────────────────────────────────────────────
        if len(accepted) >= 4:
            src = np.array([cz_zyx_lp[i] for (i, _) in accepted])
            dst = np.array([hcr_zyx[j] for (_, j) in accepted])
            tps = fit_tps(src, dst)
        else:
            tps = None

        # ── Per-round CSV ──────────────────────────────────────────────────────
        rd_rows = []
        for (i, j) in accepted:
            cz_id = int(cz_pool_ids[i])
            hcr_id = int(hcr_pool_ids[j])
            rd_rows.append(dict(
                sid=sid, round=rd,
                cz_id=cz_id, hcr_id=hcr_id,
                cz_lp_z=float(cz_zyx_lp[i, 0]),
                cz_lp_y=float(cz_zyx_lp[i, 1]),
                cz_lp_x=float(cz_zyx_lp[i, 2]),
                cz_cur_z=float(cz_cur[i, 0]),
                cz_cur_y=float(cz_cur[i, 1]),
                cz_cur_x=float(cz_cur[i, 2]),
                hcr_z=float(hcr_zyx[j, 0]),
                hcr_y=float(hcr_zyx[j, 1]),
                hcr_x=float(hcr_zyx[j, 2]),
                soma_score=_lp_soma(i, j),
                is_gt=int(coreg_lookup.get(cz_id) == hcr_id),
            ))
        pd.DataFrame(rd_rows).to_csv(sid_out / f"matches_round{rd}.csv", index=False)

        # ── Per-pair trajectory tracking ───────────────────────────────────────
        all_seen = set(pair_history.keys()) | accepted | prev_accepted
        for pair in all_seen:
            if pair not in pair_history:
                pair_history[pair] = [False] * rd  # absent in earlier rounds
            # Pad to current length if needed
            while len(pair_history[pair]) < rd:
                pair_history[pair].append(False)
            pair_history[pair].append(pair in accepted)
        # Track partner switches (cz_row → hcr partner)
        for (i, j) in accepted:
            prev_j = pair_partner.get(i, -1)
            if prev_j >= 0 and prev_j != j:
                # partner switch detected — logged in pair_trajectory later
                pass
            pair_partner[i] = j

        rounds_log.append(dict(
            round=rd,
            n_mb=len(mb_set),
            n_before_gate=n_before_gate,
            n_accepted=len(accepted),
            n_gt=n_gt,
            n_added=n_added,
            n_removed=n_removed,
            rel_delta=rel_delta,
            n_gt_removed=n_gt_removed,
            n_filtered_lf=n_filtered_lf,
            n_filtered_gate=n_filtered_gate,
            elapsed_s=round(time.time() - t_rd, 1),
        ))
        print(
            f"  rd{rd}: |MB|={len(mb_set)}  gate_kept={len(kept)}/{n_before_gate}"
            f"  |accepted|={len(accepted)} (+{n_added}/-{n_removed})"
            f"  GT={n_gt}  Δ_rel={rel_delta:.3f}"
            f"  dropped[lf={n_filtered_lf} gate={n_filtered_gate}]"
            f"  [{time.time()-t_rd:.1f}s]",
            flush=True,
        )

        if rd > 0 and rel_delta < CONVERGE_REL_DELTA:
            print(f"  converged after round {rd}")
            break
        if len(accepted) < MIN_LANDMARKS_FOR_ABORT:
            print(f"  ABORT: |accepted|={len(accepted)} < {MIN_LANDMARKS_FOR_ABORT}")
            break

    # ── Stage-2 anchor-restricted descriptor (--anchor_restricted only) ─────
    # Starts from Stage-1's converged accepted set + current TPS-warped cz_cur.
    # Uses the same anchor_vote gate (ANCHOR_VOTE_FRAC=3/5) and R_CAND_UM=150.
    # Stage-1 files are written to out_dir (already done above via per-round CSVs).
    # Stage-2 results are written to a separate _anchor_restricted output dir.
    ar_rounds_log: list[dict] = []
    if anchor_restricted:
        if len(accepted) < ANCHOR_RESTRICTED_N_ANCHORS:
            print(
                f"  Anchor-restricted Stage-2 SKIPPED: |accepted|={len(accepted)} < "
                f"{ANCHOR_RESTRICTED_N_ANCHORS} anchors needed; returning Stage-1 result.",
                flush=True,
            )
        else:
            print(
                f"\n  -- Anchor-restricted Stage-2 start: {len(accepted)} Stage-1 anchors --",
                flush=True,
            )
            # Stage-2 operates on the same coordinate arrays (cz_zyx_lp, hcr_zyx).
            # cz_cur is already TPS-warped to the Stage-1 final state.
            ar_accepted: set[tuple[int, int]] = set(accepted)
            ar_tps = tps  # carry Stage-1 TPS forward
            for w_rd in range(MAX_ROUNDS):
                t_wrd = time.time()
                # Re-warp from Stage-1 baseline LP frame using the current ar_tps.
                if ar_tps is not None:
                    cz_cur = apply_tps(ar_tps, cz_zyx_lp)

                # anchor-restricted descriptor, radius candidates.
                D_ar = anchor_restricted_score_radius(
                    cz_cur, hcr_zyx,
                    accepted=ar_accepted,
                    R_cand=R_CAND_UM,
                    n_anchors=ANCHOR_RESTRICTED_N_ANCHORS,
                )
                # Mutual-best on the anchor-restricted score matrix.
                w_mb_set, _, _ = mutual_best_pairs_from_D(D_ar)
                w_base = list(w_mb_set)

                # Same anchor_vote gate (within-round 3/5).
                w_kept, _ = anchor_vote_gate(
                    w_base, cz_cur, hcr_zyx,
                    frac_threshold=anchor_vote_frac,
                )
                # Anchor-support gate (same as Stage-1), keyed off the previous
                # Stage-2 accepted set's source hull.
                w_kept = support_gate(w_kept, cz_zyx_lp, ar_accepted,
                                      SUPPORT_HULL_MARGIN_UM)

                # Re-evaluate (no locking).
                w_prev = ar_accepted
                ar_accepted = set(w_kept)
                w_added = len(ar_accepted - w_prev)
                w_removed = len(w_prev - ar_accepted)
                w_rel_delta = (w_added + w_removed) / max(len(w_prev), 1)
                # in-pool GT count for per-round logging only (not the scoring denominator)
                w_n_gt = sum(
                    1 for (i, j) in ar_accepted
                    if coreg_lookup.get(int(cz_pool_ids[i])) == int(hcr_pool_ids[j])
                )

                # Refit TPS on Stage-2 accepted set.
                if len(ar_accepted) >= 4:
                    w_src = np.array([cz_zyx_lp[i] for (i, _) in ar_accepted])
                    w_dst = np.array([hcr_zyx[j] for (_, j) in ar_accepted])
                    ar_tps = fit_tps(w_src, w_dst)
                else:
                    ar_tps = None

                # Per-round CSV for Stage-2 (named matches_anchor_restricted_round{w_rd}.csv).
                w_rd_rows = []
                for (i, j) in ar_accepted:
                    cz_id = int(cz_pool_ids[i])
                    hcr_id = int(hcr_pool_ids[j])
                    w_rd_rows.append(dict(
                        sid=sid, round=w_rd,
                        cz_id=cz_id, hcr_id=hcr_id,
                        soma_score=_lp_soma(i, j),
                        is_gt=int(coreg_lookup.get(cz_id) == hcr_id),
                    ))
                pd.DataFrame(w_rd_rows).to_csv(
                    sid_out / f"matches_anchor_restricted_round{w_rd}.csv", index=False
                )

                ar_rounds_log.append(dict(
                    stage="anchor_restricted", round=w_rd,
                    n_mb=len(w_mb_set),
                    n_before_gate=len(w_base),
                    n_accepted=len(ar_accepted),
                    n_gt=w_n_gt,
                    n_added=w_added,
                    n_removed=w_removed,
                    rel_delta=w_rel_delta,
                    elapsed_s=round(time.time() - t_wrd, 1),
                ))
                print(
                    f"  anchor_restricted rd{w_rd}: |MB|={len(w_mb_set)}"
                    f"  gate_kept={len(w_kept)}/{len(w_base)}"
                    f"  |accepted|={len(ar_accepted)} (+{w_added}/-{w_removed})"
                    f"  GT={w_n_gt}  Δ_rel={w_rel_delta:.3f}"
                    f"  [{time.time()-t_wrd:.1f}s]",
                    flush=True,
                )
                if w_rd > 0 and w_rel_delta < CONVERGE_REL_DELTA:
                    print(f"  Anchor-restricted Stage-2 converged after round {w_rd}")
                    break
                if len(ar_accepted) < MIN_LANDMARKS_FOR_ABORT:
                    print(
                        f"  Anchor-restricted Stage-2 ABORT: |accepted|={len(ar_accepted)}"
                        f" < {MIN_LANDMARKS_FOR_ABORT}"
                    )
                    break

            # Promote Anchor-restricted Stage-2 result: overwrite accepted, cz_cur, tps for
            # final output and return value.
            accepted = ar_accepted
            tps = ar_tps

    # ── Persist TPS ───────────────────────────────────────────────────────────
    if tps is not None:
        with open(sid_out / "tps_warp_final.pkl", "wb") as f:
            pickle.dump(tps, f)

    # ── Persist rounds log ─────────────────────────────────────────────────────
    pd.DataFrame(rounds_log).to_csv(sid_out / "rounds_log.csv", index=False)
    if ar_rounds_log:
        pd.DataFrame(ar_rounds_log).to_csv(
            sid_out / "anchor_restricted_rounds_log.csv", index=False
        )

    # ── Persist local_flow round-0 drop log ───────────────────────────────────
    pd.DataFrame(lf_dropped_rows).to_csv(
        sid_out / "local_flow_rd0_dropped.csv", index=False
    )

    # ── Persist MB rejection log ───────────────────────────────────────────────
    pd.DataFrame(mb_rejection_rows).to_csv(
        sid_out / "mb_rejection_log.csv", index=False
    )

    # ── Persist pair trajectory log ────────────────────────────────────────────
    n_rounds_total = len(rounds_log)
    traj_rows = []
    # Track partner switches by scanning the per-round CSVs we just wrote.
    # Easier: reconstruct from pair_history and pair_partner tracking.
    # partner_switches: for cz_row i, count distinct hcr partners across rounds.
    partner_history: dict[int, list[int]] = {}  # cz_row → list of hcr_row per round
    for rd_idx in range(n_rounds_total):
        rd_csv = sid_out / f"matches_round{rd_idx}.csv"
        if not rd_csv.exists():
            continue
        # Guard: pd.DataFrame([]).to_csv writes a bare newline (no header);
        # pd.read_csv raises EmptyDataError on that.  Skip empty files.
        if rd_csv.stat().st_size < 2:
            continue
        df_rd = pd.read_csv(rd_csv)
        for _, row in df_rd.iterrows():
            cz_id = int(row["cz_id"]); hcr_id = int(row["hcr_id"])
            # Find pool rows
            cz_rows = np.where(cz_pool_ids == cz_id)[0]
            hcr_rows = np.where(hcr_pool_ids == hcr_id)[0]
            if cz_rows.size == 0 or hcr_rows.size == 0:
                continue
            ci = int(cz_rows[0]); cj = int(hcr_rows[0])
            if ci not in partner_history:
                partner_history[ci] = []
            partner_history[ci].append(cj)

    for (i, j), rounds_present in pair_history.items():
        # Pad to full length
        while len(rounds_present) < n_rounds_total:
            rounds_present.append(False)
        bitmask = "".join("1" if v else "0" for v in rounds_present)
        # Flips = transitions in the bitmask
        n_flips = sum(1 for a, b in zip(bitmask, bitmask[1:]) if a != b)
        # Partner switches for this cz row
        phist = partner_history.get(i, [])
        n_partner_switches = len(set(phist)) - 1 if phist else 0
        final_status = "kept" if (i, j) in accepted else "dropped"
        cz_id = int(cz_pool_ids[i])
        hcr_id = int(hcr_pool_ids[j])
        is_gt = int(coreg_lookup.get(cz_id) == hcr_id)
        traj_rows.append(dict(
            cz_id=cz_id, hcr_id=hcr_id,
            rounds_present=bitmask,
            n_flips=n_flips,
            partner_switches=max(0, n_partner_switches),
            final_status=final_status,
            is_gt=is_gt,
        ))
    pd.DataFrame(traj_rows).to_csv(sid_out / "pair_trajectory.csv", index=False)

    # Use Stage-2 final state if the anchor-restricted addendum ran; Stage-1 otherwise.
    if ar_rounds_log:
        final_accepted_count = len(accepted)
        # in-pool GT count (for in-round logging only; NOT the scoring denominator)
        final_gt_inpool = sum(
            1 for (i, j) in accepted
            if coreg_lookup.get(int(cz_pool_ids[i])) == int(hcr_pool_ids[j])
        )
    else:
        last = rounds_log[-1] if rounds_log else {}
        final_accepted_count = last.get("n_accepted", 0)
        # in-pool GT count (in-round logging only; NOT the scoring denominator)
        final_gt_inpool = last.get("n_gt", 0)

    wall_s = round(time.time() - t_subj, 1)
    prec_inpool = final_gt_inpool / max(final_accepted_count, 1)

    # ── Honest recall/precision vs pose-independent GT ────────────────────────
    # scoring_gt returns {cz_id: hcr_id} for coreg pairs whose HCR cell is
    # GFP+∩ok (no spatial/pool filter). This is the correct scoring denominator.
    # final_gt_inpool / prec_inpool above are in-pool-only and kept for backwards
    # compatibility of the per-round logs; they are NOT the headline metrics.
    G_full = scoring_gt(inp)
    nG_full = len(G_full)
    # Build final match map from the accepted set.
    M_final: dict[int, int] = {
        int(cz_pool_ids[i]): int(hcr_pool_ids[j])
        for (i, j) in accepted
    }
    tp_full = sum(1 for c, h in M_final.items() if G_full.get(c) == h)
    recall_gfpok = tp_full / nG_full if nG_full else float("nan")
    prec_gfpok = tp_full / max(len(M_final), 1)

    print(
        f"  done [{wall_s:.1f}s]  "
        f"|M|={final_accepted_count}  "
        f"GT_inpool={final_gt_inpool}  prec_inpool={prec_inpool:.3f}  |  "
        f"|G_gfpok|={nG_full}  TP={tp_full}  "
        f"recall_gfpok={recall_gfpok:.3f}  prec_gfpok={prec_gfpok:.3f}",
        flush=True,
    )
    return dict(
        sid=sid, gate=label, local_flow_rd0=int(use_local_flow_rd0),
        n_rounds=len(rounds_log) + len(ar_rounds_log),
        final_matches=final_accepted_count,
        # in-pool GT fields (pose-dependent; kept for backwards compat)
        final_gt_inpool=final_gt_inpool,
        final_precision_inpool=round(prec_inpool, 4),
        # headline metrics vs pose-independent GT (GFP+∩ok, no spatial filter)
        gt_gfpok=nG_full,
        tp_gfpok=tp_full,
        recall_gfpok=round(recall_gfpok, 4) if not (recall_gfpok != recall_gfpok) else float("nan"),
        precision_gfpok=round(prec_gfpok, 4),
        wall_s=wall_s,
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    global NCC_WORKERS
    p = argparse.ArgumentParser(
        description="Step 3 v3 — decluttered geometric matcher"
    )
    p.add_argument("--gate", choices=["likelihood_ratio", "anchor_vote", "ncc"], required=True,
                   help="Primary filter gate: likelihood_ratio | anchor_vote | ncc")
    p.add_argument("--local_flow_rd0", action="store_true", default=True,
                   help="Apply local_flow pre-filter at round 0 (default ON)")
    p.add_argument("--no_local_flow_rd0", dest="local_flow_rd0",
                   action="store_false",
                   help="Disable local_flow pre-filter at round 0")
    p.add_argument("--sids", nargs="*", default=None,
                   help="Subject ids to run (default: all 6 benchmark subjects)")
    p.add_argument("--anchor_vote_frac", type=float, default=ANCHOR_VOTE_FRAC,
                   help=f"anchor_vote gate: keep fraction threshold "
                        f"(default {ANCHOR_VOTE_FRAC})")
    p.add_argument("--ncc_workers", type=int, default=NCC_WORKERS,
                   help=f"NCC gate: parallel worker processes over candidate "
                        f"pairs (default {NCC_WORKERS}; 1 = sequential). Lower "
                        f"this if running several subjects concurrently.")
    p.add_argument("--anchor_restricted", action="store_true", default=False,
                   help="After Stage-1 converges, run a Stage-2 anchor-restricted "
                        "descriptor loop with the same gate.  Outputs go to "
                        "outputs/<gate>[_noLF]_anchor_restricted/. "
                        "Production defaults UNCHANGED; this flag is opt-in only.")
    args = p.parse_args()
    NCC_WORKERS = max(1, int(args.ncc_workers))

    sz_pins = load_sz_pins()
    sids = args.sids if args.sids else list(BENCHMARK_SUBJECTS)

    lf_suffix = "" if args.local_flow_rd0 else "_noLF"
    ar_suffix = "_anchor_restricted" if args.anchor_restricted else ""
    out_dir = OUT_BASE / f"{args.gate}{lf_suffix}{ar_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for sid in sids:
        try:
            row = run_subject(
                sid, sz_pins, gate=args.gate,
                use_local_flow_rd0=args.local_flow_rd0,
                out_dir=out_dir,
                anchor_vote_frac=args.anchor_vote_frac,
                anchor_restricted=args.anchor_restricted,
            )
            summary.append(row)
        except Exception as exc:
            import traceback
            print(f"[{sid}] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            # Write a 0/0 summary row so the subject appears in the final table
            # with a clear failure flag rather than being silently absent.
            summary.append(dict(
                sid=sid, gate=args.gate,
                local_flow_rd0=int(args.local_flow_rd0),
                n_rounds=0, final_matches=0,
                final_gt_inpool=0, final_precision_inpool=0.0,
                gt_gfpok=0, tp_gfpok=0,
                recall_gfpok=float("nan"), precision_gfpok=0.0,
                wall_s=0.0, error=str(exc),
            ))

    df = pd.DataFrame(summary)
    df.to_csv(out_dir / "summary.csv", index=False)
    print("\n=== Summary ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
