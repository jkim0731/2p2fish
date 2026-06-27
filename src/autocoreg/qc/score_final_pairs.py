"""Soma re-score every final matched pair (G2).

The production final round is the anchor-restricted Stage-2 CSV, which carries only
``sid,round,cz_id,hcr_id,is_gt`` — **no per-pair soma score**.  The QC app needs a
per-pair confidence to surface the *least-confident* pairs first (highest soma
distance).  This module recomputes, for every ``(cz_id, hcr_id)`` in a final matches
CSV, the matcher's own soma-print score — reusing the matcher's ``soma_score_radius``
(same ``SOMA_M_CZ/M_HCR/N`` descriptor).

**Frame = locked-prior (round-0), NOT TPS-warped.**  The matcher's own per-round score
collapses to ~0 once the CZ pool is TPS-warped onto HCR (the warp is fit on the very
pairs being scored — leakage; matcher round1/2 soma medians are 0.000).  The round-0 /
LP-frame score is non-circular and discriminative (790322: median 6.25, p10-p90
4.5-9.3 µm), so it is the right QC confidence: pairs that only become consistent after
heavy warping score high here and surface as least-confident.  This reproduces the
matcher's ``matches_round0.csv`` ``soma_score`` exactly (round 0 has ``tps is None`` →
``cz_cur == cz_lp``).

Soma score is a **distance — lower = better match**; the QC queue reviews the
highest-distance (least-confident) pairs first.

Output ``final_pairs.csv`` columns:
    cz_id, hcr_id, soma_score, in_pool, soma_rank_desc, soma_pct
where ``soma_rank_desc=1`` is the least-confident (largest soma distance) pair.

CLI:
    python -m autocoreg.qc.score_final_pairs <sid> <matches_csv> <out_csv>
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from autocoreg.finetune_soma_print.pool_prep import prepare_subject
from autocoreg.finetune_soma_print.matcher import soma_score_radius, SOMA_M_CZ, SOMA_M_HCR, SOMA_N, R_CAND_UM


def _resolve_sz_pins(sid: str, sz_pins: dict | None) -> dict:
    if sz_pins is not None:
        return sz_pins
    # GT-free: image-based sz estimator (cached).
    from autocoreg.io.subjects import load_subject
    from autocoreg.initial_registration.axial_scale import get_sz
    return {sid: float(get_sz(load_subject(sid))["sz_best"])}


def score_final_pairs(
    sid: str,
    matches_csv,
    *,
    sz_pins: dict | None = None,
    out_csv=None,
) -> pd.DataFrame:
    """Return a DataFrame of per-pair soma scores for ``matches_csv``.

    Reuses the matcher's pool construction (``prepare_subject``) and descriptor scoring
    (``soma_score_radius``) in the locked-prior frame.  Pairs whose cells are not in the
    matcher pool get ``soma_score=NaN`` / ``in_pool=0`` (should be ~none when the matches
    CSV was produced with the same ROI-quality asset that ``prepare_subject`` resolves).
    """
    matches_csv = Path(matches_csv)
    df = pd.read_csv(matches_csv)[["cz_id", "hcr_id"]].astype(int).drop_duplicates()

    sz_pins = _resolve_sz_pins(sid, sz_pins)
    subj = prepare_subject(sid, sz_pins=sz_pins)
    cz_lp = np.asarray(subj["cz_pool_zyx"], dtype=float)   # CZ pool in locked-prior µm
    hcr = np.asarray(subj["hcr_pool_zyx"], dtype=float)    # HCR pool µm
    cz_row = {int(c): r for r, c in enumerate(subj["cz_pool_ids"])}
    hcr_row = {int(h): r for r, h in enumerate(subj["hcr_pool_ids"])}

    pairs = [(int(c), int(h), cz_row.get(int(c)), hcr_row.get(int(h)))
             for c, h in zip(df["cz_id"], df["hcr_id"])]
    n_out = sum(1 for (_, _, i, j) in pairs if i is None or j is None)
    if n_out:
        print(f"[score_final_pairs] {sid}: {n_out}/{len(pairs)} final pairs not in "
              f"matcher pool -> soma_score=NaN (likely a matches/ROI-asset mismatch)")

    # Matcher's own descriptor scoring (same params) over the full pools, in the
    # locked-prior frame (no TPS — avoids leakage; see module docstring).
    D = soma_score_radius(cz_lp, hcr, R_CAND_UM, SOMA_M_CZ, SOMA_M_HCR, SOMA_N)

    out = []
    for (c, h, i, j) in pairs:
        s = float(D[i, j]) if (i is not None and j is not None) else float("nan")
        out.append(dict(cz_id=c, hcr_id=h, soma_score=s,
                        in_pool=int(i is not None and j is not None)))
    res = pd.DataFrame(out)

    # Rank least-confident first = largest soma distance first. NaN (not-in-pool /
    # unscored) keeps NaN rank and sorts last — NOT treated as most-uncertain.
    s = res["soma_score"]
    res["soma_rank_desc"] = s.rank(ascending=False, method="min").astype("Int64")  # 1 = least confident
    res["soma_pct"] = s.rank(pct=True)  # high pct = high distance = least confident
    res = res.sort_values(["soma_score"], ascending=False,
                          na_position="last").reset_index(drop=True)

    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(out_csv, index=False)
        print(f"[score_final_pairs] {sid}: wrote {len(res)} pairs -> {out_csv}")
    return res


def main(argv=None):
    import sys
    a = argv if argv is not None else sys.argv[1:]
    if len(a) < 2:
        print("usage: python -m autocoreg.qc.score_final_pairs <sid> <matches_csv> [out_csv]")
        sys.exit(1)
    sid, matches_csv = a[0], a[1]
    out_csv = a[2] if len(a) > 2 else None
    score_final_pairs(sid, matches_csv, out_csv=out_csv)


if __name__ == "__main__":
    main()
