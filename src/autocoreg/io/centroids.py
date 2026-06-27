"""Shared helpers for getting CZ / HCR-GFP+ centroids in µm.

Handles the fact that ``hcr_gfp_df`` contains only hcr_id + spot/density
metadata, not voxel coordinates — coordinates live in ``hcr_centroids``.

Also provides ``apply_aniso_fit`` which maps points through a
``ProcrustesFit`` (R, scales, translation).  The canonical formula from
``fit_anisotropic_similarity`` is ``dst = (src * scales) @ R.T + translation``;
candidates should call this instead of re-deriving it.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def apply_aniso_fit(pts: np.ndarray, fit) -> np.ndarray:
    """Apply a ``ProcrustesFit`` to ``pts`` (N, 3)."""
    R = np.asarray(fit.R); S = np.asarray(fit.scales); t = np.asarray(fit.translation)
    return (pts * S) @ R.T + t


def default_warmstart_zyx(cz_um_zyx: np.ndarray, hcr_um_zyx: np.ndarray,
                          *, use_icp: bool = True,
                          multistart: bool = True) -> tuple[np.ndarray, dict]:
    """Estimate a data-driven coarse warm-start in ``(z, y, x)`` convention.

    Uses the structural 180°-about-Z prior, the CZ-in-HCR-XY-centre geometric
    prior, and a data-driven scale estimate — either extent-ratio (``use_icp
    =False``) or the session-07 reciprocal-NN anisotropic ICP
    (``use_icp=True``, default).  Scales are recovered from the *subject's
    own* data, not from benchmark aggregate statistics, so the dev-rule ("no
    benchmark-priors as algorithm inputs") is preserved.

    When ``multistart=True`` (default), ICP is run from N seed translations
    (HCR-gfp centroid, gfp ± 100/200 µm along Z, Q25/Q75 of gfp).  The
    converged warps are ranked by ``reciprocal-NN count × unique-HCR-target
    fraction`` (a self-supervised signal that rewards one-to-one geometric
    consistency) and the winning seed's warp is fed into the translation
    refine + local refit pipeline.

    Returns ``(cz_init_zyx, info)`` where ``cz_init_zyx`` is ``(N, 3)`` warped
    CZ centroids in HCR µm.
    """
    from dataclasses import dataclass as _dc

    cz_um_zyx = np.asarray(cz_um_zyx, float)
    hcr_um_zyx = np.asarray(hcr_um_zyx, float)

    cz_c = cz_um_zyx.mean(0)
    hcr_c = hcr_um_zyx.mean(0)
    R_zyx = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)  # 180° about Z

    info: dict = dict(cz_mean_zyx=cz_c.tolist(), hcr_mean_zyx=hcr_c.tolist())

    if use_icp:
        try:
            # anisotropic_icp is not vendored; falls back to extent-ratio path below.
            from .anisotropic_icp import estimate_scales_icp_multi_start  # type: ignore  # noqa: F401

            cz_xyz = cz_um_zyx[:, [2, 1, 0]]
            hcr_xyz = hcr_um_zyx[:, [2, 1, 0]]
            R_xyz = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], float)

            @_dc
            class _Fit:
                R: np.ndarray
                src_mean: np.ndarray
                translation: np.ndarray
                scales: np.ndarray

            def _run_icp_from(seed_t_xyz):
                f0 = _Fit(R=R_xyz, src_mean=cz_xyz.mean(0),
                          translation=np.asarray(seed_t_xyz, float),
                          scales=np.ones(3))
                try:
                    r = estimate_scales_icp_multi_start(cz_xyz, hcr_xyz, f0)
                    if r.fit is None:
                        return None
                    return r
                except Exception:
                    return None

            def _score_recip_unique(pred_xyz):
                from scipy.spatial import cKDTree as _ck
                th = _ck(hcr_xyz); tc = _ck(pred_xyz)
                d_c2h, idx_c2h = th.query(pred_xyz, k=1)
                _, idx_h2c = tc.query(hcr_xyz, k=1)
                recip = int(((idx_h2c[idx_c2h] == np.arange(len(pred_xyz))) & (d_c2h < 30)).sum())
                uniq_frac = len(np.unique(idx_c2h)) / len(pred_xyz)
                return recip * uniq_frac, recip, uniq_frac

            if multistart:
                gfp_c = hcr_xyz.mean(0)
                seed_list = {
                    "hcr_gfp":    gfp_c,
                    "gfp_dz+100": gfp_c + [0, 0, 100],
                    "gfp_dz-100": gfp_c + [0, 0, -100],
                    "gfp_dz+200": gfp_c + [0, 0, 200],
                    "gfp_q25":    np.quantile(hcr_xyz, 0.25, axis=0),
                    "gfp_q75":    np.quantile(hcr_xyz, 0.75, axis=0),
                }
                seed_scores = []
                for name, seed_t in seed_list.items():
                    r = _run_icp_from(seed_t)
                    if r is None:
                        continue
                    pred = (cz_xyz * r.fit.scales) @ r.fit.R.T + r.fit.translation
                    rank_score, recip, uniq = _score_recip_unique(pred)
                    seed_scores.append(dict(name=name, seed_xyz=seed_t.tolist() if hasattr(seed_t, "tolist") else list(seed_t),
                                            score=rank_score, recip=recip, uniq=uniq, res=r))
                if not seed_scores:
                    raise RuntimeError("multi-start ICP: all seeds failed")
                seed_scores.sort(key=lambda d: d["score"], reverse=True)
                best = seed_scores[0]
                res = best["res"]
                info["multistart_seeds"] = [
                    {k: v for k, v in s.items() if k != "res"} for s in seed_scores
                ]
                info["multistart_winner"] = best["name"]
            else:
                fit0 = _Fit(R=R_xyz, src_mean=cz_xyz.mean(0),
                            translation=hcr_xyz.mean(0), scales=np.ones(3))
                res = estimate_scales_icp_multi_start(cz_xyz, hcr_xyz, fit0)

            if res.sxy is not None and res.sz is not None and res.fit is not None:
                # Use the full Procrustes fit (R + anisotropic scales + t) from
                # ICP's last iteration — this captures tilt + translation, not
                # just scale.
                fit = res.fit
                pred_xyz = ((cz_xyz - cz_xyz.mean(0)) @ fit.R) * fit.scales + hcr_xyz.mean(0)
                # Actually the Procrustes fit is ``dst ≈ ((src-mean_src) @ R)
                # * scales + mean_dst`` — recompute via its stored means.
                # fit_anisotropic_similarity stores ``translation = dst_mean
                # - R @ (src_mean * scales)`` internally; compose explicitly.
                pred_xyz = ((cz_xyz * fit.scales) @ fit.R.T) + fit.translation
                # Translation search: the HCR-all centroid used as the ICP
                # initial translation can be offset from the CZ sub-ROI's true
                # target centre (up to ~150 um in Z on 788406), because the CZ
                # slab does not sample the whole cortex.  Grid-search a
                # coarse-to-fine offset that maximises the number of warped CZ
                # cells landing within 50/30 um of an HCR neighbour.
                from scipy.spatial import cKDTree as _cKDTree
                _tree = _cKDTree(hcr_xyz)
                def _score(delta, radius):
                    d, _ = _tree.query(pred_xyz + delta, k=1)
                    return int((d < radius).sum())
                best = np.zeros(3); best_sc = _score(best, 50.0)
                for dz in range(-300, 301, 30):
                    for dy in range(-200, 201, 50):
                        for dx in range(-200, 201, 50):
                            cand = np.array([dx, dy, dz], dtype=float)
                            sc = _score(cand, 50.0)
                            if sc > best_sc:
                                best_sc = sc; best = cand
                # fine
                fine_best = best.copy(); fine_sc = _score(fine_best, 30.0)
                for dz in range(-60, 61, 10):
                    for dy in range(-40, 41, 10):
                        for dx in range(-40, 41, 10):
                            cand = best + np.array([dx, dy, dz], dtype=float)
                            sc = _score(cand, 30.0)
                            if sc > fine_sc:
                                fine_sc = sc; fine_best = cand
                pred_xyz = pred_xyz + fine_best
                # Iterative local refit: each pass takes reciprocal-NN pairs
                # within the current warm-start's 50-percentile distance,
                # refits anisotropic similarity, and re-warps.  Stops when
                # the inlier@30um score no longer grows.
                adopted_scales = None
                try:
                    from autocoreg.io.hcr_image import fit_anisotropic_similarity as _fas
                    best_score = int((_tree.query(pred_xyz, k=1)[0] < 30.0).sum())
                    for _it in range(5):
                        d_cz2h, idx_cz2h = _tree.query(pred_xyz, k=1)
                        tree_src = cKDTree(pred_xyz)
                        d_h2cz, idx_h2cz = tree_src.query(hcr_xyz, k=1)
                        recip = idx_h2cz[idx_cz2h] == np.arange(len(pred_xyz))
                        # threshold at median of reciprocal distances
                        r_d = d_cz2h[recip]
                        if r_d.size < 8:
                            break
                        thr = max(float(np.quantile(r_d, 0.7)), 20.0)
                        sel = recip & (d_cz2h < thr)
                        if sel.sum() < 8:
                            break
                        src_s = cz_xyz[sel]; dst_s = hcr_xyz[idx_cz2h[sel]]
                        refit = _fas(src_s, dst_s)
                        pred_new = ((cz_xyz - src_s.mean(0)) @ refit.R) * refit.scales + dst_s.mean(0)
                        sc_new = int((_tree.query(pred_new, k=1)[0] < 30.0).sum())
                        if sc_new <= best_score:
                            break
                        pred_xyz = pred_new
                        best_score = sc_new
                        adopted_scales = refit.scales
                        info.setdefault("local_refit_iters", [])
                        info["local_refit_iters"].append(dict(
                            n=int(sel.sum()), rms=float(refit.rms_um),
                            scales_xyz=refit.scales.tolist(),
                            inlier30=sc_new))
                except Exception as _exc:
                    info["local_refit_error"] = f"{type(_exc).__name__}: {_exc}"
                cz_init = pred_xyz[:, [2, 1, 0]]
                if adopted_scales is not None:
                    scales_zyx = [float(adopted_scales[2]), float(adopted_scales[1]), float(adopted_scales[0])]
                else:
                    scales_zyx = [float(fit.scales[2]), float(fit.scales[1]), float(fit.scales[0])]
                info.update(icp_sxy=float(res.sxy), icp_sz=float(res.sz),
                            icp_matched=int(res.n_matched),
                            icp_iters=int(res.iterations),
                            icp_fit_scales_xyz=fit.scales.tolist(),
                            scales_zyx=scales_zyx,
                            icp_fit_rms_um=float(fit.rms_um),
                            translation_refine_xyz=fine_best.tolist(),
                            translation_refine_score=int(fine_sc),
                            source="icp_fit")
                return cz_init, info
        except Exception as exc:
            info["icp_error"] = f"{type(exc).__name__}: {exc}"

    # Extent fallback.
    lo_cz = np.quantile(cz_um_zyx, 0.05, axis=0)
    hi_cz = np.quantile(cz_um_zyx, 0.95, axis=0)
    lo_hc = np.quantile(hcr_um_zyx, 0.05, axis=0)
    hi_hc = np.quantile(hcr_um_zyx, 0.95, axis=0)
    cz_ext = np.maximum(hi_cz - lo_cz, 1e-3)
    hc_ext = np.maximum(hi_hc - lo_hc, 1e-3)
    scales_zyx = hc_ext / cz_ext
    info.update(source="extent", scales_zyx=scales_zyx.tolist(),
                cz_extent_zyx=cz_ext.tolist(), hcr_extent_zyx=hc_ext.tolist())
    cz_init = ((cz_um_zyx - cz_c) * scales_zyx) @ R_zyx.T + hcr_c
    return cz_init, info


def centroids_um(s, modality: str = "cz") -> tuple[np.ndarray, np.ndarray]:
    """Return ``(points_um, ids)`` for the requested modality.

    ``modality`` ∈ {"cz", "hcr_gfp", "hcr_all"}.
    """
    from autocoreg.io.subjects import cz_px_to_um, hcr_px_to_um

    if modality == "cz":
        cz = s.cz_centroids
        pts = cz_px_to_um(cz[["z_px", "y_px", "x_px"]].values, s)
        return pts, cz["cz_id"].astype(int).values
    if modality == "hcr_all":
        hc = s.hcr_centroids
        pts = hcr_px_to_um(hc[["z_px", "y_px", "x_px"]].values, s)
        return pts, hc["hcr_id"].astype(int).values
    if modality == "hcr_gfp":
        # Join hcr_gfp_df with hcr_centroids on hcr_id to recover coordinates.
        df = s.hcr_gfp_df[["hcr_id"]].merge(
            s.hcr_centroids[["hcr_id", "z_px", "y_px", "x_px"]],
            on="hcr_id", how="inner",
        )
        pts = hcr_px_to_um(df[["z_px", "y_px", "x_px"]].values, s)
        return pts, df["hcr_id"].astype(int).values
    raise ValueError(f"unknown modality {modality}")
