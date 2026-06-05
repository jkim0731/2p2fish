"""Promoted iter08 CZ + HCR surfaces — main-pipeline entry points.

This module is the *public* home for the image-based surface fits that
03c session iter07/08 arrived at and validated:

* **CZ pia**: iter08 prior-selected range-relative detector
  (``TRANS_FRAC`` sweep, select closest to the experimenter's 50 µm
  target) + IRLS-Huber deg-2 bivariate polynomial, MAD-gated around
  the prior.
* **HCR pia (top)**: iter07 patch-MAX range-relative detector
  (``TRANS_FRAC=0.5``) + IRLS-Huber deg-2 poly.
* **HCR bottom**: iter08 symmetric z-flipped detector (same iter07
  detector on the reversed column), + IRLS-Huber deg-2 poly with
  ±3·MAD outlier gate.

Surfaces are returned in the canonical ``{a, b, c, p, q, r}`` dict
format used by :func:`benchmark_analysis.depth_from_surface`:

    z(x, y) = a·x + b·y + c + p·x² + q·x·y + r·y²

so every downstream consumer (``r1_coarse_align``, ``r1_revised``,
session 04/05/07 centroid scripts, …) works unchanged when handed the
promoted surface.

Results are cached as JSON under
``/root/capsule/code/dev_code/cached_surfaces/`` so downstream code
can load the fit in milliseconds instead of re-opening CZ volumes.

Public API
----------
* :func:`compute_cz_surface_iter08`
* :func:`compute_hcr_top_surface_iter07`
* :func:`compute_hcr_bottom_surface_iter08`
* :func:`get_cz_surface_iter08`  (cache-aware)
* :func:`get_hcr_top_surface_iter07`
* :func:`get_hcr_bottom_surface_iter08`
* :func:`save_cached_surface`, :func:`load_cached_surface`
* :func:`build_main_surface_store`  (batch build + summary CSV)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

# Vendored iter modules are now in-package.
from .iter07_compute import (
    EPS,
    HUBER_K,
    POLY_DEGREE,
    detect_transitions,
    fit_polysurf,
    sampling_grid,
)
from .iter08_cz_prior import (
    CZ_TARGET_Z_UM,
    CZ_THR_FLOOR,
    GATE_UM,
    TRANS_FRAC_BANK,
    _patch_log_columns,
    fit_gated_surface,
    load_cz_volume,
    select_trans_frac,
)
from .iter08_hcr_bottom import (
    MAD_GATE_K,
    detect_bottom_transitions,
    detect_top_transitions,
    mad_gate,
)

from . import config as _config
CACHE_DIR = _config.SURFACES_CACHE_DIR
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# polyfit (normalised-coord coef) → {a,b,c,p,q,r} (absolute µm)
# ------------------------------------------------------------
def _polyfit_to_abcpqr(fit: dict | None) -> dict | None:
    """Convert an ``iter07_compute.fit_polysurf`` result to the
    canonical ``{a,b,c,p,q,r}`` dict.

    ``fit_polysurf`` stores coefficients in normalised coordinates
    ``xn = (x - x0) / x_scale``, ``yn = (y - y0) / y_scale``, in the
    basis order ``[1, yn, xn, yn², xn·yn, xn²]`` (see ``_poly_basis``).
    This converter inverts that normalisation so the result lives
    directly in µm and plugs into :func:`depth_from_surface` unchanged.
    """
    if fit is None or fit.get("degree") != 2:
        return None
    c0, c1, c2, c3, c4, c5 = [float(v) for v in fit["coef"][:6]]
    x0 = float(fit["x0"]); y0 = float(fit["y0"])
    xs = float(fit["x_scale"]); ys = float(fit["y_scale"])
    p = c5 / (xs * xs)
    q = c4 / (xs * ys)
    r = c3 / (ys * ys)
    a = c2 / xs - c4 * y0 / (xs * ys) - 2.0 * c5 * x0 / (xs * xs)
    b = c1 / ys - 2.0 * c3 * y0 / (ys * ys) - c4 * x0 / (xs * ys)
    c = (c0
         - c1 * y0 / ys
         - c2 * x0 / xs
         + c3 * y0 * y0 / (ys * ys)
         + c4 * x0 * y0 / (xs * ys)
         + c5 * x0 * x0 / (xs * xs))
    return dict(a=a, b=b, c=c, p=p, q=q, r=r)


def _with_abc(fit: dict | None, **extra) -> dict | None:
    """Attach ``{a,b,c,p,q,r}`` keys to an iter08-style fit dict plus
    any provenance metadata."""
    if fit is None:
        return None
    abc = _polyfit_to_abcpqr(fit)
    if abc is None:
        return None
    out = dict(fit)  # preserve normalised-coef form
    out.update(abc)
    out.update(extra)
    return out


# ------------------------------------------------------------
# CZ pia — iter08
# ------------------------------------------------------------
def compute_cz_surface_iter08(
    s,
    target_z_um: float = CZ_TARGET_Z_UM,
    bank: tuple = TRANS_FRAC_BANK,
    thr_floor: float = CZ_THR_FLOOR,
    gate_um: float = GATE_UM,
) -> dict | None:
    """Run the iter08 CZ prior-selected detector + deg-2 IRLS-Huber fit
    for subject ``s``.  Returns ``None`` if the CZ volume is missing or
    the fit fails."""
    try:
        vol = load_cz_volume(s)
    except Exception:
        return None
    z_um = float(s.cz_z_um); xy_um = float(s.cz_xy_um)
    xi, yi = sampling_grid(vol.shape, xy_um)
    xs_um = xi * xy_um; ys_um = yi * xy_um
    log_vol = np.log(vol + EPS)
    log_cols = _patch_log_columns(log_vol, xi, yi)
    (tf, zs, _, med), sweep = select_trans_frac(
        log_cols, z_um, target_z_um, bank=bank, thr_floor=thr_floor)
    polyfit, n_in_gate, n_valid = fit_gated_surface(
        xs_um, ys_um, zs, target_z=target_z_um, gate_um=gate_um)
    return _with_abc(
        polyfit,
        method="iter08_cz_prior",
        subject_id=s.subject_id,
        cz_shape=list(vol.shape),
        cz_z_um=z_um, cz_xy_um=xy_um,
        target_z_um=target_z_um,
        selected_trans_frac=float(tf),
        median_col_z_um=float(med),
        n_valid=int(n_valid),
        n_in_gate=int(n_in_gate),
        sweep=sweep.to_dict(orient="records"),
    )


# ------------------------------------------------------------
# HCR top — iter07 detector
# ------------------------------------------------------------
def _load_hcr_combined(s, level: int = 4):
    # imported lazily to avoid a hard dependency when only CZ is needed
    from .benchmark_analysis import load_hcr_combined  # type: ignore
    return load_hcr_combined(s, level=level)


def compute_hcr_top_surface_iter07(s, level: int = 4) -> dict | None:
    vol, xy_um, z_um, _ = _load_hcr_combined(s, level=level)
    xi, yi = sampling_grid(vol.shape, xy_um)
    xs_um = xi * xy_um; ys_um = yi * xy_um
    zs, _ = detect_top_transitions(vol, z_um, xi, yi)
    polyfit = fit_polysurf(
        xs_um, ys_um, zs, degree=POLY_DEGREE, huber_k=HUBER_K)
    return _with_abc(
        polyfit,
        method="iter07_hcr_top",
        subject_id=s.subject_id,
        hcr_level=int(level),
        hcr_shape=list(vol.shape),
        hcr_z_um=float(z_um), hcr_xy_um=float(xy_um),
        median_col_z_um=float(np.nanmedian(zs)),
        n_valid=int(np.isfinite(zs).sum()),
    )


# ------------------------------------------------------------
# HCR bottom — iter08 symmetric detector
# ------------------------------------------------------------
def compute_hcr_bottom_surface_iter08(s, level: int = 4) -> dict | None:
    vol, xy_um, z_um, _ = _load_hcr_combined(s, level=level)
    xi, yi = sampling_grid(vol.shape, xy_um)
    xs_um = xi * xy_um; ys_um = yi * xy_um
    zs, _ = detect_bottom_transitions(vol, z_um, xi, yi)
    gate = mad_gate(zs, k=MAD_GATE_K)
    polyfit = fit_polysurf(
        xs_um[gate], ys_um[gate], zs[gate],
        degree=POLY_DEGREE, huber_k=HUBER_K)
    return _with_abc(
        polyfit,
        method="iter08_hcr_bottom",
        subject_id=s.subject_id,
        hcr_level=int(level),
        hcr_shape=list(vol.shape),
        hcr_z_um=float(z_um), hcr_xy_um=float(xy_um),
        median_col_z_um=float(np.nanmedian(zs)),
        n_valid=int(np.isfinite(zs).sum()),
        n_in_gate=int(gate.sum()),
    )


# ------------------------------------------------------------
# Cache I/O
# ------------------------------------------------------------
def _cache_path(subject_id: str, which: str) -> Path:
    return CACHE_DIR / f"{subject_id}_{which}.json"


def _jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def save_cached_surface(subject_id: str, which: str, surface: dict | None):
    if surface is None:
        return
    path = _cache_path(subject_id, which)
    path.write_text(json.dumps(_jsonable(surface), indent=2))


def load_cached_surface(subject_id: str, which: str) -> dict | None:
    path = _cache_path(subject_id, which)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if "coef" in data:
        data["coef"] = np.asarray(data["coef"], dtype=float)
    return data


# ------------------------------------------------------------
# Cache-aware accessors — downstream entry points
# ------------------------------------------------------------
def get_cz_surface_iter08(s, *, use_cache: bool = True,
                          write_cache: bool = True) -> dict | None:
    if use_cache:
        cached = load_cached_surface(s.subject_id, "cz_iter08")
        if cached is not None:
            return cached
    surface = compute_cz_surface_iter08(s)
    if write_cache and surface is not None:
        save_cached_surface(s.subject_id, "cz_iter08", surface)
    return surface


def get_hcr_top_surface_iter07(s, *, level: int = 4,
                               use_cache: bool = True,
                               write_cache: bool = True) -> dict | None:
    if use_cache:
        cached = load_cached_surface(s.subject_id, "hcr_top_iter07")
        if cached is not None:
            return cached
    surface = compute_hcr_top_surface_iter07(s, level=level)
    if write_cache and surface is not None:
        save_cached_surface(s.subject_id, "hcr_top_iter07", surface)
    return surface


def get_hcr_bottom_surface_iter08(s, *, level: int = 4,
                                  use_cache: bool = True,
                                  write_cache: bool = True) -> dict | None:
    if use_cache:
        cached = load_cached_surface(s.subject_id, "hcr_bottom_iter08")
        if cached is not None:
            return cached
    surface = compute_hcr_bottom_surface_iter08(s, level=level)
    if write_cache and surface is not None:
        save_cached_surface(s.subject_id, "hcr_bottom_iter08", surface)
    return surface


# ------------------------------------------------------------
# Batch entry — build the full main-pipeline surface store
# ------------------------------------------------------------
SUBJECTS_DEFAULT = ("755252", "767018", "767022", "782149", "788406", "790322")


def build_main_surface_store(subjects=SUBJECTS_DEFAULT, level: int = 4):
    """Compute CZ iter08 + HCR top/bottom iter07/08 surfaces for every
    benchmark subject, cache them, and write a combined summary CSV."""
    from .benchmark_data_loader import load_subject  # type: ignore
    rows = []
    for sid in subjects:
        print(f"=== {sid} ===", flush=True)
        s = load_subject(sid)
        cz = compute_cz_surface_iter08(s)
        save_cached_surface(sid, "cz_iter08", cz)
        top = compute_hcr_top_surface_iter07(s, level=level)
        save_cached_surface(sid, "hcr_top_iter07", top)
        bot = compute_hcr_bottom_surface_iter08(s, level=level)
        save_cached_surface(sid, "hcr_bottom_iter08", bot)

        def _pick(d, k, default=np.nan):
            return d.get(k, default) if d else default

        rows.append(dict(
            subject=sid,
            cz_trans_frac=_pick(cz, "selected_trans_frac"),
            cz_median_col_z_um=_pick(cz, "median_col_z_um"),
            cz_n_in_gate=_pick(cz, "n_in_gate"),
            cz_a=_pick(cz, "a"), cz_b=_pick(cz, "b"),
            cz_c=_pick(cz, "c"), cz_p=_pick(cz, "p"),
            cz_q=_pick(cz, "q"), cz_r=_pick(cz, "r"),
            hcr_top_median_col_z_um=_pick(top, "median_col_z_um"),
            hcr_top_n_valid=_pick(top, "n_valid"),
            hcr_top_a=_pick(top, "a"), hcr_top_b=_pick(top, "b"),
            hcr_top_c=_pick(top, "c"), hcr_top_p=_pick(top, "p"),
            hcr_top_q=_pick(top, "q"), hcr_top_r=_pick(top, "r"),
            hcr_bot_median_col_z_um=_pick(bot, "median_col_z_um"),
            hcr_bot_n_in_gate=_pick(bot, "n_in_gate"),
            hcr_bot_a=_pick(bot, "a"), hcr_bot_b=_pick(bot, "b"),
            hcr_bot_c=_pick(bot, "c"), hcr_bot_p=_pick(bot, "p"),
            hcr_bot_q=_pick(bot, "q"), hcr_bot_r=_pick(bot, "r"),
        ))
    df = pd.DataFrame(rows)
    csv_path = CACHE_DIR / "main_surfaces_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"wrote {csv_path}")
    return df


if __name__ == "__main__":
    build_main_surface_store()
