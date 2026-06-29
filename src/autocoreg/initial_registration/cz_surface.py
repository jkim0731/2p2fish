"""Iter 08 — CZ surface with the experimenter's 50 µm depth prior.

Motivation
----------
During CZ acquisition the experimenter manually positioned the pial
surface at z ≈ 50 µm from the first slice.  That is independent
metadata — not derived from the image — and we can exploit it to
disambiguate between candidate detector thresholds.

iter07_cz_proto (TRANS_FRAC = 0.5) matches the prior on 5/6 subjects
but lands at 102.5 µm on 767022 (AF/OOT slab above tissue → the
p10→p90 midpoint lands mid-ramp on strong tissue).  iter07_cz_lowfrac
(TRANS_FRAC = 0.02) always fires earlier but often drops below the
prior (34–46 µm), i.e. it latches on pre-pia AF / coverslip debris on
some subjects.  No single TRANS_FRAC works for every subject.

iter08 approach: sweep a small bank of TRANS_FRAC values and select
the one whose *median per-column transition depth* is closest to the
50 µm prior.  The prior acts as the selection score, so subjects with
a clean tissue cliff (low TRANS_FRAC fires early on coverslip) prefer
a larger fraction; subjects with an OOT slab (large TRANS_FRAC fires
deep) prefer a smaller fraction.

After selection, gate per-column transitions to |z - 50| ≤ GATE_UM
before the IRLS-Huber poly fit, so the surface is not bent by a
handful of AF-latched outliers.

Outputs
-------
* ``figures/iter08_cz_<sid>.png`` — 4-y-slice log(CZ) overlays with
  the selected surface (red), prior plane at z = 50 µm (yellow
  dashed), existing image_ceiling surface (cyan), and CZ centroid
  surface (green).
* ``data/iter08_cz_transitions_<sid>.npz`` — per-column zs + thrs for
  the selected candidate.
* ``data/iter08_cz_selection.csv`` — per-subject selected TRANS_FRAC,
  median_z, |median−prior|, n_in_gate.
* ``data/iter08_cz_sweep.csv`` — full bank (all subjects × all
  candidates) with median z and closeness-to-prior.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from autocoreg.io.hcr_image import analyze_subject, estimate_pia_surface
from autocoreg.io.subjects import cz_px_to_um, load_subject
from autocoreg.initial_registration.surface_detect import col_detect_transition, eval_polysurf, fit_polysurf, sampling_grid, HUBER_K, PATCH_W, POLY_DEGREE, SMOOTH_Z_UM, SUSTAIN_Z_UM

OUT_FIG = Path("/scratch/autocoreg_outputs/dev")
OUT_DATA = Path("/scratch/autocoreg_outputs/dev")

SUBJECTS = ["755252", "767018", "767022", "782149", "788406", "790322"]

EPS = 1e-3
CZ_THR_FLOOR = 4.5                  # ≈ log(90 DN); rejects pure-OOT columns
CZ_TARGET_Z_UM = 50.0               # experimenter prior: pia ≈ 50 µm
TRANS_FRAC_BANK = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50)
GATE_UM = 75.0                      # per-column outlier gate around prior
N_Y = 4
Y_INTERIOR_FRAC = 0.15


def load_cz_volume(s):
    files = (
        list(s.coreg_dir.glob("*reg-dim-swapped.ome.tif"))
        or list(s.coreg_dir.glob("*zstack.tif"))
    )
    if not files:
        raise FileNotFoundError(f"no CZ TIFF for {s.subject_id}")
    arr = tifffile.imread(str(files[0]))
    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"CZ TIFF shape={arr.shape} not ZYX")
    return arr.astype(np.float32, copy=False)


def _patch_log_columns(log_vol, grid_ix, grid_iy, patch_w=PATCH_W):
    """Pre-compute the (patch log-max) column for every grid point once,
    so we can re-threshold cheaply across the TRANS_FRAC bank."""
    Z, Y, X = log_vol.shape
    n = len(grid_ix)
    cols = np.empty((n, Z), dtype=np.float32)
    for i, (iy, ix) in enumerate(zip(grid_iy, grid_ix)):
        y0 = max(0, iy - patch_w); y1 = min(Y, iy + patch_w + 1)
        x0 = max(0, ix - patch_w); x1 = min(X, ix + patch_w + 1)
        cols[i] = log_vol[:, y0:y1, x0:x1].max(axis=(1, 2))
    return cols


def _detect_on_columns(log_cols, z_um, trans_frac, thr_floor):
    n = log_cols.shape[0]
    zs = np.empty(n)
    thrs = np.empty(n)
    for i in range(n):
        z_vox, _, thr_col = col_detect_transition(
            log_cols[i], z_um,
            smooth_z_um=SMOOTH_Z_UM,
            sustain_z_um=SUSTAIN_Z_UM,
            trans_frac=trans_frac,
            thr_floor=thr_floor,
            mode="range_relative",
        )
        zs[i] = z_vox * z_um if z_vox >= 0 else np.nan
        thrs[i] = thr_col
    return zs, thrs


def select_trans_frac(log_cols, z_um, target_z, bank=TRANS_FRAC_BANK,
                      thr_floor=CZ_THR_FLOOR):
    """Evaluate every TRANS_FRAC in the bank, return the one whose
    median column transition z is closest to the prior + the full sweep
    table for diagnostics."""
    rows = []
    best = None
    best_score = np.inf
    for tf in bank:
        zs, thrs = _detect_on_columns(log_cols, z_um, tf, thr_floor)
        valid = np.isfinite(zs)
        med = float(np.nanmedian(zs)) if valid.any() else np.nan
        score = abs(med - target_z) if np.isfinite(med) else np.inf
        rows.append(dict(
            trans_frac=tf, median_z=med,
            abs_dev_from_prior=score,
            n_valid=int(valid.sum()),
        ))
        if score < best_score:
            best_score = score
            best = (tf, zs, thrs, med)
    sweep = pd.DataFrame(rows)
    return best, sweep


def fit_gated_surface(xs_um, ys_um, zs_um, target_z=CZ_TARGET_Z_UM,
                      gate_um=GATE_UM):
    """Drop per-column transitions outside [target_z ± gate_um] before
    the IRLS-Huber poly fit; return the fit + number of in-gate points."""
    ok = np.isfinite(zs_um)
    if gate_um is not None and np.isfinite(target_z):
        in_gate = ok & (np.abs(zs_um - target_z) <= gate_um)
    else:
        in_gate = ok
    xs_in = xs_um[in_gate]; ys_in = ys_um[in_gate]; zs_in = zs_um[in_gate]
    polyfit = fit_polysurf(xs_in, ys_in, zs_in,
                           degree=POLY_DEGREE, huber_k=HUBER_K)
    return polyfit, int(in_gate.sum()), int(ok.sum())


def _cz_existing_surface(s):
    """Existing CZ image-based ceiling surface from analyze_subject."""
    try:
        info = analyze_subject(s)
    except Exception as exc:  # pragma: no cover
        print(f"  analyze_subject failed for {s.subject_id}: {exc}")
        return None
    return info.get("cz_surface")


def _cz_centroid_surface(s):
    cz_um = cz_px_to_um(
        s.cz_centroids[["z_px", "y_px", "x_px"]].values, s)
    cz_xyz = cz_um[:, [2, 1, 0]]
    try:
        return estimate_pia_surface(cz_xyz)
    except Exception as exc:  # pragma: no cover
        print(f"  centroid surface failed for {s.subject_id}: {exc}")
        return None


def _surface_z(surface, xs, y0):
    if surface is None:
        return np.full_like(np.asarray(xs, dtype=float), np.nan)
    a, b, c = surface["a"], surface["b"], surface["c"]
    p = surface.get("p", 0.0); q = surface.get("q", 0.0)
    r = surface.get("r", 0.0)
    xs = np.asarray(xs, dtype=float)
    return (a * xs + b * y0 + c + p * xs * xs
            + q * xs * y0 + r * y0 * y0)


def render_subject(sid):
    print(f"=== {sid} ===", flush=True)
    s = load_subject(sid)
    vol = load_cz_volume(s)
    z_um, xy_um = s.cz_z_um, s.cz_xy_um
    Z, Y, X = vol.shape
    log_vol = np.log(vol + EPS)

    xi, yi = sampling_grid(vol.shape, xy_um)
    xs_um = xi * xy_um; ys_um = yi * xy_um

    log_cols = _patch_log_columns(log_vol, xi, yi)
    (sel_tf, sel_zs, sel_thrs, sel_med), sweep = select_trans_frac(
        log_cols, z_um, CZ_TARGET_Z_UM)
    sweep["subject"] = sid
    print(f"  TRANS_FRAC sweep (target = {CZ_TARGET_Z_UM:.0f} µm):")
    for _, r in sweep.iterrows():
        mark = " *" if r["trans_frac"] == sel_tf else "  "
        print(f"  {mark} tf={r['trans_frac']:.2f}  median_z={r['median_z']:6.1f}  "
              f"|dev|={r['abs_dev_from_prior']:6.2f}  n_valid={r['n_valid']}")
    print(f"  selected TRANS_FRAC = {sel_tf}  median_z = {sel_med:.1f} µm")

    polyfit, n_in, n_valid = fit_gated_surface(xs_um, ys_um, sel_zs)
    if polyfit is None:
        print("  polyfit failed")
        return None, sweep
    print(f"  in-gate: {n_in}/{n_valid} (valid), gate ±{GATE_UM:.0f} µm")

    cz_centroid = _cz_centroid_surface(s)
    cz_existing = _cz_existing_surface(s)

    x_um = np.arange(X) * xy_um
    z_axis = np.arange(Z) * z_um
    y_lo = int(Y_INTERIOR_FRAC * Y); y_hi = Y - 1 - y_lo
    y_idx = np.linspace(y_lo, y_hi, N_Y).astype(int)

    fig, axes = plt.subplots(1, N_Y, figsize=(5 * N_Y, 4.8), sharey=True)
    for ax, iy in zip(axes, y_idx):
        y_const = iy * xy_um
        img = np.log(vol[:, iy, :] + EPS)
        vmin = float(np.percentile(img, 5))
        vmax = float(np.percentile(img, 99.5))
        ax.imshow(img, aspect="auto", cmap="gray", origin="upper",
                  extent=[x_um[0], x_um[-1], z_axis[-1], z_axis[0]],
                  vmin=vmin, vmax=vmax)
        ax.axhline(CZ_TARGET_Z_UM, color="gold", lw=1.0, ls="--",
                   label=f"prior (z = {CZ_TARGET_Z_UM:.0f} µm)")
        if cz_centroid is not None:
            ax.plot(x_um, _surface_z(cz_centroid, x_um, y_const),
                    color="tab:green", lw=1.3, label="CZ centroid")
        if cz_existing is not None:
            ax.plot(x_um, _surface_z(cz_existing, x_um, y_const),
                    color="tab:cyan", lw=1.3, label="CZ image_ceiling")
        z_poly = eval_polysurf(polyfit, x_um, np.full_like(x_um, y_const))
        ax.plot(x_um, z_poly, color="tab:red", lw=1.8,
                label=f"iter08 (tf={sel_tf})")
        ax.set_xlabel("x (µm)")
        ax.set_title(f"y = {y_const:.0f} µm")
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("z (µm)")
    fig.suptitle(
        f"{sid} — iter08 CZ (prior-selected TRANS_FRAC={sel_tf}, "
        f"poly-deg{POLY_DEGREE}, gate ±{GATE_UM:.0f} µm): "
        f"median_z = {sel_med:.1f} µm",
        fontsize=12, y=1.02)
    plt.tight_layout()
    out = OUT_FIG / f"iter08_cz_{sid}.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")

    np.savez(
        OUT_DATA / f"iter08_cz_transitions_{sid}.npz",
        xs_um=xs_um, ys_um=ys_um, zs_um=sel_zs, thrs=sel_thrs,
        selected_trans_frac=np.float32(sel_tf),
        prior_z_um=np.float32(CZ_TARGET_Z_UM),
        gate_um=np.float32(GATE_UM),
    )

    # Evaluate poly at a dense grid to report median surface z
    z_surf = eval_polysurf(polyfit, xs_um, ys_um)
    surf_med = float(np.nanmedian(z_surf))

    row = dict(
        subject=sid,
        cz_shape=f"({Z}, {Y}, {X})",
        prior_z_um=CZ_TARGET_Z_UM,
        selected_trans_frac=sel_tf,
        median_col_trans_z=sel_med,
        median_surface_z=surf_med,
        abs_dev_from_prior=abs(surf_med - CZ_TARGET_Z_UM),
        n_in_gate=n_in, n_valid=n_valid,
        gate_um=GATE_UM,
    )
    return row, sweep


def main():
    rows, sweeps = [], []
    for sid in SUBJECTS:
        row, sweep = render_subject(sid)
        sweeps.append(sweep)
        if row is not None:
            rows.append(row)

    df = pd.DataFrame(rows)
    out = OUT_DATA / "iter08_cz_selection.csv"
    df.to_csv(out, index=False)
    print("\n=== iter08 CZ selection ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:7.2f}"))
    print(f"wrote {out}")

    sweep_df = pd.concat(sweeps, ignore_index=True)
    out2 = OUT_DATA / "iter08_cz_sweep.csv"
    sweep_df.to_csv(out2, index=False)
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
