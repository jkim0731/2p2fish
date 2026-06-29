"""Iter 08 — HCR bottom (tissue→OOT) boundary via flipped iter07 detector.

Motivation
----------
iter07 locates the HCR pia (OOT→tissue transition at the top of the
volume) via patch-MAX + range-relative threshold + IRLS-Huber deg-2
poly.  ``load_hcr_combined`` clips OOT voxels to 0 on both ends of the
volume, so the bottom of the tissue block is a symmetric cliff (tissue
→ 0) — the same detector should work if we scan z in reverse.

Approach
--------
For each grid column we take the log-max over a 15×15 xy patch (same
as iter07), reverse it in z, and call ``col_detect_transition`` (the
same range-relative detector) on the reversed signal.  The reversed
signal's first-sustained-above-threshold voxel corresponds, in the
original volume, to the deepest voxel still in tissue.  We report
that index as the bottom boundary.

Surface fit is the same IRLS-Huber bivariate deg-2 polynomial
(``fit_polysurf``).  Per-column outliers are gated to |z - median_z|
≤ 3 × MAD before fitting to keep scattered deep-AF latches out of
the surface.

Outputs
-------
* ``figures/iter08_hcr_bottom_<sid>.png`` — XZ/YZ log(combined)
  panels with top (iter07 red) + bottom (iter08 magenta) + N22 top
  (green) overlaid.
* ``data/iter08_hcr_bottom_transitions_<sid>.npz`` — per-column
  bottom z (and thrs) + top z for reference.
* ``data/iter08_hcr_bottom_summary.csv`` — per-subject median
  bottom z, median tissue thickness, valid-count, HCR centroid
  bounding box for sanity.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from autocoreg.io.hcr_image import estimate_pia_surface_image_ceiling, estimate_pia_surface_quantile_ceiling, list_hcr_channels, load_hcr_combined, load_hcr_volume
from autocoreg.io.subjects import hcr_px_to_um, load_subject
from autocoreg.initial_registration.surface_detect import col_detect_transition, eval_polysurf, fit_polysurf, sampling_grid, EPS, HUBER_K, PATCH_W, POLY_DEGREE, SMOOTH_Z_UM, SUSTAIN_Z_UM, TRANS_FRAC, THR_FLOOR, detect_transitions as detect_top_transitions, get_n22

OUT_FIG = Path("/scratch/autocoreg_outputs/dev")
OUT_DATA = Path("/scratch/autocoreg_outputs/dev")

HCR = ["755252", "767018", "767022", "782149", "788406", "790322"]
MAD_GATE_K = 3.0  # per-column outlier gate for bottom-z before poly fit


def detect_bottom_transitions(vol, z_um, grid_ix, grid_iy,
                              patch_w=PATCH_W, thr_floor=THR_FLOOR,
                              trans_frac=TRANS_FRAC):
    """Same patch-MAX + col_detect_transition, but applied on each
    column reversed in z.  First sustained-above-thr on the reversed
    column = last sustained-above-thr on the original column.  Returns
    original-volume z-indices (in µm)."""
    Z, Y, X = vol.shape
    log_vol = np.log(vol + EPS)
    sustain_vox = max(1, int(SUSTAIN_Z_UM / z_um))
    zs = np.empty(len(grid_ix))
    thrs = np.empty(len(grid_ix))
    for i, (iy, ix) in enumerate(zip(grid_iy, grid_ix)):
        y0 = max(0, iy - patch_w); y1 = min(Y, iy + patch_w + 1)
        x0 = max(0, ix - patch_w); x1 = min(X, ix + patch_w + 1)
        log_col = log_vol[:, y0:y1, x0:x1].max(axis=(1, 2))
        log_col_rev = log_col[::-1]
        z_flip, _, thr_col = col_detect_transition(
            log_col_rev, z_um,
            smooth_z_um=SMOOTH_Z_UM,
            sustain_z_um=SUSTAIN_Z_UM,
            trans_frac=trans_frac,
            thr_floor=thr_floor,
            mode="range_relative",
        )
        if z_flip < 0:
            zs[i] = np.nan
        else:
            # Symmetric mapping: top detector returns the FIRST tissue
            # voxel, so the bottom detector should return the LAST
            # tissue voxel.  On the flipped column, the sustained run
            # covers flipped[z_flip : z_flip+sustain], which in
            # original indices is [Z-sustain-z_flip : Z-z_flip).
            # The LAST tissue voxel in original coordinates is
            # Z-1-z_flip.
            z_bot_vox = Z - 1 - z_flip
            zs[i] = z_bot_vox * z_um
        thrs[i] = thr_col
    return zs, thrs


def mad_gate(vals, k=MAD_GATE_K):
    ok = np.isfinite(vals)
    if ok.sum() == 0:
        return ok
    med = np.median(vals[ok])
    mad = np.median(np.abs(vals[ok] - med))
    if mad == 0:
        return ok
    return ok & (np.abs(vals - med) <= k * 1.4826 * mad)


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
    vol, xy_um, z_um, _ = load_hcr_combined(s, level=4)
    Z, Y, X = vol.shape
    print(f"  vol shape (Z,Y,X) = ({Z},{Y},{X})  z_um={z_um}  xy_um={xy_um:.3f}")

    hcr_xyz_um = hcr_px_to_um(
        s.hcr_centroids[["z_px", "y_px", "x_px"]].values, s)[:, [2, 1, 0]]
    n22 = get_n22(s, vol, xy_um, z_um, hcr_xyz_um)

    xi, yi = sampling_grid(vol.shape, xy_um)
    xs_um = xi * xy_um; ys_um = yi * xy_um

    zs_top, thrs_top = detect_top_transitions(vol, z_um, xi, yi)
    zs_bot, thrs_bot = detect_bottom_transitions(vol, z_um, xi, yi)
    valid_top = np.isfinite(zs_top)
    valid_bot = np.isfinite(zs_bot)
    print(f"  top valid: {valid_top.sum()}/{len(zs_top)} (median z = "
          f"{np.nanmedian(zs_top):.1f} µm)")
    print(f"  bot valid: {valid_bot.sum()}/{len(zs_bot)} (median z = "
          f"{np.nanmedian(zs_bot):.1f} µm)")

    top_fit = fit_polysurf(xs_um, ys_um, zs_top,
                           degree=POLY_DEGREE, huber_k=HUBER_K)

    # Bottom: MAD-gate before fit
    gated = mad_gate(zs_bot)
    n_in_gate = int(gated.sum())
    print(f"  bottom in MAD-gate: {n_in_gate}/{valid_bot.sum()}")
    bot_fit = fit_polysurf(xs_um[gated], ys_um[gated], zs_bot[gated],
                           degree=POLY_DEGREE, huber_k=HUBER_K)
    if bot_fit is None:
        print("  bottom polyfit failed")
        return None

    # HCR centroid z range (for sanity in log)
    hcr_z_um = hcr_xyz_um[:, 2]
    hcr_z_min = float(np.percentile(hcr_z_um, 1))
    hcr_z_max = float(np.percentile(hcr_z_um, 99))

    # Render
    y_mid = Y // 2; x_mid = X // 2
    x_um = np.arange(X) * xy_um
    y_um = np.arange(Y) * xy_um
    z_axis_um = np.arange(Z) * z_um

    def _line_over_x(surface_or_fit, y_const):
        if surface_or_fit is None:
            return None
        if isinstance(surface_or_fit, dict) and "coef" in surface_or_fit:
            return eval_polysurf(surface_or_fit, x_um,
                                 np.full_like(x_um, y_const))
        return _surface_z(surface_or_fit, x_um, y_const)

    def _line_over_y(surface_or_fit, x_const):
        if surface_or_fit is None:
            return None
        if isinstance(surface_or_fit, dict) and "coef" in surface_or_fit:
            return eval_polysurf(surface_or_fit,
                                 np.full_like(y_um, x_const), y_um)
        return _surface_z(surface_or_fit, x_const, y_um)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    def draw(ax, img, title, xlabel, xa, ya, overlays):
        vmin = float(np.percentile(img, 5))
        vmax = float(np.percentile(img, 99.5))
        ax.imshow(img, aspect="auto", cmap="gray", origin="upper",
                  extent=[xa[0], xa[-1], ya[-1], ya[0]],
                  vmin=vmin, vmax=vmax)
        for label, arr, color in overlays:
            if arr is None:
                continue
            ax.plot(xa, arr, color=color, lw=1.5, label=label)
        ax.set_xlabel(xlabel); ax.set_ylabel("z (µm)")
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=8)

    y_const = y_mid * xy_um; x_const = x_mid * xy_um

    ov_xz = [
        ("N22 top (truth)", _line_over_x(n22, y_const), "tab:green"),
        ("iter07 top", _line_over_x(top_fit, y_const), "tab:red"),
        ("iter08 bottom", _line_over_x(bot_fit, y_const), "magenta"),
    ]
    ov_yz = [
        ("N22 top (truth)", _line_over_y(n22, x_const), "tab:green"),
        ("iter07 top", _line_over_y(top_fit, x_const), "tab:red"),
        ("iter08 bottom", _line_over_y(bot_fit, x_const), "magenta"),
    ]

    draw(axes[0, 0], np.log(vol[:, y_mid, :] + EPS),
         f"{sid}  log(combined+ε) — XZ @ y={y_const:.0f} µm",
         "x (µm)", x_um, z_axis_um, ov_xz)
    draw(axes[0, 1], np.log(vol[:, :, x_mid] + EPS),
         f"{sid}  log(combined+ε) — YZ @ x={x_const:.0f} µm",
         "y (µm)", y_um, z_axis_um, ov_yz)
    draw(axes[1, 0], vol[:, y_mid, :],
         f"{sid}  combined — XZ @ y={y_const:.0f} µm",
         "x (µm)", x_um, z_axis_um, ov_xz)
    draw(axes[1, 1], vol[:, :, x_mid],
         f"{sid}  combined — YZ @ x={x_const:.0f} µm",
         "y (µm)", y_um, z_axis_um, ov_yz)

    z_top_grid = eval_polysurf(top_fit, xs_um, ys_um) if top_fit else None
    z_bot_grid = eval_polysurf(bot_fit, xs_um, ys_um)
    thickness = (np.nanmedian(z_bot_grid - z_top_grid)
                 if z_top_grid is not None else np.nan)

    fig.suptitle(
        f"{sid} — iter08 HCR bottom (patch-MAX on flipped column, "
        f"poly-deg{POLY_DEGREE}):  "
        f"median top = {np.nanmedian(zs_top):.1f} µm, "
        f"median bottom = {np.nanmedian(zs_bot):.1f} µm, "
        f"thickness ≈ {thickness:.0f} µm, "
        f"HCR centroid z ∈ [{hcr_z_min:.0f}, {hcr_z_max:.0f}] µm",
        fontsize=11, y=1.00)
    plt.tight_layout()
    out = OUT_FIG / f"iter08_hcr_bottom_{sid}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")

    np.savez(
        OUT_DATA / f"iter08_hcr_bottom_transitions_{sid}.npz",
        xs_um=xs_um, ys_um=ys_um,
        zs_top=zs_top, thrs_top=thrs_top,
        zs_bot=zs_bot, thrs_bot=thrs_bot,
        gated=gated,
    )

    return dict(
        subject=sid,
        z_um=z_um, xy_um=xy_um,
        vol_shape=f"({Z}, {Y}, {X})",
        median_top_z_um=float(np.nanmedian(zs_top)),
        median_bottom_z_um=float(np.nanmedian(zs_bot)),
        median_bottom_surface_z_um=float(np.nanmedian(z_bot_grid)),
        median_thickness_um=float(thickness) if np.isfinite(thickness) else np.nan,
        n_valid_top=int(valid_top.sum()),
        n_valid_bot=int(valid_bot.sum()),
        n_bottom_in_mad_gate=n_in_gate,
        hcr_centroid_z_p01=hcr_z_min,
        hcr_centroid_z_p99=hcr_z_max,
    )


def main():
    rows = []
    for sid in HCR:
        r = render_subject(sid)
        if r is not None:
            rows.append(r)
    df = pd.DataFrame(rows)
    out = OUT_DATA / "iter08_hcr_bottom_summary.csv"
    df.to_csv(out, index=False)
    print("\n=== iter08 HCR bottom summary ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:7.2f}"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
