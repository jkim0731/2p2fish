"""Session 08 — compare nonrigid stages on top of binary stage1+stage2.

Runs three nonrigid variants on the same affine-pre-warped CZ binary:

  1. ``fine_bspline``  — original (mesh ≈ min(H,W)/12 ≈ 48, single
     resolution, no smoothness penalty). Reproduces ``register_binary``
     stage 3.
  2. ``coarse_bspline``— mesh = 6, multi-resolution (4×→2×→1×) with
     Gaussian smoothing per level. The coarse mesh and resolution
     pyramid act as an implicit smoothness prior; we additionally
     smooth the resulting displacement field with a Gaussian σ=4 px to
     penalise residual roughness.
  3. ``piecewise_rigid``— Suite2p-style. Tile the canvas into
     ``n_blocks × n_blocks`` blocks, find the best (dy, dx) per block
     by NCC sweep, then interpolate block shifts to a smooth per-pixel
     displacement field via a thin-plate-style RBF smoothing
     (RegularGridInterpolator on block centres + Gaussian smoothing).

Inputs and pre-warp identical to ``register_binary.py``. Each variant
is scored by binary NCC against the same HCR target. Outputs a
single comparison summary JSON + a per-subject PNG.
"""
from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
from scipy.ndimage import (
    affine_transform, gaussian_filter, map_coordinates,
)
from scipy.interpolate import RegularGridInterpolator

# SimpleITK is only needed for the research bspline variant; allow import without it.
try:
    import SimpleITK as sitk  # type: ignore
except ImportError:
    sitk = types.SimpleNamespace()

from autocoreg.io.subjects import load_subject
from autocoreg.initial_registration.register_2d import cz_binary_top_mip, detect_objects_binary, get_sxy_with_fallback, hcr488_top_mip, ncc_overlap, rigid_to_affine_init, stage1_rigid, stage2_affine, warm_start_cz_binary
from autocoreg.initial_registration.surfaces import get_cz_surface_iter08, get_hcr_top_surface_iter07

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)
SUBJECTS = ["788406", "790322", "767018", "782149", "755252", "767022"]
HCR_LEVEL = 4

CZ_SLAB = (0.0, 50.0)
HCR_CH = "488"
HCR_SLAB = (0.0, 100.0)


# ---------------------------------------------------------------
# Variant 1 — fine-mesh BSpline (reproduce register_binary stage 3)
# ---------------------------------------------------------------
def nonrigid_fine_bspline(pre_warped: np.ndarray, hcr_bin: np.ndarray) -> dict:
    fixed_arr = gaussian_filter(hcr_bin.astype(np.float32), sigma=1.5)
    moving_arr = gaussian_filter(pre_warped.astype(np.float32), sigma=1.5)
    fixed = sitk.GetImageFromArray(fixed_arr)
    moving = sitk.GetImageFromArray(moving_arr)
    H_im, W_im = hcr_bin.shape
    mesh = max(6, int(min(H_im, W_im) // 12))
    try:
        initial_tx = sitk.BSplineTransformInitializer(
            image1=fixed, transformDomainMeshSize=[mesh, mesh], order=3,
        )
        R = sitk.ImageRegistrationMethod()
        R.SetMetricAsCorrelation()
        R.SetMetricSamplingStrategy(R.RANDOM)
        R.SetMetricSamplingPercentage(0.4)
        R.SetInterpolator(sitk.sitkLinear)
        R.SetOptimizerAsLBFGSB(
            gradientConvergenceTolerance=1e-5,
            numberOfIterations=150, maximumNumberOfCorrections=8,
        )
        R.SetInitialTransform(initial_tx, inPlace=True)
        final_tx = R.Execute(fixed, moving)
        warped_sitk = sitk.Resample(
            sitk.GetImageFromArray(pre_warped.astype(np.float32)),
            fixed, final_tx, sitk.sitkLinear, 0.0,
        )
        warped = sitk.GetArrayFromImage(warped_sitk).astype(np.float32)
        ncc = ncc_overlap(warped, hcr_bin.astype(np.float32),
                          np.ones(hcr_bin.shape, dtype=bool))
        pre_ncc = ncc_overlap(pre_warped.astype(np.float32),
                              hcr_bin.astype(np.float32),
                              np.ones(hcr_bin.shape, dtype=bool))
        if ncc < pre_ncc:
            return {"warped": pre_warped, "ncc": float(pre_ncc),
                    "ok": False, "mesh": mesh}
        return {"warped": warped, "ncc": float(ncc), "ok": True, "mesh": mesh}
    except Exception as e:
        return {"warped": pre_warped, "ncc": float("nan"),
                "ok": False, "err": str(e), "mesh": mesh}


# ---------------------------------------------------------------
# Variant 2 — coarse-mesh BSpline + multi-resolution + DF smoothing
# ---------------------------------------------------------------
def nonrigid_coarse_bspline(
    pre_warped: np.ndarray, hcr_bin: np.ndarray,
    mesh: int = 6, df_smooth_sigma_px: float = 4.0,
) -> dict:
    """Multi-resolution BSpline with a coarse mesh; the mesh size +
    pyramid act as a smoothness prior. After optimisation the
    displacement field is sampled densely and smoothed with a Gaussian
    (σ ≈ ``df_smooth_sigma_px`` px) to remove residual rough warps.
    """
    fixed_arr = gaussian_filter(hcr_bin.astype(np.float32), sigma=1.5)
    moving_arr = gaussian_filter(pre_warped.astype(np.float32), sigma=1.5)
    fixed = sitk.GetImageFromArray(fixed_arr)
    moving = sitk.GetImageFromArray(moving_arr)
    H_im, W_im = hcr_bin.shape

    try:
        initial_tx = sitk.BSplineTransformInitializer(
            image1=fixed, transformDomainMeshSize=[mesh, mesh], order=3,
        )
        R = sitk.ImageRegistrationMethod()
        R.SetMetricAsCorrelation()
        R.SetMetricSamplingStrategy(R.RANDOM)
        R.SetMetricSamplingPercentage(0.4)
        R.SetInterpolator(sitk.sitkLinear)
        R.SetOptimizerAsLBFGSB(
            gradientConvergenceTolerance=1e-5,
            numberOfIterations=200, maximumNumberOfCorrections=8,
        )
        # Multi-resolution: 4× → 2× → 1×
        R.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        R.SetSmoothingSigmasPerLevel(smoothingSigmas=[2.0, 1.0, 0.0])
        R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()
        R.SetInitialTransform(initial_tx, inPlace=True)
        final_tx = R.Execute(fixed, moving)

        # Sample displacement field densely on the fixed grid
        df_filter = sitk.TransformToDisplacementFieldFilter()
        df_filter.SetReferenceImage(fixed)
        df_img = df_filter.Execute(final_tx)
        df = sitk.GetArrayFromImage(df_img).astype(np.float32)  # (H, W, 2)

        # Smooth the DF (pixel-domain) — extra smoothness penalty
        if df_smooth_sigma_px > 0:
            df[..., 0] = gaussian_filter(df[..., 0], sigma=df_smooth_sigma_px)
            df[..., 1] = gaussian_filter(df[..., 1], sigma=df_smooth_sigma_px)

        # Apply DF (note: SITK DF is in physical x,y order; image is y,x)
        yy, xx = np.mgrid[:H_im, :W_im]
        # df[..., 0] = dx, df[..., 1] = dy in physical units (here = px)
        coords = np.array([yy + df[..., 1], xx + df[..., 0]])
        warped = map_coordinates(
            pre_warped.astype(np.float32), coords,
            order=1, mode="constant", cval=0.0,
        )
        ncc = ncc_overlap(warped.astype(np.float32),
                          hcr_bin.astype(np.float32),
                          np.ones(hcr_bin.shape, dtype=bool))
        pre_ncc = ncc_overlap(pre_warped.astype(np.float32),
                              hcr_bin.astype(np.float32),
                              np.ones(hcr_bin.shape, dtype=bool))
        if ncc < pre_ncc:
            return {"warped": pre_warped, "ncc": float(pre_ncc),
                    "ok": False, "mesh": mesh, "df_smooth_sigma_px": df_smooth_sigma_px,
                    "df_max_px": float(np.abs(df).max())}
        return {"warped": warped, "ncc": float(ncc), "ok": True,
                "mesh": mesh, "df_smooth_sigma_px": df_smooth_sigma_px,
                "df_max_px": float(np.abs(df).max())}
    except Exception as e:
        return {"warped": pre_warped, "ncc": float("nan"),
                "ok": False, "err": str(e), "mesh": mesh}


# ---------------------------------------------------------------
# Variant 3 — Suite2p-style piecewise rigid + smooth interpolation
# ---------------------------------------------------------------
def _block_best_shift(
    cz_block: np.ndarray, hcr_block: np.ndarray, max_shift: int,
) -> tuple[int, int, float]:
    """NCC sweep over (dy, dx) ∈ [-max_shift, max_shift] for one block."""
    if cz_block.std() < 1e-6 or hcr_block.std() < 1e-6:
        return 0, 0, 0.0
    H, W = cz_block.shape
    best = (0, 0, -1.0)
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            y0, y1 = max(0, dy), min(H, H + dy)
            x0, x1 = max(0, dx), min(W, W + dx)
            yy0, yy1 = max(0, -dy), min(H, H - dy)
            xx0, xx1 = max(0, -dx), min(W, W - dx)
            cz_s = cz_block[y0:y1, x0:x1]
            hcr_s = hcr_block[yy0:yy1, xx0:xx1]
            if cz_s.size < 25 or cz_s.std() < 1e-6 or hcr_s.std() < 1e-6:
                continue
            cz_z = (cz_s - cz_s.mean()) / cz_s.std()
            hcr_z = (hcr_s - hcr_s.mean()) / hcr_s.std()
            ncc = float((cz_z * hcr_z).mean())
            if ncc > best[2]:
                best = (dy, dx, ncc)
    return best


def nonrigid_piecewise_rigid(
    pre_warped: np.ndarray, hcr_bin: np.ndarray,
    n_blocks: int = 4, max_shift_px: int = 16,
    interp_smooth_sigma_px: float = 8.0,
    overlap_frac: float = 0.0, ncc_min: float = 0.05,
) -> dict:
    """Suite2p-style PWR — overlap-aware blocks + boundary-rejection gate.

    Per-block search returns ``(dy, dx)`` such that
    ``cz[y+dy, x+dx] ≈ hcr[y, x]`` (verified by synthetic-shift test in
    pwr_overlap.py); the inverse warp therefore samples
    ``cz_f[yy + dy_px, xx + dx_px]``, NOT ``cz_f[yy - dy_px, xx - dx_px]``
    as a previous version did.

    Per-block accept-gate:
      • block-NCC must exceed ``ncc_min``
      • the (dy, dx) maximum must NOT lie on the ±max_shift_px boundary
    Blocks that fail either check have their shift forced to (0, 0)
    so the smoothed displacement field doesn't propagate noise.

    overlap_frac ∈ [0, 1) is the fraction of block size shared with each
    neighbour. block_size = stride / (1 − overlap_frac); 0 reproduces
    the original non-overlapping behaviour.
    """
    H_im, W_im = hcr_bin.shape
    if not (0.0 <= overlap_frac < 1.0):
        raise ValueError("overlap_frac must be in [0, 1)")
    stride_y = H_im / n_blocks
    stride_x = W_im / n_blocks
    bh_eff = int(round(stride_y / max(1e-3, 1.0 - overlap_frac)))
    bw_eff = int(round(stride_x / max(1e-3, 1.0 - overlap_frac)))

    shifts = np.zeros((n_blocks, n_blocks, 2), dtype=np.float32)
    nccs = np.zeros((n_blocks, n_blocks), dtype=np.float32)
    accepted = np.zeros((n_blocks, n_blocks), dtype=bool)

    cz_f = pre_warped.astype(np.float32)
    hcr_f = hcr_bin.astype(np.float32)

    yc = np.array([(i + 0.5) * stride_y for i in range(n_blocks)])
    xc = np.array([(j + 0.5) * stride_x for j in range(n_blocks)])

    for i in range(n_blocks):
        for j in range(n_blocks):
            cy = int(round(yc[i])); cx = int(round(xc[j]))
            y0 = max(0, cy - bh_eff // 2); y1 = min(H_im, y0 + bh_eff)
            x0 = max(0, cx - bw_eff // 2); x1 = min(W_im, x0 + bw_eff)
            cz_b = cz_f[y0:y1, x0:x1]
            hcr_b = hcr_f[y0:y1, x0:x1]
            if cz_b.size < 100 or cz_b.std() < 1e-6 or hcr_b.std() < 1e-6:
                continue
            dy, dx, ncc = _block_best_shift(cz_b, hcr_b, max_shift_px)
            on_boundary = (abs(dy) == max_shift_px) or (abs(dx) == max_shift_px)
            nccs[i, j] = ncc
            if ncc < ncc_min or on_boundary:
                shifts[i, j] = (0.0, 0.0)
                accepted[i, j] = False
            else:
                shifts[i, j] = (dy, dx)
                accepted[i, j] = True

    fy_dy = RegularGridInterpolator(
        (yc, xc), shifts[..., 0], bounds_error=False, fill_value=None,
    )
    fy_dx = RegularGridInterpolator(
        (yc, xc), shifts[..., 1], bounds_error=False, fill_value=None,
    )
    yy, xx = np.mgrid[:H_im, :W_im]
    pts = np.stack([yy.ravel(), xx.ravel()], axis=-1).astype(np.float32)
    dy_px = fy_dy(pts).reshape(H_im, W_im).astype(np.float32)
    dx_px = fy_dx(pts).reshape(H_im, W_im).astype(np.float32)
    if interp_smooth_sigma_px > 0:
        dy_px = gaussian_filter(dy_px, sigma=interp_smooth_sigma_px)
        dx_px = gaussian_filter(dx_px, sigma=interp_smooth_sigma_px)

    # CORRECTED SIGN: warped[y, x] = cz[y + dy_px, x + dx_px]
    coords = np.array([yy + dy_px, xx + dx_px])
    warped = map_coordinates(
        pre_warped.astype(np.float32), coords,
        order=1, mode="constant", cval=0.0,
    )

    # Pearson NCC accept-gate (works for binary OR intensity HCR targets;
    # for binary-vs-binary input it reduces to the standard binary NCC).
    def _ncc(a, b):
        av = a.astype(np.float64).ravel()
        bv = b.astype(np.float64).ravel()
        av = av - av.mean(); bv = bv - bv.mean()
        den = av.std() * bv.std()
        return -1.0 if den <= 1e-12 else float((av * bv).mean() / den)

    pre_ncc = _ncc(cz_f, hcr_f)
    new_ncc = _ncc(warped, hcr_f)
    ok = new_ncc >= pre_ncc
    if not ok:
        warped = pre_warped
        new_ncc = pre_ncc
    return {
        "warped": warped, "ncc": float(new_ncc), "ok": bool(ok),
        "n_blocks": n_blocks, "max_shift_px": max_shift_px,
        "interp_smooth_sigma_px": interp_smooth_sigma_px,
        "overlap_frac": float(overlap_frac), "ncc_min": float(ncc_min),
        "shifts": shifts, "block_nccs": nccs, "accepted": accepted,
        "n_accepted": int(accepted.sum()),
        "df_max_dy_px": float(np.abs(dy_px).max()),
        "df_max_dx_px": float(np.abs(dx_px).max()),
    }


# ---------------------------------------------------------------
# Render: per-subject 4-up overlay (HCR target | pre-warped | each variant)
# ---------------------------------------------------------------
def render_variants(
    sid: str, hcr_bin, pre_warped, fine_warp, coarse_warp, pwr_warp,
    fine_ncc, coarse_ncc, pwr_ncc, pre_ncc, hcr_xy_um, sxy, out_path: Path,
):
    H, W = hcr_bin.shape
    extent = [0, W * hcr_xy_um, H * hcr_xy_um, 0]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    def overlay(ax, warped, title):
        rgb = np.zeros((H, W, 3), dtype=np.float32)
        rgb[..., 0] = np.clip(hcr_bin.astype(np.float32), 0, 1)
        rgb[..., 1] = np.clip(warped.astype(np.float32), 0, 1)
        ax.imshow(rgb, extent=extent, origin="upper")
        ax.set_title(title)
        ax.set_xlabel("x (µm)")

    axes[0, 0].imshow(hcr_bin, cmap="gray", extent=extent, origin="upper")
    axes[0, 0].set_title("HCR ch488 binary (target)")
    axes[0, 0].set_xlabel("x (µm)"); axes[0, 0].set_ylabel("y (µm)")

    overlay(axes[0, 1], pre_warped, f"affine pre-warp (CZ→HCR)  NCC={pre_ncc:.3f}")
    overlay(axes[0, 2], fine_warp, f"fine BSpline (mesh≈48)  NCC={fine_ncc:.3f}")
    overlay(axes[1, 0], coarse_warp, f"coarse BSpline (mesh=6, multi-res, smooth)  NCC={coarse_ncc:.3f}")
    overlay(axes[1, 1], pwr_warp, f"piecewise-rigid (Suite2p-style)  NCC={pwr_ncc:.3f}")

    axes[1, 2].axis("off")
    text = (
        f"{sid}\n"
        f"sxy = {sxy:.3f}\n\n"
        f"Pre-warp (affine)         NCC = {pre_ncc:.3f}\n"
        f"fine BSpline   (≈48)      NCC = {fine_ncc:.3f}   Δ = {fine_ncc - pre_ncc:+.3f}\n"
        f"coarse BSpline (6, smooth) NCC = {coarse_ncc:.3f}   Δ = {coarse_ncc - pre_ncc:+.3f}\n"
        f"piecewise-rigid (6×6, σ=6) NCC = {pwr_ncc:.3f}   Δ = {pwr_ncc - pre_ncc:+.3f}"
    )
    axes[1, 2].text(0.05, 0.95, text, family="monospace", fontsize=11,
                    transform=axes[1, 2].transAxes, va="top")

    fig.suptitle(f"{sid} — nonrigid variants on binary affine pre-warp")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------
# Driver
# ---------------------------------------------------------------
def process_subject(sid: str) -> dict:
    print(f"\n=== {sid} ===", flush=True)
    s = load_subject(sid)
    sxy, sxy_src = get_sxy_with_fallback(sid, s)
    print(f"  sxy={sxy:.4f}  src={sxy_src}", flush=True)

    hcr_surf = get_hcr_top_surface_iter07(s, level=HCR_LEVEL)
    cz_surf = get_cz_surface_iter08(s)

    hcr_488_mip, hcr_xy_um = hcr488_top_mip(s, hcr_surf, *HCR_SLAB)
    hcr_bin = detect_objects_binary(hcr_488_mip, hcr_xy_um)

    cz_bin = cz_binary_top_mip(s, cz_surf, *CZ_SLAB)
    cz_bin_warm = warm_start_cz_binary(
        cz_bin, cz_xy_um=float(s.cz_xy_um), hcr_xy_um=hcr_xy_um, sxy=sxy,
    )

    rigid = stage1_rigid(cz_bin_warm, hcr_bin)
    print(f"  RIGID:    θ={rigid['theta']:+.2f}°  NCC={rigid['ncc']:.3f}", flush=True)

    H_im, W_im = hcr_bin.shape
    H_t, W_t = cz_bin_warm.shape
    th_in = rigid["template_shape"]
    th = np.deg2rad(rigid["theta"])
    c, s_ = np.cos(th), np.sin(th)
    M0 = np.array([[c, s_], [-s_, c]], dtype=np.float64)
    cy_out = rigid["oy"] + th_in[0] / 2.0
    cx_out = rigid["ox"] + th_in[1] / 2.0
    cy_in = H_t / 2.0; cx_in = W_t / 2.0
    off0 = np.array([cy_in, cx_in]) - M0 @ np.array([cy_out, cx_out])

    affine = stage2_affine(cz_bin_warm, hcr_bin, M0, off0)
    print(f"  AFFINE:   NCC={affine['ncc']:.3f}", flush=True)

    pre_warped = affine_transform(
        cz_bin_warm.astype(np.float32),
        affine["M"], offset=affine["offset"],
        output_shape=hcr_bin.shape, order=1, mode="constant", cval=0.0,
    )
    pre_ncc = ncc_overlap(pre_warped, hcr_bin.astype(np.float32),
                          np.ones(hcr_bin.shape, dtype=bool))

    fine = nonrigid_fine_bspline(pre_warped, hcr_bin)
    print(f"  fine BSpline    NCC={fine['ncc']:.3f}  ok={fine.get('ok')}", flush=True)
    coarse = nonrigid_coarse_bspline(pre_warped, hcr_bin)
    print(f"  coarse BSpline  NCC={coarse['ncc']:.3f}  ok={coarse.get('ok')}  "
          f"df_max={coarse.get('df_max_px', float('nan')):.1f} px", flush=True)
    pwr = nonrigid_piecewise_rigid(pre_warped, hcr_bin)
    print(f"  piecewise-rigid NCC={pwr['ncc']:.3f}  ok={pwr.get('ok')}  "
          f"df_max=(dy {pwr['df_max_dy_px']:.1f}, dx {pwr['df_max_dx_px']:.1f}) px",
          flush=True)

    out_png = FIG / f"register_nonrigid_variants_{sid}.png"
    render_variants(
        sid, hcr_bin, pre_warped, fine["warped"], coarse["warped"], pwr["warped"],
        fine["ncc"], coarse["ncc"], pwr["ncc"], pre_ncc,
        hcr_xy_um, sxy, out_png,
    )

    return {
        "sid": sid, "sxy": sxy, "hcr_xy_um": hcr_xy_um,
        "rigid_ncc": float(rigid["ncc"]),
        "affine_ncc": float(affine["ncc"]),
        "pre_warp_ncc": float(pre_ncc),
        "fine_bspline":    {"ncc": fine["ncc"], "ok": fine.get("ok"), "mesh": fine.get("mesh")},
        "coarse_bspline":  {"ncc": coarse["ncc"], "ok": coarse.get("ok"),
                            "mesh": coarse.get("mesh"),
                            "df_smooth_sigma_px": coarse.get("df_smooth_sigma_px"),
                            "df_max_px": coarse.get("df_max_px")},
        "piecewise_rigid": {"ncc": pwr["ncc"], "ok": pwr.get("ok"),
                            "n_blocks": pwr.get("n_blocks"),
                            "max_shift_px": pwr.get("max_shift_px"),
                            "interp_smooth_sigma_px": pwr.get("interp_smooth_sigma_px"),
                            "df_max_dy_px": pwr.get("df_max_dy_px"),
                            "df_max_dx_px": pwr.get("df_max_dx_px")},
        "wrote": str(out_png.relative_to(HERE)),
    }


def main():
    all_results = {}
    for sid in SUBJECTS:
        try:
            all_results[sid] = process_subject(sid)
        except Exception as e:
            print(f"  ERROR {sid}: {type(e).__name__}: {e}", flush=True)
            all_results[sid] = {"sid": sid, "error": str(e)}
    out = HERE / "register_nonrigid_variants_summary.json"
    out.write_text(json.dumps(all_results, indent=2, default=float))
    print(f"\nwrote {out.name}")


if __name__ == "__main__":
    main()
