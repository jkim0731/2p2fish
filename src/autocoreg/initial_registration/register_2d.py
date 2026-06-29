"""Session 08 — V3 binary-mask registration with rigid → affine → nonrigid.

Pipeline (V3 geometry):
  • CZ:  fill outlines in `seg-mask-outline.tif`, pia-flatten, max-MIP at
    depth 0-50 µm  → CZ binary mask (Y, X) on CZ pixel grid.
  • HCR ch488: pia-flattened max-MIP at 0-100 µm (already in
    ``projections_{sid}.npz``), then "object detection" binarisation:
    Gaussian blur (≈cell radius) → percentile threshold → small-object
    removal → fill holes.

  Warm start: rotate CZ binary 180° and rescale by ``sxy`` so that one
  warm CZ-pixel = one HCR-L4 pixel in HCR-frame µm.

  Stage 1 — RIGID:    FFT-NCC sweep over θ ∈ [-30°, 30°] (coarse 2°,
                       fine ±2° at 0.25°); best (θ, tx, ty).
  Stage 2 — AFFINE:    6-param x' = A·x + t (init from Stage 1), Powell
                       optimise -NCC on full HCR canvas.
  Stage 3 — NONRIGID:  SimpleITK BSpline (init from Stage 2 affine);
                       NCC of deformed template vs HCR.

  Best-NCC stage is reported per-subject per-variant (only V3 here).
"""
from __future__ import annotations

import json
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import (
    affine_transform,
    binary_fill_holes,
    binary_opening,
    gaussian_filter,
    label,
    rotate as nd_rotate,
    zoom,
)
from scipy.optimize import minimize
from scipy.signal import correlate

# SimpleITK is only needed for the research-only stage3_bspline function;
# allow import without it for the core pipeline.
try:
    import SimpleITK as sitk  # type: ignore
except ImportError:
    sitk = types.SimpleNamespace()

from autocoreg.io.hcr_image import load_hcr_volume
from autocoreg.io.subjects import load_subject
from autocoreg.initial_registration.lateral_scale import estimate_sxy_roi_area
from autocoreg.initial_registration.surfaces import get_cz_surface_iter08, get_hcr_top_surface_iter07
from autocoreg.initial_registration.projections import top_slab_projection

HERE = Path(__file__).resolve().parent
FIG = Path("/scratch/autocoreg_outputs/dev")  # figures go to /scratch in package context (was HERE/figures)
SUBJECTS = ["788406", "790322", "767018", "782149", "755252", "767022"]
HCR_LEVEL = 4

CZ_SLAB = (0.0, 50.0)
HCR_CH = "488"
HCR_SLAB = (0.0, 100.0)
MODE = "max"


# --------------------------------------------------------------
# CZ binary volume + pia-flat MIP
# --------------------------------------------------------------
def load_cz_binary_volume(s) -> np.ndarray:
    files = list(s.coreg_dir.glob("*seg-mask-outline.tif"))
    if not files:
        files = list(s.coreg_dir.glob("*seg_masks_outline.tif"))
    if not files:
        files = list(s.coreg_dir.glob("*seg*outline*.tif"))
    if not files:
        raise FileNotFoundError(f"No seg outline tif in {s.coreg_dir}")
    with tifffile.TiffFile(files[0]) as tf:
        out = tf.asarray()
    if out.dtype != np.uint8:
        out = (out > 0).astype(np.uint8)
    out = (out > 0).astype(np.uint8)
    # Fill outlines per-slice → solid cell bodies
    filled = np.empty_like(out)
    for z in range(out.shape[0]):
        filled[z] = binary_fill_holes(out[z]).astype(np.uint8)
    return filled


def cz_binary_top_mip(s, cz_surf, d_lo: float, d_hi: float) -> np.ndarray:
    bin_vol = load_cz_binary_volume(s)
    proj = top_slab_projection(
        bin_vol.astype(np.float32),
        cz_surf, xy_um=float(s.cz_xy_um), z_um=float(s.cz_z_um),
        d_lo=d_lo, d_hi=d_hi, mode="max",
    )
    return (np.nan_to_num(proj, nan=0.0) > 0.5).astype(np.uint8)


# --------------------------------------------------------------
# HCR 488 MIP → "object detection" binarisation
# --------------------------------------------------------------
def hcr488_top_mip(s, hcr_surf, d_lo: float, d_hi: float):
    vol, xy_um, z_um = load_hcr_volume(s, channel=HCR_CH, level=HCR_LEVEL)
    proj = top_slab_projection(
        vol.astype(np.float32),
        hcr_surf, xy_um=float(xy_um), z_um=float(z_um),
        d_lo=d_lo, d_hi=d_hi, mode=MODE,
    )
    return proj, float(xy_um)


def detect_objects_binary(
    mip: np.ndarray, xy_um: float, sigma_um: float = 4.0,
    pct: float = 88.0, min_area_um2: float = 30.0,
) -> np.ndarray:
    """Object-detection binarisation for HCR 488 MIP.

    1. Gaussian-blur (~cell-radius scale) to suppress noise/seams.
    2. Percentile threshold (default p88) → bright pixels.
    3. Small-object removal (drop CCs < `min_area_um2`).
    4. Fill holes inside surviving objects.

    Output: uint8 mask of shape ``mip.shape``.
    """
    a = np.asarray(mip, dtype=np.float32)
    finite = np.isfinite(a)
    a_f = np.where(finite, a, np.nanmin(a[finite]) if finite.any() else 0.0)
    sig_px = max(sigma_um / xy_um, 0.5)
    blur = gaussian_filter(a_f, sigma=sig_px)
    thr = np.percentile(blur[finite], pct)
    raw = (blur > thr).astype(np.uint8)
    raw = binary_opening(raw, iterations=1).astype(np.uint8)
    raw = binary_fill_holes(raw).astype(np.uint8)
    lab, n = label(raw)
    if n == 0:
        return raw
    sizes_px = np.bincount(lab.ravel())[1:]
    px_area_um2 = xy_um * xy_um
    keep = np.where(sizes_px * px_area_um2 >= min_area_um2)[0] + 1
    out = np.isin(lab, keep).astype(np.uint8)
    return out


# --------------------------------------------------------------
# Warm start: 180° + sxy scale → HCR pixel grid
# --------------------------------------------------------------
def warm_start_cz_binary(
    cz_bin: np.ndarray, cz_xy_um: float, hcr_xy_um: float, sxy: float,
) -> np.ndarray:
    rot = nd_rotate(
        cz_bin.astype(np.float32), 180.0, reshape=True, order=0,
        mode="constant", cval=0.0,
    )
    f = (cz_xy_um * sxy) / hcr_xy_um
    z = zoom(rot, f, order=0)
    return (z > 0.5).astype(np.uint8)


# --------------------------------------------------------------
# NCC utilities
# --------------------------------------------------------------
def fft_ncc(template: np.ndarray, image: np.ndarray) -> np.ndarray:
    h, w = template.shape
    n = float(h * w)
    mT = float(template.mean())
    t_centered = (template - mT).astype(np.float32)
    t_norm = float(np.sqrt((t_centered ** 2).sum()))
    if t_norm <= 0:
        return np.zeros(
            (image.shape[0] - h + 1, image.shape[1] - w + 1), dtype=np.float32
        )
    box = np.ones((h, w), dtype=np.float32)
    img = image.astype(np.float32)
    sum_TI = correlate(img, t_centered, mode="valid", method="fft")
    sum_I = correlate(img, box, mode="valid", method="fft")
    sum_I2 = correlate(img * img, box, mode="valid", method="fft")
    mean_I = sum_I / n
    var_I = np.maximum(sum_I2 / n - mean_I ** 2, 0.0)
    std_I_norm = np.sqrt(var_I) * np.sqrt(n)
    return (sum_TI / (t_norm * std_I_norm + 1e-10)).astype(np.float32)


def ncc_overlap(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """NCC of two same-shape arrays evaluated only inside `mask`."""
    if mask.sum() < 50:
        return -1.0
    av = a[mask].astype(np.float64)
    bv = b[mask].astype(np.float64)
    av -= av.mean(); bv -= bv.mean()
    den = (av.std() * bv.std())
    if den <= 1e-12:
        return -1.0
    return float((av * bv).mean() / den)


# --------------------------------------------------------------
# Stage 1 — rigid (FFT-NCC θ sweep)
# --------------------------------------------------------------
def stage1_rigid(template: np.ndarray, image: np.ndarray) -> dict:
    image_f = image.astype(np.float32)
    coarse = np.arange(-30.0, 30.0 + 1e-9, 2.0)
    fine_pad, fine_step = 3.0, 0.25

    def sweep(thetas):
        best = {"ncc": -np.inf, "log": []}
        for th in thetas:
            t_rot = nd_rotate(
                template.astype(np.float32), float(th), reshape=True,
                order=0, mode="constant", cval=0.0,
            )
            t_rot = (t_rot > 0.5).astype(np.float32)
            if t_rot.shape[0] >= image_f.shape[0] or t_rot.shape[1] >= image_f.shape[1]:
                best["log"].append((float(th), float("nan"), -1, -1))
                continue
            ncc = fft_ncc(t_rot, image_f)
            idx = int(np.argmax(ncc))
            oy, ox = np.unravel_index(idx, ncc.shape)
            v = float(ncc[oy, ox])
            best["log"].append((float(th), v, int(oy), int(ox)))
            if v > best["ncc"]:
                best.update(
                    theta=float(th), oy=int(oy), ox=int(ox), ncc=v,
                    template_shape=tuple(int(x) for x in t_rot.shape),
                )
        return best

    coarse_b = sweep(coarse)
    fine = np.arange(coarse_b["theta"] - fine_pad, coarse_b["theta"] + fine_pad + 1e-9, fine_step)
    fine_b = sweep(fine)
    best = fine_b if fine_b["ncc"] >= coarse_b["ncc"] else coarse_b
    best["sweep_log"] = coarse_b["log"] + fine_b["log"]
    return best


def rigid_to_affine_init(
    rigid: dict, template_shape, image_shape,
) -> np.ndarray:
    """Return 2×3 affine matrix mapping HCR (output) → template (input)
    such that template, rotated by `theta` and placed at (oy, ox) in
    HCR, produces the rigid alignment.

    Uses scipy convention for `affine_transform`:
        input_coords = M @ output_coords + offset
    We package as homogeneous [[A, t], [0, 1]] (3×3) and we'll split.
    """
    th = np.deg2rad(rigid["theta"])
    th_in = template_shape  # (h, w) of rotated template
    H, W = image_shape
    # Centre of rotated template in image coords
    cy_out = rigid["oy"] + th_in[0] / 2.0
    cx_out = rigid["ox"] + th_in[1] / 2.0
    # Centre of un-rotated warm template
    H_in, W_in = template_shape  # caller passes original (non-rotated)
    cy_in = H_in / 2.0
    cx_in = W_in / 2.0
    c, s = np.cos(th), np.sin(th)
    # Output→input: rotate by +θ around image centre then translate
    # M (input = M @ output + offset)
    M = np.array([[c,  s], [-s, c]], dtype=np.float64)
    offset = np.array([cy_in, cx_in]) - M @ np.array([cy_out, cx_out])
    return M, offset


# --------------------------------------------------------------
# Stage 2 — affine refinement
# --------------------------------------------------------------
def stage2_affine(
    template: np.ndarray, image: np.ndarray,
    M0: np.ndarray, offset0: np.ndarray,
    min_overlap_frac: float = 0.5,
    scale_bounds: tuple = (0.90, 1.10),
    theta_bounds_deg: float = 10.0,
) -> dict:
    """4-DOF SIMILARITY refinement (rotation + uniform scale + tx + ty)
    on top of the rigid init (M0, offset0). Optimised via Powell on
    -NCC.

    Parameter vector p ∈ R^4:
       [d_theta, d_log_scale, d_oy, d_ox]
    The transform actually applied is:
       M_new = (1 + d_log_scale) · R(d_theta) · M0
       offset_new = offset0 + (d_oy, d_ox)
    Solutions that scale outside ``scale_bounds`` or leave less than
    ``min_overlap_frac`` of the original template area inside the image
    receive a heavy penalty (prevents the corner-overfit pathology
    seen with unconstrained 6-DOF affine on 782149).
    """
    template_f = template.astype(np.float32)
    image_f = image.astype(np.float32)
    H_im, W_im = image_f.shape
    H_t, W_t = template_f.shape
    template_area_target = float((template_f > 1e-6).sum() * min_overlap_frac)

    log_s_lo = np.log(scale_bounds[0])
    log_s_hi = np.log(scale_bounds[1])
    th_lim = np.deg2rad(theta_bounds_deg)

    def warp(p):
        d_th, d_log_s, d_oy, d_ox = p
        s = np.exp(d_log_s)
        Rd = np.array([[np.cos(d_th), -np.sin(d_th)],
                       [np.sin(d_th),  np.cos(d_th)]])
        M = s * (Rd @ M0)
        off = offset0 + np.array([d_oy, d_ox])
        warped = affine_transform(
            template_f, M, offset=off,
            output_shape=(H_im, W_im), order=1, mode="constant", cval=-1.0,
        )
        valid = warped >= 0
        warped = np.where(valid, warped, 0.0)
        return warped, valid, M, off

    def neg_ncc(p):
        d_th, d_log_s, _, _ = p
        if not (log_s_lo <= d_log_s <= log_s_hi) or abs(d_th) > th_lim:
            return 1.0  # Out-of-bounds penalty
        warped, valid, _, _ = warp(p)
        if (warped > 1e-6).sum() < template_area_target:
            return 1.0  # Too little overlap → penalty
        return -ncc_overlap(warped, image_f, valid)

    p0 = np.zeros(4)
    base_ncc = -neg_ncc(p0)
    res = minimize(
        neg_ncc, p0, method="Powell",
        options={"xtol": 1e-3, "ftol": 1e-4, "maxiter": 800},
    )
    final_ncc = -res.fun
    if final_ncc < base_ncc:
        p_best = p0; final_ncc = base_ncc
    else:
        p_best = res.x
    warped, _, M, off = warp(p_best)
    return {"M": M, "offset": off, "ncc": float(final_ncc), "warped": warped,
            "params": p_best.tolist()}


# --------------------------------------------------------------
# Stage 3 — non-rigid (SITK BSpline)
# --------------------------------------------------------------
def stage3_nonrigid(
    template: np.ndarray, image: np.ndarray,
    M_aff: np.ndarray, offset_aff: np.ndarray,
) -> dict:
    """BSpline registration in SITK, initialised from affine warp.
    Affine result is used to pre-warp the moving image into HCR space;
    BSpline then refines locally.
    """
    image_f = image.astype(np.float32)
    template_f = template.astype(np.float32)
    H_im, W_im = image_f.shape

    pre_warped = affine_transform(
        template_f, M_aff, offset=offset_aff,
        output_shape=(H_im, W_im), order=1, mode="constant", cval=0.0,
    ).astype(np.float32)
    pre_warped_ncc = ncc_overlap(pre_warped, image_f, np.ones_like(image_f, dtype=bool))

    # Light blur both inputs to give the optimiser a continuous metric
    fixed_arr = gaussian_filter(image_f, sigma=1.5)
    moving_arr = gaussian_filter(pre_warped, sigma=1.5)
    fixed = sitk.GetImageFromArray(fixed_arr.astype(np.float32))
    moving = sitk.GetImageFromArray(moving_arr.astype(np.float32))

    mesh = max(6, int(min(H_im, W_im) // 12))
    err = None
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
            numberOfIterations=150,
            maximumNumberOfCorrections=8,
        )
        R.SetInitialTransform(initial_tx, inPlace=True)
        final_tx = R.Execute(fixed, moving)
        warped_sitk = sitk.Resample(
            moving, fixed, final_tx, sitk.sitkLinear, 0.0, moving.GetPixelID()
        )
        warped = sitk.GetArrayFromImage(warped_sitk).astype(np.float32)
        ncc = ncc_overlap(warped, image_f, np.ones_like(image_f, dtype=bool))
        # Accept only if it improved
        if ncc >= pre_warped_ncc:
            return {"warped": warped, "ncc": float(ncc), "ok": True}
        else:
            return {"warped": pre_warped, "ncc": float(pre_warped_ncc),
                    "ok": False, "err": f"nonrigid did not improve ({ncc:.3f} < {pre_warped_ncc:.3f})"}
    except Exception as e:
        err = str(e)
        return {"warped": pre_warped, "ncc": float(pre_warped_ncc),
                "ok": False, "err": err}


# --------------------------------------------------------------
# Visualisation
# --------------------------------------------------------------
def stretch01(img):
    a = np.asarray(img, dtype=np.float32)
    return np.clip(a, 0, 1)


def render_full(
    sid: str, hcr_bin, cz_bin_warm, rigid_warp, affine_warp, nonrigid_warp,
    out_path: Path, theta: float, sxy: float, hcr_xy_um: float,
    nccs: dict, hcr_488_mip,
):
    H, W = hcr_bin.shape
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    extent = [0, W * hcr_xy_um, H * hcr_xy_um, 0]

    # Top row: HCR 488 raw, HCR binary, CZ warm-start binary, NCC bar chart
    axes[0, 0].imshow(np.log1p(hcr_488_mip), cmap="gray", extent=extent, origin="upper")
    axes[0, 0].set_title("HCR ch488 0-100 max (log)"); axes[0, 0].set_xlabel("x (µm)")

    axes[0, 1].imshow(hcr_bin, cmap="gray", extent=extent, origin="upper")
    axes[0, 1].set_title("HCR ch488 binarised (objects)")

    h, w = cz_bin_warm.shape
    extent_cz = [0, w * hcr_xy_um, h * hcr_xy_um, 0]
    axes[0, 2].imshow(cz_bin_warm, cmap="gray", extent=extent_cz, origin="upper")
    axes[0, 2].set_title(f"CZ binary warm-start  (180°, ×sxy={sxy:.3f})")

    keys = ["rigid", "affine", "nonrigid"]
    vals = [nccs[k] for k in keys]
    axes[0, 3].bar(keys, vals, color=["#7f7", "#7af", "#f97"])
    axes[0, 3].set_title("NCC by stage")
    axes[0, 3].set_ylim(0, max(0.6, max(vals) * 1.1))
    for i, v in enumerate(vals):
        axes[0, 3].text(i, v + 0.01, f"{v:.3f}", ha="center")

    # Bottom row: overlays per stage + final
    def overlay(ax, warped, title):
        rgb = np.zeros((H, W, 3), dtype=np.float32)
        rgb[..., 0] = stretch01(hcr_bin)  # red = HCR
        rgb[..., 1] = stretch01(warped)   # green = CZ warped
        ax.imshow(rgb, extent=extent, origin="upper")
        ax.set_title(title); ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")

    overlay(axes[1, 0], rigid_warp,    f"RIGID  θ={theta:+.2f}°  NCC={nccs['rigid']:.3f}")
    overlay(axes[1, 1], affine_warp,   f"AFFINE  NCC={nccs['affine']:.3f}")
    overlay(axes[1, 2], nonrigid_warp, f"NONRIGID  NCC={nccs['nonrigid']:.3f}")

    best_stage = max(keys, key=lambda k: nccs[k])
    best_warp = {"rigid": rigid_warp, "affine": affine_warp, "nonrigid": nonrigid_warp}[best_stage]
    overlay(axes[1, 3], best_warp, f"BEST: {best_stage}  NCC={nccs[best_stage]:.3f}")

    fig.suptitle(f"{sid} — V3 binary registration (HCR ch488 0-100 vs CZ ROI 0-50)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------
# Driver
# --------------------------------------------------------------
def get_sxy_with_fallback(sid: str, s) -> tuple[float, str]:
    """Return (sxy, source). Falls back to landmark-GT sxy for subjects
    that lack the HCR cell_body_segmentation/metrics.pickle (755252,
    767022) — flagged so the comparison can mark these as warm-start-
    leaky."""
    try:
        return float(estimate_sxy_roi_area(sid)["sxy_median"]), "bbox"
    except Exception:
        from autocoreg.io.hcr_image import fit_anisotropic_similarity
        from autocoreg.io.subjects import landmark_pairs_um
        cz_um, hcr_um = landmark_pairs_um(s, active_only=True)
        fit = fit_anisotropic_similarity(cz_um, hcr_um)
        return float(np.sqrt(fit.scales[0] * fit.scales[1])), "landmark_gt_fallback"


def process_subject(sid: str) -> dict:
    print(f"\n=== {sid} ===", flush=True)
    s = load_subject(sid)
    sxy, sxy_src = get_sxy_with_fallback(sid, s)
    print(f"  sxy={sxy:.4f}  source={sxy_src}", flush=True)

    hcr_surf = get_hcr_top_surface_iter07(s, level=HCR_LEVEL)
    cz_surf = get_cz_surface_iter08(s)

    hcr_488_mip, hcr_xy_um = hcr488_top_mip(s, hcr_surf, *HCR_SLAB)
    hcr_bin = detect_objects_binary(hcr_488_mip, hcr_xy_um)
    print(f"  HCR ch488 MIP: shape={hcr_488_mip.shape} xy={hcr_xy_um:.3f}  "
          f"binary fill={(hcr_bin>0).mean()*100:.1f}%", flush=True)

    cz_bin = cz_binary_top_mip(s, cz_surf, *CZ_SLAB)
    cz_bin_warm = warm_start_cz_binary(
        cz_bin, cz_xy_um=float(s.cz_xy_um), hcr_xy_um=hcr_xy_um, sxy=sxy,
    )
    print(f"  CZ binary MIP: shape={cz_bin.shape} fill={(cz_bin>0).mean()*100:.1f}%   "
          f"warm shape={cz_bin_warm.shape} fill={(cz_bin_warm>0).mean()*100:.1f}%",
          flush=True)

    # Stage 1 — RIGID
    rigid = stage1_rigid(cz_bin_warm, hcr_bin)
    print(f"  RIGID:    θ={rigid['theta']:+.2f}°  (oy,ox)=({rigid['oy']},{rigid['ox']}) px  "
          f"NCC={rigid['ncc']:.3f}", flush=True)

    H_im, W_im = hcr_bin.shape
    H_t, W_t = cz_bin_warm.shape
    th_in = rigid["template_shape"]  # rotated template shape
    M0, off0 = rigid_to_affine_init(rigid, template_shape=(H_t, W_t),
                                    image_shape=(H_im, W_im))
    # Re-fetch via more careful function:
    th = np.deg2rad(rigid["theta"])
    c, s_ = np.cos(th), np.sin(th)
    M0 = np.array([[c,  s_], [-s_, c]], dtype=np.float64)
    cy_out = rigid["oy"] + th_in[0] / 2.0
    cx_out = rigid["ox"] + th_in[1] / 2.0
    cy_in = H_t / 2.0; cx_in = W_t / 2.0
    off0 = np.array([cy_in, cx_in]) - M0 @ np.array([cy_out, cx_out])

    # Stage 2 — AFFINE
    affine = stage2_affine(cz_bin_warm, hcr_bin, M0, off0)
    print(f"  AFFINE:   NCC={affine['ncc']:.3f}", flush=True)

    # Stage 3 — NONRIGID
    nonrigid = stage3_nonrigid(cz_bin_warm, hcr_bin, affine["M"], affine["offset"])
    print(f"  NONRIGID: NCC={nonrigid['ncc']:.3f} ok={nonrigid.get('ok')}", flush=True)

    # Build rigid overlay (warped to HCR canvas)
    rigid_warp = affine_transform(
        cz_bin_warm.astype(np.float32), M0, offset=off0,
        output_shape=hcr_bin.shape, order=1, mode="constant", cval=0.0,
    )

    nccs = {"rigid": float(rigid["ncc"]), "affine": float(affine["ncc"]),
            "nonrigid": float(nonrigid["ncc"])}
    best_stage = max(nccs, key=nccs.get)

    out_png = FIG / f"register_binary_{sid}_V3.png"
    render_full(
        sid, hcr_bin, cz_bin_warm, rigid_warp, affine["warped"], nonrigid["warped"],
        out_png, rigid["theta"], sxy, hcr_xy_um, nccs, hcr_488_mip,
    )

    out = {
        "sid": sid, "sxy": sxy, "sxy_source": sxy_src, "hcr_xy_um": hcr_xy_um,
        "rigid": {"theta_deg": rigid["theta"], "oy": rigid["oy"], "ox": rigid["ox"],
                  "ncc": float(rigid["ncc"])},
        "affine": {"ncc": float(affine["ncc"]),
                   "M": affine["M"].tolist(), "offset": affine["offset"].tolist()},
        "nonrigid": {"ncc": float(nonrigid["ncc"]), "ok": bool(nonrigid.get("ok"))},
        "best_stage": best_stage, "best_ncc": nccs[best_stage],
        "wrote": str(out_png.relative_to(HERE)),
        "params": {"hcr_ch": HCR_CH, "hcr_slab_um": list(HCR_SLAB),
                   "cz_slab_um": list(CZ_SLAB), "mode": MODE},
    }
    print(f"  BEST stage: {best_stage}  NCC={nccs[best_stage]:.3f}", flush=True)
    return out


def main():
    all_results = {}
    for sid in SUBJECTS:
        all_results[sid] = process_subject(sid)
    summary = HERE / "register_binary_summary.json"
    summary.write_text(json.dumps(all_results, indent=2, default=float))
    print(f"\nwrote {summary.name}")


if __name__ == "__main__":
    main()
