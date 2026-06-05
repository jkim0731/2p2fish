"""Stage A — locked-prior warm-start.

Consumes:
  - `surface_registration_v2.get_surface_registration(s)` — 2-D PWR fit on
    the raw 488 MIP. **Authoritative source of sxy** (PWR-affine
    `base_sxy * exp(d_log_scale)`) and of (t_x, t_y) plus a residual θ
    correction.
  - `surfaces_iter08.get_cz_surface_iter08`, `get_hcr_top_surface_iter07` —
    image-only pia surfaces; used for the tilt-residual rotation R_tilt and
    the t_z anchor (CZ pia mapped onto HCR pia at the CZ centroid xy).

The per-cell xy-bbox area ratio (`roi_area_sxy`) is intentionally NOT
used here — it has known per-subject biases (e.g. 782149 −15 %) and is
only acceptable as a bootstrap initial guess for rigid registration.

Pins all 7 affine DOF as a usable warm-start before any 3-D centroid
matching:
  R   = R_180 @ R_tilt @ R_pwr_θ
  s_x = s_y = sxy           (PWR-affine, image-NCC)
  s_z = HCR_mean_depth / CZ_mean_depth  (basin-correct prior; Stage B
       refines via image-NCC sweep before any sz-locking downstream)
  (t_x, t_y) from PWR rigid+affine + crop_bbox offset
  t_z so the warped CZ centroid mean lands on
       HCR_pia(x_hcr, y_hcr) + sz · CZ_centroid_mean_depth.

API
---
* :class:`LockedPriorWarmStart` — dataclass with everything pinned.
* :func:`compute_locked_prior_warm_start(s)` — run from scratch.
* :func:`apply_to_cz_um(lp, cz_zyx_um)` — produce warped CZ centroids in HCR µm.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .benchmark_data_loader import HCR_SEG_XY_DOWNSAMPLE  # noqa: F401
from .r1_revised import (
    _plane_normal_from_surface,
    _rotation_about_z_row,
    _rotation_between_row,
    _surface_z_at,
)
from .surface_registration_v2 import get_surface_registration
from .surfaces_iter08 import (
    get_cz_surface_iter08,
    get_hcr_top_surface_iter07,
)

PRIOR_ROTATION_DEG_Z = 180.0

# Stage A's sz prior: HCR_mean_depth_below_pia / CZ_mean_depth_below_pia
# at the registered target. This puts the warped cloud in the same
# axial basin v1's ICP converges to. Stage B's image NCC sweep refines
# sz; this is just the initial guess so the LP is a usable warm-start.
SZ_INIT = "depth_ratio"


@dataclass
class LockedPriorWarmStart:
    """Pinned 7-DOF affine in (z, y, x) HCR µm convention.

    Apply to a CZ centroid array shaped (N, 3) in (z, y, x) CZ µm via
    ``apply_to_cz_um``.  Compatible with the candidate convention in
    `bench/candidate_impls/_p1_teaser.py` etc.
    """

    subject_id: str
    R: np.ndarray              # (3, 3) row-vec, applied AFTER mean-centering & scale
    scales: np.ndarray         # (3,) (sz, sy, sx) — note (z, y, x) order
    translation: np.ndarray    # (3,) (tz, ty, tx) — added LAST
    src_mean: np.ndarray       # (3,) CZ centroid mean (z, y, x) µm
    sxy_value: float
    sxy_source: str
    rotation_deg_z: float = 180.0
    pwr_ncc: float = 0.0
    pwr_method: str = ""
    diagnostics: dict = field(default_factory=dict)


# ----------------------------------------------------------------------
# sxy resolution
# ----------------------------------------------------------------------
def resolve_sxy(s, *, registration: dict | None = None) -> tuple[float, str]:
    """Return the PWR-affine sxy from `surface_registration_v2`.

    For ALL 6 subjects the final sxy is the image-NCC-driven PWR fit:
    `base_sxy * exp(d_log_scale)`. The per-cell xy-bbox area ratio
    (`roi_area_sxy`) has known per-subject biases and is intentionally
    not consulted here — see feedback memory
    `feedback_sxy_source_of_truth`.
    """
    reg = registration if registration is not None else get_surface_registration(s)
    affine = reg["methods"].get("affine", {})
    params = affine.get("params") or [0.0, 0.0, 0.0, 0.0]
    d_log_s = float(params[1])  # d_log_scale at stage2
    base_sxy = float(reg["sxy"])  # warm-start sxy used to render cz_warm
    sxy_img = base_sxy * math.exp(d_log_s)
    return sxy_img, "pwr_affine_sxy"


# ----------------------------------------------------------------------
# (t_x, t_y) from the PWR rigid+affine offset, in HCR µm
# ----------------------------------------------------------------------
def _affine_centroid_translation_um(
    s, reg: dict, sxy: float
) -> tuple[float, float, float, float]:
    """Apply the PWR rigid+affine 2-D fit to the CZ centroid mean and
    return (mean_x_hcr_um, mean_y_hcr_um, cz_mean_x_um, cz_mean_y_um),
    in absolute HCR µm (i.e. on the same frame as `s.hcr_centroids`
    converted via `hcr_xy_um`).

    The pipeline used by `surface_registration_v2`:

      cz_bin (CZ-px)          shape (H_b, W_b),  1 px = cz_xy_um µm
        ↓ nd_rotate 180° (reshape=True)            same shape
      rot           : (i, j) ↔ (H_b-1-i, W_b-1-j) in cz_bin
        ↓ zoom by f = cz_xy_um*sxy/hcr_xy_um
      cz_warm       shape (H_w, W_w),  1 px = hcr_xy_um µm
        ↓ M, offset (scipy: p_in_warm = M @ p_out_crop + offset)
      cropped HCR   1 px = hcr_xy_um µm, origin at crop_bbox[y0,x0]
        + [y0, x0] → HCR-full level-4 pixel grid
        × hcr_xy_um → absolute HCR µm.
    """
    cz_xy_um = float(s.cz_xy_um)
    hcr_xy_um = float(reg["hcr_xy_um"])
    cz_warm_h, cz_warm_w = reg["cz_warm_shape"]
    y0, _, x0, _ = reg["crop_bbox"]

    f = (cz_xy_um * sxy) / hcr_xy_um  # cz_bin → cz_warm zoom factor
    # cz_bin shape recovered from cz_warm shape (round-trip through scipy.zoom)
    H_b = int(round(cz_warm_h / f))
    W_b = int(round(cz_warm_w / f))

    cz = s.cz_centroids[["y_px", "x_px"]].to_numpy(float)
    cz_mean_y_um = float(cz[:, 0].mean()) * cz_xy_um
    cz_mean_x_um = float(cz[:, 1].mean()) * cz_xy_um

    # 1. CZ µm → cz_bin px.
    y_b = cz_mean_y_um / cz_xy_um
    x_b = cz_mean_x_um / cz_xy_um
    # 2. 180° rotation about cz_bin centre.
    y_rot = (H_b - 1) - y_b
    x_rot = (W_b - 1) - x_b
    # 3. Zoom by f → cz_warm px.
    yr_pix = y_rot * f
    xr_pix = x_rot * f

    # 4. Apply M, offset to get cropped-HCR px.  Prefer affine over rigid.
    method_key = "affine" if "affine" in reg["methods"] else "rigid"
    m = reg["methods"][method_key]
    M = np.asarray(m["M"], dtype=float)
    off = np.asarray(m["offset"], dtype=float)
    # scipy: p_in = M @ p_out + offset  →  p_out = M^{-1} (p_in - offset)
    p_in = np.array([yr_pix, xr_pix], dtype=float)
    p_out_crop = np.linalg.solve(M, p_in - off)

    # 5. Add crop offset → HCR-full level-4 px → absolute HCR µm.
    y_hcr_um = (float(p_out_crop[0]) + y0) * hcr_xy_um
    x_hcr_um = (float(p_out_crop[1]) + x0) * hcr_xy_um
    return x_hcr_um, y_hcr_um, cz_mean_x_um, cz_mean_y_um


# ----------------------------------------------------------------------
# (t_z) — anchor CZ pia onto HCR pia at the CZ centroid xy
# ----------------------------------------------------------------------
def _cz_mean_depth_um(s, src_mean_zyx_um: np.ndarray, cz_surface: dict) -> float:
    """Mean CZ centroid depth (µm) below CZ pia at the centroid mean xy."""
    from .benchmark_analysis import depth_from_surface
    cz_mean_xyz = src_mean_zyx_um[[2, 1, 0]]
    return float(depth_from_surface(cz_mean_xyz[None, :], cz_surface)[0])


def _hcr_mean_depth_um(s, hcr_surface: dict) -> float:
    """Mean HCR centroid depth (µm) below HCR pia, computed per-cell."""
    from .benchmark_analysis import depth_from_surface
    hcr_xyz = np.column_stack([
        s.hcr_centroids["x_px"].to_numpy(float) * float(s.hcr_xy_um),
        s.hcr_centroids["y_px"].to_numpy(float) * float(s.hcr_xy_um),
        s.hcr_centroids["z_px"].to_numpy(float) * float(s.hcr_z_um),
    ])
    depths = depth_from_surface(hcr_xyz, hcr_surface)
    return float(np.nanmean(depths))


def _resolve_sz_init(
    sz_init,
    s,
    cz_surface: dict,
    hcr_surface: dict,
    src_mean_zyx_um: np.ndarray,
) -> tuple[float, str]:
    """Resolve the Stage-A sz prior. Returns (sz, source)."""
    if isinstance(sz_init, (int, float)):
        return float(sz_init), "fixed"
    if sz_init == "depth_ratio":
        cz_d = _cz_mean_depth_um(s, src_mean_zyx_um, cz_surface)
        hcr_d = _hcr_mean_depth_um(s, hcr_surface)
        if cz_d > 0:
            return float(hcr_d / cz_d), "depth_ratio"
        return 1.0, "depth_ratio_fallback"
    raise ValueError(f"Unknown sz_init: {sz_init!r}")


def _tz_at_pia(
    s, sxy: float, R: np.ndarray, src_mean_zyx_um: np.ndarray,
    cz_surface: dict, hcr_surface: dict,
    target_x_hcr_um: float, target_y_hcr_um: float,
    sz: float,
) -> float:
    """Choose t_z so the CZ centroid maps to the HCR pia at the registered
    (x_hcr, y_hcr).  In zyx convention the candidate consumer applies:

        cz_warped = ((cz - src_mean) * scales) @ R.T + translation

    with translation = (tz, ty, tx).  We want the predicted z of the
    centroid mean to equal hcr_pia(x_hcr, y_hcr) + sz * cz_mean_depth.
    """
    cz_mean_depth = _cz_mean_depth_um(s, src_mean_zyx_um, cz_surface)
    pia_z = float(_surface_z_at(
        hcr_surface,
        np.array([target_x_hcr_um]),
        np.array([target_y_hcr_um]),
    )[0])
    return pia_z + sz * cz_mean_depth


# ----------------------------------------------------------------------
# top-level driver
# ----------------------------------------------------------------------
def compute_locked_prior_warm_start(
    s,
    *,
    sz_init=SZ_INIT,
    use_pwr_residual_theta: bool = True,
    registration: dict | None = None,
) -> LockedPriorWarmStart:
    """Build the locked-prior warm-start for one subject.

    Parameters
    ----------
    s
        ``SubjectData``.
    sz_init
        Initial sz (Stage B is what eventually sets it).  Default 1.0
        leaves the axial DOF for ICP to refine within Stage C bounds.
    use_pwr_residual_theta
        If True (default), incorporate the small θ residual recovered by
        the PWR rigid stage as an extra rotation about Z.
    registration
        Pre-computed ``surface_registration_v2`` dict; loaded from cache
        when None.
    """
    sid = s.subject_id
    if registration is None:
        registration = get_surface_registration(s)

    # ---- Surfaces ----
    cz_surface = get_cz_surface_iter08(s)
    hcr_surface = get_hcr_top_surface_iter07(s)

    # ---- sxy ----
    sxy, sxy_source = resolve_sxy(s, registration=registration)

    # ---- R = R_180 @ R_tilt(cz_normal -> hcr_normal) [@ R_pwr_theta] ----
    cz_xy = np.column_stack([
        s.cz_centroids["x_px"].to_numpy(float) * float(s.cz_xy_um),
        s.cz_centroids["y_px"].to_numpy(float) * float(s.cz_xy_um),
    ])
    hcr_xy = np.column_stack([
        s.hcr_centroids["x_px"].to_numpy(float) * float(s.hcr_xy_um),
        s.hcr_centroids["y_px"].to_numpy(float) * float(s.hcr_xy_um),
    ])
    n_cz, _ = _plane_normal_from_surface(cz_surface, cz_xy)
    n_hcr, _ = _plane_normal_from_surface(hcr_surface, hcr_xy)

    # Build R in xyz row-vec convention (matches the helpers + the surface
    # normals which are returned as (n_x, n_y, n_z)).
    R_180 = _rotation_about_z_row(PRIOR_ROTATION_DEG_Z)
    n_cz_rot = n_cz @ R_180
    R_tilt = _rotation_between_row(n_cz_rot, n_hcr)
    R_xyz = R_180 @ R_tilt
    if use_pwr_residual_theta:
        theta_pwr = float(registration["methods"]["rigid"]["theta_deg"])
        # The PWR rigid theta is in the 2-D image plane; add it as a small
        # rotation about Z on top of the 180° flip.
        R_pwr = _rotation_about_z_row(theta_pwr)
        R_xyz = R_xyz @ R_pwr
        rot_deg_z = PRIOR_ROTATION_DEG_Z + theta_pwr
    else:
        rot_deg_z = PRIOR_ROTATION_DEG_Z

    # Convert R to zyx row-vec so it matches src_mean / scales / translation
    # (all stored in zyx).  P is the anti-diagonal permutation; R_zyx =
    # P @ R_xyz @ P satisfies (v @ P) @ R_xyz @ P = v @ R_zyx, i.e. applying
    # R_zyx to a zyx row-vec yields the same rotation as R_xyz on the xyz
    # equivalent.  Without this conversion `apply_to_cz_um` was rotating
    # around the wrong axis (verified on 788406: median LP-vs-landmark error
    # 711 µm → 128 µm after the permutation).
    P = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=float)
    R = P @ R_xyz @ P

    # ---- (t_x, t_y) from PWR rigid+affine ----
    x_hcr_target_um, y_hcr_target_um, _, _ = (
        _affine_centroid_translation_um(s, registration, sxy)
    )

    # ---- src_mean (centroid mean in CZ µm, zyx) ----
    cz_zyx_px = s.cz_centroids[["z_px", "y_px", "x_px"]].to_numpy(float)
    src_mean_zyx = np.array([
        float(cz_zyx_px[:, 0].mean()) * float(s.cz_z_um),
        float(cz_zyx_px[:, 1].mean()) * float(s.cz_xy_um),
        float(cz_zyx_px[:, 2].mean()) * float(s.cz_xy_um),
    ])

    # ---- sz prior (depth ratio) ----
    sz_value, sz_source = _resolve_sz_init(
        sz_init, s, cz_surface, hcr_surface, src_mean_zyx
    )

    # ---- t_z so CZ pia → HCR pia ----
    tz = _tz_at_pia(
        s, sxy, R, src_mean_zyx, cz_surface, hcr_surface,
        x_hcr_target_um, y_hcr_target_um, sz_value,
    )

    scales = np.array([sz_value, sxy, sxy], dtype=float)  # (sz, sy, sx)
    translation = np.array([tz, y_hcr_target_um, x_hcr_target_um])

    return LockedPriorWarmStart(
        subject_id=sid,
        R=R,
        scales=scales,
        translation=translation,
        src_mean=src_mean_zyx,
        sxy_value=float(sxy),
        sxy_source=sxy_source,
        rotation_deg_z=float(rot_deg_z),
        pwr_ncc=float(registration["best_ncc"]),
        pwr_method=str(registration["best_method"]),
        diagnostics={
            "sz_init": float(sz_value),
            "sz_source": sz_source,
            "x_hcr_target_um": float(x_hcr_target_um),
            "y_hcr_target_um": float(y_hcr_target_um),
            "tz_um": float(tz),
            "n_cz_normal": n_cz.tolist(),
            "n_hcr_normal": n_hcr.tolist(),
            "use_pwr_residual_theta": bool(use_pwr_residual_theta),
            "pwr_rigid_theta_deg": float(
                registration["methods"]["rigid"]["theta_deg"]
            ),
        },
    )


def apply_to_cz_um(
    lp: LockedPriorWarmStart, cz_zyx_um: np.ndarray
) -> np.ndarray:
    """Warp CZ centroids to HCR µm using the locked-prior affine.

    ``cz_zyx_um`` shape (N, 3), columns (z, y, x).  Returns same shape in
    HCR µm.  Convention matches the candidate's expectation:

        cz_warped = ((cz_zyx - src_mean) * scales) @ R.T + translation
    """
    cz = np.asarray(cz_zyx_um, dtype=float)
    return ((cz - lp.src_mean) * lp.scales) @ lp.R.T + lp.translation


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from .benchmark_data_loader import BENCHMARK_SUBJECTS, load_subject

    sids = sys.argv[1:] or list(BENCHMARK_SUBJECTS)
    for sid in sids:
        s = load_subject(sid)
        lp = compute_locked_prior_warm_start(s)
        print(
            f"{sid}  sxy={lp.sxy_value:.3f} ({lp.sxy_source})  "
            f"theta_z={lp.rotation_deg_z:.2f}°  "
            f"t=(z={lp.translation[0]:+.0f}, y={lp.translation[1]:+.0f}, "
            f"x={lp.translation[2]:+.0f}) µm  "
            f"pwr_ncc={lp.pwr_ncc:.3f} [{lp.pwr_method}]"
        )
