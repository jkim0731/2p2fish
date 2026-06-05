"""3-D overlap crop: axis-aligned bounding box of the warped CZ volume in HCR µm.

``get_overlap_crop(s)`` answers "what region of the HCR volume does the CZ
sample actually cover after the locked-prior + Stage B sz are applied?"  It is
the canonical "where should I work?" call for any session that needs to localise
effort to the overlap region before cell-matching.

Coordinate frames
-----------------
* CZ µm (z, y, x)   — raw CZ volume corners, stored in CZ voxel-size units
* HCR µm (z, y, x)  — output of ``apply_to_cz_um``; same convention as
                       ``s.hcr_centroids`` converted by (hcr_z_um, hcr_xy_um)
* HCR level-2 vox   — ``hcr_xy_um = s.hcr_xy_um``, ``hcr_z_um = s.hcr_z_um``
                       This is the centroid / seg-zarr frame.  The pickle-bbox
                       coords used by seg zarr are also level-2 (divide by 4
                       from level-0 to match; see reference_hcr_seg_zarr_levels).

``load_hcr_volume(s, channel="488", level=2)`` is used solely to get the
volume shape; the voxel sizes come from ``s.hcr_xy_um`` / ``s.hcr_z_um``
(which are the level-2 values stored on ``SubjectData``).
"""
from __future__ import annotations

import numpy as np

from .benchmark_analysis import load_hcr_volume
from .cz_volume import load_cz_volume
from .locked_prior_warm import apply_to_cz_um, compute_locked_prior_warm_start
from .sz_estimator import get_sz

# Level-2 is the centroid / seg-zarr frame.  Voxel sizes at level-2 are
# s.hcr_xy_um and s.hcr_z_um (stored directly on SubjectData — no rescaling
# needed because hcr_level_resolution at level=2 gives xy_um * 2^0 = xy_um).
_HCR_BOUNDS_LEVEL = 2


def _cz_volume_corners_zyx_um(s) -> np.ndarray:
    """Return the 8 corners of the CZ volume in CZ µm (z, y, x), shape (8, 3)."""
    cz_vol = load_cz_volume(s)
    Z, Y, X = cz_vol.shape
    cz_z_um = float(s.cz_z_um)
    cz_xy_um = float(s.cz_xy_um)
    # Corners: z ∈ {0, (Z-1)*cz_z_um}, y ∈ {0, (Y-1)*cz_xy_um}, etc.
    z_edges = np.array([0.0, (Z - 1) * cz_z_um])
    y_edges = np.array([0.0, (Y - 1) * cz_xy_um])
    x_edges = np.array([0.0, (X - 1) * cz_xy_um])
    corners = np.array([
        [z, y, x]
        for z in z_edges
        for y in y_edges
        for x in x_edges
    ])
    return corners


def get_overlap_crop(
    s,
    margin_frac: float = 0.10,
) -> dict:
    """Compute the 3-D overlap bounding box of the warped CZ volume in HCR µm.

    Steps
    -----
    1. Compute the locked-prior warm-start (cached via its own mechanism).
    2. Get the locked sz from Stage B (``get_sz``, JSON-cached).
    3. Override ``lp.scales[0]`` with the locked sz and warp the 8 corners
       of the CZ volume bounding box into HCR µm.
    4. Take the axis-aligned bounding box, expand by ``margin_frac`` per axis
       on each side, clip against the HCR volume extent.
    5. Convert to level-2 voxel indices (useful for slicing seg zarr).

    Parameters
    ----------
    s
        ``SubjectData`` from ``benchmark_data_loader.load_subject``.
    margin_frac
        Fractional expansion on each side per axis (default 10 %).

    Returns
    -------
    dict with keys:
        ``bbox_hcr_um``        — [z0, z1, y0, y1, x0, x1] in HCR µm (float)
        ``bbox_hcr_l2_vox``    — [z0, z1, y0, y1, x0, x1] in level-2 voxel
                                 indices (int, inclusive-floor / exclusive-ceil)
        ``margin_frac``        — the margin_frac used
        ``sz_used``            — the sz value from Stage B (or sz_lp on failure)
        ``hcr_voxel_um``       — (z_um, xy_um) at level 2
        ``hcr_volume_shape``   — (Z, Y, X) of level-2 HCR 488 volume
        ``cz_corners_hcr_um``  — (8, 3) warped corners in HCR µm, for debugging
    """
    # ---- Stage A: locked-prior affine (cached inside compute_LP) ----
    lp = compute_locked_prior_warm_start(s)

    # ---- Stage B: locked sz ----
    sz_dict = get_sz(s)
    sz_used = sz_dict.get("sz_best")
    if sz_used is None:
        # Stage B failed; fall back to the LP depth-ratio prior and log it
        sz_used = float(lp.scales[0])
        print(
            f"[overlap_crop] WARNING: Stage B sz estimation failed for "
            f"{s.subject_id} (reason: {sz_dict.get('fail_reason','?')}); "
            f"falling back to sz_lp={sz_used:.3f}"
        )
    else:
        sz_used = float(sz_used)

    # ---- Override sz in a copy of lp.scales ----
    # apply_to_cz_um uses lp.scales[0] as sz; we swap it for the locked value.
    import copy
    lp_locked = copy.copy(lp)
    lp_locked.scales = lp.scales.copy()
    lp_locked.scales[0] = sz_used

    # ---- Warp 8 corners of CZ volume ----
    corners_cz_um = _cz_volume_corners_zyx_um(s)  # (8, 3) in CZ µm
    corners_hcr_um = apply_to_cz_um(lp_locked, corners_cz_um)  # (8, 3) HCR µm

    # ---- Axis-aligned bounding box in HCR µm ----
    z_min = float(corners_hcr_um[:, 0].min())
    z_max = float(corners_hcr_um[:, 0].max())
    y_min = float(corners_hcr_um[:, 1].min())
    y_max = float(corners_hcr_um[:, 1].max())
    x_min = float(corners_hcr_um[:, 2].min())
    x_max = float(corners_hcr_um[:, 2].max())

    print(
        f"[overlap_crop] {s.subject_id}: raw CZ bbox in HCR µm "
        f"z=[{z_min:.0f}, {z_max:.0f}] "
        f"y=[{y_min:.0f}, {y_max:.0f}] "
        f"x=[{x_min:.0f}, {x_max:.0f}]  "
        f"extents: dz={z_max-z_min:.0f} dy={y_max-y_min:.0f} dx={x_max-x_min:.0f} µm"
    )

    # ---- Load HCR volume shape for bounds clipping (level-2) ----
    hcr_vol, hcr_xy_um, hcr_z_um = load_hcr_volume(
        s, channel="488", level=_HCR_BOUNDS_LEVEL
    )
    hcr_shape = hcr_vol.shape  # (Z, Y, X)
    del hcr_vol  # free memory immediately; we only needed the shape

    # Level-2 voxel sizes match s.hcr_z_um / s.hcr_xy_um directly.
    assert abs(hcr_xy_um - float(s.hcr_xy_um)) < 1e-4, (
        f"level-2 xy_um={hcr_xy_um:.4f} != s.hcr_xy_um={s.hcr_xy_um:.4f}"
    )
    assert abs(hcr_z_um - float(s.hcr_z_um)) < 1e-4, (
        f"level-2 z_um={hcr_z_um:.4f} != s.hcr_z_um={s.hcr_z_um:.4f}"
    )

    hcr_z_extent_um = hcr_shape[0] * hcr_z_um
    hcr_y_extent_um = hcr_shape[1] * hcr_xy_um
    hcr_x_extent_um = hcr_shape[2] * hcr_xy_um

    # ---- Expand by margin_frac on each side ----
    dz = z_max - z_min
    dy = y_max - y_min
    dx = x_max - x_min
    mz = margin_frac * dz
    my = margin_frac * dy
    mx = margin_frac * dx

    z0 = max(0.0, z_min - mz)
    z1 = min(hcr_z_extent_um, z_max + mz)
    y0 = max(0.0, y_min - my)
    y1 = min(hcr_y_extent_um, y_max + my)
    x0 = max(0.0, x_min - mx)
    x1 = min(hcr_x_extent_um, x_max + mx)

    print(
        f"[overlap_crop] {s.subject_id}: final bbox (margin {margin_frac*100:.0f}%) "
        f"z=[{z0:.0f}, {z1:.0f}] ({z1-z0:.0f} µm) "
        f"y=[{y0:.0f}, {y1:.0f}] ({y1-y0:.0f} µm) "
        f"x=[{x0:.0f}, {x1:.0f}] ({x1-x0:.0f} µm)"
    )

    # ---- Convert to level-2 voxel indices ----
    # Floor for start, ceil for end (inclusive range).
    z0v = int(np.floor(z0 / hcr_z_um))
    z1v = int(np.ceil(z1 / hcr_z_um))
    y0v = int(np.floor(y0 / hcr_xy_um))
    y1v = int(np.ceil(y1 / hcr_xy_um))
    x0v = int(np.floor(x0 / hcr_xy_um))
    x1v = int(np.ceil(x1 / hcr_xy_um))

    # Clip against volume dimensions.
    z0v = max(0, z0v); z1v = min(hcr_shape[0], z1v)
    y0v = max(0, y0v); y1v = min(hcr_shape[1], y1v)
    x0v = max(0, x0v); x1v = min(hcr_shape[2], x1v)

    return dict(
        bbox_hcr_um=[z0, z1, y0, y1, x0, x1],
        bbox_hcr_l2_vox=[z0v, z1v, y0v, y1v, x0v, x1v],
        margin_frac=margin_frac,
        sz_used=sz_used,
        hcr_voxel_um=(hcr_z_um, hcr_xy_um),
        hcr_volume_shape=list(hcr_shape),
        cz_corners_hcr_um=corners_hcr_um.tolist(),
    )


def crop_hcr_volume(
    s,
    vol: np.ndarray,
    margin_frac: float = 0.10,
) -> np.ndarray:
    """Return the slab of ``vol`` (Z, Y, X, level-2) covering the overlap crop.

    ``vol`` must be a 3-D array in level-2 HCR voxel space — e.g. the output
    of ``load_hcr_volume(s, channel="488", level=2)``.  The crop indices are
    the ``bbox_hcr_l2_vox`` values from ``get_overlap_crop``.
    """
    crop = get_overlap_crop(s, margin_frac=margin_frac)
    z0, z1, y0, y1, x0, x1 = crop["bbox_hcr_l2_vox"]
    return vol[z0:z1, y0:y1, x0:x1]
