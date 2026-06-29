"""Leave-one-out image NCC δ helper for Step 3 v2 Path B.

For a candidate pair (i, j) in the current accepted set:
  • Take the K=20 nearest accepted anchors (CZ side) other than i
  • Fit two local INVERSE TPS warps:
      TPS_plus  = HCR µm → CZ µm using the K neighbours + (i, j)
      TPS_minus = HCR µm → CZ µm using only the K neighbours
  • Sample a `patch_um` cube around HCR cell j on a `step_um` grid
  • Look up the CZ image at the back-warped points (linear interp) for both
    TPS_plus and TPS_minus
  • Look up the HCR-488 patch in HCR µm directly
  • Δ_NCC(i, j) = NCC(warped_plus, hcr_patch) − NCC(warped_minus, hcr_patch)

Δ_NCC > 0 → adding (i, j) makes the local warp look more like HCR-488.
Filter: keep pairs with Δ_NCC ≥ threshold (default 0).

3-slab mode (n_slabs=3):
  Instead of a single z-MIP NCC, compute the mean of three orthogonal slabs:
    z-slab : 160×160 µm view-plane × 24 µm thin, MIP over z → xy image
    y-slab : 160×160 µm view-plane × 24 µm thin, MIP over y → xz image
    x-slab : 160×160 µm view-plane × 24 µm thin, MIP over x → yz image
  The TPS is fit once per (plus/minus) and evaluated across all three slab
  grids in a single concatenated call (same pattern as nearmiss_ncc_patchwidth).
  δ = mean3slab(NCC_plus) − mean3slab(NCC_minus).
  This scored 86.4% in pre-checks vs 82% for the single z-slab.

Pia cap (pia_surf != None):
  Grid voxels whose HCR z-coordinate is LESS THAN the pia surface z at that
  (x, y) position are above the pia (empty tissue) and are zeroed in both
  HCR and CZ patches before MIP.  Convention: pia_z = a*x + b*y + c + p*x²
  + q*x*y + r*y²; z < pia_z means above (shallower than) the pia.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import Rbf
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    a -= a.mean(); b -= b.mean()
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _fit_inverse_tps(src_hcr, dst_cz):
    """Fit per-axis thin-plate Rbf taking HCR µm → CZ-native µm."""
    if len(src_hcr) < 4:
        return None
    try:
        rz = Rbf(src_hcr[:, 0], src_hcr[:, 1], src_hcr[:, 2], dst_cz[:, 0],
                  function="thin_plate")
        ry = Rbf(src_hcr[:, 0], src_hcr[:, 1], src_hcr[:, 2], dst_cz[:, 1],
                  function="thin_plate")
        rx = Rbf(src_hcr[:, 0], src_hcr[:, 1], src_hcr[:, 2], dst_cz[:, 2],
                  function="thin_plate")
        return rz, ry, rx
    except Exception:
        return None


def _sample_image(vol, grid_zyx_world, origin_world, vox_size):
    """Linear-interp sample of a (Z, Y, X) volume at world µm positions.

    origin_world: (oz, oy, ox) µm position of voxel (0, 0, 0)
    vox_size: (vz, vy, vx) µm/voxel
    grid_zyx_world: (3, N) array of world µm points
    Returns sampled values as a 1-D array of length N.
    """
    oz, oy, ox = origin_world
    vz, vy, vx = vox_size
    coords = np.empty_like(grid_zyx_world)
    coords[0] = (grid_zyx_world[0] - oz) / vz
    coords[1] = (grid_zyx_world[1] - oy) / vy
    coords[2] = (grid_zyx_world[2] - ox) / vx
    return map_coordinates(vol, coords, order=1, mode="constant", cval=0.0)


def _pia_z_at_xy(surf: dict, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Evaluate HCR pia z at (x, y) positions (HCR µm axis-2 and axis-1).

    Convention matches run_step2p5_refined.hcr_pia_z_over_region:
      x = HCR µm axis-2 (X), y = HCR µm axis-1 (Y).
    pia_z = a*x + b*y + c + p*x² + q*x*y + r*y²
    z < pia_z ⟹ the point is above (shallower than) the pia surface.
    """
    a = surf["a"]; b = surf["b"]; c = surf["c"]
    p = surf.get("p", 0.0); q = surf.get("q", 0.0); r_c = surf.get("r", 0.0)
    return a * x + b * y + c + p * x * x + q * x * y + r_c * y * y


def _build_slab_pts(
    centre_zyx: np.ndarray,
    slab_axis: int,
    patch_xy_um: float,
    patch_z_um: float,
    step_um: float,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Return (pts_zyx (3, N), shape (n_thin, n_a, n_b)) for one oriented slab.

    slab_axis: 0=z (xy view), 1=y (xz view), 2=x (yz view).
    View-plane axes are the two axes other than slab_axis, in order (lower idx first).
    View-plane size: patch_xy_um × patch_xy_um.  Thin-axis size: patch_z_um.
    """
    half_vp = patch_xy_um / 2.0
    half_th = patch_z_um / 2.0
    n_vp = max(2, int(round(patch_xy_um / step_um)))
    n_th = max(2, int(round(patch_z_um / step_um)))

    coords_vp = np.linspace(-half_vp, half_vp, n_vp)
    coords_th = np.linspace(-half_th, half_th, n_th)

    vp_axes = [a for a in (0, 1, 2) if a != slab_axis]

    A_th, A_a, A_b = np.meshgrid(coords_th, coords_vp, coords_vp, indexing="ij")

    coord_zyx: list = [None, None, None]
    coord_zyx[slab_axis]  = A_th.ravel() + centre_zyx[slab_axis]
    coord_zyx[vp_axes[0]] = A_a.ravel() + centre_zyx[vp_axes[0]]
    coord_zyx[vp_axes[1]] = A_b.ravel() + centre_zyx[vp_axes[1]]

    pts = np.stack([coord_zyx[0], coord_zyx[1], coord_zyx[2]])  # (3, N)
    return pts, (n_th, n_vp, n_vp)


def _mip_ncc_from_samples(
    hcr_flat: np.ndarray,
    cz_flat: np.ndarray,
    pts: np.ndarray,
    shape: tuple[int, int, int],
    pia_surf: dict | None,
) -> float:
    """MIP + NCC from pre-sampled flat arrays, with optional pia masking.

    Above-pia voxels (pts[0] < pia_z(pts[2], pts[1])) are zeroed before MIP.
    """
    n_th, n_a, n_b = shape
    if pia_surf is not None:
        # pts[0]=z, pts[1]=y, pts[2]=x in HCR µm
        pia_z_pts = _pia_z_at_xy(pia_surf, pts[2], pts[1])
        above_pia = pts[0] < pia_z_pts
        hcr_flat = hcr_flat.copy()
        cz_flat  = cz_flat.copy()
        hcr_flat[above_pia] = 0.0
        cz_flat[above_pia]  = 0.0
    hcr_2d = hcr_flat.reshape(n_th, n_a, n_b).max(axis=0)
    cz_2d  = cz_flat.reshape(n_th, n_a, n_b).max(axis=0)
    return _ncc(cz_2d, hcr_2d)


def _three_slab_mean_ncc(
    tps,
    centre_hcr: np.ndarray,
    hcr488_vol: np.ndarray,
    hcr488_origin: tuple,
    hcr488_voxel: tuple,
    cz_vol: np.ndarray,
    cz_voxel: tuple,
    patch_xy_um: float,
    patch_z_um: float,
    step_um: float,
    pia_surf: dict | None,
) -> float:
    """Compute mean of z/y/x-slab MIP-NCCs with a single batched TPS evaluation.

    All three slab grids are concatenated, the TPS is called once per component,
    then the results are split back and each slab is MIP-NCC'd.
    Returns (ncc_z + ncc_y + ncc_x) / 3.
    """
    pts_list: list[np.ndarray] = []
    shape_list: list[tuple[int, int, int]] = []
    counts: list[int] = []

    for slab_axis in (0, 1, 2):
        pts, shape = _build_slab_pts(centre_hcr, slab_axis, patch_xy_um,
                                     patch_z_um, step_um)
        pts_list.append(pts)
        shape_list.append(shape)
        counts.append(pts.shape[1])

    all_pts = np.concatenate(pts_list, axis=1)  # (3, N_total)

    # Sample HCR 488 at all points in one call
    all_hcr = _sample_image(hcr488_vol, all_pts, hcr488_origin, hcr488_voxel)

    # Warp to CZ native µm — one TPS call per axis component
    rz, ry, rx = tps
    all_cz_pts = np.stack([rz(all_pts[0], all_pts[1], all_pts[2]),
                           ry(all_pts[0], all_pts[1], all_pts[2]),
                           rx(all_pts[0], all_pts[1], all_pts[2])])
    all_cz = _sample_image(cz_vol, all_cz_pts,
                           origin_world=(0.0, 0.0, 0.0),
                           vox_size=cz_voxel)

    # Split back and compute per-slab MIP-NCC
    ncc_vals: list[float] = []
    offset = 0
    for pts, shape, cnt in zip(pts_list, shape_list, counts):
        hcr_seg = all_hcr[offset: offset + cnt]
        cz_seg  = all_cz[offset:  offset + cnt]
        offset += cnt
        ncc_vals.append(_mip_ncc_from_samples(hcr_seg, cz_seg, pts, shape, pia_surf))

    return float(sum(ncc_vals) / 3.0)


def loo_delta_ncc(
    pair_idx: tuple[int, int],
    accepted: set[tuple[int, int]],
    cz_zyx_native_um: np.ndarray,
    hcr_zyx_um: np.ndarray,
    hcr488_vol: np.ndarray,
    hcr488_origin: tuple,
    hcr488_voxel: tuple,
    cz_vol: np.ndarray,
    cz_voxel: tuple,
    *,
    k_neighbours: int = 20,
    k_skip_nearest: int = 5,
    skip_radius_um: float | None = None,
    patch_xy_um: float = 160.0,
    patch_z_um: float = 24.0,
    step_um: float = 4.0,
    mip_2d: bool = True,
    n_slabs: int = 1,
    pia_surf: dict | None = None,
) -> float | None:
    """LOO Δ_NCC for candidate pair (i, j).

    Anchor-skip modes (choose one):
      skip_radius_um (preferred): exclude all accepted anchors whose HCR
        position is within `skip_radius_um` of the candidate.  Density-
        independent — consistent with the fixed physical R_cand design.
        The K nearest remaining anchors are then used for the local TPS.
      k_skip_nearest (legacy fallback, used when skip_radius_um is None):
        take the K nearest anchors, then DROP the `k_skip_nearest` closest.

    n_slabs=1 (default, backward compat):
      Single z-MIP NCC patch (160×160 µm xy × 24 µm z, MIP over z).
      mip_2d flag applies; pia_surf is ignored.

    n_slabs=3 (production NCC gate):
      Mean of three orthogonal slab MIP-NCCs (z→xy, y→xz, x→yz; each
      160×160 µm view-plane × 24 µm thin, 4 µm step).  The TPS is fit once
      and evaluated across all three slabs in a single batched call.
      If pia_surf is provided, above-pia voxels are zeroed before MIP.
      δ = mean3slab(NCC_plus) − mean3slab(NCC_minus).

    Returns Δ_NCC or None if not computable (out of bounds / too few anchors).
    """
    i, j = pair_idx
    others = [(a, b) for (a, b) in accepted if a != i]
    others_cz_idx = np.array([a for (a, _) in others], dtype=int)
    others_hcr_idx = np.array([b for (_, b) in others], dtype=int)
    centre_hcr = hcr_zyx_um[j]
    others_hcr_pos = hcr_zyx_um[others_hcr_idx]

    if skip_radius_um is not None:
        # Physical-radius skip: exclude anchors too close to the candidate.
        # The remaining pool is then queried for the K nearest supports.
        dists_to_cand = np.linalg.norm(others_hcr_pos - centre_hcr, axis=1)
        far_mask = dists_to_cand > skip_radius_um
        far_cz_idx = others_cz_idx[far_mask]
        far_hcr_idx = others_hcr_idx[far_mask]
        far_hcr_pos = others_hcr_pos[far_mask]
        if len(far_hcr_pos) < k_neighbours:
            return None
        tree = cKDTree(far_hcr_pos)
        _, nbr_local = tree.query(centre_hcr, k=min(k_neighbours, len(far_hcr_pos)))
        nbr_local = np.atleast_1d(nbr_local)
        if len(nbr_local) < 6:
            return None
        cz_nbr_idx = far_cz_idx[nbr_local]
        hcr_nbr_idx = far_hcr_idx[nbr_local]
    else:
        # Legacy k_skip_nearest path (v2 default).
        if len(others) < k_neighbours:
            return None
        tree = cKDTree(others_hcr_pos)
        _, nbr_local = tree.query(centre_hcr, k=min(k_neighbours, len(others)))
        nbr_local = np.atleast_1d(nbr_local)
        # Drop the k_skip_nearest closest anchors — see original docstring.
        if k_skip_nearest > 0 and len(nbr_local) > k_skip_nearest:
            nbr_local = nbr_local[k_skip_nearest:]
        if len(nbr_local) < 6:
            return None
        cz_nbr_idx = others_cz_idx[nbr_local]
        hcr_nbr_idx = others_hcr_idx[nbr_local]

    hcr_minus = hcr_zyx_um[hcr_nbr_idx]
    cz_minus  = cz_zyx_native_um[cz_nbr_idx]
    hcr_plus  = np.vstack([hcr_minus, hcr_zyx_um[j][None]])
    cz_plus   = np.vstack([cz_minus, cz_zyx_native_um[i][None]])

    tps_plus  = _fit_inverse_tps(hcr_plus, cz_plus)
    tps_minus = _fit_inverse_tps(hcr_minus, cz_minus)
    if tps_plus is None or tps_minus is None:
        return None

    if n_slabs == 3:
        # 3-slab mean: batched TPS evaluation across z/y/x slabs.
        # pia_surf applied per-slab inside _three_slab_mean_ncc.
        ncc_p = _three_slab_mean_ncc(
            tps_plus, centre_hcr,
            hcr488_vol, hcr488_origin, hcr488_voxel,
            cz_vol, cz_voxel,
            patch_xy_um, patch_z_um, step_um, pia_surf,
        )
        ncc_m = _three_slab_mean_ncc(
            tps_minus, centre_hcr,
            hcr488_vol, hcr488_origin, hcr488_voxel,
            cz_vol, cz_voxel,
            patch_xy_um, patch_z_um, step_um, pia_surf,
        )
        return ncc_p - ncc_m

    # Single z-slab path (n_slabs=1, original behaviour).
    half_xy = patch_xy_um / 2.0
    half_z = patch_z_um / 2.0
    n_xy = max(2, int(round(patch_xy_um / step_um)))
    n_z = max(2, int(round(patch_z_um / step_um)))
    coords_xy = np.linspace(-half_xy, half_xy, n_xy)
    coords_z = np.linspace(-half_z, half_z, n_z)
    Z, Y, X = np.meshgrid(coords_z, coords_xy, coords_xy, indexing="ij")
    pts_world = np.stack([Z.ravel() + centre_hcr[0],
                          Y.ravel() + centre_hcr[1],
                          X.ravel() + centre_hcr[2]])

    hcr_patch_flat = _sample_image(hcr488_vol, pts_world, hcr488_origin, hcr488_voxel)

    def warp_cz(tps):
        rz, ry, rx = tps
        cz_z = rz(pts_world[0], pts_world[1], pts_world[2])
        cz_y = ry(pts_world[0], pts_world[1], pts_world[2])
        cz_x = rx(pts_world[0], pts_world[1], pts_world[2])
        cz_pts = np.stack([cz_z, cz_y, cz_x])
        return _sample_image(
            cz_vol, cz_pts,
            origin_world=(0.0, 0.0, 0.0),
            vox_size=cz_voxel,
        )

    warped_plus_flat  = warp_cz(tps_plus)
    warped_minus_flat = warp_cz(tps_minus)

    if mip_2d:
        # Reshape to (n_z, n_xy, n_xy), MIP over z
        hcr_patch = hcr_patch_flat.reshape(n_z, n_xy, n_xy).max(axis=0)
        warped_plus = warped_plus_flat.reshape(n_z, n_xy, n_xy).max(axis=0)
        warped_minus = warped_minus_flat.reshape(n_z, n_xy, n_xy).max(axis=0)
    else:
        hcr_patch = hcr_patch_flat
        warped_plus = warped_plus_flat
        warped_minus = warped_minus_flat

    return _ncc(warped_plus, hcr_patch) - _ncc(warped_minus, hcr_patch)
