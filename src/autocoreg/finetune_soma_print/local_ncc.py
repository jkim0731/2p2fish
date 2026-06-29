"""Per-pair local image NCC.

For each candidate match (cz_position_in_HCR_µm, hcr_position_in_HCR_µm),
extract a small 3-D patch from each modality and return Pearson NCC.

* CZ patch: sampled from the original CZ z-stack using the cz cell's
  *original* CZ pixel position (no warping of the image).  Patch in CZ
  voxel coordinates of size (patch_z_um/cz_z_um, patch_xy_um/cz_xy_um, ...).
* HCR patch: sampled from the HCR 488 volume at the matched HCR cell's
  position, in HCR voxel coordinates.

The two patches are NOT in registered coordinates per voxel — they are
both isotropic *in µm* via nearest/linear resample, then NCC is computed
on the resampled patches.

This is a sharper signal than hull-restricted NCC because it tests
whether the actual cells look similar at the matched site, not whether
the larger neighborhood aligns.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates


def _sample_isotropic_patch(
    vol: np.ndarray,
    center_um_zyx: np.ndarray,
    *,
    voxel_um_z: float,
    voxel_um_xy: float,
    patch_um: tuple[float, float, float] = (20.0, 20.0, 20.0),
    target_um: float = 1.0,
    order: int = 1,
) -> np.ndarray | None:
    """Resample a 3-D patch from ``vol`` at ``center_um_zyx`` to an isotropic
    ``target_um`` voxel grid of shape ~ ``patch_um / target_um``.

    Returns the resampled patch as float32, or None if the patch falls
    fully outside the volume.
    """
    pz, py, px = patch_um
    nz = int(round(pz / target_um))
    ny = int(round(py / target_um))
    nx = int(round(px / target_um))
    # Build target grid in µm around center
    z_um = (np.arange(nz) - nz / 2.0) * target_um + center_um_zyx[0]
    y_um = (np.arange(ny) - ny / 2.0) * target_um + center_um_zyx[1]
    x_um = (np.arange(nx) - nx / 2.0) * target_um + center_um_zyx[2]
    # Convert µm coords to voxel coords
    z_vx = z_um / voxel_um_z
    y_vx = y_um / voxel_um_xy
    x_vx = x_um / voxel_um_xy
    # All-pairs grid
    Z, Y, X = np.meshgrid(z_vx, y_vx, x_vx, indexing="ij")
    coords = np.array([Z.ravel(), Y.ravel(), X.ravel()])
    # Reject if entire patch is outside
    if (coords[0].max() < 0 or coords[0].min() >= vol.shape[0]
        or coords[1].max() < 0 or coords[1].min() >= vol.shape[1]
        or coords[2].max() < 0 or coords[2].min() >= vol.shape[2]):
        return None
    samp = map_coordinates(vol, coords, order=order, mode="constant", cval=0.0)
    return samp.reshape(nz, ny, nx).astype(np.float32)


def pearson_ncc(a: np.ndarray, b: np.ndarray, *, min_std: float = 1e-6) -> float:
    a = a.astype(np.float64, copy=False); b = b.astype(np.float64, copy=False)
    a_mean = a.mean(); b_mean = b.mean()
    a_d = a - a_mean; b_d = b - b_mean
    a_n = np.sqrt(np.sum(a_d * a_d))
    b_n = np.sqrt(np.sum(b_d * b_d))
    if a_n < min_std or b_n < min_std:
        return float("nan")
    return float(np.sum(a_d * b_d) / (a_n * b_n))


def per_pair_local_ncc(
    cz_vol: np.ndarray, hcr_vol: np.ndarray,
    cz_centers_um: np.ndarray,    # (N, 3) CZ centers in CZ µm
    hcr_centers_um: np.ndarray,   # (N, 3) HCR centers in HCR µm
    *,
    cz_voxel_um_z: float, cz_voxel_um_xy: float,
    hcr_voxel_um_z: float, hcr_voxel_um_xy: float,
    patch_um: tuple[float, float, float] = (20.0, 20.0, 20.0),
    target_um: float = 1.0,
) -> np.ndarray:
    """Vectorised per-pair local Pearson NCC.  Returns array of NCC values
    (NaN where any patch fell outside its volume).
    """
    n = cz_centers_um.shape[0]
    out = np.full(n, np.nan, dtype=np.float32)
    for k in range(n):
        cz_p = _sample_isotropic_patch(
            cz_vol, cz_centers_um[k],
            voxel_um_z=cz_voxel_um_z, voxel_um_xy=cz_voxel_um_xy,
            patch_um=patch_um, target_um=target_um, order=1,
        )
        if cz_p is None:
            continue
        hcr_p = _sample_isotropic_patch(
            hcr_vol, hcr_centers_um[k],
            voxel_um_z=hcr_voxel_um_z, voxel_um_xy=hcr_voxel_um_xy,
            patch_um=patch_um, target_um=target_um, order=1,
        )
        if hcr_p is None:
            continue
        out[k] = pearson_ncc(cz_p, hcr_p)
    return out
