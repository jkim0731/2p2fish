"""Compare HCR ch488 binarisation variants across all subjects.

For each subject we render a row of panels:
  raw 488 MIP (log) | current (σ=4 µm, p88) | tight (σ=2 µm, p92) |
  current+erode (1 px) | h-maxima watershed (split touching blobs)

Goal: pick a binarisation that keeps GFP+ cell-bodies separated rather
than fusing them into shapeless blobs (which is what the current
register_binary pipeline often does for densely-labelled subjects).

Output:
  • figures/compare_binarization_grid.png — 6 rows × 5 cols
  • compare_binarization_summary.csv — blob count / mean area / fill
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import (
    binary_erosion,
    binary_fill_holes,
    binary_opening,
    distance_transform_edt,
    gaussian_filter,
    label,
)
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from .benchmark_data_loader import load_subject
from .surfaces_iter08 import get_hcr_top_surface_iter07
from .register_binary import detect_objects_binary, hcr488_top_mip

HERE = Path(__file__).resolve().parent
FIG = Path("/tmp")  # figures go to /tmp in package context (was HERE/figures)
SUBJECTS = ["788406", "790322", "767018", "782149", "755252", "767022"]


# --------------------------------------------------------------
# Variants
# --------------------------------------------------------------
def variant_current(mip, xy_um):
    """Existing register_binary pipeline (σ=4, p88, open, fill, ≥30 µm²)."""
    return detect_objects_binary(mip, xy_um, sigma_um=4.0, pct=88.0,
                                 min_area_um2=30.0)


def variant_tight(mip, xy_um, sigma_um=2.0, pct=92.0, min_area_um2=20.0):
    """Smaller Gaussian (closer to single-cell scale) + higher percentile.

    With σ≈cell radius/2 nearby cells aren't blurred together, and the
    higher percentile (p92) keeps only brighter cores; min-area lowered
    accordingly so we don't lose them again."""
    a = np.asarray(mip, dtype=np.float32)
    finite = np.isfinite(a)
    a_f = np.where(finite, a, np.nanmin(a[finite]) if finite.any() else 0.0)
    sig_px = max(sigma_um / xy_um, 0.5)
    blur = gaussian_filter(a_f, sigma=sig_px)
    thr = np.percentile(blur[finite], pct)
    raw = (blur > thr).astype(np.uint8)
    raw = binary_opening(raw, iterations=1).astype(np.uint8)
    # NOTE: skip fill_holes — it merges touching blobs
    lab, n = label(raw)
    if n == 0:
        return raw
    sizes = np.bincount(lab.ravel())[1:]
    keep = np.where(sizes * (xy_um ** 2) >= min_area_um2)[0] + 1
    return np.isin(lab, keep).astype(np.uint8)


def variant_current_erode(mip, xy_um, erode_px=1, min_area_um2=20.0):
    """Current binarisation followed by 1-px binary erosion to break thin
    bridges between touching cells."""
    base = variant_current(mip, xy_um)
    if base.sum() == 0:
        return base
    eroded = binary_erosion(base, iterations=erode_px).astype(np.uint8)
    lab, n = label(eroded)
    if n == 0:
        return eroded
    sizes = np.bincount(lab.ravel())[1:]
    keep = np.where(sizes * (xy_um ** 2) >= min_area_um2)[0] + 1
    return np.isin(lab, keep).astype(np.uint8)


def variant_watershed(mip, xy_um, sigma_um=2.0, pct=88.0, min_area_um2=15.0,
                      peak_min_dist_um=6.0):
    """h-maxima / watershed split of `variant_current`.

    Uses distance-transform-derived peaks as seeds and watersheds the
    foreground mask from `variant_current` so that two touching cells
    each get their own basin. We then return the watershed-labelled
    mask binarised again (foreground / background) — the *gain* over
    `current` is that two seeds in one merged blob now have a thin
    background line between them, which propagates to RIGID/AFFINE
    NCC by giving each cell a distinct centroid."""
    fg = variant_current(mip, xy_um).astype(bool)
    if fg.sum() == 0:
        return fg.astype(np.uint8)
    # Distance transform inside foreground; peaks ≈ cell centres
    dist = distance_transform_edt(fg)
    min_dist_px = max(int(round(peak_min_dist_um / xy_um)), 2)
    coords = peak_local_max(
        dist, min_distance=min_dist_px, labels=fg, exclude_border=False,
    )
    if coords.size == 0:
        return fg.astype(np.uint8)
    seeds = np.zeros(fg.shape, dtype=np.int32)
    for i, (yy, xx) in enumerate(coords, start=1):
        seeds[yy, xx] = i
    ws = watershed(-dist, markers=seeds, mask=fg, watershed_line=True)
    out = (ws > 0).astype(np.uint8)
    lab, n = label(out)
    if n == 0:
        return out
    sizes = np.bincount(lab.ravel())[1:]
    keep = np.where(sizes * (xy_um ** 2) >= min_area_um2)[0] + 1
    return np.isin(lab, keep).astype(np.uint8)


VARIANTS = [
    ("current", variant_current),
    ("tight (σ=2µm, p92)", variant_tight),
    ("current+erode 1px", variant_current_erode),
    ("watershed split", variant_watershed),
]


def stats(mask, xy_um):
    lab, n = label(mask)
    if n == 0:
        return dict(n_blobs=0, mean_area_um2=0.0, fill_pct=0.0)
    sizes = np.bincount(lab.ravel())[1:]
    return dict(
        n_blobs=int(n),
        mean_area_um2=float(sizes.mean() * xy_um ** 2),
        fill_pct=float(mask.mean() * 100.0),
    )


# --------------------------------------------------------------
# Driver
# --------------------------------------------------------------
def main():
    rows = []
    n_subj = len(SUBJECTS)
    n_col = 1 + len(VARIANTS)
    fig, axes = plt.subplots(n_subj, n_col, figsize=(4 * n_col, 4 * n_subj))
    if n_subj == 1:
        axes = axes[None, :]

    for i, sid in enumerate(SUBJECTS):
        print(f"\n=== {sid} ===", flush=True)
        s = load_subject(sid)
        hcr_surf = get_hcr_top_surface_iter07(s, level=4)
        mip, xy_um = hcr488_top_mip(s, hcr_surf, 0.0, 100.0)

        # Raw 488
        ax = axes[i, 0]
        ax.imshow(np.log1p(np.nan_to_num(mip)), cmap="gray", origin="upper")
        ax.set_ylabel(f"{sid}\nxy={xy_um:.3f} µm", fontsize=9)
        if i == 0:
            ax.set_title("raw 488 (log)")
        ax.set_xticks([]); ax.set_yticks([])

        for j, (name, fn) in enumerate(VARIANTS, start=1):
            mask = fn(mip, xy_um)
            st = stats(mask, xy_um)
            ax = axes[i, j]
            ax.imshow(mask, cmap="gray", origin="upper", vmin=0, vmax=1)
            if i == 0:
                ax.set_title(name)
            ax.set_xlabel(
                f"n={st['n_blobs']}  ⟨A⟩={st['mean_area_um2']:.0f}µm²  "
                f"fill={st['fill_pct']:.1f}%",
                fontsize=8,
            )
            ax.set_xticks([]); ax.set_yticks([])
            rows.append({"sid": sid, "variant": name, **st})
            print(f"  {name:25s}  n={st['n_blobs']:4d}  "
                  f"<A>={st['mean_area_um2']:6.1f} µm²  "
                  f"fill={st['fill_pct']:5.2f}%", flush=True)

    fig.suptitle("HCR ch488 binarisation variants — per-subject comparison",
                 fontsize=14)
    fig.tight_layout()
    out = FIG / "compare_binarization_grid.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out.name}")

    df = pd.DataFrame(rows)
    csv = HERE / "compare_binarization_summary.csv"
    df.to_csv(csv, index=False)
    print(f"wrote {csv.name}")
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
