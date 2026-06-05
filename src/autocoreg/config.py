"""Central path configuration for mfish-autocoreg.

All cache and data directories are resolved from environment variables with
fallbacks that match the original capsule layout so the package works
read-only against existing caches when MFISH_CACHE_DIR is not overridden.

Environment variables:
    MFISH_DATA_ROOT     Root of the data tree (default: /root/capsule/data)
    MFISH_CACHE_DIR     Root of mutable cache dirs (default:
                        /root/capsule/code/dev_code so existing cached_*
                        dirs are found transparently)
    MFISH_ROI_QUALITY_DIR   Directory holding per-subject v5d parquet files
                            produced by repo B (default:
                            MFISH_CACHE_DIR/cached_roi_quality)
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Primary roots
# ---------------------------------------------------------------------------
DATA_ROOT = Path(os.environ.get("MFISH_DATA_ROOT", "/root/capsule/data"))

# Default cache root: the existing dev_code dir so cached_surfaces, cached_sz,
# cached_surface_registration, cached_hcr_cell_* dirs are found without any
# env-var configuration on the capsule machine.
_DEFAULT_CACHE = "/root/capsule/code/dev_code"
CACHE_DIR = Path(os.environ.get("MFISH_CACHE_DIR", _DEFAULT_CACHE))

# ---------------------------------------------------------------------------
# Contract dir (repo B → repo A interface)
# ---------------------------------------------------------------------------
_DEFAULT_ROI_QUALITY = str(CACHE_DIR / "cached_roi_quality")
ROI_QUALITY_DIR = Path(
    os.environ.get("MFISH_ROI_QUALITY_DIR", _DEFAULT_ROI_QUALITY)
)

# ---------------------------------------------------------------------------
# Per-module cache subdirs (created on first use by their owning modules)
# ---------------------------------------------------------------------------
# surfaces_iter08.py
SURFACES_CACHE_DIR = CACHE_DIR / "cached_surfaces"

# surface_registration_v2.py
SURFACE_REGISTRATION_CACHE_DIR = CACHE_DIR / "cached_surface_registration"

# roi_area_sxy.py (tight bbox and max-xsection per-cell caches)
HCR_TIGHT_BBOX_CACHE_DIR = CACHE_DIR / "cached_hcr_cell_tight_bbox"
HCR_MAX_XSECTION_CACHE_DIR = CACHE_DIR / "cached_hcr_cell_max_xsection"

# sz_estimator.py
SZ_CACHE_DIR = CACHE_DIR / "cached_sz"

# ---------------------------------------------------------------------------
# Optional data files (non-fatal if absent; module should degrade gracefully)
# ---------------------------------------------------------------------------
# GMM threshold JSON produced by session 07b.  Used by strict_gfp_ids().
# Falls back to per-subject GMM recompute when absent.
_DATA_RO = Path(
    os.environ.get(
        "MFISH_DATA_RO",
        str(DATA_ROOT / "claude-data_ophys-mfish-autocoreg_260503"),
    )
)
GMM_THRESH_JSON = Path(
    os.environ.get(
        "MFISH_GMM_THRESH_JSON",
        str(_DATA_RO / "sessions/07b_scale_clean_gfp/gmm_threshold_results.json"),
    )
)

# sz pin CSV (validation only — used only when sz_pins kwarg passed from benchmark/)
SZ_TABLE_CSV = Path(
    os.environ.get(
        "MFISH_SZ_TABLE_CSV",
        str(
            _DATA_RO
            / "full_automatic_execution_02/sessions/v2_S02_sz_image_sweep/"
            "results_iter7_summary.csv"
        ),
    )
)

# cell_data_mean CSVs for intensity-only subjects (755252, 767022)
DATA_LEVEL = Path(os.environ.get("MFISH_DATA_LEVEL", str(_DATA_RO)))
