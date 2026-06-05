# mfish-autocoreg

Automated coregistration pipeline for matching calcium-imaging (CZ) cells to
multiplexed FISH (HCR) cells in mouse cortex.

## Install

```bash
pip install -e .
# With QC viewer (PyQt5 required):
pip install -e ".[qc]"
```

## Configure

| Env var | Default | Description |
|---|---|---|
| `MFISH_DATA_ROOT` | `/root/capsule/data` | Root of the data tree |
| `MFISH_CACHE_DIR` | `/root/capsule/code/dev_code` | Root cache dir (cached_* live here) |
| `MFISH_ROI_QUALITY_DIR` | `$MFISH_CACHE_DIR/cached_roi_quality` | v5d parquets from repo B |

## Run

```bash
# Full pipeline on one subject (GT-free):
autocoreg run 790322

# With QC viewer:
autocoreg run 790322 --qc
```

## Pipeline stages

1. Surface fitting (CZ iter08 + HCR iter07)
2. sxy estimation — min-rule 2× quarter-FOV ROI cross-section ratio
3. Surface registration — MIP-based affine/PWR on 488 channel
4. sz estimation — FFT-NCC slab-side-view sweep
5. Cell matching — soma-print + shape-context + local NCC

## Data contract with mfish-roi-classifier (repo B)

Place `{sid}_stage2_4class_proba_v5d_um.parquet` in `MFISH_ROI_QUALITY_DIR`.
Columns: `hcr_id, p_bad, p_bad_ok, p_good, p_merged`.
