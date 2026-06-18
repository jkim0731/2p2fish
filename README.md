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
| `MFISH_QC_ARTIFACT_DIR` | `/tmp/autocoreg_outputs/qc` | Where QC label volumes + warped CZ are written/read (`<variant>/<sid>/`) |
| `MFISH_QC_LABELS_DIR` | `/tmp/autocoreg_outputs/qc_labels` | QC pass/fail labels + manual-match logs |

## Run

```bash
# Full pipeline on one subject (GT-free): rough reg -> matcher
autocoreg run 790322

# + build the QC input artifacts (seg volumes + warped CZ), no viewer:
autocoreg run 790322 --build-qc

# + build artifacts and launch the QC viewer:
autocoreg run 790322 --qc
```

## Pipeline stages

1. Surface fitting (CZ iter08 + HCR iter07)
2. sxy estimation — min-rule 2× quarter-FOV ROI cross-section ratio
3. Surface registration — MIP-based affine/PWR on 488 channel
4. sz estimation — FFT-NCC slab-side-view sweep (`get_sz()["sz_best"]`)
5. Cell matching — `run_step3_v3` (anchor_vote gate + local-flow rd0 + Wang
   Stage-2), GT-free → `step3_v3_anchor_vote_wang_end/<sid>/matches_*.csv`
6. QC artifacts (`--qc`/`--build-qc`) — `qc/build_artifacts.py` warps the CZ
   label + 488 image into HCR µm and crops the HCR seg, from the matcher's
   final-round matches; outputs feed `qc/app.py`
7. QC viewer (`--qc`) — pass/fail labeller + manual add-match mode (press `a`)

The matcher takes ~40 s/subject (warm caches); building QC artifacts is the slow
step (~15 min, dominated by the per-plane CZ inverse-TPS warp).

## Data contract with mfish-roi-classifier (repo B)

Place `{sid}_stage2_4class_proba_v5d_um.parquet` in `MFISH_ROI_QUALITY_DIR`.
Columns: `hcr_id, p_bad, p_bad_ok, p_good, p_merged`.
