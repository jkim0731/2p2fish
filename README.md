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
| `MFISH_ROI_QUALITY_DIR` | `$MFISH_CACHE_DIR/cached_roi_quality` | per-subject ROI-quality parquet from repo B (auto-resolves an attached `*_HCR-ROI-label_*` asset if present) |
| `MFISH_MATCHER_OUT_BASE` | `/scratch/autocoreg_outputs/matches` | Matcher CSV output root (`<variant>/<sid>/`) |
| `MFISH_QC_ARTIFACT_DIR` | `/scratch/autocoreg_outputs/qc` | QC label volumes + warped CZ (`<variant>/<sid>/`) |
| `MFISH_QC_LABELS_DIR` | `/scratch/autocoreg_outputs/qc_labels` | QC pass/fail labels + manual-match logs |
| `MFISH_QC_CACHE_DIR` | `/scratch/autocoreg_qc_cache` | QC viewer launch caches (`<sid>/`) |

Outputs default to `/scratch` (never `/tmp` or `/`).

## Run

```bash
# Full pipeline on one subject (GT-free): rough reg -> matcher
autocoreg run 790322

# + build the QC input artifacts (seg volumes + warped CZ), no viewer:
autocoreg run 790322 --build-qc

# + build artifacts and launch the QC viewer:
autocoreg run 790322 --qc

# QC viewer with a subject picker (labeled subjects auto-discovered):
autocoreg_qc

# Matcher options:
autocoreg run 790322 --gate {likelihood_ratio,anchor_vote,ncc}   # default anchor_vote
autocoreg run 790322 --no-anchor-restricted                      # skip the Stage-2 addendum
```

## Package layout

```
autocoreg/
  initial_registration/   rough/warm registration (CZ -> HCR µm):
                          surfaces · lateral_scale (sxy) · surface_registration ·
                          axial_scale (sz) · locked_prior · overlap_crop · coarse_align ·
                          surface_detect · cz_surface · hcr_bottom_surface · binarize ·
                          projections · register_2d · register_nonrigid
  finetune_soma_print/    3-D soma-print cell-cell fine matching:
                          descriptor · pool_prep · tps · scoring · matcher ·
                          local_ncc · loo_image_ncc
  io/                     shared loaders: subjects · inputs · centroids ·
                          hcr_image · cz_volume · gfp_threshold
  qc/                     PyQt5 QC viewer (app) + artifact builder + launcher
  archive/                superseded protocols kept for comparison, never imported by
                          production: shape_context · oracle_benchmark · locked_benchmark ·
                          refined_benchmark · iterative_matcher
  benchmark/              GT-based validation API (validation data only)
```

## Pipeline stages

1. Surface fitting — `initial_registration.surfaces` (CZ + HCR)
2. sxy estimation — `initial_registration.lateral_scale.estimate_sxy_min_rule`
   (min-rule 2× quarter-FOV ROI cross-section ratio)
3. Surface registration — `initial_registration.surface_registration` (MIP-based
   affine/PWR on the 488 channel)
4. sz estimation — `initial_registration.axial_scale.get_sz` (FFT-NCC slab-side-view sweep)
5. Cell matching — `finetune_soma_print.matcher.run_subject`: candidate radius →
   mutual-best → round-0 local-flow → `anchor_vote` gate → iterative TPS, then the
   Stage-2 `anchor_restricted` addendum (re-describe each cell from its nearest accepted
   anchors). GT-free → `anchor_vote_anchor_restricted/<sid>/matches_anchor_restricted_round*.csv`
6. QC artifacts (`--qc`/`--build-qc`) — `qc/build_artifacts.py` warps the CZ label +
   488 image into HCR µm and crops the HCR seg from the matcher's final-round matches;
   outputs feed `qc/app.py`
7. QC viewer (`--qc` or `autocoreg_qc`) — least-confident-first pass/fail labeller +
   manual add-match mode

The matcher takes ~40–70 s/subject (warm caches); building QC artifacts is the slow
step (~15 min, dominated by the per-plane CZ inverse-TPS warp).

The soma-print descriptor follows Wang et al. (2026), adapted — see
`finetune_soma_print/descriptor.py`.

## Data contract with mfish-roi-classifier (repo B)

Place the per-subject ROI-quality parquet (`{sid}_roi_quality_proba.parquet`, columns
`hcr_id, p_bad, p_bad_ok, p_good, p_merged`) in `MFISH_ROI_QUALITY_DIR`, or attach the
labeled `*_HCR-ROI-label_*` data asset and it resolves automatically.
