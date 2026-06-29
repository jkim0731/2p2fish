"""mfish-autocoreg: automated coregistration of mFISH (HCR) and Ca-imaging (CZ).

Two-stage, GT-free protocol:

initial_registration/    — rough/warm registration (CZ → HCR µm)
    surfaces             — CZ + HCR surface fitting
    lateral_scale        — sxy from ROI cross-sections
    surface_registration — MIP-based surface registration (affine/PWR)
    axial_scale          — sz from FFT-NCC slab sweep
    locked_prior         — locked-prior warm-start pose
    overlap_crop         — overlap bounding box
    coarse_align         — scale-free coarse affine
    surface_detect, cz_surface, hcr_bottom_surface, binarize,
    projections, register_2d, register_nonrigid  — surface/registration helpers

finetune_soma_print/     — 3-D soma-print cell-cell fine matching
    descriptor           — m-NN soma-print descriptor
    pool_prep            — candidate-pool construction (locked-prior frame)
    tps                  — thin-plate-spline warp + neighbour scoring / anchor-vote
    scoring              — vectorised soma-print scoring
    matcher              — production matcher (gate: anchor_vote + Stage-2 anchor_restricted)
    local_ncc, loo_image_ncc  — image-NCC gate helpers

io/                      — shared data loading
    subjects, inputs, centroids, hcr_image, cz_volume, gfp_threshold

qc/                      — PyQt5 QC viewer + artifacts (requires [qc] extras)

archive/                 — superseded protocols kept for comparison
    shape_context, oracle_benchmark, locked_benchmark,
    refined_benchmark, iterative_matcher

benchmark/               — GT-based validation API (validation data only)
"""
__version__ = "0.1.0"
