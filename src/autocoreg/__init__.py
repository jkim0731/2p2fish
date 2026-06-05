"""mfish-autocoreg: automated coregistration pipeline for mFISH / Ca-imaging data.

Core pipeline (GT-free):
    surfaces_iter08     — CZ + HCR surface fitting (iter07/08)
    surface_registration_v2 — MIP-based surface registration (affine/PWR)
    roi_area_sxy        — sxy estimation from ROI cross-sections
    sz_estimator        — sz estimation from FFT-NCC slab sweep
    locked_prior_warm   — locked-prior warm-start pose
    overlap_crop        — overlap bounding box

Matching:
    data                — SubjectInputs bundle (GT-optional)
    soma_print, shape_context, local_ncc, loo_image_ncc
    run_step1_oracle, run_step2_locked, run_step2p5_refined,
    run_step3_iterative, run_step3_v3

QC (requires [qc] extras):
    qc.app              — PyQt5 viewer

Validation only (benchmark/):
    benchmark.scoring_gt, benchmark.BENCHMARK_SUBJECTS, etc.
"""
__version__ = "0.1.0"
