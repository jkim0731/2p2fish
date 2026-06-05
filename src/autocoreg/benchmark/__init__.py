"""Validation-only helpers for the benchmark dataset.

The core pipeline (surfaces → sxy → registration → sz → matcher) does NOT
import anything from this subpackage.  Benchmark/ is for GT-based evaluation
only and requires the coreg tables, landmark files, and sz_pins CSV that are
not present for new subjects.

Public API mirrors the GT-gated items from autocoreg.data:
    BENCHMARK_SUBJECTS
    load_sz_pins()         — reads SZ_TABLE_CSV (validation data)
    scoring_gt(inp)        — pose-independent GT {cz_id: hcr_id}
    strict_gfp_ids(sid)    — 07b GMM-intersection GFP+ (validation)
"""
from ..data import (
    BENCHMARK_SUBJECTS,
    load_sz_pins,
    scoring_gt,
    strict_gfp_ids,
    subject_inputs,
)

__all__ = [
    "BENCHMARK_SUBJECTS",
    "load_sz_pins",
    "scoring_gt",
    "strict_gfp_ids",
    "subject_inputs",
]
