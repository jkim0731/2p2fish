"""Minimal CLI entry point for mfish-autocoreg.

Usage:
    autocoreg run <subject_id>  [--qc] [--out-dir DIR]
    autocoreg run 790322 --qc

Pipeline stages executed by ``run``:
    1. Surface fitting  (surfaces_iter08)
    2. sxy estimation   (roi_area_sxy.estimate_sxy_min_rule)
    3. Surface registration (surface_registration_v2)
    4. sz estimation    (sz_estimator.get_sz)
    5. Matcher          (run_step3_v3.run_subject) [if available]

``--qc`` launches qc.app after matching completes.
"""
from __future__ import annotations

import argparse
import sys


def _run(sid: str, out_dir: str | None, launch_qc: bool) -> None:
    from .benchmark_data_loader import load_subject
    from .surfaces_iter08 import get_cz_surface_iter08, get_hcr_top_surface_iter07
    from .roi_area_sxy import estimate_sxy_min_rule
    from .surface_registration_v2 import get_surface_registration
    from .sz_estimator import get_sz

    print(f"[autocoreg] loading subject {sid}")
    s = load_subject(sid)

    print(f"[autocoreg] fitting surfaces")
    cz_surf = get_cz_surface_iter08(s)
    hcr_surf = get_hcr_top_surface_iter07(s)
    print(f"  CZ surface: {cz_surf is not None}  HCR top surface: {hcr_surf is not None}")

    print(f"[autocoreg] estimating sxy (min-rule)")
    sxy_result = estimate_sxy_min_rule(sid)
    sxy = sxy_result["sxy_median"]
    print(f"  sxy = {sxy:.4f}  (method={sxy_result['method']})")

    print(f"[autocoreg] surface registration")
    reg = get_surface_registration(s)
    print(f"  registration method: {reg.get('method') if reg else 'None'}")

    print(f"[autocoreg] estimating sz")
    sz = get_sz(s)
    print(f"  sz = {sz:.4f}")

    print(f"[autocoreg] pipeline complete for {sid}: sxy={sxy:.4f} sz={sz:.4f}")

    if launch_qc:
        try:
            from PyQt5 import QtWidgets
        except ImportError:
            print("[autocoreg] ERROR: --qc requires PyQt5; install with: pip install mfish-autocoreg[qc]")
            sys.exit(1)
        from .qc.app import main as qc_main
        qc_main(sid)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="autocoreg",
        description="mFISH automated coregistration pipeline",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run the full pipeline on a subject")
    run_parser.add_argument("subject_id", help="Subject ID (e.g. 790322)")
    run_parser.add_argument("--qc", action="store_true", help="Launch QC viewer after matching")
    run_parser.add_argument("--out-dir", default=None, help="Output directory for results")

    args = parser.parse_args()

    if args.command == "run":
        _run(args.subject_id, args.out_dir, args.qc)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
