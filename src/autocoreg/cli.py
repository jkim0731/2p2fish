"""Minimal CLI entry point for mfish-autocoreg.

Usage:
    autocoreg run <subject_id> [--qc] [--build-qc] [--gate GATE] [--no-wang]
    autocoreg run 790322 --qc

Pipeline stages executed by ``run``:
    1. Surface fitting       (surfaces_iter08)
    2. sxy estimation        (roi_area_sxy.estimate_sxy_min_rule)
    3. Surface registration  (surface_registration_v2)
    4. sz estimation         (sz_estimator.get_sz)
    5. Matcher               (run_step3_v3.run_subject)  -> matches CSVs
    6. QC artifacts          (qc.build_artifacts)        [if --qc or --build-qc]
    7. QC viewer             (qc.app)                    [if --qc]

The matcher + artifact builder + viewer all use the same final-round matches
CSV, so the QC app is fed exactly what the matcher produced.  All outputs are
GT-free (sz pinned from the image-based estimator, not the validation table).
"""
from __future__ import annotations

import argparse
import sys


# Production matcher configuration (matches the validated
# `step3_v3_anchor_vote_wang_end` setup).
_DEFAULT_GATE = "anchor_vote"


def _matcher_variant(gate: str, use_local_flow_rd0: bool, wang: bool) -> str:
    """Output-dir / QC-variant name `run_step3_v3` uses for this config."""
    return (f"step3_v3_{gate}"
            + ("" if use_local_flow_rd0 else "_noLF")
            + ("_wang_end" if wang else ""))


def _run(sid: str, out_dir: str | None, launch_qc: bool, *,
         gate: str = _DEFAULT_GATE, wang: bool = True, build_qc: bool = False) -> None:
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
    sz = float(get_sz(s)["sz_best"])  # get_sz returns a dict; sz_best == table sz_peak
    print(f"  sz = {sz:.4f}")

    print(f"[autocoreg] pipeline (rough reg) complete: sxy={sxy:.4f} sz={sz:.4f}")

    # ---- Step 5: matcher (GT-free; sz pinned from the image estimator) ----
    from .run_step3_v3 import run_subject, OUT_BASE
    use_local_flow_rd0 = True
    variant = _matcher_variant(gate, use_local_flow_rd0, wang)
    matcher_out = OUT_BASE / variant
    matcher_out.mkdir(parents=True, exist_ok=True)
    sz_pins = {sid: sz}
    print(f"[autocoreg] matching (gate={gate}, local_flow_rd0={use_local_flow_rd0}, "
          f"wang_addendum={wang}) -> {matcher_out / sid}")
    run_subject(sid, sz_pins, gate=gate, use_local_flow_rd0=use_local_flow_rd0,
                out_dir=matcher_out, wang_addendum=wang)

    from .qc.build_artifacts import final_round_csv, build_qc_artifacts
    matches_csv = final_round_csv(matcher_out / sid)
    print(f"[autocoreg] final-round matches: {matches_csv}")

    # ---- Step 6: build QC artifacts (only when needed) ----
    if launch_qc or build_qc:
        from . import config as _config
        artifact_dir = _config.QC_ARTIFACT_DIR / variant / sid
        print(f"[autocoreg] building QC artifacts -> {artifact_dir}")
        build_qc_artifacts(sid, matches_csv=matches_csv, out_dir=artifact_dir,
                           sz_pins=sz_pins)

    # ---- Step 7: launch the QC viewer ----
    if launch_qc:
        try:
            from PyQt5 import QtWidgets  # noqa: F401
        except ImportError:
            print("[autocoreg] ERROR: --qc requires PyQt5; install with: "
                  "pip install mfish-autocoreg[qc]")
            sys.exit(1)
        from .qc.app import launch as qc_launch
        app, _win = qc_launch(sid, variant=variant, matches_csv=str(matches_csv))
        sys.exit(app.exec_())


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="autocoreg",
        description="mFISH automated coregistration pipeline",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run the full pipeline on a subject")
    run_parser.add_argument("subject_id", help="Subject ID (e.g. 790322)")
    run_parser.add_argument("--qc", action="store_true",
                            help="Build QC artifacts and launch the QC viewer after matching")
    run_parser.add_argument("--build-qc", action="store_true",
                            help="Build QC artifacts after matching but do NOT launch the viewer")
    run_parser.add_argument("--gate", default=_DEFAULT_GATE,
                            choices=["lr", "anchor_vote", "ncc"],
                            help=f"Matcher primary gate (default {_DEFAULT_GATE})")
    run_parser.add_argument("--no-wang", dest="wang", action="store_false",
                            help="Disable the Stage-2 Wang anchor-restricted addendum")
    run_parser.add_argument("--out-dir", default=None, help="Output directory for results")

    args = parser.parse_args()

    if args.command == "run":
        _run(args.subject_id, args.out_dir, args.qc,
             gate=args.gate, wang=args.wang, build_qc=args.build_qc)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
