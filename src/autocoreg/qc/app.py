"""Qt-based per-CZ-ROI QC app.

For each CZ ROI (matched + unmatched), shows a cube view around it:
  • HCR 488 background slice
  • Current CZ ROI contour (yellow, thick)
  • Other CZ ROI contours within the cube (red = matched, magenta = unmatched) — togglable
  • Matched HCR ROI contour (white, thick) if matched
  • Other HCR ROI contours within the cube (green = matched, cyan = unmatched) — togglable

Per-ROI label is logged to outputs/qc_labels/qt_<variant>_<sid>.csv:
  • Matched CZ ROIs: radio → good / bad / unsure
  • Unmatched CZ ROIs: radio → matched roi visible / matched roi not visible

Auto-save on every radio click.

Add-match mode (function 2 — manual registration improvement):
  Toggle with 'a'.  Mimics the BigWarp landmark workflow: the active landmark
  set is seeded from the matcher's accepted pairs and a per-axis thin-plate-
  spline (CZ-native µm -> HCR µm) is fit from it.  Click a CZ ROI (snaps to the
  nearest warped CZ centroid, gold markers) then its HCR cell (label under the
  cursor, else nearest HCR centroid); "Add pair" appends the landmark, refits
  the TPS, and re-warps the CZ centroid markers so alignment visibly improves.
  Each add/undo is auto-saved (append-only) to
  qc_labels/manual_matches_<variant>_<sid>.csv and replayed on relaunch.

Keys:
  → / ←  next / prev CZ ROI
  ↑ / ↓  Z slice ±1 (slice mode only)
  s / m  switch to slice / MIP-cube-Z mode
  1 / 2 / 3   select radio option (top to bottom)
  4          toggle HCR 488 image
  w          toggle CZ warped image
  c          toggle "other CZ ROIs"
  h          toggle "other HCR ROIs"
  a          toggle add-match mode
  Enter      add the pending CZ↔HCR pair (add-match mode)
  Backspace  undo last added pair (add-match mode)
  Esc        reset the pending selection (add-match mode)

Usage:
  autocoreg run <sid> --qc [--qc-variant local_flow_wang_end]
  python -c "from autocoreg.qc.app import main; main(['--sid','790322','--variant','local_flow_wang_end'])"
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tifffile
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from .. import config as _config
from ..data import load_sz_pins, subject_inputs
from ..benchmark_analysis import load_hcr_volume

# Artifact inputs + label outputs are resolved from config (env-overridable).
OUT_ROOT = _config.QC_ARTIFACT_DIR
LABELS_ROOT = _config.QC_LABELS_DIR
CUBE_HALF_UM = 60.0
HCR_LEVEL = 2  # default pyramid level; --level overrides

# Colors (RGBA, 0..255)
COLOR_CUR_CZ     = (255, 255, 0,   255)  # yellow
COLOR_CUR_HCR    = (255, 255, 255, 255)  # white
COLOR_OTHER_CZM  = (255, 60,  60,  230)  # red     (other matched CZ)
COLOR_OTHER_CZU  = (255, 80,  255, 230)  # magenta (other unmatched CZ)
COLOR_OTHER_HCRM = (80,  255, 80,  230)  # green   (other matched HCR)
COLOR_OTHER_HCRU = (60,  220, 255, 230)  # cyan    (other unmatched HCR)
COLOR_HCR_FAIL_GFP = (255, 140, 0,  200)  # orange  (HCR failed GFP+)
COLOR_HCR_FAIL_CLS = (180,  60, 200, 200)  # purple (HCR failed ROI classifier)

# Contour widths
WIDTH_CUR   = 4.0
WIDTH_OTHER = 3.0

# Add-match-mode overlay colors (RGBA)
COLOR_WARP_CZ      = (255, 215, 0,   255)  # gold    (warped CZ centroid markers)
COLOR_PICK_CZ      = (0,   255, 255, 255)  # cyan    (CZ pick highlight)
COLOR_PICK_HCR     = (255, 0,   255, 255)  # magenta (HCR pick highlight)
COLOR_LANDMARK     = (255, 255, 255, 200)  # white   (active landmark link line)
COLOR_ADDED_LINK   = (0,   255, 0,   220)  # green   (session-added landmark link)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--sid", required=True)
    p.add_argument("--variant", default="local_flow",
                   choices=["local_flow", "lr_mb", "lr_only",
                            "local_flow_wang_end", "lr_mb_wang_end", "lr_only_wang_end"])
    p.add_argument("--cube_um", type=float, default=CUBE_HALF_UM,
                   help="Cube half-extent in µm (default 60).")
    p.add_argument("--level", type=int, default=HCR_LEVEL,
                   help="HCR 488 pyramid level (2 ≈ 1µm, 3 ≈ 2µm, 4 ≈ 4µm). "
                        "Higher = faster load, less detail.")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--cz_list", default=None,
                   help="Path to a CSV with a 'cz_id' column; iteration is "
                        "restricted to those IDs (in CSV order).")
    p.add_argument("--matches-csv", dest="matches_csv", default=None,
                   help="Explicit matcher-output CSV (cz_id,hcr_id[,soma_score]). "
                        "Overrides the find_final_round_csv lookup under "
                        "MFISH_QC_MATCHES_DIR.")
    return p.parse_args(argv)


def find_final_round_csv(sid: str, variant: str) -> Path:
    """Locate the final-round matcher CSV for (sid, variant).

    Looks under ``MFISH_QC_MATCHES_DIR/step3_v2_path_a_<variant>/<sid>/`` and
    returns the last ``matches_wang_round*.csv`` if any, else the last
    ``matches_round*.csv``.  (Vendored from the session-15 qc_pair_app so the
    repo has no flat-module dependency.)
    """
    d = _config.QC_MATCHES_DIR / f"step3_v2_path_a_{variant}" / sid
    wang_rounds = sorted(d.glob("matches_wang_round*.csv"),
                         key=lambda p: int(p.stem.replace("matches_wang_round", "")))
    if wang_rounds:
        return wang_rounds[-1]
    rounds = sorted(d.glob("matches_round*.csv"),
                    key=lambda p: int(p.stem.replace("matches_round", "")))
    if not rounds:
        raise FileNotFoundError(f"No matches CSVs under {d}")
    return rounds[-1]


def compute_centroids(label_arr: np.ndarray, ids: list[int]) -> dict[int, np.ndarray]:
    flat = label_arr.ravel()
    mask = np.isin(flat, ids)
    if not mask.any():
        return {}
    labels = flat[mask]
    z_idx, y_idx, x_idx = np.unravel_index(np.flatnonzero(mask), label_arr.shape)
    sums_z = np.bincount(labels, weights=z_idx, minlength=int(labels.max()) + 1)
    sums_y = np.bincount(labels, weights=y_idx, minlength=int(labels.max()) + 1)
    sums_x = np.bincount(labels, weights=x_idx, minlength=int(labels.max()) + 1)
    counts = np.bincount(labels, minlength=int(labels.max()) + 1)
    out = {}
    for v in set(ids):
        if v < len(counts) and counts[v] > 0:
            out[int(v)] = np.array(
                [sums_z[v] / counts[v],
                 sums_y[v] / counts[v],
                 sums_x[v] / counts[v]], dtype=float
            )
    return out


def _make_lut(rgb):
    """Build a 256-entry LUT going from black → rgb."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        f = i / 255.0
        lut[i] = [int(f * rgb[0]), int(f * rgb[1]), int(f * rgb[2])]
    return lut


class CubeView(QtWidgets.QWidget):
    """A 2D pyqtgraph image plot with two background images (HCR 488 + warped CZ)
    and contour overlays."""

    def __init__(self):
        super().__init__()
        self.plot = pg.PlotWidget()
        self.plot.setAspectLocked(True)
        self.plot.invertY(True)
        self.plot.setBackground("k")
        # Allow free pan + zoom (incl. zooming out past the cube extents).
        vb = self.plot.getViewBox()
        vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None,
                     minXRange=1.0, maxXRange=10000.0,
                     minYRange=1.0, maxYRange=10000.0)
        vb.setMouseEnabled(x=True, y=True)
        self._range_cb = None  # set by outer widget
        # HCR 488 (green)
        self.img_hcr = pg.ImageItem(axisOrder="row-major")
        self.img_hcr.setLookupTable(_make_lut((0, 255, 0)))
        self.plot.addItem(self.img_hcr)
        # Warped CZ (red, additive blend)
        self.img_cz = pg.ImageItem(axisOrder="row-major")
        self.img_cz.setLookupTable(_make_lut((255, 0, 0)))
        self.img_cz.setCompositionMode(QtGui.QPainter.CompositionMode_Plus)
        self.plot.addItem(self.img_cz)
        self.contour_items: list[pg.PlotDataItem] = []
        # Add-match-mode overlay (warped CZ centroid markers, selection
        # highlights, landmark links) — kept separate from contour_items so it
        # can be cleared/redrawn independently of the pass/fail contours.
        self.overlay_items: list[pg.GraphicsObject] = []
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot)
        self.setLayout(layout)

    def clear_contours(self):
        for it in self.contour_items:
            self.plot.removeItem(it)
        self.contour_items.clear()

    def clear_overlay(self):
        for it in self.overlay_items:
            self.plot.removeItem(it)
        self.overlay_items.clear()

    def add_scatter(self, xs, ys, *, color, size=10, symbol="o", pen=None):
        spi = pg.ScatterPlotItem(
            x=list(xs), y=list(ys), size=size, symbol=symbol,
            brush=pg.mkBrush(color), pen=(pen if pen is not None else pg.mkPen(None)),
        )
        self.plot.addItem(spi)
        self.overlay_items.append(spi)
        return spi

    def add_overlay_line(self, x_pts, y_pts, *, color, width=1.5, style=None):
        pen = pg.mkPen(color=color, width=width)
        if style is not None:
            pen.setStyle(style)
        pdi = pg.PlotDataItem(x_pts, y_pts, pen=pen, connect="all")
        self.plot.addItem(pdi)
        self.overlay_items.append(pdi)
        return pdi

    def add_contour(self, x_pts, y_pts, color, width=1.5):
        pen = pg.mkPen(color=color, width=width)
        pdi = pg.PlotDataItem(x_pts, y_pts, pen=pen, connect="all")
        self.plot.addItem(pdi)
        self.contour_items.append(pdi)

    def set_hcr_image(self, arr_2d, *, x_lo, y_lo, xy_um):
        h, w = arr_2d.shape
        self.img_hcr.setImage(arr_2d, autoLevels=False)
        self.img_hcr.setRect(QtCore.QRectF(x_lo, y_lo, w * xy_um, h * xy_um))

    def set_cz_image(self, arr_2d, *, x_lo, y_lo, xy_um):
        if arr_2d is None:
            self.img_cz.clear()
            return
        h, w = arr_2d.shape
        self.img_cz.setImage(arr_2d, autoLevels=False)
        self.img_cz.setRect(QtCore.QRectF(x_lo, y_lo, w * xy_um, h * xy_um))


class QCApp(QtWidgets.QMainWindow):
    def __init__(self, sid: str, variant: str, cube_um: float, start: int,
                 hcr_level: int = HCR_LEVEL, cz_list_path: str | None = None,
                 matches_csv: str | None = None):
        super().__init__()
        self.sid = sid
        self.variant = variant
        self.cube_half = float(cube_um)
        self.hcr_level = int(hcr_level)
        self.cz_list_path = cz_list_path
        self.matches_csv = matches_csv
        self.setWindowTitle(f"QC Qt — {sid} / {variant}")
        self._load_data()
        self._init_state()
        self._build_ui()
        # initial show
        start = max(0, min(start, len(self.cz_order) - 1))
        self.show_idx = start
        self._refresh_pair()

    # ---------------- data loading ----------------
    def _load_data(self):
        qc_dir = OUT_ROOT / self.variant / self.sid
        if not qc_dir.exists():
            sys.exit(f"missing QC artifacts: {qc_dir}")
        seg_meta = json.loads((qc_dir / "seg_volumes_meta.json").read_text())
        cz_bb = seg_meta["bbox_cz_warped"]
        cz_vox = float(seg_meta["voxel_um_cz_warped"])
        hbb = seg_meta["bbox_hcr_seg"]
        hcr_vox_xy = float(seg_meta["voxel_um_hcr_seg_xy"])
        hcr_vox_z = float(seg_meta["voxel_um_hcr_seg_z"])
        self.cz_bb = cz_bb; self.cz_vox = cz_vox
        self.hbb = hbb; self.hcr_vox_xy = hcr_vox_xy; self.hcr_vox_z = hcr_vox_z

        self.cz_matched_arr   = tifffile.imread(str(qc_dir / "cz_matched_seg.tif"))
        self.cz_unmatched_arr = tifffile.imread(str(qc_dir / "cz_unmatched_seg.tif"))
        self.hcr_matched_arr   = tifffile.imread(str(qc_dir / "hcr_matched_seg.tif"))
        self.hcr_unmatched_arr = tifffile.imread(str(qc_dir / "hcr_unmatched_seg.tif"))
        # Optional: HCR cells that failed each filter (NOT in pool).
        fg_path = qc_dir / "hcr_failed_gfp_seg.tif"
        fc_path = qc_dir / "hcr_failed_classifier_seg.tif"
        self.hcr_failed_gfp_arr = (
            tifffile.imread(str(fg_path)) if fg_path.exists() else None
        )
        self.hcr_failed_cls_arr = (
            tifffile.imread(str(fc_path)) if fc_path.exists() else None
        )

        # Matches CSV: explicit override, else final-round lookup.
        matches_csv = (
            Path(self.matches_csv) if self.matches_csv
            else find_final_round_csv(self.sid, self.variant)
        )
        print(f"[qt] matches CSV: {matches_csv}")
        self.df_matches = pd.read_csv(matches_csv)
        self.df_matches["cz_id"] = self.df_matches["cz_id"].astype(int)
        self.df_matches["hcr_id"] = self.df_matches["hcr_id"].astype(int)
        self.cz_to_hcr = dict(zip(
            self.df_matches["cz_id"], self.df_matches["hcr_id"]
        ))
        self.cz_to_soma = dict(zip(
            self.df_matches["cz_id"],
            self.df_matches.get("soma_score", pd.Series([np.nan] * len(self.df_matches))),
        ))
        # cz_id iteration order: matched first by row order, then unmatched
        cz_matched_ids = list(self.df_matches["cz_id"])
        unmatched_uniq = sorted(set(int(v) for v in np.unique(self.cz_unmatched_arr) if v != 0))
        # avoid double-counting
        matched_set = set(cz_matched_ids)
        unmatched_only = [v for v in unmatched_uniq if v not in matched_set]
        full_order = list(cz_matched_ids) + unmatched_only
        if self.cz_list_path:
            ext = pd.read_csv(self.cz_list_path)
            wanted = [int(c) for c in ext["cz_id"].astype(int).tolist()]
            full_set = set(full_order)
            self.cz_order = [c for c in wanted if c in full_set]
            missing = [c for c in wanted if c not in full_set]
            print(f"[qt] --cz_list restricts to {len(self.cz_order)}/{len(wanted)} CZ "
                  f"IDs (skipped {len(missing)} not present in seg)")
        else:
            self.cz_order = full_order

        # CZ centroids in HCR µm (warped grid → world)
        all_cz_uniq = sorted(set(int(v) for v in np.unique(self.cz_matched_arr) if v != 0) |
                             set(int(v) for v in np.unique(self.cz_unmatched_arr) if v != 0))
        cz_combo = np.where(self.cz_matched_arr > 0, self.cz_matched_arr, self.cz_unmatched_arr)
        cz_centroids_vox = compute_centroids(cz_combo, all_cz_uniq)
        self.cz_world = {}
        for v, c in cz_centroids_vox.items():
            self.cz_world[int(v)] = np.array([
                cz_bb["z_lo"] + c[0] * cz_vox,
                cz_bb["y_lo"] + c[1] * cz_vox,
                cz_bb["x_lo"] + c[2] * cz_vox,
            ])

        # HCR 488 volume cropped to overlap bbox. Required — retry on transient OSError.
        import time as _time
        sz_pins = load_sz_pins()
        inp = subject_inputs(self.sid, sz_pins=sz_pins)
        # CZ-native + HCR centroids (zyx µm, keyed by id) for the manual-match
        # TPS.  The app's display "world" is HCR µm, so hcr_um is already in
        # world coords; the TPS maps CZ-native µm -> HCR µm (same convention as
        # build_warped_cz_volume).
        self.cz_native_by_id = {
            int(i): np.asarray(p, dtype=float)
            for i, p in zip(inp.cz_ids, inp.cz_um)
        }
        self.hcr_by_id = {
            int(i): np.asarray(p, dtype=float)
            for i, p in zip(inp.hcr_ids, inp.hcr_um)
        }
        last_exc = None
        for attempt in range(3):
            try:
                print(f"[qt] loading HCR 488 level {self.hcr_level} "
                      f"(attempt {attempt + 1}/3) ...", flush=True)
                vol, xy_um, z_um = load_hcr_volume(
                    inp.s, channel="488", level=self.hcr_level
                )
                break
            except (OSError, IOError) as exc:
                last_exc = exc
                print(f"[qt] HCR 488 read failed ({type(exc).__name__}: {exc}); "
                      f"sleeping 5s then retrying", file=sys.stderr)
                _time.sleep(5)
        else:
            raise RuntimeError(
                f"HCR 488 zarr read failed after 3 attempts: {last_exc}"
            )
        cz_lp = inp.cz_lp_um
        # HCR 488 bbox: same Z extent as CZ + 30 µm margin; XY extends 10% of
        # CZ extent further on each side (so HCR shows context outside CZ).
        margin_z = 30.0
        y_ext = float(cz_lp[:, 1].max() - cz_lp[:, 1].min())
        x_ext = float(cz_lp[:, 2].max() - cz_lp[:, 2].min())
        margin_y = 30.0 + 0.10 * y_ext
        margin_x = 30.0 + 0.10 * x_ext
        z_lo = float(cz_lp[:, 0].min()) - margin_z
        z_hi = float(cz_lp[:, 0].max()) + margin_z
        y_lo = float(cz_lp[:, 1].min()) - margin_y
        y_hi = float(cz_lp[:, 1].max()) + margin_y
        x_lo = float(cz_lp[:, 2].min()) - margin_x
        x_hi = float(cz_lp[:, 2].max()) + margin_x
        z0 = max(0, int(z_lo / z_um));   z1 = min(vol.shape[0], int(z_hi / z_um) + 1)
        y0 = max(0, int(y_lo / xy_um));  y1 = min(vol.shape[1], int(y_hi / xy_um) + 1)
        x0 = max(0, int(x_lo / xy_um));  x1 = min(vol.shape[2], int(x_hi / xy_um) + 1)
        self.hcr488 = vol[z0:z1, y0:y1, x0:x1].astype(np.float32)
        self.hcr488_origin = (z0 * z_um, y0 * xy_um, x0 * xy_um)
        self.hcr488_voxel = (float(z_um), float(xy_um), float(xy_um))
        self.hcr488_levels = (
            float(np.percentile(self.hcr488, 5)),
            float(np.percentile(self.hcr488, 99.5)),
        )
        print(f"[qt] HCR 488 loaded: {self.hcr488.shape} at {xy_um:.3f} µm/vox xy, "
              f"{z_um:.3f} µm/vox z ({self.hcr488.nbytes / 1e6:.0f} MB)")

        # Warped CZ stack image (TPS-transformed into HCR µm), if present
        cz_warp_tif = qc_dir / "cz_warped_in_hcr_um.tif"
        cz_warp_meta = qc_dir / "cz_warped_meta.json"
        if cz_warp_tif.exists() and cz_warp_meta.exists():
            cz_meta = json.loads(cz_warp_meta.read_text())
            self.czw = tifffile.imread(str(cz_warp_tif)).astype(np.float32)
            self.czw_voxel = float(cz_meta["voxel_um"])
            self.czw_origin = (
                float(cz_meta["bbox_um"]["z_lo"]),
                float(cz_meta["bbox_um"]["y_lo"]),
                float(cz_meta["bbox_um"]["x_lo"]),
            )
            self.czw_levels = (
                float(np.percentile(self.czw, 5)),
                float(np.percentile(self.czw, 99.5)),
            )
        else:
            self.czw = None

        # Labels CSV
        LABELS_ROOT.mkdir(parents=True, exist_ok=True)
        self.labels_path = LABELS_ROOT / f"qt_{self.variant}_{self.sid}.csv"
        if self.labels_path.exists():
            prior = pd.read_csv(self.labels_path)
            prior = prior.sort_values("timestamp").drop_duplicates(
                subset=["cz_id"], keep="last"
            )
            self.labels_state = {
                int(r.cz_id): str(r.label) for r in prior.itertuples(index=False)
            }
        else:
            self.labels_state = {}

        # Manual-match CSV (function 2, append-only) + prior add/undo events to
        # replay so the active landmark set resumes across sessions.
        self.manual_matches_path = (
            LABELS_ROOT / f"manual_matches_{self.variant}_{self.sid}.csv"
        )
        self._prior_manual_events = []
        if self.manual_matches_path.exists():
            mm = pd.read_csv(self.manual_matches_path).sort_values("timestamp")
            self._prior_manual_events = [
                (str(r.action), int(r.cz_id), int(r.hcr_id))
                for r in mm.itertuples(index=False)
            ]

        print(f"[qt] {len(self.cz_order)} CZ ROIs ({len(cz_matched_ids)} matched, "
              f"{len(unmatched_only)} unmatched).  "
              f"HCR 488 cube {self.hcr488.shape}, "
              f"{len(self.labels_state)} prior labels.")

    def _init_state(self):
        self.show_idx = 0
        self.cur_z_world = 0.0
        self.show_other_cz = True
        self.show_other_hcr = True
        self.show_czw = True
        self.show_hcr488 = True
        self.show_hcr_fail_gfp = False
        self.show_hcr_fail_cls = False
        self.mip_mode = False  # toggled by 'm' / radio

        # ---- Add-match mode (function 2) ----
        self.add_match_mode = False
        # Active landmark set used to fit the TPS.  Seeded from the matcher's
        # accepted matches; the operator appends new pairs.  {cz_id: hcr_id}.
        self.active_pairs: dict[int, int] = {}
        # Session-added pairs (subset of active_pairs the operator created), in
        # add order, for undo + audit.
        self.added_order: list[int] = []
        # Pending click selection.
        self.pending_cz_id: int | None = None
        self.pending_hcr_id: int | None = None
        # Fitted per-axis TPS (callables) and cache of warped CZ centroids.
        self._tps = None
        self.cz_warped_by_id: dict[int, np.ndarray] = {}

        # Seed the active set from accepted matches whose endpoints both have
        # known native centroids.
        for cz_id, hcr_id in self.cz_to_hcr.items():
            if int(cz_id) in self.cz_native_by_id and int(hcr_id) in self.hcr_by_id:
                self.active_pairs[int(cz_id)] = int(hcr_id)
        # Replay prior manual add/undo events (resume across sessions).
        for action, cz, hc in getattr(self, "_prior_manual_events", []):
            if action == "add" and cz in self.cz_native_by_id and hc in self.hcr_by_id:
                self.active_pairs[cz] = hc
                if cz in self.added_order:
                    self.added_order.remove(cz)
                self.added_order.append(cz)
            elif action == "undo" and self.added_order:
                if cz in self.active_pairs:
                    self.active_pairs.pop(cz, None)
                if cz in self.added_order:
                    self.added_order.remove(cz)
                orig = self.cz_to_hcr.get(cz)
                if orig is not None and int(orig) in self.hcr_by_id:
                    self.active_pairs[cz] = int(orig)
        self._refit_tps()

    # ---------------- manual-match (function 2) ----------------
    def _refit_tps(self):
        """Fit per-axis thin-plate Rbf (CZ-native µm -> HCR µm) from the active
        landmark set, then refresh the warped CZ centroids.  Needs >= 4 pairs;
        below that the TPS is left unfit and no warped markers are shown."""
        from scipy.interpolate import Rbf
        pairs = [(c, h) for c, h in self.active_pairs.items()
                 if c in self.cz_native_by_id and h in self.hcr_by_id]
        if len(pairs) < 4:
            self._tps = None
            self.cz_warped_by_id = {}
            return
        src = np.array([self.cz_native_by_id[c] for c, _ in pairs])  # (N,3) zyx
        dst = np.array([self.hcr_by_id[h] for _, h in pairs])        # (N,3) zyx
        self._tps = [
            Rbf(src[:, 0], src[:, 1], src[:, 2], dst[:, a],
                function="thin_plate", smooth=0.0)
            for a in range(3)
        ]
        self._warp_cz_centroids()

    def _warp_cz_centroids(self):
        """Apply the current TPS to every CZ-native centroid -> world (HCR µm)."""
        if self._tps is None or not self.cz_native_by_id:
            self.cz_warped_by_id = {}
            return
        ids = list(self.cz_native_by_id.keys())
        src = np.array([self.cz_native_by_id[i] for i in ids])
        warped = np.stack(
            [self._tps[a](src[:, 0], src[:, 1], src[:, 2]) for a in range(3)],
            axis=1,
        )
        self.cz_warped_by_id = {int(i): warped[k] for k, i in enumerate(ids)}

    def _cz_pos(self, cz_id: int) -> np.ndarray | None:
        """Current world (HCR µm) position of a CZ ROI: TPS-warped centroid if
        available, else the baked warped-seg centroid."""
        p = self.cz_warped_by_id.get(int(cz_id))
        if p is not None:
            return p
        return self.cz_world.get(int(cz_id))

    def _nearest_id(self, by_id: dict, x: float, y: float, z: float,
                    z_tol: float, r_max: float):
        """Nearest id in a {id: zyx} dict to (x,y) at slice z, within z_tol in z
        and r_max in-plane.  Returns (id, dist) or (None, inf)."""
        best, best_d = None, float("inf")
        for i, p in by_id.items():
            if abs(p[0] - z) > z_tol:
                continue
            d = ((p[2] - x) ** 2 + (p[1] - y) ** 2) ** 0.5
            if d < best_d:
                best, best_d = int(i), d
        if best is None or best_d > r_max:
            return None, float("inf")
        return best, best_d

    def _hcr_label_at(self, x: float, y: float, z: float) -> int:
        """HCR label id at world (x,y,z) µm, looked up in the matched/unmatched
        HCR seg volumes (0 = background)."""
        for arr in (self.hcr_matched_arr, self.hcr_unmatched_arr):
            zv = int(round((z - self.hbb["z_lo"]) / self.hcr_vox_z))
            yv = int(round((y - self.hbb["y_lo"]) / self.hcr_vox_xy))
            xv = int(round((x - self.hbb["x_lo"]) / self.hcr_vox_xy))
            if (0 <= zv < arr.shape[0] and 0 <= yv < arr.shape[1]
                    and 0 <= xv < arr.shape[2]):
                v = int(arr[zv, yv, xv])
                if v != 0:
                    return v
        return 0

    def _on_canvas_click(self, ev):
        """Left-click in add-match mode: first click picks the CZ ROI (nearest
        warped CZ centroid), second picks the HCR ROI (label under cursor, else
        nearest HCR centroid)."""
        if not self.add_match_mode:
            return
        if ev.button() != QtCore.Qt.LeftButton:
            return
        vb = self.view.plot.getViewBox()
        if not self.view.plot.sceneBoundingRect().contains(ev.scenePos()):
            return
        pt = vb.mapSceneToView(ev.scenePos())
        x, y, z = float(pt.x()), float(pt.y()), self.cur_z_world
        z_tol = max(self.cube_half, 8.0)
        if self.pending_cz_id is None:
            cz_id, d = self._nearest_id(
                self.cz_warped_by_id or self.cz_world, x, y, z,
                z_tol=z_tol, r_max=self.cube_half)
            if cz_id is None:
                self._set_match_status("no CZ ROI near click")
                return
            self.pending_cz_id = cz_id
            self._set_match_status(f"CZ picked: cz_id={cz_id} (d={d:.1f}µm). "
                                   f"Now click its HCR cell.")
        elif self.pending_hcr_id is None:
            hcr_id = self._hcr_label_at(x, y, z)
            if hcr_id == 0:
                hcr_id, _ = self._nearest_id(self.hcr_by_id, x, y, z,
                                             z_tol=z_tol, r_max=self.cube_half)
            if hcr_id is None or hcr_id == 0:
                self._set_match_status("no HCR ROI near click")
                return
            self.pending_hcr_id = int(hcr_id)
            self._set_match_status(self._pending_summary())
        else:
            # Both already set — restart selection at this click.
            self.pending_cz_id = None
            self.pending_hcr_id = None
            self._on_canvas_click(ev)
            return
        self._update_match_buttons()
        self._redraw_contours_only()

    def _pending_summary(self) -> str:
        cz, hc = self.pending_cz_id, self.pending_hcr_id
        if cz is None or hc is None:
            return ""
        wp = self._cz_pos(cz)
        hp = self.hcr_by_id.get(hc)
        dist = (float(np.linalg.norm(wp - hp)) if wp is not None and hp is not None
                else float("nan"))
        existing = ""
        if cz in self.active_pairs and self.active_pairs[cz] != hc:
            existing = f"  (replaces hcr_id={self.active_pairs[cz]})"
        return (f"pair: cz_id={cz} ↔ hcr_id={hc}  warp-dist={dist:.1f}µm{existing}"
                f"\nEnter=add  Esc=reset")

    def _add_pair(self):
        if self.pending_cz_id is None or self.pending_hcr_id is None:
            self._set_match_status("pick a CZ ROI then an HCR ROI first")
            return
        cz, hc = self.pending_cz_id, self.pending_hcr_id
        if cz not in self.cz_native_by_id or hc not in self.hcr_by_id:
            self._set_match_status(f"missing centroid for cz_id={cz}/hcr_id={hc}")
            return
        self.active_pairs[cz] = hc
        if cz in self.added_order:
            self.added_order.remove(cz)
        self.added_order.append(cz)
        self._refit_tps()
        self._save_manual_match(cz, hc, "add")
        self.pending_cz_id = None
        self.pending_hcr_id = None
        self._set_match_status(self._counts_summary() + "  (added, TPS refit)")
        self._update_match_buttons()
        self._redraw()

    def _undo_pair(self):
        if not self.added_order:
            self._set_match_status("nothing to undo")
            return
        cz = self.added_order.pop()
        hc = self.active_pairs.pop(cz, None)
        # If the cz was originally a seeded match, restore that mapping.
        orig = self.cz_to_hcr.get(cz)
        if orig is not None and int(orig) in self.hcr_by_id:
            self.active_pairs[cz] = int(orig)
        self._refit_tps()
        self._save_manual_match(cz, hc if hc is not None else -1, "undo")
        self._set_match_status(self._counts_summary() + f"  (undid cz_id={cz})")
        self._update_match_buttons()
        self._redraw()

    def _reset_selection(self):
        self.pending_cz_id = None
        self.pending_hcr_id = None
        self._set_match_status("selection reset")
        self._update_match_buttons()
        self._redraw_contours_only()

    def _counts_summary(self) -> str:
        return (f"active landmarks: {len(self.active_pairs)}  |  "
                f"session-added: {len(self.added_order)}")

    def _save_manual_match(self, cz_id: int, hcr_id: int, action: str):
        LABELS_ROOT.mkdir(parents=True, exist_ok=True)
        ts = _dt.datetime.now().isoformat(timespec="seconds")
        write_header = not self.manual_matches_path.exists()
        with open(self.manual_matches_path, "a") as f:
            if write_header:
                f.write("timestamp,action,cz_id,hcr_id,n_active\n")
            f.write(f"{ts},{action},{cz_id},{hcr_id},{len(self.active_pairs)}\n")
        print(f"[match] {action} cz_id={cz_id} hcr_id={hcr_id} "
              f"n_active={len(self.active_pairs)}")

    def _draw_match_overlay(self):
        """Overlay for add-match mode: warped CZ centroid markers (gold) visible
        in the current slice, plus highlights + a link line for the pending
        selection.  Cleared (and skipped) when add-match mode is off."""
        self.view.clear_overlay()
        if not self.add_match_mode:
            return
        (vy_lo, vy_hi), (vx_lo, vx_hi) = self._viewport_xy_range()
        if self.mip_mode:
            z_lo, z_hi = self.mip_z_world
        else:
            z_tol = max(self.cube_half, 8.0)
            z_lo, z_hi = self.cur_z_world - z_tol, self.cur_z_world + z_tol
        xs, ys = [], []
        for cz_id, p in self.cz_warped_by_id.items():
            if not (z_lo <= p[0] <= z_hi):
                continue
            if not (vx_lo <= p[2] <= vx_hi and vy_lo <= p[1] <= vy_hi):
                continue
            if cz_id == self.pending_cz_id:
                continue
            xs.append(p[2]); ys.append(p[1])
        if xs:
            self.view.add_scatter(xs, ys, color=COLOR_WARP_CZ, size=7, symbol="o")
        if self.pending_cz_id is not None:
            cp = self._cz_pos(self.pending_cz_id)
            if cp is not None:
                self.view.add_scatter([cp[2]], [cp[1]], color=(0, 0, 0, 0),
                                      size=18, symbol="o",
                                      pen=pg.mkPen(COLOR_PICK_CZ, width=3))
        if self.pending_hcr_id is not None:
            hp = self.hcr_by_id.get(self.pending_hcr_id)
            if hp is not None:
                self.view.add_scatter([hp[2]], [hp[1]], color=(0, 0, 0, 0),
                                      size=18, symbol="o",
                                      pen=pg.mkPen(COLOR_PICK_HCR, width=3))
                cp = self._cz_pos(self.pending_cz_id) if self.pending_cz_id else None
                if cp is not None:
                    self.view.add_overlay_line([cp[2], hp[2]], [cp[1], hp[1]],
                                               color=COLOR_ADDED_LINK, width=2)

    def _toggle_add_match(self):
        self.add_match_mode = self.chk_add_match.isChecked()
        self.match_box.setVisible(self.add_match_mode)
        # Disable the pass/fail radio while matching (avoid stray labels).
        self.radio_box.setEnabled(not self.add_match_mode)
        if not self.add_match_mode:
            self.pending_cz_id = None
            self.pending_hcr_id = None
        else:
            self._set_match_status(
                self._counts_summary() + "\nclick a CZ ROI, then its HCR cell")
        self._update_match_buttons()
        self._redraw()

    def _set_match_status(self, msg: str):
        if hasattr(self, "lbl_match"):
            self.lbl_match.setText(msg)

    def _update_match_buttons(self):
        if not hasattr(self, "btn_add_pair"):
            return
        self.btn_add_pair.setEnabled(
            self.pending_cz_id is not None and self.pending_hcr_id is not None)
        self.btn_undo_pair.setEnabled(len(self.added_order) > 0)

    # ---------------- UI ----------------
    def _build_ui(self):
        cw = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout()
        cw.setLayout(h)
        self.setCentralWidget(cw)

        # Image area
        self.view = CubeView()
        h.addWidget(self.view, stretch=4)

        # Right control panel
        panel = QtWidgets.QWidget()
        pl = QtWidgets.QVBoxLayout()
        panel.setLayout(pl)
        h.addWidget(panel, stretch=1)

        self.lbl_status = QtWidgets.QLabel("…")
        self.lbl_status.setStyleSheet("font-weight: bold;")
        pl.addWidget(self.lbl_status)

        # Z slider
        z_lbl_row = QtWidgets.QHBoxLayout()
        z_lbl_row.addWidget(QtWidgets.QLabel("Z (µm):"))
        self.lbl_z = QtWidgets.QLabel("—")
        z_lbl_row.addWidget(self.lbl_z)
        pl.addLayout(z_lbl_row)
        self.z_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.z_slider.valueChanged.connect(self._on_z_slider)
        pl.addWidget(self.z_slider)

        # Contrast widgets (compact): histogram + region + min/max spinboxes
        # + Auto button.  Auto re-derives 5/99.5 percentiles from the current
        # cube; drag region or type to override.
        self.hist_hcr, self.spin_hcr_min, self.spin_hcr_max, self.btn_hcr_auto = \
            self._build_contrast_panel(
                pl, label="HCR 488", img=self.view.img_hcr,
                gradient_rgb=(0, 255, 0),
                auto_fn=lambda: self._auto_contrast("hcr"),
            )
        self.hist_cz, self.spin_cz_min, self.spin_cz_max, self.btn_cz_auto = \
            self._build_contrast_panel(
                pl, label="CZ warped", img=self.view.img_cz,
                gradient_rgb=(255, 0, 0),
                auto_fn=lambda: self._auto_contrast("cz"),
            )

        # Toggles
        self.chk_hcr488 = QtWidgets.QCheckBox("HCR 488 image (4)")
        self.chk_hcr488.setChecked(True)
        self.chk_hcr488.stateChanged.connect(lambda _: self._toggle_hcr488())
        pl.addWidget(self.chk_hcr488)
        self.chk_czw = QtWidgets.QCheckBox("CZ warped image (w)")
        self.chk_czw.setChecked(True)
        self.chk_czw.stateChanged.connect(lambda _: self._toggle_czw())
        pl.addWidget(self.chk_czw)
        # Mode: slice / MIP
        mode_box = QtWidgets.QGroupBox("View mode")
        mode_layout = QtWidgets.QHBoxLayout()
        mode_box.setLayout(mode_layout)
        self.rad_slice = QtWidgets.QRadioButton("Slice (s)")
        self.rad_slice.setChecked(True)
        self.rad_mip = QtWidgets.QRadioButton("MIP cube Z (m)")
        self.rad_slice.toggled.connect(lambda checked: checked and self._set_mode(False))
        self.rad_mip.toggled.connect(lambda checked: checked and self._set_mode(True))
        mode_layout.addWidget(self.rad_slice)
        mode_layout.addWidget(self.rad_mip)
        pl.addWidget(mode_box)

        self.chk_other_cz = QtWidgets.QCheckBox("Other CZ ROIs (c)")
        self.chk_other_cz.setChecked(True)
        self.chk_other_cz.stateChanged.connect(lambda _: self._toggle_other_cz())
        pl.addWidget(self.chk_other_cz)
        self.chk_other_hcr = QtWidgets.QCheckBox("Other HCR ROIs (h)")
        self.chk_other_hcr.setChecked(True)
        self.chk_other_hcr.stateChanged.connect(lambda _: self._toggle_other_hcr())
        pl.addWidget(self.chk_other_hcr)
        self.chk_hcr_fail_gfp = QtWidgets.QCheckBox("HCR failed GFP+ (f)")
        self.chk_hcr_fail_gfp.setChecked(False)
        self.chk_hcr_fail_gfp.stateChanged.connect(lambda _: self._toggle_hcr_fail_gfp())
        pl.addWidget(self.chk_hcr_fail_gfp)
        self.chk_hcr_fail_cls = QtWidgets.QCheckBox("HCR failed ROI classifier (r)")
        self.chk_hcr_fail_cls.setChecked(False)
        self.chk_hcr_fail_cls.stateChanged.connect(lambda _: self._toggle_hcr_fail_cls())
        pl.addWidget(self.chk_hcr_fail_cls)

        # Radio group — built lazily per-pair (different options for matched/unmatched)
        self.radio_box = QtWidgets.QGroupBox("Label (auto-save)")
        self.radio_layout = QtWidgets.QVBoxLayout()
        self.radio_box.setLayout(self.radio_layout)
        self.radio_buttons: list[QtWidgets.QRadioButton] = []
        self.radio_group = QtWidgets.QButtonGroup()
        pl.addWidget(self.radio_box)

        # ---- Add-match mode (function 2) ----
        self.chk_add_match = QtWidgets.QCheckBox("Add-match mode (a)")
        self.chk_add_match.setChecked(False)
        self.chk_add_match.stateChanged.connect(lambda _: self._toggle_add_match())
        pl.addWidget(self.chk_add_match)

        self.match_box = QtWidgets.QGroupBox("Manual match (TPS re-warp)")
        mbl = QtWidgets.QVBoxLayout()
        self.match_box.setLayout(mbl)
        self.lbl_match = QtWidgets.QLabel("click a CZ ROI, then its HCR cell")
        self.lbl_match.setWordWrap(True)
        mbl.addWidget(self.lbl_match)
        mrow = QtWidgets.QHBoxLayout()
        self.btn_add_pair = QtWidgets.QPushButton("Add pair (Enter)")
        self.btn_add_pair.clicked.connect(self._add_pair)
        self.btn_reset_sel = QtWidgets.QPushButton("Reset (Esc)")
        self.btn_reset_sel.clicked.connect(self._reset_selection)
        mrow.addWidget(self.btn_add_pair)
        mrow.addWidget(self.btn_reset_sel)
        mbl.addLayout(mrow)
        self.btn_undo_pair = QtWidgets.QPushButton("Undo last add (Backspace)")
        self.btn_undo_pair.clicked.connect(self._undo_pair)
        mbl.addWidget(self.btn_undo_pair)
        self.match_box.setVisible(False)
        pl.addWidget(self.match_box)

        # Nav buttons
        nav_row = QtWidgets.QHBoxLayout()
        b_prev = QtWidgets.QPushButton("← Prev")
        b_prev.clicked.connect(self._prev)
        b_next = QtWidgets.QPushButton("Next →")
        b_next.clicked.connect(self._next)
        nav_row.addWidget(b_prev)
        nav_row.addWidget(b_next)
        pl.addLayout(nav_row)

        # Labels file path display
        pl.addStretch(1)
        path_lbl = QtWidgets.QLabel(
            f"<small>Labels: {self.labels_path}</small>"
        )
        path_lbl.setWordWrap(True)
        pl.addWidget(path_lbl)

        # Keyboard shortcuts
        self._mk_shortcut("Right", self._next)
        self._mk_shortcut("Left", self._prev)
        self._mk_shortcut("Up", self._z_up)
        self._mk_shortcut("Down", self._z_down)
        self._mk_shortcut("1", lambda: self._click_radio(0))
        self._mk_shortcut("2", lambda: self._click_radio(1))
        self._mk_shortcut("3", lambda: self._click_radio(2))
        self._mk_shortcut("c", lambda: self.chk_other_cz.toggle())
        self._mk_shortcut("h", lambda: self.chk_other_hcr.toggle())
        self._mk_shortcut("4", lambda: self.chk_hcr488.toggle())  # 4 → 488 image
        self._mk_shortcut("w", lambda: self.chk_czw.toggle())     # w → warped CZ image
        self._mk_shortcut("m", lambda: self.rad_mip.setChecked(True))
        self._mk_shortcut("s", lambda: self.rad_slice.setChecked(True))
        self._mk_shortcut("f", lambda: self.chk_hcr_fail_gfp.toggle())
        self._mk_shortcut("r", lambda: self.chk_hcr_fail_cls.toggle())
        self._mk_shortcut("a", lambda: self.chk_add_match.toggle())
        self._mk_shortcut("Return", self._add_pair)
        self._mk_shortcut("Enter", self._add_pair)
        self._mk_shortcut("Backspace", self._undo_pair)
        self._mk_shortcut("Escape", self._reset_selection)

        # Mouse picking for add-match mode.
        self.view.plot.scene().sigMouseClicked.connect(self._on_canvas_click)

        # Debounced re-draw of contours on pan/zoom (so contours always cover
        # the visible viewport, not just the cube).
        self._range_timer = QtCore.QTimer(self)
        self._range_timer.setSingleShot(True)
        self._range_timer.setInterval(120)
        self._range_timer.timeout.connect(self._redraw_contours_only)
        vb = self.view.plot.getViewBox()
        vb.sigRangeChanged.connect(lambda *a, **k: self._range_timer.start())

        self.resize(1100, 800)

    def _redraw_contours_only(self):
        """Refresh contours for the current viewport without touching images."""
        self.view.clear_contours()
        if self.mip_mode:
            self._draw_cz_contours_mip()
            self._draw_hcr_contours_mip()
        else:
            self._draw_cz_contours_at_z(self.cur_z_world)
            self._draw_hcr_contours_at_z(self.cur_z_world)
        self._draw_match_overlay()

    def _mk_shortcut(self, key, fn):
        sc = QtWidgets.QShortcut(QtGui.QKeySequence(key), self)
        sc.activated.connect(fn)

    def _build_contrast_panel(self, parent_layout, *, label, img, gradient_rgb, auto_fn):
        """Compact contrast UI: header + histogram (height 90) + min/max spinboxes
        + Auto button.  Returns (hist_widget, spin_min, spin_max, btn_auto).
        Spinboxes are kept in sync with the histogram's region."""
        parent_layout.addWidget(QtWidgets.QLabel(f"<b>Contrast — {label}</b>"))
        hist = pg.HistogramLUTWidget()
        hist.setImageItem(img)
        hist.gradient.restoreState({
            "mode": "rgb",
            "ticks": [(0.0, (0, 0, 0, 255)), (1.0, (*gradient_rgb, 255))],
        })
        hist.setMaximumHeight(110)
        parent_layout.addWidget(hist)

        row = QtWidgets.QHBoxLayout()
        spin_min = QtWidgets.QDoubleSpinBox(); spin_min.setRange(-1e9, 1e9)
        spin_min.setDecimals(0); spin_min.setSingleStep(1); spin_min.setMaximumWidth(80)
        spin_max = QtWidgets.QDoubleSpinBox(); spin_max.setRange(-1e9, 1e9)
        spin_max.setDecimals(0); spin_max.setSingleStep(1); spin_max.setMaximumWidth(80)
        btn = QtWidgets.QPushButton("Auto")
        btn.setMaximumWidth(60)
        btn.clicked.connect(auto_fn)
        row.addWidget(QtWidgets.QLabel("min:")); row.addWidget(spin_min)
        row.addWidget(QtWidgets.QLabel("max:")); row.addWidget(spin_max)
        row.addWidget(btn)
        parent_layout.addLayout(row)

        # Bidirectional sync: histogram region ↔ spinboxes
        syncing = {"flag": False}
        def from_region():
            if syncing["flag"]:
                return
            lo, hi = hist.getLevels()
            syncing["flag"] = True
            try:
                spin_min.setValue(float(lo))
                spin_max.setValue(float(hi))
            finally:
                syncing["flag"] = False
        def from_spin():
            if syncing["flag"]:
                return
            lo = float(spin_min.value()); hi = float(spin_max.value())
            if hi <= lo:
                hi = lo + 1
            syncing["flag"] = True
            try:
                hist.setLevels(lo, hi)
            finally:
                syncing["flag"] = False
        hist.sigLevelsChanged.connect(from_region)
        spin_min.valueChanged.connect(from_spin)
        spin_max.valueChanged.connect(from_spin)
        return hist, spin_min, spin_max, btn

    def _auto_contrast(self, which: str):
        """Re-derive 5/99.5 percentiles from current cube and apply."""
        self._compute_local_levels(self.cube_bb)
        if which == "hcr":
            self.hist_hcr.setLevels(*self.hcr488_levels)
        elif which == "cz" and self.czw is not None:
            self.hist_cz.setLevels(*self.czw_levels)

    # ---------------- pair display ----------------
    def _refresh_pair(self):
        cz_id = self.cz_order[self.show_idx]
        matched = cz_id in self.cz_to_hcr
        hcr_id = int(self.cz_to_hcr[cz_id]) if matched else None
        soma = float(self.cz_to_soma.get(cz_id, np.nan)) if matched else float("nan")

        # World cube around CZ centroid
        cz_c = self.cz_world.get(cz_id)
        if cz_c is None:
            print(f"[qt] cz_id={cz_id} no centroid (skip)")
            return
        h = self.cube_half
        bb = dict(
            z_lo=cz_c[0] - h, z_hi=cz_c[0] + h,
            y_lo=cz_c[1] - h, y_hi=cz_c[1] + h,
            x_lo=cz_c[2] - h, x_hi=cz_c[2] + h,
        )
        self.cube_bb = bb
        self.cur_cz_id = cz_id
        self.cur_hcr_id = hcr_id
        self.cur_matched = matched
        self.cur_soma = soma

        # MIP z-range = central 80% of the current CZ ROI's z extent (in world µm)
        self.mip_z_world = self._compute_mip_z_range(cz_id)

        # Per-cube contrast levels (HCR 488 + warped CZ).  Push them through
        # the histogram widgets so the draggable region matches the auto values;
        # user can drag to override afterwards.
        self._compute_local_levels(bb)
        self.hist_hcr.setLevels(*self.hcr488_levels)
        if self.czw is not None:
            self.hist_cz.setLevels(*self.czw_levels)

        # Z slider range
        z_steps = int(np.ceil((bb["z_hi"] - bb["z_lo"]) / self.hcr488_voxel[0]))
        self.z_slider.blockSignals(True)
        self.z_slider.setMinimum(0)
        self.z_slider.setMaximum(max(0, z_steps - 1))
        # Center initial Z on CZ centroid
        z_init_step = int(round((cz_c[0] - bb["z_lo"]) / self.hcr488_voxel[0]))
        z_init_step = max(0, min(z_init_step, z_steps - 1))
        self.z_slider.setValue(z_init_step)
        self.z_slider.blockSignals(False)
        self._set_z_step(z_init_step)

        # Build label radio
        self._build_radio(matched)

        # Status label
        status = (
            f"CZ ROI {self.show_idx + 1}/{len(self.cz_order)}  "
            f"cz_id={cz_id}  "
            f"{'matched → hcr_id=' + str(hcr_id) + f'  soma={soma:.2f}' if matched else 'UNMATCHED'}"
        )
        cur_label = self.labels_state.get(cz_id, "—")
        status += f"\nlabel: <b>{cur_label}</b>"
        self.lbl_status.setText(status)

    def _build_radio(self, matched: bool):
        # Clear existing
        for b in self.radio_buttons:
            self.radio_layout.removeWidget(b)
            self.radio_group.removeButton(b)
            b.deleteLater()
        self.radio_buttons.clear()

        if matched:
            options = ["good", "bad", "unsure"]
        else:
            options = ["matched roi visible", "matched roi not visible"]

        cz_id = self.cur_cz_id
        prior = self.labels_state.get(cz_id, None)
        for i, opt in enumerate(options):
            rb = QtWidgets.QRadioButton(f"{i+1}. {opt}")
            if prior == opt:
                rb.setChecked(True)
            rb.toggled.connect(lambda checked, o=opt: checked and self._save_label(o))
            self.radio_layout.addWidget(rb)
            self.radio_buttons.append(rb)
            self.radio_group.addButton(rb, i)

    def _compute_mip_z_range(self, cz_id: int) -> tuple[float, float]:
        """Central 80% of the current CZ ROI's z extent (world µm).
        Look in cz_matched_arr first, else cz_unmatched_arr."""
        for arr in (self.cz_matched_arr, self.cz_unmatched_arr):
            zs = np.argwhere(arr == cz_id)[:, 0] if arr.size else np.array([])
            if zs.size > 0:
                z10 = float(np.percentile(zs, 10))
                z90 = float(np.percentile(zs, 90))
                z_lo = self.cz_bb["z_lo"] + z10 * self.cz_vox
                z_hi = self.cz_bb["z_lo"] + (z90 + 1) * self.cz_vox
                return (z_lo, z_hi)
        # Fallback: cube range
        return (self.cube_bb["z_lo"], self.cube_bb["z_hi"])

    def _compute_local_levels(self, bb):
        """Set self.hcr488_levels and self.czw_levels from the current cube's
        sub-volume so contrast is appropriate for the local region.
        Falls back to global levels if the cube is outside the volume."""
        # HCR 488
        z_um, xy_um, _ = self.hcr488_voxel
        oz, oy, ox = self.hcr488_origin
        z0v = max(0, int((bb["z_lo"] - oz) / z_um))
        z1v = min(self.hcr488.shape[0], int((bb["z_hi"] - oz) / z_um) + 1)
        y0v = max(0, int((bb["y_lo"] - oy) / xy_um))
        y1v = min(self.hcr488.shape[1], int((bb["y_hi"] - oy) / xy_um) + 1)
        x0v = max(0, int((bb["x_lo"] - ox) / xy_um))
        x1v = min(self.hcr488.shape[2], int((bb["x_hi"] - ox) / xy_um) + 1)
        if z0v < z1v and y0v < y1v and x0v < x1v:
            sub = self.hcr488[z0v:z1v, y0v:y1v, x0v:x1v]
            self.hcr488_levels = (
                float(np.percentile(sub, 5)),
                float(np.percentile(sub, 99.5)),
            )
        # Warped CZ
        if self.czw is not None:
            cvox = self.czw_voxel
            oz, oy, ox = self.czw_origin
            z0v = max(0, int((bb["z_lo"] - oz) / cvox))
            z1v = min(self.czw.shape[0], int((bb["z_hi"] - oz) / cvox) + 1)
            y0v = max(0, int((bb["y_lo"] - oy) / cvox))
            y1v = min(self.czw.shape[1], int((bb["y_hi"] - oy) / cvox) + 1)
            x0v = max(0, int((bb["x_lo"] - ox) / cvox))
            x1v = min(self.czw.shape[2], int((bb["x_hi"] - ox) / cvox) + 1)
            if z0v < z1v and y0v < y1v and x0v < x1v:
                sub = self.czw[z0v:z1v, y0v:y1v, x0v:x1v]
                self.czw_levels = (
                    float(np.percentile(sub, 5)),
                    float(np.percentile(sub, 99.5)),
                )

    # ---------------- Z step / image ----------------
    def _on_z_slider(self, step_idx: int):
        self._set_z_step(step_idx)

    def _set_z_step(self, step_idx: int):
        bb = self.cube_bb
        z_world = bb["z_lo"] + step_idx * self.hcr488_voxel[0]
        self.cur_z_world = z_world
        self.lbl_z.setText(f"{z_world:.0f}")
        self._redraw()

    def _z_up(self):
        self.z_slider.setValue(min(self.z_slider.value() + 1, self.z_slider.maximum()))

    def _z_down(self):
        self.z_slider.setValue(max(self.z_slider.value() - 1, 0))

    def _redraw(self):
        bb = self.cube_bb
        z_world = self.cur_z_world
        # ----- HCR 488 -----
        if self.show_hcr488:
            z_um, xy_um, _ = self.hcr488_voxel
            oz, oy, ox = self.hcr488_origin
            if self.mip_mode:
                mlo, mhi = self.mip_z_world
                z0v = max(0, int((mlo - oz) / z_um))
                z1v = min(self.hcr488.shape[0], int((mhi - oz) / z_um) + 1)
                if z0v < z1v:
                    slab = self.hcr488[z0v:z1v].max(axis=0)
                    self.view.set_hcr_image(slab, x_lo=ox, y_lo=oy, xy_um=xy_um)
                else:
                    self.view.img_hcr.clear()
            else:
                z_vox = int(round((z_world - oz) / z_um))
                if 0 <= z_vox < self.hcr488.shape[0]:
                    self.view.set_hcr_image(self.hcr488[z_vox], x_lo=ox, y_lo=oy, xy_um=xy_um)
                else:
                    self.view.img_hcr.clear()
        else:
            self.view.img_hcr.clear()
        # ----- Warped CZ stack -----
        if self.show_czw and self.czw is not None:
            cvox = self.czw_voxel
            oz, oy, ox = self.czw_origin
            if self.mip_mode:
                mlo, mhi = self.mip_z_world
                z0v = max(0, int((mlo - oz) / cvox))
                z1v = min(self.czw.shape[0], int((mhi - oz) / cvox) + 1)
                if z0v < z1v:
                    slab = self.czw[z0v:z1v].max(axis=0)
                    self.view.set_cz_image(slab, x_lo=ox, y_lo=oy, xy_um=cvox)
                else:
                    self.view.img_cz.clear()
            else:
                z_vox = int(round((z_world - oz) / cvox))
                if 0 <= z_vox < self.czw.shape[0]:
                    self.view.set_cz_image(self.czw[z_vox], x_lo=ox, y_lo=oy, xy_um=cvox)
                else:
                    self.view.img_cz.clear()
        else:
            self.view.img_cz.clear()
        # ----- Contours -----
        self.view.clear_contours()
        if self.mip_mode:
            self._draw_cz_contours_mip()
            self._draw_hcr_contours_mip()
        else:
            self._draw_cz_contours_at_z(z_world)
            self._draw_hcr_contours_at_z(z_world)
        self._draw_match_overlay()
        # Initial viewport: cube ± 10% margin (20% larger than cube extent)
        if not getattr(self, "_viewport_set_for_idx", None) == self.show_idx:
            ex = bb["x_hi"] - bb["x_lo"]; ey = bb["y_hi"] - bb["y_lo"]
            self.view.plot.setXRange(bb["x_lo"] - 0.1 * ex, bb["x_hi"] + 0.1 * ex, padding=0)
            self.view.plot.setYRange(bb["y_lo"] - 0.1 * ey, bb["y_hi"] + 0.1 * ey, padding=0)
            self._viewport_set_for_idx = self.show_idx

    def _viewport_xy_range(self):
        """Current visible XY range in world µm: ((y_lo, y_hi), (x_lo, x_hi))."""
        try:
            (x_lo, x_hi), (y_lo, y_hi) = self.view.plot.viewRange()
        except Exception:
            bb = self.cube_bb
            x_lo, x_hi = bb["x_lo"], bb["x_hi"]
            y_lo, y_hi = bb["y_lo"], bb["y_hi"]
        return (float(y_lo), float(y_hi)), (float(x_lo), float(x_hi))

    def _draw_cz_contours_at_z(self, z_world: float):
        cv = self.cz_vox
        z_vox = int(round((z_world - self.cz_bb["z_lo"]) / cv))
        (vy_lo, vy_hi), (vx_lo, vx_hi) = self._viewport_xy_range()
        for arr, base_color in (
            (self.cz_matched_arr, COLOR_OTHER_CZM),
            (self.cz_unmatched_arr, COLOR_OTHER_CZU),
        ):
            if not (0 <= z_vox < arr.shape[0]):
                continue
            y0 = max(0, int((vy_lo - self.cz_bb["y_lo"]) / cv))
            y1 = min(arr.shape[1], int((vy_hi - self.cz_bb["y_lo"]) / cv) + 1)
            x0 = max(0, int((vx_lo - self.cz_bb["x_lo"]) / cv))
            x1 = min(arr.shape[2], int((vx_hi - self.cz_bb["x_lo"]) / cv) + 1)
            slab = arr[z_vox, y0:y1, x0:x1]
            uniq = np.unique(slab); uniq = uniq[uniq != 0]
            for v in uniq.tolist():
                v = int(v)
                if v == self.cur_cz_id:
                    color = COLOR_CUR_CZ; width = WIDTH_CUR
                elif not self.show_other_cz:
                    continue
                else:
                    color = base_color; width = WIDTH_OTHER
                mask = (slab == v).astype(np.uint8)
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                for c in contours:
                    if len(c) < 2: continue
                    pts = c.reshape(-1, 2).astype(float)
                    x = self.cz_bb["x_lo"] + (x0 + pts[:, 0]) * cv
                    y = self.cz_bb["y_lo"] + (y0 + pts[:, 1]) * cv
                    x = np.append(x, x[0]); y = np.append(y, y[0])
                    self.view.add_contour(x, y, color=color, width=width)

    def _draw_hcr_contours_at_z(self, z_world: float):
        z_vox = int(round((z_world - self.hbb["z_lo"]) / self.hcr_vox_z))
        (vy_lo, vy_hi), (vx_lo, vx_hi) = self._viewport_xy_range()
        # Build (arr, color, is_failed) sources.  Failed arrays only included
        # when their toggle is on.
        sources = [
            (self.hcr_matched_arr, COLOR_OTHER_HCRM, False),
            (self.hcr_unmatched_arr, COLOR_OTHER_HCRU, False),
        ]
        if self.show_hcr_fail_gfp and self.hcr_failed_gfp_arr is not None:
            sources.append((self.hcr_failed_gfp_arr, COLOR_HCR_FAIL_GFP, True))
        if self.show_hcr_fail_cls and self.hcr_failed_cls_arr is not None:
            sources.append((self.hcr_failed_cls_arr, COLOR_HCR_FAIL_CLS, True))
        for arr, base_color, is_failed in sources:
            if not (0 <= z_vox < arr.shape[0]):
                continue
            y0 = max(0, int((vy_lo - self.hbb["y_lo"]) / self.hcr_vox_xy))
            y1 = min(arr.shape[1], int((vy_hi - self.hbb["y_lo"]) / self.hcr_vox_xy) + 1)
            x0 = max(0, int((vx_lo - self.hbb["x_lo"]) / self.hcr_vox_xy))
            x1 = min(arr.shape[2], int((vx_hi - self.hbb["x_lo"]) / self.hcr_vox_xy) + 1)
            slab = arr[z_vox, y0:y1, x0:x1]
            uniq = np.unique(slab); uniq = uniq[uniq != 0]
            for v in uniq.tolist():
                v = int(v)
                if not is_failed:
                    is_matched_hcr = (self.cur_matched and v == self.cur_hcr_id)
                    if is_matched_hcr:
                        color = COLOR_CUR_HCR; width = WIDTH_CUR
                    elif not self.show_other_hcr:
                        continue
                    else:
                        color = base_color; width = WIDTH_OTHER
                else:
                    color = base_color; width = WIDTH_OTHER
                mask = (slab == v).astype(np.uint8)
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                for c in contours:
                    if len(c) < 2: continue
                    pts = c.reshape(-1, 2).astype(float)
                    x = self.hbb["x_lo"] + (x0 + pts[:, 0]) * self.hcr_vox_xy
                    y = self.hbb["y_lo"] + (y0 + pts[:, 1]) * self.hcr_vox_xy
                    x = np.append(x, x[0]); y = np.append(y, y[0])
                    self.view.add_contour(x, y, color=color, width=width)

    def _draw_cz_contours_mip(self):
        cv = self.cz_vox
        (vy_lo, vy_hi), (vx_lo, vx_hi) = self._viewport_xy_range()
        # Use the current CZ ROI's 80% z-extent for MIP, not the full cube z
        mip_z_lo, mip_z_hi = self.mip_z_world
        z0 = max(0, int((mip_z_lo - self.cz_bb["z_lo"]) / cv))
        for arr, base_color in (
            (self.cz_matched_arr, COLOR_OTHER_CZM),
            (self.cz_unmatched_arr, COLOR_OTHER_CZU),
        ):
            z1 = min(arr.shape[0], int((mip_z_hi - self.cz_bb["z_lo"]) / cv) + 1)
            if z0 >= z1: continue
            y0 = max(0, int((vy_lo - self.cz_bb["y_lo"]) / cv))
            y1 = min(arr.shape[1], int((vy_hi - self.cz_bb["y_lo"]) / cv) + 1)
            x0 = max(0, int((vx_lo - self.cz_bb["x_lo"]) / cv))
            x1 = min(arr.shape[2], int((vx_hi - self.cz_bb["x_lo"]) / cv) + 1)
            slab = arr[z0:z1, y0:y1, x0:x1]
            uniq = np.unique(slab); uniq = uniq[uniq != 0]
            for v in uniq.tolist():
                v = int(v)
                if v == self.cur_cz_id:
                    color = COLOR_CUR_CZ; width = WIDTH_CUR
                elif not self.show_other_cz:
                    continue
                else:
                    color = base_color; width = WIDTH_OTHER
                mask = (slab == v).any(axis=0).astype(np.uint8)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in contours:
                    if len(c) < 2: continue
                    pts = c.reshape(-1, 2).astype(float)
                    x = self.cz_bb["x_lo"] + (x0 + pts[:, 0]) * cv
                    y = self.cz_bb["y_lo"] + (y0 + pts[:, 1]) * cv
                    x = np.append(x, x[0]); y = np.append(y, y[0])
                    self.view.add_contour(x, y, color=color, width=width)

    def _draw_hcr_contours_mip(self):
        (vy_lo, vy_hi), (vx_lo, vx_hi) = self._viewport_xy_range()
        mip_z_lo, mip_z_hi = self.mip_z_world
        sources = [
            (self.hcr_matched_arr, COLOR_OTHER_HCRM, False),
            (self.hcr_unmatched_arr, COLOR_OTHER_HCRU, False),
        ]
        if self.show_hcr_fail_gfp and self.hcr_failed_gfp_arr is not None:
            sources.append((self.hcr_failed_gfp_arr, COLOR_HCR_FAIL_GFP, True))
        if self.show_hcr_fail_cls and self.hcr_failed_cls_arr is not None:
            sources.append((self.hcr_failed_cls_arr, COLOR_HCR_FAIL_CLS, True))
        for arr, base_color, is_failed in sources:
            z0 = max(0, int((mip_z_lo - self.hbb["z_lo"]) / self.hcr_vox_z))
            z1 = min(arr.shape[0], int((mip_z_hi - self.hbb["z_lo"]) / self.hcr_vox_z) + 1)
            if z0 >= z1: continue
            y0 = max(0, int((vy_lo - self.hbb["y_lo"]) / self.hcr_vox_xy))
            y1 = min(arr.shape[1], int((vy_hi - self.hbb["y_lo"]) / self.hcr_vox_xy) + 1)
            x0 = max(0, int((vx_lo - self.hbb["x_lo"]) / self.hcr_vox_xy))
            x1 = min(arr.shape[2], int((vx_hi - self.hbb["x_lo"]) / self.hcr_vox_xy) + 1)
            slab = arr[z0:z1, y0:y1, x0:x1]
            uniq = np.unique(slab); uniq = uniq[uniq != 0]
            for v in uniq.tolist():
                v = int(v)
                if not is_failed:
                    is_matched_hcr = (self.cur_matched and v == self.cur_hcr_id)
                    if is_matched_hcr:
                        color = COLOR_CUR_HCR; width = WIDTH_CUR
                    elif not self.show_other_hcr:
                        continue
                    else:
                        color = base_color; width = WIDTH_OTHER
                else:
                    color = base_color; width = WIDTH_OTHER
                mask = (slab == v).any(axis=0).astype(np.uint8)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in contours:
                    if len(c) < 2: continue
                    pts = c.reshape(-1, 2).astype(float)
                    x = self.hbb["x_lo"] + (x0 + pts[:, 0]) * self.hcr_vox_xy
                    y = self.hbb["y_lo"] + (y0 + pts[:, 1]) * self.hcr_vox_xy
                    x = np.append(x, x[0]); y = np.append(y, y[0])
                    self.view.add_contour(x, y, color=color, width=width)

    # ---------------- nav + toggles + label ----------------
    def _next(self):
        self.show_idx = (self.show_idx + 1) % len(self.cz_order)
        self._refresh_pair()

    def _prev(self):
        self.show_idx = (self.show_idx - 1) % len(self.cz_order)
        self._refresh_pair()

    def _toggle_other_cz(self):
        self.show_other_cz = self.chk_other_cz.isChecked()
        self._redraw()

    def _toggle_other_hcr(self):
        self.show_other_hcr = self.chk_other_hcr.isChecked()
        self._redraw()

    def _toggle_hcr488(self):
        self.show_hcr488 = self.chk_hcr488.isChecked()
        self._redraw()

    def _toggle_czw(self):
        self.show_czw = self.chk_czw.isChecked()
        self._redraw()

    def _toggle_hcr_fail_gfp(self):
        self.show_hcr_fail_gfp = self.chk_hcr_fail_gfp.isChecked()
        self._redraw_contours_only()

    def _toggle_hcr_fail_cls(self):
        self.show_hcr_fail_cls = self.chk_hcr_fail_cls.isChecked()
        self._redraw_contours_only()

    def _set_mode(self, mip: bool):
        self.mip_mode = bool(mip)
        # Z slider only meaningful in slice mode; gray out in MIP mode
        self.z_slider.setEnabled(not self.mip_mode)
        self._redraw()

    def _click_radio(self, i):
        if 0 <= i < len(self.radio_buttons):
            self.radio_buttons[i].setChecked(True)

    def _save_label(self, label: str):
        cz_id = self.cur_cz_id
        hcr_id = self.cur_hcr_id if self.cur_matched else -1
        ts = _dt.datetime.now().isoformat(timespec="seconds")
        soma = self.cur_soma if self.cur_matched else float("nan")
        row = f"{ts},{self.show_idx},{cz_id},{hcr_id},{soma:.4f},{'matched' if self.cur_matched else 'unmatched'},{label}\n"
        write_header = not self.labels_path.exists()
        with open(self.labels_path, "a") as f:
            if write_header:
                f.write("timestamp,idx,cz_id,hcr_id,soma_score,kind,label\n")
            f.write(row)
        self.labels_state[cz_id] = label
        # Update status label
        s = self.lbl_status.text().split("\n")[0]
        self.lbl_status.setText(s + f"\nlabel: <b>{label}</b>  (saved)")
        print(f"[label] idx={self.show_idx+1} cz_id={cz_id} {'matched' if self.cur_matched else 'unmatched'} → {label}")


def launch(sid: str, variant: str = "local_flow", cube_um: float = CUBE_HALF_UM,
           start: int = 0, hcr_level: int = HCR_LEVEL,
           cz_list_path: str | None = None, matches_csv: str | None = None):
    """Build the QC window and return (QApplication, QCApp) without entering the
    event loop.  Callers that want a blocking GUI call ``app.exec_()`` after."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = QCApp(sid, variant, cube_um, start, hcr_level,
                cz_list_path=cz_list_path, matches_csv=matches_csv)
    win.show()
    return app, win


def main(argv=None):
    args = parse_args(argv)
    app, _win = launch(args.sid, args.variant, args.cube_um, args.start,
                       args.level, cz_list_path=args.cz_list,
                       matches_csv=args.matches_csv)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
