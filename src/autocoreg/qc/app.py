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
  wheel        Z slice ±1 (slice mode) ;  Shift+wheel = zoom
  s / d  switch to slice / MIP-cube-Z mode
  m          toggle QC'd markers
  1 / 2 / 3   matched pair label: good / bad / unsure
  4 / 5       unmatched ROI label: matched-roi visible / not-visible
  q          toggle HCR 488 image
  w          toggle CZ warped image
  c          toggle "other CZ ROIs"
  v          toggle "other HCR ROIs"
  b / n      toggle HCR failed-GFP+ / failed-classifier overlays
  u          toggle add-match mode
  i          toggle batch-accept (MIP: left-click a CZ/HCR overlap to accept/remove)
  Enter      next ROI if labeled (QC mode) / add pending pair (add-match mode)
  Shift+right-click   report CZ + HCR ROI IDs overlapping the point
  right-click         context menu: "Show IDs" + "QC CZ <id>" submenu for the CZ ROI under
                      the cursor (Go to it, or label it in place: good/bad/unsure, or the
                      unmatched options) — QC any specific ROI you point at, out of queue order
  Backspace  undo last added pair (add-match mode)
  Esc        reset the pending selection (add-match mode)

Usage:
  autocoreg run <sid> --qc [--qc-variant anchor_vote_anchor_restricted]
  python -c "from autocoreg.qc.app import main; main(['--sid','790322','--variant','anchor_vote_anchor_restricted'])"
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tifffile
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from autocoreg import config as _config
from autocoreg.io.subjects import load_subject
from autocoreg.io.centroids import centroids_um
from autocoreg.io.hcr_image import hcr_level_resolution

# Artifact inputs + label outputs are resolved from config (env-overridable).
OUT_ROOT = _config.QC_ARTIFACT_DIR
LABELS_ROOT = _config.QC_LABELS_DIR
CUBE_HALF_UM = 60.0
HCR_LEVEL = 2  # default pyramid level; --level overrides

# Auto-contrast clip percentiles (lower, upper) for the 488 / warped-CZ backgrounds. The display
# maps [pct_lo, pct_hi] -> [black, full-colour], so RAISING the upper percentile pushes the white
# point up => a DIMMER image. Default upper raised 99.5 -> 99.9 (the old 99.5 read too bright).
# Env-overridable so operators can tune without an env rebuild (set in qc.sh):
#   MFISH_QC_CLIP_LO (higher = darker floor)  /  MFISH_QC_CLIP_HI (higher = dimmer).
AUTO_CLIP_LO = float(os.environ.get("MFISH_QC_CLIP_LO", "5"))
AUTO_CLIP_HI = float(os.environ.get("MFISH_QC_CLIP_HI", "99.9"))

# ROI-boundary rendering mode:
#   "image"  (default) — draw ALL ROI boundaries for the slice as ONE cached RGBA ImageItem
#            per view (edge overlay). Zoom/pan needs NO rebuild (pyqtgraph transforms the
#            ImageItem); slices are cached; the per-slice build is a single vectorized boundary
#            pass (no per-ROI cv2.findContours). Fixes the zoom/pan/slice clunkiness.
#   "vector" — legacy: one pyqtgraph PlotDataItem per ROI contour, re-extracted (cv2.findContours)
#            on every pan/zoom/slice. Kept as a fallback (set MFISH_QC_CONTOUR_MODE=vector).
CONTOUR_MODE = os.environ.get("MFISH_QC_CONTOUR_MODE", "image").strip().lower()
# Boundary thickness in *data* pixels (voxels), added by dilation. NOTE: unlike the legacy
# vector contours (whose pen width was a constant number of *screen* px regardless of zoom),
# an image-overlay edge is a fixed number of *data* px, so it magnifies as you zoom in.
# Default 0 = a crisp 1-voxel rim (thinnest, closest to the old look at normal zoom); raise to
# 1/2 via MFISH_QC_EDGE_PX only if you want a heavier line.
EDGE_PX = int(os.environ.get("MFISH_QC_EDGE_PX", "0"))
# MFISH_QC_PROFILE=1 prints per-interaction wall-clock timings (full redraw, edge build,
# pan/zoom) so we can tell whether the bottleneck is CPU (boundary extraction) or the
# remote-display repaint.  Off by default (zero overhead when off).
PROFILE = os.environ.get("MFISH_QC_PROFILE", "0").strip().lower() not in ("0", "", "false", "no")

# Orthoview axis map: for each view, (slice_axis, row_axis, col_axis) in volume [z,y,x] order,
# where row→vertical (plot y) and col→horizontal (plot x).  A plane is extracted along
# slice_axis; the two remaining axes come out ascending, so if row_axis > col_axis the plane is
# transposed (see _plane2d).  Layout (radiological, axes visibly shared):
#     [ YZ ][ XY ]     XY (axial, main): slice z; rows=y, cols=x
#     [    ][ XZ ]     XZ (coronal):     slice y; rows=z, cols=x   -> shares x with XY (below it)
#                      YZ (sagittal):    slice x; rows=y, cols=z   -> shares y with XY (left of it)
_VIEW_AX = {"xy": (0, 1, 2), "xz": (1, 0, 2), "yz": (2, 1, 0)}

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


from .positions import (
    POS_COLS,
    compute_centroids,
    fmt_val as _fmt_val,
    position_row as _position_row,
    compute_pair_positions as _compute_pair_positions,
    write_positions_csv as _write_positions_csv,
    cz_world_from_seg as _cz_world_from_seg,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--sid", required=True)
    # Free-text (not choices): must accept the production variant
    # anchor_vote_anchor_restricted as well as the legacy session variants.
    p.add_argument("--variant", default="anchor_vote_anchor_restricted",
                   help="Matcher/QC variant dir name (e.g. anchor_vote_anchor_restricted).")
    p.add_argument("--final-pairs", dest="final_pairs", default=None,
                   help="Path to final_pairs.csv (cz_id,hcr_id,soma_score). If omitted, "
                        "the app looks in the artifact dir and computes+caches it if absent.")
    p.add_argument("--sort", default="soma_desc",
                   choices=["soma_desc", "soma_asc", "matcher"],
                   help="QC queue order: soma_desc = least-confident first (default), "
                        "soma_asc = most-confident first, matcher = matcher row order.")
    p.add_argument("--worst-pct", dest="worst_pct", type=float, default=None,
                   help="Restrict the queue to the worst (highest-soma-distance) N%% of "
                        "matched pairs (matched-only). Omit for the full queue.")
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

    Looks under ``MFISH_QC_MATCHES_DIR/<variant>/<sid>/`` and returns the last
    ``matches_anchor_restricted_round*.csv`` if any, else the last
    ``matches_round*.csv``.  ``matches_wang_round*`` accepted as legacy.
    """
    import re
    d = _config.QC_MATCHES_DIR / variant / sid
    for pat in ("matches_anchor_restricted_round*.csv", "matches_wang_round*.csv"):
        cands = sorted(d.glob(pat), key=lambda p: int(re.findall(r"\d+", p.stem)[-1]))
        if cands:
            return cands[-1]
    rounds = sorted(d.glob("matches_round*.csv"),
                    key=lambda p: int(re.findall(r"\d+", p.stem)[-1]))
    if not rounds:
        raise FileNotFoundError(f"No matches CSVs under {d}")
    return rounds[-1]


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
        # Edge-overlay ImageItems (CONTOUR_MODE == "image"): one RGBA image per view holding
        # ALL ROI boundaries for the current slice. Added last (top zValue) so they render over
        # the two background images; normal source-over compositing (not additive). These are
        # full-slice, positioned in world µm via setRect, so pan/zoom transforms them for free.
        self.edge_hcr = pg.ImageItem(axisOrder="row-major")
        self.edge_hcr.setZValue(10)
        self.plot.addItem(self.edge_hcr)
        self.edge_cz = pg.ImageItem(axisOrder="row-major")
        self.edge_cz.setZValue(11)
        self.plot.addItem(self.edge_cz)
        # Linked-view crosshair (orthoview): dashed lines marking the other two views' planes.
        _chpen = pg.mkPen((255, 255, 255, 150), width=1, style=QtCore.Qt.DashLine)
        self.vline = pg.InfiniteLine(angle=90, movable=False, pen=_chpen)
        self.hline = pg.InfiniteLine(angle=0, movable=False, pen=_chpen)
        self.vline.setZValue(20); self.hline.setZValue(20)
        self.vline.setVisible(False); self.hline.setVisible(False)
        self.plot.addItem(self.vline); self.plot.addItem(self.hline)
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

    def set_edge_image(self, item, rgba, *, x_lo, y_lo, xy_um, y_um=None):
        """Set an RGBA (H,W,4 uint8) edge overlay, positioned in world µm. rgba=None clears.
        xy_um scales cols (horizontal); y_um scales rows (vertical), defaulting to xy_um."""
        if rgba is None:
            item.clear()
            return
        h, w = rgba.shape[:2]
        item.setImage(rgba, autoLevels=False)
        item.setRect(QtCore.QRectF(x_lo, y_lo, w * xy_um, h * (y_um or xy_um)))

    def clear_edges(self):
        self.edge_hcr.clear()
        self.edge_cz.clear()

    def set_crosshair(self, x, y):
        """Show a linked-view crosshair: vertical line at world x (col), horizontal at y (row)."""
        self.vline.setPos(x); self.hline.setPos(y)
        self.vline.setVisible(True); self.hline.setVisible(True)

    def hide_crosshair(self):
        self.vline.setVisible(False); self.hline.setVisible(False)

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

    def set_hcr_image(self, arr_2d, *, x_lo, y_lo, xy_um, y_um=None):
        # xy_um scales the horizontal (col) axis; y_um the vertical (row) axis. y_um defaults
        # to xy_um (isotropic XY view); side views pass y_um = the z-voxel (anisotropic).
        h, w = arr_2d.shape
        self.img_hcr.setImage(arr_2d, autoLevels=False)
        self.img_hcr.setRect(QtCore.QRectF(x_lo, y_lo, w * xy_um, h * (y_um or xy_um)))

    def set_cz_image(self, arr_2d, *, x_lo, y_lo, xy_um, y_um=None):
        if arr_2d is None:
            self.img_cz.clear()
            return
        h, w = arr_2d.shape
        self.img_cz.setImage(arr_2d, autoLevels=False)
        self.img_cz.setRect(QtCore.QRectF(x_lo, y_lo, w * xy_um, h * (y_um or xy_um)))


class QCApp(QtWidgets.QMainWindow):
    def __init__(self, sid: str, variant: str, cube_um: float, start: int,
                 hcr_level: int = HCR_LEVEL, cz_list_path: str | None = None,
                 matches_csv: str | None = None, final_pairs_path: str | None = None,
                 sort_mode: str = "soma_desc", worst_pct: float | None = None):
        super().__init__()
        self.sid = sid
        self.variant = variant
        self.cube_half = float(cube_um)
        self.hcr_level = int(hcr_level)
        self.cz_list_path = cz_list_path
        self.matches_csv = matches_csv
        self.final_pairs_path = final_pairs_path
        self.sort_mode = sort_mode          # soma_desc | soma_asc | matcher
        self.worst_pct = worst_pct          # None = full queue
        self.setWindowTitle(f"QC Qt — {sid} / {variant}")
        self._load_data()
        self._init_state()
        self._build_ui()
        self._update_queue_label()
        self._refresh_qcd_list()
        # initial show
        start = max(0, min(start, len(self.cz_order) - 1))
        self.show_idx = start
        self._refresh_pair()
        # Once shown + laid out, widen the window so the image area is square.
        QtCore.QTimer.singleShot(0, self._square_image)

    # ---------------- data loading ----------------
    def _load_data(self):
        qc_dir = OUT_ROOT / self.variant / self.sid
        if not qc_dir.exists():
            sys.exit(f"missing QC artifacts: {qc_dir}")
        # Launch caches (HCR-488 crop + derived data) live under a dedicated per-subject
        # dir, separate from the artifacts.
        self.cache_dir = _config.QC_CACHE_DIR / self.sid
        self.cache_dir.mkdir(parents=True, exist_ok=True)
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
        # Snapshot the matcher's original mapping so add-match overrides can be undone.
        self._auto_cz_to_hcr = dict(self.cz_to_hcr)
        # Per-pair soma-print score (G2): the production anchor-restricted final CSV has no
        # soma_score column, so load/compute final_pairs.csv (cz_id,hcr_id,soma_score)
        # — soma is a DISTANCE, lower = better match.  cz_to_soma drives the queue sort.
        self.cz_to_soma = self._load_soma_scores(qc_dir, matches_csv)
        finite_soma = {c: s for c, s in self.cz_to_soma.items()
                       if s is not None and np.isfinite(s)}
        self.cz_to_somarank, self.cz_to_somapct = {}, {}
        if finite_soma:
            order_desc = sorted(finite_soma, key=lambda c: finite_soma[c], reverse=True)
            n = len(order_desc)
            for r, c in enumerate(order_desc):
                self.cz_to_somarank[c] = r + 1                  # 1 = least confident
                self.cz_to_somapct[c] = 1.0 - r / max(1, n - 1)  # ~1 = least confident

        # Matched CZ ids (queue built from these + unmatched).
        self.cz_matched_ids = [int(c) for c in self.df_matches["cz_id"]]
        # Derived data (CZ warped centroids, HCR ROI z-ranges, unmatched ids, native
        # centroids, resolutions, HCR dir) — cached to /scratch so repeat launches skip
        # the seg-volume numpy reductions AND load_subject.
        self._load_or_compute_derived(qc_dir, seg_meta)
        self._build_cz_order()
        self.export_dir = LABELS_ROOT / f"exports_{self.sid}"
        self.export_dir.mkdir(parents=True, exist_ok=True)

        # HCR 488 background: read only the crop region (the warped-CZ bbox from the
        # seg meta), and cache it to /scratch so subsequent launches memmap it.
        self._load_hcr488(qc_dir)

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
                float(np.percentile(self.czw, AUTO_CLIP_LO)),
                float(np.percentile(self.czw, AUTO_CLIP_HI)),
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
                if str(r.label) not in ("removed", "", "nan")
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

        print(f"[qt] {len(self.cz_order)} CZ ROIs in queue ({len(self.cz_matched_ids)} matched, "
              f"{len(self.unmatched_only)} unmatched).  "
              f"HCR 488 cube {self.hcr488.shape}, "
              f"{len(self.labels_state)} prior labels.")

    def _load_or_compute_derived(self, qc_dir, seg_meta):
        """Set per-launch derived data: cz_world (CZ warped centroids), hcr_zrange
        (matched HCR ROI z-extents), unmatched_only, native CZ/HCR centroids,
        resolutions, and the HCR dir.  Cached to qc_dir/launch_cache.{json,npz} keyed by
        the seg counts so repeat launches skip the seg-volume reductions + load_subject."""
        cache_json = self.cache_dir / f"launch_cache_{self.variant}.json"
        cache_npz = self.cache_dir / f"launch_cache_{self.variant}.npz"
        key = {"sid": self.sid, "counts": seg_meta.get("counts"),
               "cz_vox": self.cz_vox, "hcr_vox_xy": self.hcr_vox_xy}
        if cache_json.exists() and cache_npz.exists():
            try:
                m = json.loads(cache_json.read_text())
                if m.get("key") == key:
                    d = np.load(cache_npz)
                    self.cz_world = {int(i): p for i, p in zip(d["cw_ids"], d["cw_pos"])}
                    self.hcr_zrange = {int(i): (float(a), float(b))
                                       for i, (a, b) in zip(d["zr_ids"], d["zr"])}
                    self.unmatched_only = [int(v) for v in d["unmatched_only"]]
                    self.cz_native_by_id = {int(i): p for i, p in zip(d["cz_ids"], d["cz_um"])}
                    self.hcr_by_id = {int(i): p for i, p in zip(d["hcr_ids"], d["hcr_um"])}
                    self.cz_xy_um = m["cz_xy_um"]; self.cz_z_um = m["cz_z_um"]
                    self.hcr_xy_um = m["hcr_xy_um"]; self.hcr_z_um = m["hcr_z_um"]
                    self._hcr_dir = Path(m["hcr_dir"])
                    self._hcr_seg_xy_um = m["hcr_seg_xy_um"]
                    self._hcr_seg_z_um = m["hcr_seg_z_um"]
                    self.s = None
                    print("[qt] derived data from launch cache")
                    return
            except Exception as exc:
                print(f"[qt] launch cache unreadable ({exc}); recomputing")
        # ---- cold path: compute everything ----
        from scipy.ndimage import find_objects
        self.hcr_zrange = {}
        for k, sl in enumerate(find_objects(self.hcr_matched_arr)):
            if sl is None:
                continue
            self.hcr_zrange[k + 1] = (self.hbb["z_lo"] + sl[0].start * self.hcr_vox_z,
                                      self.hbb["z_lo"] + sl[0].stop * self.hcr_vox_z)
        unmatched_uniq = sorted(set(int(v) for v in np.unique(self.cz_unmatched_arr) if v != 0))
        matched_set = set(self.cz_matched_ids)
        self.unmatched_only = [v for v in unmatched_uniq if v not in matched_set]
        self.cz_world = _cz_world_from_seg(
            self.cz_matched_arr, self.cz_unmatched_arr, self.cz_bb, self.cz_vox
        )
        s = load_subject(self.sid)
        cz_um, cz_ids = centroids_um(s, "cz")
        hcr_um, hcr_ids = centroids_um(s, "hcr_all")
        self.cz_native_by_id = {int(i): np.asarray(p, float) for i, p in zip(cz_ids, cz_um)}
        self.hcr_by_id = {int(i): np.asarray(p, float) for i, p in zip(hcr_ids, hcr_um)}
        self.cz_xy_um = float(s.cz_xy_um); self.cz_z_um = float(s.cz_z_um)
        self.hcr_xy_um = float(s.hcr_xy_um); self.hcr_z_um = float(s.hcr_z_um)
        self._hcr_dir = Path(s.hcr_dir)
        self._hcr_seg_xy_um = float(s.hcr_seg_xy_um)
        self._hcr_seg_z_um = float(s.hcr_seg_z_um)
        self.s = s
        try:
            cw_ids = np.fromiter(self.cz_world.keys(), dtype=np.int64)
            cw_pos = (np.array([self.cz_world[int(i)] for i in cw_ids], dtype=np.float32)
                      if len(cw_ids) else np.zeros((0, 3), np.float32))
            zr_ids = np.fromiter(self.hcr_zrange.keys(), dtype=np.int64)
            zr = (np.array([self.hcr_zrange[int(i)] for i in zr_ids], dtype=np.float32)
                  if len(zr_ids) else np.zeros((0, 2), np.float32))
            np.savez(cache_npz, cw_ids=cw_ids, cw_pos=cw_pos, zr_ids=zr_ids, zr=zr,
                     unmatched_only=np.array(self.unmatched_only, dtype=np.int64),
                     cz_ids=np.asarray(cz_ids, np.int64), cz_um=np.asarray(cz_um, np.float32),
                     hcr_ids=np.asarray(hcr_ids, np.int64), hcr_um=np.asarray(hcr_um, np.float32))
            cache_json.write_text(json.dumps({
                "key": key, "cz_xy_um": self.cz_xy_um, "cz_z_um": self.cz_z_um,
                "hcr_xy_um": self.hcr_xy_um, "hcr_z_um": self.hcr_z_um,
                "hcr_dir": str(self._hcr_dir), "hcr_seg_xy_um": self._hcr_seg_xy_um,
                "hcr_seg_z_um": self._hcr_seg_z_um}))
        except Exception as exc:
            print(f"[qt] WARN: could not write launch cache ({exc})")

    def _seg_subject(self):
        """A subject-like object exposing the fields open_hcr_seg_zarr_array needs —
        the real SubjectData if loaded, else a stub from the cached HCR dir + seg res
        (so HCR-seg export works on a warm launch without load_subject)."""
        if getattr(self, "s", None) is not None:
            return self.s
        import types
        return types.SimpleNamespace(hcr_dir=self._hcr_dir,
                                     hcr_seg_xy_um=self._hcr_seg_xy_um,
                                     hcr_seg_z_um=self._hcr_seg_z_um)

    def _load_hcr488(self, qc_dir):
        """Load the HCR 488 background cropped to the warped-CZ bbox (self.cz_bb).
        Reads ONLY the crop region from the zarr (not the full pyramid level), then
        caches it under qc_dir as .npy so later launches reload it quickly.

        The cached crop is loaded FULLY INTO RAM (not memmapped). The QC cache dir is
        normally on a network filesystem (/scratch, NFS), where a memmap faults each
        touched slice in over the network on demand -> laggy z-navigation/pan. One bulk
        read into RAM up front makes every later slice access local (instant); the crop
        is small (see the 'MB' in the caching log), so lightly-used RAM has ample room."""
        import time as _time
        bb = self.cz_bb
        level = self.hcr_level
        cache_npy = self.cache_dir / f"hcr488_{self.variant}_L{level}.npy"
        cache_meta = self.cache_dir / f"hcr488_{self.variant}_L{level}.json"
        key = {"bbox": [bb["z_lo"], bb["z_hi"], bb["y_lo"], bb["y_hi"],
                        bb["x_lo"], bb["x_hi"]], "level": int(level)}
        # Fast path: cached crop matching this bbox+level -> load fully into RAM.
        if cache_npy.exists() and cache_meta.exists():
            try:
                m = json.loads(cache_meta.read_text())
                if m.get("key") == key:
                    self.hcr488 = np.load(cache_npy)   # in RAM, NOT mmap (see docstring)
                    self.hcr488_origin = tuple(m["origin"])
                    self.hcr488_voxel = tuple(m["voxel"])
                    self.hcr488_levels = tuple(m["levels"])
                    print(f"[qt] HCR 488 from cache {cache_npy.name} {self.hcr488.shape} "
                          f"(loaded into RAM)")
                    return
            except Exception as exc:
                print(f"[qt] HCR488 cache unreadable ({exc}); re-reading")
        # Slow path: read just the crop from the zarr (uses cached HCR dir/res — no
        # dependency on self.s, so it works whether or not the subject was loaded).
        import zarr
        zpath = self._hcr_dir / "image_tile_fusing" / "fused" / "channel_488.zarr"
        xy_um = self.hcr_xy_um * (2 ** (level - 2))
        z_um = self.hcr_z_um * (2 ** max(0, level - 2))
        last_exc = None
        for attempt in range(3):
            try:
                print(f"[qt] reading HCR 488 crop (level {level}, attempt {attempt+1}/3) ...",
                      flush=True)
                node = zarr.open(str(zpath), mode="r")[str(level)]
                Z, Y, X = node.shape[-3:]
                z0 = max(0, int(bb["z_lo"] / z_um)); z1 = min(Z, int(bb["z_hi"] / z_um) + 1)
                y0 = max(0, int(bb["y_lo"] / xy_um)); y1 = min(Y, int(bb["y_hi"] / xy_um) + 1)
                x0 = max(0, int(bb["x_lo"] / xy_um)); x1 = min(X, int(bb["x_hi"] / xy_um) + 1)
                crop = np.asarray(node[0, 0, z0:z1, y0:y1, x0:x1]).astype(np.float32)
                break
            except (OSError, IOError) as exc:
                last_exc = exc
                print(f"[qt] HCR 488 read failed ({type(exc).__name__}: {exc}); retry 5s",
                      file=sys.stderr)
                _time.sleep(5)
        else:
            raise RuntimeError(f"HCR 488 zarr read failed after 3 attempts: {last_exc}")
        self.hcr488 = crop
        self.hcr488_origin = (z0 * z_um, y0 * xy_um, x0 * xy_um)
        self.hcr488_voxel = (float(z_um), float(xy_um), float(xy_um))
        self.hcr488_levels = (float(np.percentile(crop, AUTO_CLIP_LO)),
                              float(np.percentile(crop, AUTO_CLIP_HI)))
        print(f"[qt] HCR 488 crop {crop.shape} ({crop.nbytes/1e6:.0f} MB) "
              f"@ {xy_um:.3f}µm xy; caching ...", flush=True)
        try:
            np.save(cache_npy, crop)
            cache_meta.write_text(json.dumps({
                "key": key, "origin": list(self.hcr488_origin),
                "voxel": list(self.hcr488_voxel), "levels": list(self.hcr488_levels),
                "shape": list(crop.shape)}))
        except Exception as exc:
            print(f"[qt] WARN: could not cache HCR 488 ({exc})")

    # ---------------- soma scores + review queue (G3) ----------------
    def _load_soma_scores(self, qc_dir, matches_csv) -> dict:
        """Resolve {cz_id: soma_score}.  Order: (1) the matcher's own ``soma_score``
        column if present (written during autocoreg — no recompute); else (2)
        --final-pairs / <qc_dir>/final_pairs.csv; else (3) compute via score_final_pairs.
        Soma is a DISTANCE (lower = better)."""
        if "soma_score" in self.df_matches.columns:
            soma = {int(c): float(s) for c, s in
                    zip(self.df_matches["cz_id"], self.df_matches["soma_score"])}
            n_fin = int(np.isfinite(np.fromiter(soma.values(), dtype=float)).sum())
            print(f"[qt] soma scores from matcher CSV ({n_fin} finite / {len(soma)})")
            return soma
        fp = Path(self.final_pairs_path) if self.final_pairs_path else (qc_dir / "final_pairs.csv")
        if not fp.exists():
            try:
                from .score_final_pairs import score_final_pairs
                print(f"[qt] computing soma scores (no {fp.name}); one-time ~1 min ...",
                      flush=True)
                score_final_pairs(self.sid, matches_csv, out_csv=fp)
            except Exception as e:
                print(f"[qt] WARN: soma scoring unavailable ({e}); queue = matcher order")
                return {int(c): float("nan") for c in self.df_matches["cz_id"]}
        try:
            d = pd.read_csv(fp)
            soma = {int(c): float(s) for c, s in zip(d["cz_id"], d["soma_score"])}
            n_fin = int(np.isfinite(np.fromiter(soma.values(), dtype=float)).sum())
            print(f"[qt] soma scores: {fp} ({n_fin} finite / {len(soma)})")
            return soma
        except Exception as e:
            print(f"[qt] WARN: could not read {fp} ({e}); queue = matcher order")
            return {int(c): float("nan") for c in self.df_matches["cz_id"]}

    def _soma_of(self, cz_id):
        s = self.cz_to_soma.get(int(cz_id), float("nan"))
        return s if (s is not None and np.isfinite(s)) else None

    def _build_cz_order(self):
        """Build self.cz_order from matched (soma-sorted) + unmatched, honouring
        sort_mode (soma_desc=least-confident first), worst_pct, and optional --cz_list."""
        matched = list(self.cz_matched_ids)
        finite = [c for c in matched if self._soma_of(c) is not None]
        nan_m = [c for c in matched if self._soma_of(c) is None]
        if self.sort_mode == "soma_desc":
            finite.sort(key=self._soma_of, reverse=True)
            matched_sorted = finite + nan_m
        elif self.sort_mode == "soma_asc":
            finite.sort(key=self._soma_of)
            matched_sorted = finite + nan_m
        else:  # matcher row order
            matched_sorted = matched
        if self.worst_pct is not None and finite:
            k = max(1, int(np.ceil(self.worst_pct / 100.0 * len(finite))))
            worst = set(sorted(finite, key=self._soma_of, reverse=True)[:k])
            order = [c for c in matched_sorted if c in worst]  # matched-only focus
        else:
            order = matched_sorted + list(self.unmatched_only)
        if self.cz_list_path:
            ext = pd.read_csv(self.cz_list_path)
            wanted = [int(c) for c in ext["cz_id"].astype(int).tolist()]
            avail = set(matched) | set(self.unmatched_only)
            order = [c for c in wanted if c in avail]
            print(f"[qt] --cz_list restricts to {len(order)}/{len(wanted)} present CZ ids")
        self.cz_order = order
        print(f"[qt] queue: {len(order)} ROIs (sort={self.sort_mode}"
              + (f", worst {self.worst_pct:.0f}%" if self.worst_pct is not None else "")
              + f"; {len(finite)} matched w/soma, {len(self.unmatched_only)} unmatched)")

    def _rebuild_queue(self):
        """Re-read the queue widgets, rebuild cz_order, stay on the current ROI if
        still present (else reset to 0), refresh."""
        idx = self.cmb_sort.currentIndex()
        self.sort_mode = {0: "soma_desc", 1: "soma_asc", 2: "matcher"}[idx]
        w = float(self.spin_worst.value())
        self.worst_pct = w if w > 0 else None
        cur = self.cz_order[self.show_idx] if self.cz_order else None
        self._build_cz_order()
        self._update_queue_label()
        self.show_idx = self.cz_order.index(cur) if (cur in self.cz_order) else 0
        if self.cz_order:
            self._refresh_pair()

    def _update_queue_label(self):
        if hasattr(self, "lbl_queue"):
            self.lbl_queue.setText(f"{len(self.cz_order)} ROIs in queue")

    def _init_state(self):
        self.show_idx = 0
        self.cur_z_world = 0.0
        self.cur_y_world = 0.0   # XZ side-view plane (sliced along y) + crosshair y
        self.cur_x_world = 0.0   # YZ side-view plane (sliced along x) + crosshair x
        self.show_side_views = False  # linked-crosshair orthoview (toggle ` / top-left button)
        self.show_other_cz = True
        self.show_other_hcr = True
        self.show_cur_cz = True   # current pair's CZ ROI contour
        self.show_cur_hcr = True  # current pair's matched HCR ROI contour
        self.show_czw = True
        self.show_hcr488 = True
        self.show_hcr_fail_gfp = False
        self.show_hcr_fail_cls = False
        self.show_qc_markers = True   # spatial dots for QC'd pairs (good/bad/unsure)
        self.skip_qcd = True          # Next/Prev skip already-QC'd pairs
        self.batch_accept_mode = False  # MIP left-click accept (function 3)
        self._id_text_item = None     # ephemeral on-image ROI-ID text (right-click)
        self.mip_mode = False  # toggled by 'm' / radio
        # CONTOUR_MODE=="image": per-(view, slice, toggle-state, current-id) RGBA edge cache.
        self._edge_cache: dict = {}

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
        # Replay prior manual add/undo events (resume across sessions).  Also re-apply
        # the cz->hcr override so the QC'd list / export show the manual partner.
        for action, cz, hc in getattr(self, "_prior_manual_events", []):
            if action == "add" and cz in self.cz_native_by_id and hc in self.hcr_by_id:
                self.active_pairs[cz] = hc
                self.cz_to_hcr[cz] = hc
                if cz in self.added_order:
                    self.added_order.remove(cz)
                self.added_order.append(cz)
            elif action == "undo" and self.added_order:
                if cz in self.active_pairs:
                    self.active_pairs.pop(cz, None)
                if cz in self.added_order:
                    self.added_order.remove(cz)
                orig = self._auto_cz_to_hcr.get(cz)
                if orig is not None:
                    self.cz_to_hcr[cz] = int(orig)
                    if int(orig) in self.hcr_by_id:
                        self.active_pairs[cz] = int(orig)
                else:
                    self.cz_to_hcr.pop(cz, None)
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
        """Left-click drives add-match / batch-accept modes.  Right-click ROI-ID
        inspection is handled in eventFilter (Shift) + the right-click menu."""
        if ev.button() != QtCore.Qt.LeftButton:
            return
        if not (self.add_match_mode or self.batch_accept_mode):
            return
        if not self.view.plot.sceneBoundingRect().contains(ev.scenePos()):
            return
        vb = self.view.plot.getViewBox()
        pt = vb.mapSceneToView(ev.scenePos())
        # Batch-accept (MIP): click a CZ/HCR overlap to accept the pair.
        if self.batch_accept_mode:
            if not self.mip_mode:
                self._notify("Batch-accept: switch to MIP mode (m) first")
                return
            self._batch_click(float(pt.x()), float(pt.y()))
            return
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
        # Make the manual link the pair's HCR partner everywhere, and record it as a
        # QC'd pair (label "manual") so it shows in the QC'd list + exports.
        self.cz_to_hcr[cz] = hc
        self._write_label(cz, hc, float(self.cz_to_soma.get(cz, float("nan"))),
                          "manual", "manual")
        self.pending_cz_id = None
        self.pending_hcr_id = None
        self._set_match_status(self._counts_summary() + "  (added, TPS refit)")
        self._notify(f"✓ manual cz{cz} ↔ hcr{hc}   (QC'd: {len(self.labels_state)})",
                     kind="ok")
        self._update_match_buttons()
        self._redraw()

    def _undo_pair(self):
        if not self.added_order:
            self._set_match_status("nothing to undo")
            return
        cz = self.added_order.pop()
        hc = self.active_pairs.pop(cz, None)
        # Restore the original (matcher) mapping for this cz, and drop the manual QC entry.
        orig = self._auto_cz_to_hcr.get(cz)
        if orig is not None:
            self.cz_to_hcr[cz] = int(orig)
            if int(orig) in self.hcr_by_id:
                self.active_pairs[cz] = int(orig)
        else:
            self.cz_to_hcr.pop(cz, None)
        if self.labels_state.get(cz) == "manual":
            self._remove_label(cz)
        self._refit_tps()
        self._save_manual_match(cz, hc if hc is not None else -1, "undo")
        self._set_match_status(self._counts_summary() + f"  (undid cz_id={cz})")
        self._notify(f"✗ undid manual cz{cz}", kind="warn")
        self._update_match_buttons()
        self._redraw()

    def _reset_selection(self):
        self.pending_cz_id = None
        self.pending_hcr_id = None
        self._set_match_status("selection reset")
        self._update_match_buttons()
        self._redraw_contours_only()

    # ---------------- on-demand CZ volume re-warp (apply manual TPS to image + seg) --------
    def _native_cz_data(self):
        """Native CZ segmentation labels + 488 image (both ZYX), loaded once and cached.
        Same sources build_qc_artifacts uses (find_cz_seg_tiff + load_cz_volume)."""
        if getattr(self, "_native_cz_seg", None) is None:
            from autocoreg.qc.build_artifacts import find_cz_seg_tiff
            from autocoreg.io.cz_volume import load_cz_volume
            seg = tifffile.imread(str(find_cz_seg_tiff(self.sid))).astype(np.int32, copy=False)
            while seg.ndim > 3 and seg.shape[0] == 1:
                seg = seg[0]
            if getattr(self, "s", None) is None:   # warm-cache launch skipped load_subject
                self.s = load_subject(self.sid)
            self._native_cz_seg = seg
            self._native_cz_vol = load_cz_volume(self.s)
        return self._native_cz_seg, self._native_cz_vol

    def _rewarp_cz_view(self):
        """Re-resample the warped-CZ background image + CZ ROI segmentation through the
        CURRENT (manual) landmark TPS, on the existing display grid (cz_bb @ cz_vox — shared
        by czw and cz_*_arr), replacing the pre-baked ones in memory.  This is the inverse-TPS
        pass from build_artifacts, but driven by ``active_pairs`` instead of the matches CSV,
        so the red CZ image + its outlines reflect the manual landmarks (adding a landmark
        alone only moves the CZ centroid markers).  On-demand: it is a full volume resample.
        Restored to the baked warp on relaunch (the .tif files on disk are untouched)."""
        from scipy.interpolate import Rbf
        from scipy.ndimage import map_coordinates
        pairs = [(c, h) for c, h in self.active_pairs.items()
                 if c in self.cz_native_by_id and h in self.hcr_by_id]
        if len(pairs) < 4:
            self._notify(f"re-warp needs ≥4 active landmarks (have {len(pairs)})", kind="warn")
            return
        nz, ny, nx = self.cz_matched_arr.shape   # display grid (czw shares it exactly)
        bb, vox = self.cz_bb, self.cz_vox
        z_lo, y_lo, x_lo = bb["z_lo"], bb["y_lo"], bb["x_lo"]
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            self._notify(f"re-warping CZ view through {len(pairs)} landmarks…", kind="info")
            QtWidgets.QApplication.processEvents()
            # Inverse TPS: HCR µm -> CZ-native µm (per build_artifacts._fit_inverse_tps).
            hcr = np.array([self.hcr_by_id[h] for _, h in pairs], dtype=float)       # (N,3) zyx
            czn = np.array([self.cz_native_by_id[c] for c, _ in pairs], dtype=float)  # (N,3) zyx
            rbf = [Rbf(hcr[:, 0], hcr[:, 1], hcr[:, 2], czn[:, a], function="thin_plate")
                   for a in range(3)]
            try:
                cz_seg, cz_vol = self._native_cz_data()
            except Exception as e:
                # Re-warp needs the raw CZ seg TIFF + 488 stack under DATA_ROOT; the interactive
                # QC capsule may only attach the pre-built QC artifacts.  Report, don't crash.
                import traceback; traceback.print_exc()
                self._notify(f"re-warp unavailable: native CZ data not found ({type(e).__name__}). "
                             f"Attach the multiplane-ophys segmentation + CZ z-stack asset.",
                             kind="err")
                return
            y_centers = y_lo + (np.arange(ny) + 0.5) * vox
            x_centers = x_lo + (np.arange(nx) + 0.5) * vox
            Y, X = np.meshgrid(y_centers, x_centers, indexing="ij")
            Y_flat, X_flat = Y.ravel(), X.ravel()
            warped_cz = np.zeros((nz, ny, nx), dtype=np.int32)
            warped_img = np.zeros((nz, ny, nx), dtype=np.float32)
            for k in range(nz):
                z = z_lo + (k + 0.5) * vox
                Z_flat = np.full_like(Y_flat, z)
                cz_z = rbf[0](Z_flat, Y_flat, X_flat) / self.cz_z_um
                cz_y = rbf[1](Z_flat, Y_flat, X_flat) / self.cz_xy_um
                cz_x = rbf[2](Z_flat, Y_flat, X_flat) / self.cz_xy_um
                coords = np.stack([cz_z, cz_y, cz_x])
                warped_cz[k] = map_coordinates(
                    cz_seg, coords, order=0, mode="constant", cval=0).reshape(ny, nx)
                warped_img[k] = map_coordinates(
                    cz_vol, coords, order=1, mode="constant", cval=0.0).reshape(ny, nx)
                if k % 8 == 0:
                    self._notify(f"re-warping CZ view… z {k + 1}/{nz}", kind="info")
                    QtWidgets.QApplication.processEvents()
            # Split by the CURRENT matched set (manual adds updated cz_to_hcr) + replace.
            matched_cz_ids = np.fromiter(
                (int(c) for c in self.cz_to_hcr.keys()), dtype=np.int32)
            m = np.isin(warped_cz, matched_cz_ids)
            self.cz_matched_arr = np.where(m, warped_cz, 0).astype(np.int32)
            self.cz_unmatched_arr = np.where(~m & (warped_cz > 0), warped_cz, 0).astype(np.int32)
            self.czw = warped_img
            self.czw_voxel = vox
            self.czw_origin = (z_lo, y_lo, x_lo)
            self.czw_levels = (float(np.percentile(warped_img, AUTO_CLIP_LO)),
                               float(np.percentile(warped_img, AUTO_CLIP_HI)))
            self._edge_cache.clear()   # CZ seg changed → drop cached edge overlays
            self._redraw()
            self._notify(f"✓ CZ view re-warped through {len(pairs)} landmarks "
                         f"(relaunch to restore baked warp)", kind="ok")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

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
        """Overlay: QC'd-pair markers (always) + add-match markers (add-match mode)."""
        self.view.clear_overlay()
        self._draw_qc_markers()
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
        if self.add_match_mode and self.chk_batch.isChecked():
            self.chk_batch.setChecked(False)  # mutually exclusive click modes
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
        # Three resizable panes (image | controls | QC'd) — drag the handles to
        # adjust each pane's width.
        h = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        h.setChildrenCollapsible(False)
        h.setHandleWidth(6)
        self._splitter = h
        self.setCentralWidget(h)

        # Left column: compact horizontal toggle toolbar above the image area.
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(); lv.setContentsMargins(0, 0, 0, 0); lv.setSpacing(2)
        left.setLayout(lv)
        bar = QtWidgets.QHBoxLayout(); bar.setSpacing(10); bar.setContentsMargins(2, 0, 2, 0)
        self.view = CubeView()          # main XY (axial)
        self.view_xz = CubeView()       # side: rows=z, cols=x  (coronal), plane at cur_y
        self.view_yz = CubeView()       # side: rows=z, cols=y  (sagittal), plane at cur_x

        def _tb(text, checked, slot, tip=""):
            cb = QtWidgets.QCheckBox(text)
            cb.setChecked(checked)          # set before connecting -> no slot call now
            if tip:
                cb.setToolTip(tip)
            cb.stateChanged.connect(lambda _=0: slot())
            bar.addWidget(cb)
            return cb
        # Top-left: checkable BUTTON toggling the linked-crosshair orthoview (XZ + YZ). Shortcut `.
        self.btn_ortho = QtWidgets.QPushButton("ortho (`)")
        self.btn_ortho.setCheckable(True)
        self.btn_ortho.setChecked(False)
        self.btn_ortho.setToolTip("Linked-crosshair side views (XZ below, YZ left; axes shared). "
                                  "Scroll a side view to move its plane. Shortcut: `")
        self.btn_ortho.toggled.connect(lambda _=0: self._toggle_side_views())
        bar.addWidget(self.btn_ortho)
        self.chk_hcr488 = _tb("488 (q)", True, self._toggle_hcr488, "HCR 488 image")
        self.chk_czw = _tb("CZ img (w)", True, self._toggle_czw, "CZ warped image")
        self.chk_cur_cz = _tb("pair CZ (z)", True, self._toggle_cur_cz)
        self.chk_cur_hcr = _tb("pair HCR (x)", True, self._toggle_cur_hcr)
        self.chk_other_cz = _tb("other CZ (c)", True, self._toggle_other_cz)
        self.chk_other_hcr = _tb("other HCR (v)", True, self._toggle_other_hcr)
        self.chk_hcr_fail_gfp = _tb("failGFP (b)", False, self._toggle_hcr_fail_gfp)
        self.chk_hcr_fail_cls = _tb("failCLS (n)", False, self._toggle_hcr_fail_cls)
        self.chk_qc_markers = _tb("markers (m)", True, self._toggle_qc_markers,
                                  "QC'd pair dots: good=green, bad=red, unsure=yellow")
        bar.addStretch(1)
        lv.addLayout(bar)
        # Image area (radiological orthoview grid), so the shared axes line up pixel-for-pixel:
        #     [ YZ ][ XY ]     XY & XZ share column 1  -> x aligned
        #     [    ][ XZ ]     YZ & XY share row 0      -> y aligned
        # Side views start hidden + their row/col stretch 0, so XY fills the whole area until the
        # orthoview is toggled on (_toggle_side_views flips visibility + stretch).
        image_area = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(image_area)
        grid.setContentsMargins(0, 0, 0, 0); grid.setSpacing(2)
        grid.addWidget(self.view_yz, 0, 0)
        grid.addWidget(self.view, 0, 1)
        grid.addWidget(self.view_xz, 1, 1)
        grid.setColumnStretch(0, 0); grid.setColumnStretch(1, 1)
        grid.setRowStretch(0, 1); grid.setRowStretch(1, 0)
        self._img_grid = grid
        self.view_yz.setVisible(False)
        self.view_xz.setVisible(False)
        lv.addWidget(image_area, stretch=1)
        h.addWidget(left)

        # Right control panel — scrollable so the window can shrink and add-match
        # mode can't push controls off-screen.
        panel = QtWidgets.QWidget()
        pl = QtWidgets.QVBoxLayout()
        pl.setContentsMargins(4, 4, 18, 4)  # right margin leaves room for the scrollbar
        pl.setSpacing(4)
        panel.setLayout(pl)
        panel_scroll = QtWidgets.QScrollArea()
        panel_scroll.setWidgetResizable(True)
        panel_scroll.setWidget(panel)
        panel_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        panel_scroll.setMinimumWidth(180)  # splitter-resizable; no max so it can widen
        h.addWidget(panel_scroll)

        self.lbl_status = QtWidgets.QLabel("…")
        self.lbl_status.setStyleSheet("font-weight: bold;")
        self.lbl_status.setWordWrap(True)   # wrap instead of clipping on the right
        pl.addWidget(self.lbl_status)

        # ---- Review queue controls (G3): soma-sorted, least-confident first ----
        q_box = QtWidgets.QGroupBox("Review queue")
        ql = QtWidgets.QVBoxLayout(); q_box.setLayout(ql)
        ql.addWidget(QtWidgets.QLabel("Sort (soma distance: lower = better match):"))
        self.cmb_sort = QtWidgets.QComboBox()
        self.cmb_sort.addItems(["Least-confident first", "Most-confident first",
                                "Matcher order"])
        self.cmb_sort.setCurrentIndex(
            {"soma_desc": 0, "soma_asc": 1, "matcher": 2}.get(self.sort_mode, 0))
        self.cmb_sort.currentIndexChanged.connect(lambda _: self._rebuild_queue())
        ql.addWidget(self.cmb_sort)
        wrow = QtWidgets.QHBoxLayout()
        wrow.addWidget(QtWidgets.QLabel("Worst %:"))
        self.spin_worst = QtWidgets.QDoubleSpinBox()
        self.spin_worst.setRange(0, 100); self.spin_worst.setDecimals(0)
        self.spin_worst.setSingleStep(5)
        self.spin_worst.setValue(self.worst_pct if self.worst_pct is not None else 0)
        self.spin_worst.setToolTip("0 = full queue; N = only the worst N% matched pairs")
        btn_worst = QtWidgets.QPushButton("Apply")
        btn_worst.clicked.connect(self._rebuild_queue)
        wrow.addWidget(self.spin_worst); wrow.addWidget(btn_worst)
        ql.addLayout(wrow)
        self.chk_skip_qcd = QtWidgets.QCheckBox("Un-QC'd only (Next/Prev)")
        self.chk_skip_qcd.setChecked(self.skip_qcd)
        self.chk_skip_qcd.setToolTip("On: Next/Prev skip already-QC'd pairs (walk only "
                                     "undecided ones).  Off: step through all in sort order.")
        self.chk_skip_qcd.stateChanged.connect(
            lambda _: setattr(self, "skip_qcd", self.chk_skip_qcd.isChecked()))
        ql.addWidget(self.chk_skip_qcd)
        self.lbl_queue = QtWidgets.QLabel("")
        self.lbl_queue.setWordWrap(True)
        ql.addWidget(self.lbl_queue)
        pl.addWidget(q_box)

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

        # View mode: slice / MIP.  (Image/ROI toggle checkboxes are in the top toolbar.)
        mode_box = QtWidgets.QGroupBox("View mode")
        mode_layout = QtWidgets.QHBoxLayout()
        mode_box.setLayout(mode_layout)
        self.rad_slice = QtWidgets.QRadioButton("Slice (s)")
        self.rad_slice.setChecked(True)
        self.rad_mip = QtWidgets.QRadioButton("MIP cube Z (d)")
        self.rad_slice.toggled.connect(lambda checked: checked and self._set_mode(False))
        self.rad_mip.toggled.connect(lambda checked: checked and self._set_mode(True))
        mode_layout.addWidget(self.rad_slice)
        mode_layout.addWidget(self.rad_mip)
        pl.addWidget(mode_box)

        # Radio group — built lazily per-pair (different options for matched/unmatched)
        self.radio_box = QtWidgets.QGroupBox("Label (auto-save)")
        self.radio_layout = QtWidgets.QVBoxLayout()
        self.radio_box.setLayout(self.radio_layout)
        self.radio_buttons: list[QtWidgets.QRadioButton] = []
        self.radio_group = QtWidgets.QButtonGroup()
        pl.addWidget(self.radio_box)

        # ---- Add-match mode (function 2) ----
        self.chk_add_match = QtWidgets.QCheckBox("Add-match mode (u)")
        self.chk_add_match.setChecked(False)
        self.chk_add_match.stateChanged.connect(lambda _: self._toggle_add_match())
        pl.addWidget(self.chk_add_match)
        self.chk_batch = QtWidgets.QCheckBox("Batch-accept MIP (i)")
        self.chk_batch.setToolTip("MIP mode: left-click a CZ/HCR overlap to accept "
                                  "(label good); click again to remove.")
        self.chk_batch.setChecked(False)
        self.chk_batch.stateChanged.connect(lambda _: self._toggle_batch_accept())
        pl.addWidget(self.chk_batch)

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
        # On-demand: re-resample the CZ background image + CZ ROI segmentation through the
        # CURRENT (manual) TPS so the red image + its outlines reflect the added landmarks.
        # Not per-add — it is a full inverse-TPS volume resample (seconds).
        self.btn_rewarp = QtWidgets.QPushButton("Re-warp CZ view (apply TPS)")
        self.btn_rewarp.setToolTip(
            "Re-resample the warped-CZ image + CZ ROI outlines through the current manual "
            "landmark TPS, on the display grid. Adding a landmark only moves the CZ centroid "
            "markers live; this updates the image + contours too. Restored to the baked warp "
            "on relaunch.")
        self.btn_rewarp.clicked.connect(self._rewarp_cz_view)
        mbl.addWidget(self.btn_rewarp)
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

        pl.addStretch(1)

        # ---- Far-right column: QC'd pairs (scrollable) + last action + save target ----
        qcd_panel = QtWidgets.QWidget()
        qcd_panel.setMinimumWidth(180)  # splitter-resizable; no max so it can widen
        qv = QtWidgets.QVBoxLayout()
        qv.setContentsMargins(4, 4, 4, 4); qv.setSpacing(4)
        qcd_panel.setLayout(qv)
        # last-action notification (coloured: green=accepted, orange=removed, red=rejected)
        self.lbl_action = QtWidgets.QLabel("")
        self.lbl_action.setWordWrap(True)
        self.lbl_action.setStyleSheet("font-weight: bold;")
        qv.addWidget(self.lbl_action)
        self.qcd_box = QtWidgets.QGroupBox("QC'd pairs (0)")
        ql2 = QtWidgets.QVBoxLayout(); self.qcd_box.setLayout(ql2)
        self.qcd_list = QtWidgets.QListWidget()
        self._qcd_items = {}  # cz_id -> QListWidgetItem (incremental updates)
        self.qcd_list.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.qcd_list.setToolTip("Double-click to jump to a pair")
        self.qcd_list.itemDoubleClicked.connect(self._on_qcd_double)
        ql2.addWidget(self.qcd_list, stretch=1)
        b_sort = QtWidgets.QPushButton("Sort")
        b_sort.setToolTip("Sort the list by label then cz_id (else newest-last order)")
        b_sort.clicked.connect(self._sort_qcd_list)
        ql2.addWidget(b_sort)
        b_rm = QtWidgets.QPushButton("Remove selected")
        b_rm.clicked.connect(self._remove_selected_qcd)
        ql2.addWidget(b_rm)
        qv.addWidget(self.qcd_box, stretch=1)
        # ---- Export (XYZ positions + HCR seg crop) ----
        exp_box = QtWidgets.QGroupBox("Export")
        el = QtWidgets.QVBoxLayout(); exp_box.setLayout(el)
        b_copy = QtWidgets.QPushButton("Copy pair XYZ (p)")
        b_copy.setToolTip("Copy current pair's XYZ in czstack-native + HCR coords to clipboard")
        b_copy.clicked.connect(self._copy_positions)
        b_allpos = QtWidgets.QPushButton("Export all positions CSV")
        b_allpos.clicked.connect(self._export_all_positions)
        b_segcrop = QtWidgets.QPushButton("Export HCR seg crop")
        b_segcrop.setToolTip("Save raw HCR segmentation labels near this pair as a tif")
        b_segcrop.clicked.connect(self._export_hcr_seg_crop)
        el.addWidget(b_copy); el.addWidget(b_allpos); el.addWidget(b_segcrop)
        self.lbl_export = QtWidgets.QLabel("")
        self.lbl_export.setWordWrap(True)
        el.addWidget(self.lbl_export)
        qv.addWidget(exp_box)
        # save target: show path in a read-only, horizontally-scrollable field (drag /
        # arrow-keys to see the full path; tooltip has it in full) + a chooser button.
        qv.addWidget(QtWidgets.QLabel("Saving QC labels to:"))
        self.savepath_edit = QtWidgets.QLineEdit()
        self.savepath_edit.setReadOnly(True)
        self.savepath_edit.setStyleSheet("color: #555;")
        qv.addWidget(self.savepath_edit)
        b_save = QtWidgets.QPushButton("Set save file…")
        b_save.clicked.connect(self._choose_save_file)
        qv.addWidget(b_save)
        h.addWidget(qcd_panel)
        # Only the image pane stretches when the window resizes; panels keep width
        # until dragged.  Initial pane widths:
        h.setStretchFactor(0, 1)
        h.setStretchFactor(1, 0)
        h.setStretchFactor(2, 0)
        h.setSizes([760, 250, 250])
        self._update_savepath_label()

        # Keyboard shortcuts
        self._mk_shortcut("Right", self._next)
        self._mk_shortcut("Left", self._prev)
        self._mk_shortcut("Up", self._z_up)
        self._mk_shortcut("Down", self._z_down)
        self._mk_shortcut("`", lambda: self.btn_ortho.toggle())  # toggle orthoview
        # Radio labels: matched 1/2/3 (good/bad/unsure); unmatched 4/5 (visible/not).
        self._mk_shortcut("1", lambda: self._radio_key(1))
        self._mk_shortcut("2", lambda: self._radio_key(2))
        self._mk_shortcut("3", lambda: self._radio_key(3))
        # 5 selects "not visible" on unmatched ROIs (no-op otherwise).
        self._mk_shortcut("5", lambda: self._radio_key(5))
        self._mk_shortcut("c", lambda: self.chk_other_cz.toggle())
        self._mk_shortcut("v", lambda: self.chk_other_hcr.toggle())
        self._mk_shortcut("z", lambda: self.chk_cur_cz.toggle())
        self._mk_shortcut("x", lambda: self.chk_cur_hcr.toggle())
        self._mk_shortcut("p", self._copy_positions)
        # 4 selects "visible" on unmatched ROIs (no-op on matched).
        self._mk_shortcut("4", lambda: self._radio_key(4))
        self._mk_shortcut("q", lambda: self.chk_hcr488.toggle())  # q → 488 image
        self._mk_shortcut("w", lambda: self.chk_czw.toggle())     # w → warped CZ image
        self._mk_shortcut("m", lambda: self.chk_qc_markers.toggle())  # m → QC'd markers
        self._mk_shortcut("s", lambda: self.rad_slice.setChecked(True))
        self._mk_shortcut("d", lambda: self.rad_mip.setChecked(True))  # d → MIP cube
        self._mk_shortcut("b", lambda: self.chk_hcr_fail_gfp.toggle())
        self._mk_shortcut("n", lambda: self.chk_hcr_fail_cls.toggle())
        self._mk_shortcut("u", lambda: self.chk_add_match.toggle())
        self._mk_shortcut("i", lambda: self.chk_batch.toggle())
        self._mk_shortcut("Return", self._enter_pressed)
        self._mk_shortcut("Enter", self._enter_pressed)
        self._mk_shortcut("Backspace", self._undo_pair)
        self._mk_shortcut("Escape", self._reset_selection)

        # Mouse picking for add-match mode (left-click).
        self.view.plot.scene().sigMouseClicked.connect(self._on_canvas_click)
        # Keep pyqtgraph's right-click menu; add a "Show IDs" action to it.
        self._last_rc_world = None
        _vbmenu = self.view.plot.getViewBox().menu
        _vbmenu.addAction("Show IDs", self._show_ids_from_menu)
        # Right-click a CZ ROI -> "QC CZ <id>" submenu: jump to it and/or label it in place.
        # Rebuilt on each open (aboutToShow) from the CZ ROI under the right-click position.
        self._qc_roi_menu = QtWidgets.QMenu("QC CZ ROI", self)
        _vbmenu.addMenu(self._qc_roi_menu)
        _vbmenu.aboutToShow.connect(self._populate_qc_roi_menu)
        # Wheel steps the slice; Shift+wheel zooms (handled in eventFilter).  All three
        # viewports are filtered; eventFilter dispatches the wheel by which view it is over.
        self._vp_main = self.view.plot.viewport()
        self._vp_xz = self.view_xz.plot.viewport()
        self._vp_yz = self.view_yz.plot.viewport()
        for vp in (self._vp_main, self._vp_xz, self._vp_yz):
            vp.installEventFilter(self)
        # Share axes with the main view: XZ (below) locks its x to XY's x; YZ (left) locks its
        # y to XY's y.  So pan/zoom in the shared axis stays in sync + features line up.  The
        # non-shared axis (z) is fit independently, so aspect-lock is off on the side views.
        _mvb = self.view.plot.getViewBox()
        self.view_xz.plot.getViewBox().setXLink(_mvb)
        self.view_yz.plot.getViewBox().setYLink(_mvb)
        self.view_xz.plot.setAspectLocked(False)
        self.view_yz.plot.setAspectLocked(False)

        # Pan/zoom re-draw.  In "vector" mode contours only cover the viewport (clipped to
        # the visible XY range) so they must be re-extracted on pan/zoom — debounced 120ms.
        # In "image" mode the edge overlay is the FULL slice as one ImageItem, which
        # pyqtgraph transforms with the view for free, so pan/zoom needs NO rebuild.
        self._range_timer = QtCore.QTimer(self)
        self._range_timer.setSingleShot(True)
        self._range_timer.setInterval(120)
        self._range_timer.timeout.connect(self._on_view_range_changed)
        vb = self.view.plot.getViewBox()
        vb.sigRangeChanged.connect(lambda *a, **k: self._range_timer.start())

        self.setMinimumSize(720, 480)  # allow the user to shrink the window
        self.resize(1000, 740)

    def _square_image(self):
        """Widen the window so the image area (CubeView) is ~square (side panels are
        fixed-width), clamped to fit the screen in both width and height."""
        vw, vh = self.view.width(), self.view.height()
        if vw <= 0 or vh <= 0:
            return
        new_w = self.width() + (vh - vw)
        new_h = self.height()
        screen = QtWidgets.QApplication.primaryScreen()
        if screen is not None:
            a = screen.availableGeometry()
            new_w = min(new_w, int(a.width() * 0.98))
            new_h = min(new_h, int(a.height() * 0.95))
        self.resize(max(self.minimumWidth(), new_w), max(self.minimumHeight(), new_h))

    def _redraw_contours_only(self):
        """Refresh ROI boundaries + match overlay without touching the background images.
        Called by toggles (cur/other/fail) and manual-match edits — i.e. STATE changes that
        alter which boundaries/colors are shown (as opposed to pan/zoom)."""
        if CONTOUR_MODE == "image":
            self.view.clear_contours()   # drop any stale vector contours
            self._draw_edges()
            self._draw_match_overlay()
            self._redraw_side_views()    # keep the orthoview in sync with toggles (no-op if off)
            return
        self.view.clear_contours()
        if self.mip_mode:
            self._draw_cz_contours_mip()
            self._draw_hcr_contours_mip()
        else:
            self._draw_cz_contours_at_z(self.cur_z_world)
            self._draw_hcr_contours_at_z(self.cur_z_world)
        self._draw_match_overlay()

    def _on_view_range_changed(self):
        """Fired (debounced) on pan/zoom.  In "image" mode the full-slice edge overlay is
        transformed by pyqtgraph automatically, so the expensive per-ROI boundary extraction
        is NOT rebuilt — this is the core fix for zoom/pan clunkiness.  The QC/add-match
        marker overlay, however, IS viewport-clipped (see _draw_qc_markers / _draw_match_overlay),
        so refresh only that — it is sparse scatter points, cheap to redraw.  In "vector" mode
        contours are viewport-clipped too, so the whole thing is re-extracted."""
        if CONTOUR_MODE == "image":
            t0 = time.perf_counter() if PROFILE else 0.0
            self._draw_match_overlay()
            if PROFILE:
                print(f"[qc-profile] pan/zoom (markers only, no edge rebuild): "
                      f"{1e3*(time.perf_counter()-t0):.1f} ms")
            return
        self._redraw_contours_only()

    def _mk_shortcut(self, key, fn):
        sc = QtWidgets.QShortcut(QtGui.QKeySequence(key), self)
        sc.activated.connect(fn)

    def _build_contrast_panel(self, parent_layout, *, label, img, gradient_rgb, auto_fn):
        """Compact contrast UI: header + histogram (height 90) + min/max spinboxes
        + Auto button.  Returns (hist_widget, spin_min, spin_max, btn_auto).
        Spinboxes are kept in sync with the histogram's region."""
        parent_layout.addWidget(QtWidgets.QLabel(f"<b>Contrast — {label}</b>"))
        try:                       # horizontal histogram is short -> saves vertical space
            hist = pg.HistogramLUTWidget(orientation="horizontal")
        except TypeError:          # older pyqtgraph without the orientation kwarg
            hist = pg.HistogramLUTWidget()
        hist.setImageItem(img)
        hist.gradient.restoreState({
            "mode": "rgb",
            "ticks": [(0.0, (0, 0, 0, 255)), (1.0, (*gradient_rgb, 255))],
        })
        hist.setMaximumHeight(90)
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
        # Center the orthoview crosshair (side-view planes) on the CZ ROI centroid.
        self.cur_y_world = float(cz_c[1])
        self.cur_x_world = float(cz_c[2])

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
        if matched:
            rk = self.cz_to_somarank.get(cz_id)
            pct = self.cz_to_somapct.get(cz_id)
            soma_str = f"soma={soma:.2f}µm" if np.isfinite(soma) else "soma=NA"
            if rk is not None:
                soma_str += (f"  (least-conf rank {rk}/{len(self.cz_to_somarank)}, "
                             f"pct {pct:.2f})")
            pair_str = f"matched → hcr_id={hcr_id}  {soma_str}"
        else:
            pair_str = "UNMATCHED"
        status = (f"CZ ROI {self.show_idx + 1}/{len(self.cz_order)}  "
                  f"cz_id={cz_id}  {pair_str}")
        cur_label = self.labels_state.get(cz_id, "—")
        n_left = sum(1 for c in self.cz_order if c not in self.labels_state)
        status += (f"\nlabel: {cur_label}   |   "
                   f"{len(self.labels_state)} QC'd, {n_left} left")
        self.lbl_status.setText(status)

    def _build_radio(self, matched: bool):
        # Clear existing
        for b in self.radio_buttons:
            self.radio_layout.removeWidget(b)
            self.radio_group.removeButton(b)
            b.deleteLater()
        self.radio_buttons.clear()

        if matched:
            options = ["good", "bad", "unsure"]; keys = [1, 2, 3]
        else:
            options = ["matched roi visible", "matched roi not visible"]; keys = [4, 5]
        # key digit -> option index, used by the number-key shortcuts.
        self._radio_key_for = {k: i for i, k in enumerate(keys)}

        cz_id = self.cur_cz_id
        prior = self.labels_state.get(cz_id, None)
        for i, opt in enumerate(options):
            rb = QtWidgets.QRadioButton(f"{keys[i]}. {opt}")
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
                float(np.percentile(sub, AUTO_CLIP_LO)),
                float(np.percentile(sub, AUTO_CLIP_HI)),
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
                    float(np.percentile(sub, AUTO_CLIP_LO)),
                    float(np.percentile(sub, AUTO_CLIP_HI)),
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

    def eventFilter(self, obj, ev):
        """Wheel = step the slice of the view it is over (main→z, XZ→y, YZ→x); Shift+wheel
        zooms (falls through to pyqtgraph).  In MIP mode plain wheel also zooms (the slab is
        pinned to the main view's range, so side views don't scroll independently).  Right-click
        (main view only): record the position for the menu's 'Show IDs'; Shift+right-click
        shows IDs and is consumed."""
        t = ev.type()
        is_main = obj is getattr(self, "_vp_main", None)
        is_xz = obj is getattr(self, "_vp_xz", None)
        is_yz = obj is getattr(self, "_vp_yz", None)
        if (t == QtCore.QEvent.MouseButtonPress and ev.button() == QtCore.Qt.RightButton
                and is_main):
            sp = self.view.plot.mapToScene(ev.pos())
            wp = self.view.plot.getViewBox().mapSceneToView(sp)
            self._last_rc_world = (float(wp.x()), float(wp.y()))
            if ev.modifiers() & QtCore.Qt.ShiftModifier:
                self._show_roi_ids_at(*self._last_rc_world)
                return True  # consume -> no context menu
            # plain right-click falls through -> pyqtgraph menu (with "Show IDs")
        elif t == QtCore.QEvent.Wheel:
            shift = bool(ev.modifiers() & QtCore.Qt.ShiftModifier)
            # Plain wheel steps the slice of whichever view it is over — but NOT in MIP (the
            # side views' slab is locked to the main view's range) and NOT with Shift (zoom).
            if not shift and not self.mip_mode:
                dy = ev.angleDelta().y()
                if dy != 0:
                    d = 1 if dy > 0 else -1
                    if is_main:
                        self._z_up() if d > 0 else self._z_down()
                    elif is_xz:
                        self._y_step(d)
                    elif is_yz:
                        self._x_step(d)
                return True   # consume -> plain wheel = slice step, no zoom
            # Shift+wheel (or any wheel in MIP) -> fall through to pyqtgraph zoom.
        return super().eventFilter(obj, ev)

    def _show_ids_from_menu(self):
        if getattr(self, "_last_rc_world", None) is not None:
            self._show_roi_ids_at(*self._last_rc_world)

    def _redraw(self):
        _t_redraw0 = time.perf_counter() if PROFILE else 0.0
        self._clear_id_text()  # ROI-ID text is ephemeral: drop it when the image changes
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
        # ----- ROI boundaries -----
        if CONTOUR_MODE == "image":
            self.view.clear_contours()   # image mode: one cached RGBA edge overlay per view
            self._draw_edges()
        else:
            self.view.clear_edges()
            self.view.clear_contours()
            if self.mip_mode:
                self._draw_cz_contours_mip()
                self._draw_hcr_contours_mip()
            else:
                self._draw_cz_contours_at_z(z_world)
                self._draw_hcr_contours_at_z(z_world)
        self._draw_match_overlay()
        # Viewport on ROI change: first ROI -> cube ± 10%; later ROIs -> keep the
        # current zoom (span) and just recenter on the new ROI.
        if not getattr(self, "_viewport_set_for_idx", None) == self.show_idx:
            cx = 0.5 * (bb["x_lo"] + bb["x_hi"]); cy = 0.5 * (bb["y_lo"] + bb["y_hi"])
            if getattr(self, "_have_view", False):
                (vx_lo, vx_hi), (vy_lo, vy_hi) = self.view.plot.viewRange()
                hx = 0.5 * (vx_hi - vx_lo); hy = 0.5 * (vy_hi - vy_lo)
            else:
                ex = bb["x_hi"] - bb["x_lo"]; ey = bb["y_hi"] - bb["y_lo"]
                hx = 0.6 * ex; hy = 0.6 * ey  # cube + 10% margin each side
                self._have_view = True
            self.view.plot.setXRange(cx - hx, cx + hx, padding=0)
            self.view.plot.setYRange(cy - hy, cy + hy, padding=0)
            self._viewport_set_for_idx = self.show_idx
            self._fit_side_viewports()   # fit XZ/YZ to the new cube on ROI change
        # Side panels + linked crosshair (both no-op / hidden when the orthoview is off).
        self._redraw_side_views()
        self._update_crosshairs()
        if PROFILE:
            print(f"[qc-profile] full redraw (mode={CONTOUR_MODE}, mip={self.mip_mode}): "
                  f"{1e3*(time.perf_counter()-_t_redraw0):.1f} ms")

    def _fit_side_viewports(self):
        """Fit the side views' z (non-shared) axis to the current cube.  The shared axes are
        LINKED to the main view (XZ.x↔XY.x, YZ.y↔XY.y), so they follow XY automatically and are
        not set here."""
        if not self.show_side_views:
            return
        bb = self.cube_bb
        self.view_xz.plot.setYRange(bb["z_lo"], bb["z_hi"], padding=0.1)  # XZ vertical = z
        self.view_yz.plot.setXRange(bb["z_lo"], bb["z_hi"], padding=0.1)  # YZ horizontal = z

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
                    if not self.show_cur_cz:
                        continue
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
                        if not self.show_cur_hcr:
                            continue
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
                    if not self.show_cur_cz:
                        continue
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
                        if not self.show_cur_hcr:
                            continue
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

    # ---------------- edge-overlay rendering (CONTOUR_MODE == "image") ----------------
    @staticmethod
    def _boundary_mask(lbl):
        """True at every labeled pixel adjacent to a different label/background — all ROI
        boundaries in one vectorized pass (replaces per-ROI cv2.findContours)."""
        b = np.zeros(lbl.shape, dtype=bool)
        d = lbl[:-1, :] != lbl[1:, :]
        b[:-1, :] |= d; b[1:, :] |= d
        d = lbl[:, :-1] != lbl[:, 1:]
        b[:, :-1] |= d; b[:, 1:] |= d
        return b & (lbl != 0)

    def _paint_edge(self, rgba, mask, color):
        """Paint RGBA 4-tuple ``color`` where ``mask`` is True, dilating by EDGE_PX for width."""
        if not mask.any():
            return
        if EDGE_PX > 0:
            k = 2 * EDGE_PX + 1
            mask = cv2.dilate(mask.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
        rgba[mask] = color

    def _mip_slab_world(self, view):
        """World-µm (lo, hi) slab along ``view``'s slice axis in MIP mode.  The main (xy) view
        keeps its exact ROI z-extent (unchanged behavior).  Side views use the SAME slab
        THICKNESS as the main view, centered on the crosshair — "the same range as in the main
        view" — so all three projections cover the current ROI's slab from their own direction."""
        sa = _VIEW_AX[view][0]
        lo, hi = self.mip_z_world
        if view == "xy":
            return lo, hi
        T = hi - lo
        c = (self.cur_z_world, self.cur_y_world, self.cur_x_world)[sa]
        return c - 0.5 * T, c + 0.5 * T

    def _plane2d(self, arr, origin3, vox3, view):
        """The [row, col] label/image plane of a volume for ``view``: a single slice at the
        crosshair position along the slice axis, or (MIP mode) a max-projection over the slab.
        ``origin3``/``vox3`` are per-axis (z, y, x) world origins + voxel sizes.  None if the
        slice/slab is out of range.

        MIP uses one ``max(axis=slice)`` (one label per output pixel) — a faithful union-of-
        footprints boundary, and a single vectorized reduction."""
        sa, ra, ca = _VIEW_AX[view]
        if self.mip_mode:
            lo, hi = self._mip_slab_world(view)
            i0 = max(0, int((lo - origin3[sa]) / vox3[sa]))
            i1 = min(arr.shape[sa], int((hi - origin3[sa]) / vox3[sa]) + 1)
            if i0 >= i1:
                return None
            sl = [slice(None), slice(None), slice(None)]
            sl[sa] = slice(i0, i1)
            plane = arr[tuple(sl)].max(axis=sa)
        else:
            cur = (self.cur_z_world, self.cur_y_world, self.cur_x_world)[sa]
            idx = int(round((cur - origin3[sa]) / vox3[sa]))
            if not (0 <= idx < arr.shape[sa]):
                return None
            plane = np.take(arr, idx, axis=sa)
        # The extracted plane's axes are the two non-slice axes in ascending order; transpose
        # to (row_axis, col_axis) when the desired row axis is the higher-numbered one (YZ).
        return plane.T if ra > ca else plane

    def _bg_plane(self, arr, origin3, vox3, view):
        """Background-image plane for ``view`` + its display rect.
        Returns (plane_2d | None, col_lo, row_lo, col_um, row_um)."""
        plane = self._plane2d(arr, origin3, vox3, view)
        if plane is None:
            return None, 0.0, 0.0, 1.0, 1.0
        _, ra, ca = _VIEW_AX[view]
        return plane, origin3[ca], origin3[ra], vox3[ca], vox3[ra]

    def _edges_rgba_for_view(self, which, view="xy"):
        """RGBA (H,W,4 uint8) ROI-boundary overlay for the CZ or HCR segmentation in ``view``,
        colored by category with the current ROI on top.  Mirrors the color/toggle/current-ROI
        semantics of the vector contour methods (COLOR_* constants, show_* gating), as one
        vectorized boundary pass per source array (no cv2.findContours per ROI).  Uses the
        seg-array bbox + per-axis voxel size, so it aligns on the same world-µm grid as the
        background images.  Returns (rgba, col_lo, row_lo, col_um, row_um).

        The current ROI is painted LAST so its highlight is always visible over neighbors."""
        if which == "cz":
            origin3 = (self.cz_bb["z_lo"], self.cz_bb["y_lo"], self.cz_bb["x_lo"])
            vox3 = (self.cz_vox, self.cz_vox, self.cz_vox)
            sources = [(self.cz_matched_arr, COLOR_OTHER_CZM, True),
                       (self.cz_unmatched_arr, COLOR_OTHER_CZU, True)]
            show_other = self.show_other_cz
            cur_id = self.cur_cz_id if self.show_cur_cz else None
            cur_color = COLOR_CUR_CZ
        else:
            origin3 = (self.hbb["z_lo"], self.hbb["y_lo"], self.hbb["x_lo"])
            vox3 = (self.hcr_vox_z, self.hcr_vox_xy, self.hcr_vox_xy)
            sources = [(self.hcr_matched_arr, COLOR_OTHER_HCRM, True),
                       (self.hcr_unmatched_arr, COLOR_OTHER_HCRU, True)]
            # Failed arrays (eligible=False): drawn only when their toggle is on, never
            # "other"-gated, and never eligible for the current-ROI highlight.
            if self.show_hcr_fail_gfp and self.hcr_failed_gfp_arr is not None:
                sources.append((self.hcr_failed_gfp_arr, COLOR_HCR_FAIL_GFP, False))
            if self.show_hcr_fail_cls and self.hcr_failed_cls_arr is not None:
                sources.append((self.hcr_failed_cls_arr, COLOR_HCR_FAIL_CLS, False))
            show_other = self.show_other_hcr
            cur_id = self.cur_hcr_id if (self.show_cur_hcr and self.cur_matched) else None
            cur_color = COLOR_CUR_HCR
        _, ra, ca = _VIEW_AX[view]
        arr0 = sources[0][0]
        H, W = int(arr0.shape[ra]), int(arr0.shape[ca])
        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        cur_bnd = np.zeros((H, W), dtype=bool)   # current-ROI boundary, accumulated then on top
        for arr, color, eligible in sources:
            L = self._plane2d(arr, origin3, vox3, view)
            if L is None or L.shape != (H, W):
                continue
            b = self._boundary_mask(L)
            if eligible and cur_id is not None:
                here_cur = (L == cur_id)
                cur_bnd |= (b & here_cur)
                b_other = b & (~here_cur)
            else:
                b_other = b
            # matched/unmatched "other" edges honor show_other; failed arrays always draw.
            if (not eligible) or show_other:
                self._paint_edge(rgba, b_other, color)
        if cur_id is not None:
            self._paint_edge(rgba, cur_bnd, cur_color)
        return rgba, origin3[ca], origin3[ra], vox3[ca], vox3[ra]

    def _edge_state_key(self, which, view="xy"):
        """Cache key = view + slice (or MIP slab) along the view's slice axis + the toggles/
        current-id that change the overlay.  Deliberately viewport-INDEPENDENT: pan/zoom never
        invalidates the cache (the ImageItem is transformed instead of rebuilt)."""
        sa = _VIEW_AX[view][0]
        if self.mip_mode:
            lo, hi = self._mip_slab_world(view)
            zk = ("mip", round(lo, 3), round(hi, 3))
        else:
            cur = (self.cur_z_world, self.cur_y_world, self.cur_x_world)[sa]
            zk = ("sl", round(cur, 3))
        if which == "cz":
            return ("cz", view, zk, self.show_cur_cz, self.show_other_cz,
                    (self.cur_cz_id if self.show_cur_cz else None), EDGE_PX)
        return ("hcr", view, zk, self.show_cur_hcr, self.show_other_hcr,
                self.show_hcr_fail_gfp, self.show_hcr_fail_cls, self.cur_matched,
                (self.cur_hcr_id if (self.show_cur_hcr and self.cur_matched) else None), EDGE_PX)

    def _draw_edges(self, view="xy", widget=None):
        """Set the CZ + HCR edge-overlay ImageItems for ``view`` on ``widget`` (default main),
        via a per-(view, slice, state) RGBA cache.  Called on slice/ROI/toggle change — NOT on
        pan/zoom."""
        widget = widget if widget is not None else self.view
        for which, item in (("hcr", widget.edge_hcr), ("cz", widget.edge_cz)):
            key = self._edge_state_key(which, view)
            cached = self._edge_cache.get(key)
            if cached is None:
                t0 = time.perf_counter() if PROFILE else 0.0
                cached = self._edges_rgba_for_view(which, view)
                if len(self._edge_cache) > 512:   # bound memory over a long session (3 views)
                    self._edge_cache.clear()
                self._edge_cache[key] = cached
                if PROFILE:
                    print(f"[qc-profile] edge build {which}/{view}: "
                          f"{1e3*(time.perf_counter()-t0):.1f} ms (miss; cache={len(self._edge_cache)})")
            elif PROFILE:
                print(f"[qc-profile] edge {which}/{view}: cache hit")
            rgba, col_lo, row_lo, col_um, row_um = cached
            widget.set_edge_image(item, rgba, x_lo=col_lo, y_lo=row_lo, xy_um=col_um, y_um=row_um)

    # ---------------- linked-crosshair orthoview (side views) ----------------
    def _toggle_side_views(self):
        self.show_side_views = self.btn_ortho.isChecked()
        self.view_yz.setVisible(self.show_side_views)
        self.view_xz.setVisible(self.show_side_views)
        if self.show_side_views:
            self._img_grid.setColumnStretch(0, 1); self._img_grid.setColumnStretch(1, 3)
            self._img_grid.setRowStretch(0, 3); self._img_grid.setRowStretch(1, 1)
            self._fit_side_viewports()
            self._redraw_side_views()
        else:
            # Collapse the YZ column + XZ row so the main XY view fills the whole area.
            self._img_grid.setColumnStretch(0, 0); self._img_grid.setColumnStretch(1, 1)
            self._img_grid.setRowStretch(0, 1); self._img_grid.setRowStretch(1, 0)
        self._update_crosshairs()

    def _update_crosshairs(self):
        """Draw the linked crosshair in every view (or hide it when the orthoview is off).
        Each view's (vertical=col, horizontal=row) lines mark the OTHER two planes."""
        if not self.show_side_views:
            for w in (self.view, self.view_xz, self.view_yz):
                w.hide_crosshair()
            return
        cz, cy, cx = self.cur_z_world, self.cur_y_world, self.cur_x_world
        self.view.set_crosshair(cx, cy)      # XY: cols=x, rows=y
        self.view_xz.set_crosshair(cx, cz)   # XZ: cols=x, rows=z
        self.view_yz.set_crosshair(cz, cy)   # YZ: cols=z, rows=y

    def _redraw_side_views(self):
        """Refresh the XZ + YZ side panels (background images + edge overlays) and crosshairs.
        No-op when the orthoview is off.  Cheap: same in-RAM slice + vectorized boundary pass as
        the main view, just along y/x instead of z."""
        if not self.show_side_views:
            return
        z_um, xy_um, _ = self.hcr488_voxel
        hcr_vox3 = (z_um, xy_um, xy_um)
        # Match the MAIN view's current contrast (incl. live histogram drags), not just auto.
        lv_hcr = self._cur_levels(self.view.img_hcr, self.hcr488_levels)
        lv_cz = self._cur_levels(self.view.img_cz, getattr(self, "czw_levels", (0.0, 1.0)))
        for view, w in (("xz", self.view_xz), ("yz", self.view_yz)):
            # HCR 488 background
            if self.show_hcr488:
                plane, cl, rl, cu, ru = self._bg_plane(self.hcr488, self.hcr488_origin, hcr_vox3, view)
                if plane is not None:
                    w.set_hcr_image(plane, x_lo=cl, y_lo=rl, xy_um=cu, y_um=ru)
                    w.img_hcr.setLevels(lv_hcr)
                else:
                    w.img_hcr.clear()
            else:
                w.img_hcr.clear()
            # Warped CZ background
            if self.show_czw and self.czw is not None:
                cvox3 = (self.czw_voxel, self.czw_voxel, self.czw_voxel)
                plane, cl, rl, cu, ru = self._bg_plane(self.czw, self.czw_origin, cvox3, view)
                if plane is not None:
                    w.set_cz_image(plane, x_lo=cl, y_lo=rl, xy_um=cu, y_um=ru)
                    w.img_cz.setLevels(lv_cz)
                else:
                    w.img_cz.clear()
            else:
                w.img_cz.clear()
            # ROI boundaries: always the edge-overlay for side views (the vector-contour path is
            # XY-only; the edge builder is mode-independent, so side views work in both modes).
            w.clear_contours()
            self._draw_edges(view, w)
        self._update_crosshairs()

    @staticmethod
    def _cur_levels(item, fallback):
        """The ImageItem's current display levels (min,max), else ``fallback``."""
        lv = getattr(item, "levels", None)
        if lv is None:
            return fallback
        try:
            return (float(lv[0]), float(lv[1]))
        except (TypeError, IndexError, ValueError):
            return fallback

    def _y_step(self, delta):
        """Move the XZ side-view plane (world y) by one HCR-xy voxel, clamped to the cube."""
        bb = self.cube_bb
        ny = self.cur_y_world + delta * self.hcr488_voxel[1]
        self.cur_y_world = float(min(max(ny, bb["y_lo"]), bb["y_hi"]))
        self._redraw_side_views()

    def _x_step(self, delta):
        """Move the YZ side-view plane (world x) by one HCR-xy voxel, clamped to the cube."""
        bb = self.cube_bb
        nx = self.cur_x_world + delta * self.hcr488_voxel[1]
        self.cur_x_world = float(min(max(nx, bb["x_lo"]), bb["x_hi"]))
        self._redraw_side_views()

    # ---------------- nav + toggles + label ----------------
    def _step_to_unqcd(self, direction: int):
        """Move one ROI in ``direction`` (+1 next, -1 prev).  If 'Un-QC'd only' is on
        (self.skip_qcd), skip already-QC'd pairs so nav walks only the undecided ones;
        otherwise step through every pair in sort order."""
        n = len(self.cz_order)
        if n == 0:
            return
        if not getattr(self, "skip_qcd", True):
            self.show_idx = (self.show_idx + direction) % n
            self._refresh_pair()
            return
        for step in range(1, n + 1):
            idx = (self.show_idx + direction * step) % n
            if self.cz_order[idx] not in self.labels_state:
                self.show_idx = idx
                self._refresh_pair()
                return
        self._notify("All pairs in the queue are QC'd ✓", kind="ok")

    def _next(self):
        self._step_to_unqcd(+1)

    def _prev(self):
        self._step_to_unqcd(-1)

    def _enter_pressed(self):
        """Enter = add pair (add-match mode), else advance only if the current ROI
        has a decision (label)."""
        if self.add_match_mode:
            self._add_pair()
        else:
            self._next_if_labeled()

    def _next_if_labeled(self):
        if self.labels_state.get(self.cur_cz_id):
            self._next()
        else:
            kind = "good/bad/unsure" if self.cur_matched else "visible/not-visible"
            self._notify(f"Not labeled — pick {kind} before Enter (use → to skip).")

    def _notify(self, msg, ms=3000, kind="info"):
        """Status-bar message + a persistent coloured label in the QC'd panel."""
        self.statusBar().showMessage(msg, ms)
        if hasattr(self, "lbl_action"):
            col = {"ok": "#1a8a1a", "warn": "#c87000", "err": "#c00000",
                   "info": "#333333"}.get(kind, "#333333")
            self.lbl_action.setText(msg)
            self.lbl_action.setStyleSheet(f"font-weight: bold; color: {col};")
        print("[notify]", msg)

    def _update_savepath_label(self):
        if hasattr(self, "savepath_edit"):
            p = str(self.labels_path)
            self.savepath_edit.setText(p)
            self.savepath_edit.setToolTip(p)
            self.savepath_edit.setCursorPosition(len(p))  # show the filename end first

    def _dump_labels_to(self, path):
        """Write the current QC state as a fresh labels CSV (snapshot)."""
        ts = _dt.datetime.now().isoformat(timespec="seconds")
        with open(path, "w") as f:
            f.write("timestamp,idx,cz_id,hcr_id,soma_score,kind,label\n")
            for cz, lab in self.labels_state.items():
                hcr = self.cz_to_hcr.get(int(cz))
                kind = "matched" if hcr is not None else "unmatched"
                soma = float(self.cz_to_soma.get(int(cz), float("nan")))
                f.write(f"{ts},-1,{int(cz)},{int(hcr) if hcr is not None else -1},"
                        f"{soma:.4f},{kind},{lab}\n")

    def _choose_save_file(self):
        """Let the user pick the folder + filename for the QC labels CSV; snapshot
        the current QC state there and direct future writes to it."""
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Choose QC labels CSV", str(self.labels_path),
            "CSV files (*.csv);;All files (*)")
        if not path:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.labels_path = path
        self._dump_labels_to(path)
        self._update_savepath_label()
        self._notify(f"Saving QC labels to {path.name}", kind="info")

    def _toggle_cur_cz(self):
        self.show_cur_cz = self.chk_cur_cz.isChecked()
        self._redraw_contours_only()

    def _toggle_cur_hcr(self):
        self.show_cur_hcr = self.chk_cur_hcr.isChecked()
        self._redraw_contours_only()

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

    def _toggle_qc_markers(self):
        self.show_qc_markers = self.chk_qc_markers.isChecked()
        self._draw_match_overlay()

    def _set_mode(self, mip: bool):
        self.mip_mode = bool(mip)
        # Z slider only meaningful in slice mode; gray out in MIP mode
        self.z_slider.setEnabled(not self.mip_mode)
        self._redraw()

    def _click_radio(self, i):
        if 0 <= i < len(self.radio_buttons):
            self.radio_buttons[i].setChecked(True)

    def _radio_key(self, digit: int) -> bool:
        """Select the radio option bound to a number key for the current pair
        (matched: 1/2/3; unmatched: 4/5).  Returns True if it matched an option."""
        idx = getattr(self, "_radio_key_for", {}).get(digit)
        if idx is not None and idx < len(self.radio_buttons):
            self._click_radio(idx)
            return True
        return False

    # ---------------- label write (any pair) + QC'd list + markers ----------------
    def _write_label(self, cz_id, hcr_id, soma, kind, label):
        """Append a label row (append-only) for ANY cz_id and update in-memory state +
        the QC'd-pairs list.  label='removed' un-QCs the pair (dropped on reload)."""
        LABELS_ROOT.mkdir(parents=True, exist_ok=True)
        ts = _dt.datetime.now().isoformat(timespec="seconds")
        write_header = not self.labels_path.exists()
        with open(self.labels_path, "a") as f:
            if write_header:
                f.write("timestamp,idx,cz_id,hcr_id,soma_score,kind,label\n")
            f.write(f"{ts},{self.show_idx},{int(cz_id)},{int(hcr_id)},"
                    f"{soma:.4f},{kind},{label}\n")
        self.labels_state[int(cz_id)] = label
        self._qcd_upsert(int(cz_id), label)

    def _save_label(self, label: str):
        kind = "matched" if self.cur_matched else "unmatched"
        soma = self.cur_soma if self.cur_matched else float("nan")
        hcr_id = self.cur_hcr_id if self.cur_matched else -1
        self._write_label(self.cur_cz_id, hcr_id, soma, kind, label)
        s = self.lbl_status.text().split("\n")[0]
        self.lbl_status.setText(s + f"\nlabel: {label}  (saved)")
        print(f"[label] idx={self.show_idx+1} cz_id={self.cur_cz_id} {kind} → {label}")
        nk = {"good": "ok", "bad": "err", "unsure": "info",
              "matched roi visible": "ok", "matched roi not visible": "warn"}.get(label, "info")
        self._notify(f"labeled cz{self.cur_cz_id}: {label}   (QC'd: {len(self.labels_state)})",
                     kind=nk)
        self._draw_match_overlay()  # marker-only refresh (no costly contour re-extract)
        # "matched roi visible" on an unmatched ROI -> the true HCR cell IS present, so
        # jump into add-match mode with this CZ pre-selected; the user just clicks the HCR.
        if label == "matched roi visible":
            self._start_manual_link_for_current()

    def _start_manual_link_for_current(self):
        """Enable add-match mode (if needed) and pre-select the current CZ ROI so the
        operator can immediately click its HCR cell to link the pair."""
        cz = self.cur_cz_id
        if not self.add_match_mode:
            self.chk_add_match.setChecked(True)   # triggers _toggle_add_match
        self.pending_cz_id = cz
        self.pending_hcr_id = None
        self._set_match_status(f"cz_id={cz} selected — click its HCR cell to link "
                               f"(Enter to add).")
        self._update_match_buttons()
        self._redraw_contours_only()

    def _accept_pair_label(self, cz_id):
        """Accept a matched pair (label 'good'), for batch-click; no current-ROI state."""
        cz_id = int(cz_id)
        hcr_id = int(self.cz_to_hcr.get(cz_id, -1))
        soma = float(self.cz_to_soma.get(cz_id, float("nan")))
        self._write_label(cz_id, hcr_id, soma, "matched", "good")

    def _remove_label(self, cz_id):
        """Un-QC a pair: drop it from memory AND from the labels CSV (removal is NOT
        logged — no 'removed' row is written), so it stays removed on reload."""
        cz_id = int(cz_id)
        self.labels_state.pop(cz_id, None)
        if self.labels_path.exists():
            df = pd.read_csv(self.labels_path)
            if "cz_id" in df.columns:
                df = df[df["cz_id"].astype(int) != cz_id]
                df.to_csv(self.labels_path, index=False)
        self._qcd_remove(cz_id)

    def _qcd_color(self, lab):
        return {"good": (40, 200, 40), "bad": (220, 60, 60), "unsure": (200, 180, 40),
                "manual": (0, 190, 210),
                "matched roi visible": (40, 200, 40),
                "matched roi not visible": (220, 140, 0)}.get(lab)

    def _qcd_text(self, cz, lab):
        hcr = self.cz_to_hcr.get(int(cz))
        base = f"cz{cz} ↔ hcr{hcr}" if hcr is not None else f"cz{cz} (unmatched)"
        return f"{base}  [{lab}]"

    def _qcd_upsert(self, cz, lab, scroll=True):
        """Add or update a single QC'd-list row in place (O(1)) — avoids the full
        rebuild that lagged batch-accept; newest stays scrolled into view."""
        if not hasattr(self, "qcd_list"):
            return
        cz = int(cz)
        it = self._qcd_items.get(cz)
        if it is None:
            it = QtWidgets.QListWidgetItem()
            it.setData(QtCore.Qt.UserRole, cz)
            self.qcd_list.addItem(it)
            self._qcd_items[cz] = it
        it.setText(self._qcd_text(cz, lab))
        c = self._qcd_color(lab)
        if c:
            it.setForeground(QtGui.QColor(*c))
        if scroll:
            self.qcd_list.scrollToItem(it)
        self.qcd_box.setTitle(f"QC'd pairs ({len(self.labels_state)})")

    def _qcd_remove(self, cz):
        if not hasattr(self, "qcd_list"):
            return
        it = self._qcd_items.pop(int(cz), None)
        if it is not None:
            self.qcd_list.takeItem(self.qcd_list.row(it))
        self.qcd_box.setTitle(f"QC'd pairs ({len(self.labels_state)})")

    def _refresh_qcd_list(self):
        """Full rebuild (init / reload only)."""
        if not hasattr(self, "qcd_list"):
            return
        self.qcd_list.clear()
        self._qcd_items = {}
        for cz, lab in self.labels_state.items():
            self._qcd_upsert(cz, lab, scroll=False)

    def _sort_qcd_list(self):
        """Rebuild the QC'd list sorted by label then cz_id (instead of QC order)."""
        if not hasattr(self, "qcd_list"):
            return
        self.qcd_list.clear()
        self._qcd_items = {}
        for cz, lab in sorted(self.labels_state.items(),
                              key=lambda kv: (kv[1], int(kv[0]))):
            self._qcd_upsert(cz, lab, scroll=False)

    def _on_qcd_double(self, item):
        cz = int(item.data(QtCore.Qt.UserRole))
        if cz in self.cz_order:
            self.show_idx = self.cz_order.index(cz)
            self._refresh_pair()

    def _remove_selected_qcd(self):
        it = self.qcd_list.currentItem()
        if it is not None:
            self._remove_label(int(it.data(QtCore.Qt.UserRole)))
            self._draw_match_overlay()

    # ---------------- batch-accept (function 3, MIP) ----------------
    def _label_at_view(self, arr, bb, vox_xy, vox_z, x, y):
        """Label id at world (x,y) in the current view: MIP-projected over the ROI's
        z-extent if in MIP mode, else at the current z slice.  0/out-of-bounds -> None."""
        yi = int(round((y - bb["y_lo"]) / vox_xy))
        xi = int(round((x - bb["x_lo"]) / vox_xy))
        if not (0 <= yi < arr.shape[1] and 0 <= xi < arr.shape[2]):
            return None
        if self.mip_mode:
            mlo, mhi = self.mip_z_world
            z0 = max(0, int((mlo - bb["z_lo"]) / vox_z))
            z1 = min(arr.shape[0], int((mhi - bb["z_lo"]) / vox_z) + 1)
            col = arr[z0:z1, yi, xi]
            nz = col[col != 0]
            return int(nz[0]) if nz.size else None
        zi = int(round((self.cur_z_world - bb["z_lo"]) / vox_z))
        if 0 <= zi < arr.shape[0]:
            v = int(arr[zi, yi, xi])
            return v if v != 0 else None
        return None

    def _clear_id_text(self):
        if getattr(self, "_id_text_item", None) is not None:
            self.view.plot.removeItem(self._id_text_item)
            self._id_text_item = None

    def _cz_id_at(self, x, y):
        """CZ ROI id under world (x, y) in the current view (matched first, then unmatched),
        or None. Same resolution the 'Show IDs' report uses."""
        for arr in (self.cz_matched_arr, self.cz_unmatched_arr):
            v = self._label_at_view(arr, self.cz_bb, self.cz_vox, self.cz_vox, x, y)
            if v:
                return int(v)
        return None

    def _populate_qc_roi_menu(self):
        """(Re)build the right-click 'QC CZ <id>' submenu for the CZ ROI under the last
        right-click position: a 'Go to' entry + the label options valid for that ROI
        (matched -> good/bad/unsure; unmatched -> matched roi visible / not visible)."""
        m = self._qc_roi_menu
        m.clear()
        wp = getattr(self, "_last_rc_world", None)
        cz = self._cz_id_at(*wp) if wp is not None else None
        if cz is None:
            m.setTitle("QC CZ ROI — right-click on a CZ ROI")
            m.setEnabled(False)
            return
        m.setEnabled(True)
        matched = cz in self.cz_to_hcr
        m.setTitle(f"QC CZ {cz}" + (f"  (→ hcr {self.cz_to_hcr[cz]})" if matched
                                    else "  (unmatched)"))
        in_queue = cz in self.cz_order
        go = m.addAction(f"Go to CZ {cz}" + ("" if in_queue else "  — not in queue"))
        go.setEnabled(in_queue)
        go.triggered.connect(lambda _checked=False, c=cz: self._goto_cz(c))
        m.addSeparator()
        options = (["good", "bad", "unsure"] if matched
                   else ["matched roi visible", "matched roi not visible"])
        for opt in options:
            a = m.addAction(f"label: {opt}")
            a.setEnabled(in_queue)
            a.triggered.connect(lambda _checked=False, c=cz, o=opt: self._qc_roi(c, o))

    def _goto_cz(self, cz_id):
        """Jump the QC queue/view to cz_id (make it current) without labeling."""
        if cz_id in self.cz_order:
            self.show_idx = self.cz_order.index(cz_id)
            self._refresh_pair()
        else:
            self._notify(f"cz {cz_id} is not in the current QC queue", kind="warn")

    def _qc_roi(self, cz_id, label):
        """Right-click QC: jump to cz_id (make it current), then apply the QC label to it,
        reusing the normal per-ROI save path so the label is valid for its matched-state."""
        if cz_id not in self.cz_order:
            self._notify(f"cz {cz_id} is not in the current QC queue", kind="warn")
            return
        self.show_idx = self.cz_order.index(cz_id)
        self._refresh_pair()      # sets cur_cz_id / cur_matched / cur_hcr_id / cur_soma
        self._save_label(label)

    def _show_roi_ids_at(self, x, y):
        """Right-click: report CZ + HCR ROI IDs overlapping (x, y) as ephemeral text
        drawn on the image at the click (cleared when the image changes) + the action
        label.  If a clicked CZ pair is already QC'd, select its row in the QC'd list."""
        cz_ids = []
        for arr in (self.cz_matched_arr, self.cz_unmatched_arr):
            v = self._label_at_view(arr, self.cz_bb, self.cz_vox, self.cz_vox, x, y)
            if v and v not in cz_ids:
                cz_ids.append(v)
        sources = [(self.hcr_matched_arr, "matched"),
                   (self.hcr_unmatched_arr, "in-pool unmatched")]
        if self.hcr_failed_gfp_arr is not None:
            sources.append((self.hcr_failed_gfp_arr, "failed GFP+"))
        if self.hcr_failed_cls_arr is not None:
            sources.append((self.hcr_failed_cls_arr, "failed classifier"))
        hcr_hits, seen = [], set()
        for arr, cat in sources:
            v = self._label_at_view(arr, self.hbb, self.hcr_vox_xy, self.hcr_vox_z, x, y)
            if v and v not in seen:
                seen.add(v); hcr_hits.append((v, cat))
        lines = []
        for c in cz_ids:
            partner = self.cz_to_hcr.get(int(c))
            lines.append(f"CZ {c}" + (f" → hcr {partner}" if partner is not None
                                      else " (unmatched)"))
        for v, cat in hcr_hits:
            extra = ""
            if cat == "matched":
                czp = next((cc for cc, hh in self.cz_to_hcr.items() if hh == v), None)
                if czp is not None:
                    extra = f" ← cz {czp}"
            lines.append(f"HCR {v} ({cat}){extra}")
        msg = " | ".join(lines) if lines else "no ROI at this location"
        # Ephemeral on-image text at the click (cleared on next image change).
        self._clear_id_text()
        ti = pg.TextItem(text=msg.replace(" | ", "\n"), color=(255, 255, 255),
                         anchor=(0, 0), fill=pg.mkBrush(0, 0, 0, 170))
        ti.setPos(x, y)
        self.view.plot.addItem(ti)
        self._id_text_item = ti
        self._notify(msg, kind="info")
        # If a clicked CZ pair is already QC'd, jump to its row in the QC'd list.
        for c in cz_ids:
            it = self._qcd_items.get(int(c))
            if it is not None:
                self.qcd_list.setCurrentItem(it)
                self.qcd_list.scrollToItem(it)
                break

    def _batch_click(self, x, y):
        """Accept (or toggle-remove) the matched pair whose CZ and HCR ROIs both cover
        the click point.  Notifies if the click is not on such an overlap."""
        cz = self._label_at_view(self.cz_matched_arr, self.cz_bb, self.cz_vox,
                                 self.cz_vox, x, y)
        hcr = self._label_at_view(self.hcr_matched_arr, self.hbb, self.hcr_vox_xy,
                                  self.hcr_vox_z, x, y)
        if cz is None or hcr is None:
            self._notify("✗ Not added — click within an overlap of CZ & HCR ROI boundaries",
                         kind="err")
            return
        if self.cz_to_hcr.get(int(cz)) != int(hcr):
            self._notify(f"✗ Not added — cz{cz} & hcr{hcr} do not form a matched pair here",
                         kind="err")
            return
        cz = int(cz)
        if self.labels_state.get(cz) == "good":
            self._remove_label(cz)
            self._notify(f"✗ removed  cz{cz} ↔ hcr{hcr}", kind="warn")
        else:
            self._accept_pair_label(cz)
            self._notify(f"✓ accepted  cz{cz} ↔ hcr{hcr}   (QC'd: {len(self.labels_state)})",
                         kind="ok")
        self._draw_match_overlay()  # marker-only refresh (contours unchanged) — no lag

    def _toggle_batch_accept(self):
        self.batch_accept_mode = self.chk_batch.isChecked()
        if self.batch_accept_mode:
            # Mutually exclusive with add-match; nudge into MIP mode.
            if self.add_match_mode:
                self.chk_add_match.setChecked(False)
            if not self.mip_mode:
                self.rad_mip.setChecked(True)
            self._notify("Batch-accept ON — left-click a CZ/HCR overlap to accept; "
                         "click again to remove")
        self._redraw_contours_only()

    def _draw_qc_markers(self):
        """Spatial dots for QC'd pairs in view: good=green o, bad=red x, unsure=yellow o."""
        if not self.show_qc_markers:
            return
        (vy_lo, vy_hi), (vx_lo, vx_hi) = self._viewport_xy_range()
        # Displayed z interval: MIP slab thickness, or the single slice plane.
        if self.mip_mode:
            zlo, zhi = self.mip_z_world
        else:
            zlo = zhi = self.cur_z_world
        groups = {"good": [], "bad": [], "unsure": []}
        for cz, lab in self.labels_state.items():
            if lab not in groups:
                continue
            p = self._cz_pos(cz)
            if p is None:
                continue
            if not (vx_lo <= p[2] <= vx_hi and vy_lo <= p[1] <= vy_hi):
                continue
            # Show only when the matched HCR ROI is actually visualised here, i.e. its
            # z-extent overlaps the displayed slice/slab (same rule for slice + MIP).
            hcr = self.cz_to_hcr.get(int(cz))
            zr = self.hcr_zrange.get(int(hcr)) if hcr is not None else None
            if zr is None or zr[0] > zhi or zr[1] < zlo:
                continue
            groups[lab].append((p[2], p[1]))
        # White ring so markers are visible on the green/magenta image.
        edge = pg.mkPen((255, 255, 255, 255), width=2.0)
        style = {"good": ((40, 235, 40), "o", 16), "bad": ((255, 50, 50), "x", 16),
                 "unsure": ((245, 220, 40), "o", 14)}
        for lab, pts in groups.items():
            if not pts:
                continue
            col, sym, sz = style[lab]
            self.view.add_scatter([a for a, _ in pts], [b for _, b in pts],
                                  color=col, size=sz, symbol=sym, pen=edge)

    # ---------------- export (positions + HCR seg crop) ----------------
    def _positions_for(self, cz_id) -> dict:
        """XYZ for a CZ ROI in BOTH frames: czstack-native (µm + px), CZ warped into
        HCR µm, and the matched HCR cell (HCR µm + px). Missing parts omitted.

        cz_in_hcr comes from self._cz_pos (TPS-aware) so manual-match refits are
        reflected.  A single-element dict wraps the one CZ so position_row can look
        it up by id, matching the derived-dict contract.
        """
        cz_id = int(cz_id)
        derived = {
            "cz_native_by_id": self.cz_native_by_id,
            "cz_in_hcr_by_id": {cz_id: self._cz_pos(cz_id)},
            "hcr_by_id": self.hcr_by_id,
            "cz_xy_um": self.cz_xy_um,
            "cz_z_um": self.cz_z_um,
            "hcr_xy_um": self.hcr_xy_um,
            "hcr_z_um": self.hcr_z_um,
        }
        return _position_row(
            cz_id,
            self.cz_to_hcr.get(cz_id),
            self.cz_to_soma.get(cz_id, float("nan")),
            derived,
        )

    def _copy_positions(self):
        d = self._positions_for(self.cur_cz_id)
        hdr = ",".join(POS_COLS)
        row = ",".join(_fmt_val(d.get(c, "")) for c in POS_COLS)
        QtWidgets.QApplication.clipboard().setText(hdr + "\n" + row)
        self._set_export_status("copied pair XYZ (both frames) to clipboard")
        print("[export] copied pair XYZ:\n" + hdr + "\n" + row)

    def _export_all_positions(self):
        out = self.export_dir / f"positions_{self.sid}.csv"
        # Build cz_in_hcr_by_id using _cz_pos (TPS-aware) for the full queue so
        # manual-match refits are reflected in every row.
        derived = {
            "cz_native_by_id": self.cz_native_by_id,
            "cz_in_hcr_by_id": {c: self._cz_pos(c) for c in self.cz_order},
            "hcr_by_id": self.hcr_by_id,
            "cz_xy_um": self.cz_xy_um,
            "cz_z_um": self.cz_z_um,
            "hcr_xy_um": self.hcr_xy_um,
            "hcr_z_um": self.hcr_z_um,
        }
        df = _compute_pair_positions(self.cz_to_hcr, self.cz_to_soma, derived,
                                     cz_ids=self.cz_order)
        _write_positions_csv(df, out)
        self._set_export_status(f"wrote {len(df)} positions → {out.name}")
        print(f"[export] {len(df)} positions -> {out}")

    def _export_hcr_seg_crop(self):
        """Raw HCR segmentation labels cropped ±cube_half µm around the current pair
        (HCR µm frame), as a label tif + meta json — for downstream analysis/plots."""
        center = (self.hcr_by_id.get(self.cur_hcr_id) if self.cur_matched
                  else self._cz_pos(self.cur_cz_id))
        if center is None:
            self._set_export_status("no HCR location for this ROI")
            return
        from .build_artifacts import open_hcr_seg_zarr_array
        slicer, (Z, Y, X), xy_um, z_um = open_hcr_seg_zarr_array(self._seg_subject())
        h = float(self.cube_half)
        z0 = max(0, int((center[0] - h) / z_um)); z1 = min(Z, int((center[0] + h) / z_um) + 1)
        y0 = max(0, int((center[1] - h) / xy_um)); y1 = min(Y, int((center[1] + h) / xy_um) + 1)
        x0 = max(0, int((center[2] - h) / xy_um)); x1 = min(X, int((center[2] + h) / xy_um) + 1)
        crop = np.asarray(slicer(slice(z0, z1), slice(y0, y1), slice(x0, x1)),
                          dtype=np.int32)
        tag = f"cz{self.cur_cz_id}" + (f"_hcr{self.cur_hcr_id}" if self.cur_matched
                                       else "_unmatched")
        tif = self.export_dir / f"hcr_seg_crop_{self.sid}_{tag}.tif"
        # Native-res labels are ~99% background -> zlib shrinks the file ~50x.
        tifffile.imwrite(str(tif), crop, compression="zlib")
        n_labels = int(np.count_nonzero(np.unique(crop)))
        meta = dict(
            sid=self.sid, cz_id=int(self.cur_cz_id),
            hcr_id=int(self.cur_hcr_id) if self.cur_matched else None,
            center_hcr_um=[float(v) for v in center],
            origin_um=[z0 * z_um, y0 * xy_um, x0 * xy_um],
            voxel_um=[float(z_um), float(xy_um), float(xy_um)],
            shape=list(crop.shape), half_um=h, n_labels=n_labels,
        )
        tif.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        self._set_export_status(f"HCR seg crop {crop.shape} ({n_labels} cells) → {tif.name}")
        print(f"[export] HCR seg crop {crop.shape} ({n_labels} cells) -> {tif}")

    def _set_export_status(self, msg):
        if hasattr(self, "lbl_export"):
            self.lbl_export.setText(msg)


def launch(sid: str, variant: str = "step3_v3_anchor_vote_wang_end",
           cube_um: float = CUBE_HALF_UM, start: int = 0, hcr_level: int = HCR_LEVEL,
           cz_list_path: str | None = None, matches_csv: str | None = None,
           final_pairs_path: str | None = None, sort_mode: str = "soma_desc",
           worst_pct: float | None = None):
    """Build the QC window and return (QApplication, QCApp) without entering the
    event loop.  Callers that want a blocking GUI call ``app.exec_()`` after."""
    # World-writable outputs: /scratch is shared across uids (the GUI often runs as
    # root while setup ran as claude-user); umask 0 makes labels/exports writable by all.
    os.umask(0)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = QCApp(sid, variant, cube_um, start, hcr_level,
                cz_list_path=cz_list_path, matches_csv=matches_csv,
                final_pairs_path=final_pairs_path, sort_mode=sort_mode,
                worst_pct=worst_pct)
    win.show()
    return app, win


def main(argv=None):
    args = parse_args(argv)
    app, _win = launch(args.sid, args.variant, args.cube_um, args.start,
                       args.level, cz_list_path=args.cz_list,
                       matches_csv=args.matches_csv,
                       final_pairs_path=args.final_pairs, sort_mode=args.sort,
                       worst_pct=args.worst_pct)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
