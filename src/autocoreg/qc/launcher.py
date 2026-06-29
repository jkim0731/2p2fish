"""``autocoreg_qc`` — interactive launcher for the QC app.

Opens a small window listing the subjects that have an attached **labeled
ROI-quality asset** (Capsule-1 output: ``HCR_<sid>_..._HCR-ROI-label_*/
<sid>_roi_quality_proba.parquet``) under ``DATA_ROOT``.  Click one to open the QC
app for that subject — no CLI arguments needed.

Console entry point (pyproject ``[project.scripts]``):  ``autocoreg_qc``.

Output dirs come from the ``MFISH_*`` env / config defaults (all under ``/scratch``),
so ``source qc_env.sh; autocoreg_qc`` uses the session dirs; bare ``autocoreg_qc``
uses the ``/scratch/autocoreg_outputs`` defaults.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from PyQt5 import QtWidgets

from autocoreg import config as _config

DEFAULT_VARIANT = "anchor_vote_anchor_restricted"


def discover_labeled_subjects() -> list[tuple[str, Path]]:
    """[(sid, asset_dir)] for every labeled ROI-quality asset under DATA_ROOT."""
    out, seen = [], set()
    for proba in sorted(_config.DATA_ROOT.glob(
            "HCR_*_HCR-ROI-label_*/*_roi_quality_proba.parquet")):
        m = re.match(r"(\d+)_roi_quality_proba\.parquet", proba.name)
        if not m or m.group(1) in seen:
            continue
        seen.add(m.group(1))
        out.append((m.group(1), proba.parent))
    return out


def find_matches_csv(sid: str, variant: str) -> Path | None:
    """Latest final-round matcher CSV for (sid, variant) under QC_MATCHES_DIR."""
    d = _config.QC_MATCHES_DIR / variant / sid
    for pat in ("matches_anchor_restricted_round[0-9]*.csv", "matches_wang_round[0-9]*.csv", "matches_round[0-9]*.csv"):
        cands = sorted(d.glob(pat),
                       key=lambda p: int(re.findall(r"\d+", p.stem)[-1] or 0))
        if cands:
            return cands[-1]
    return None


class LauncherDialog(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("autocoreg QC — pick a labeled subject")
        self.selected = None
        self.subjects = discover_labeled_subjects()

        v = QtWidgets.QVBoxLayout(self)
        v.addWidget(QtWidgets.QLabel(
            f"Labeled ROI-quality assets under {_config.DATA_ROOT}:"))
        self.listw = QtWidgets.QListWidget()
        for sid, path in self.subjects:
            has = find_matches_csv(sid, DEFAULT_VARIANT) is not None
            tag = "" if has else "   [no matcher output — run `autocoreg run <sid>`]"
            self.listw.addItem(f"{sid}    ({path.name}){tag}")
        self.listw.itemDoubleClicked.connect(lambda _i: self._open())
        v.addWidget(self.listw)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Variant:"))
        self.var_edit = QtWidgets.QLineEdit(DEFAULT_VARIANT)
        row.addWidget(self.var_edit)
        v.addLayout(row)

        row2 = QtWidgets.QHBoxLayout()
        row2.addWidget(QtWidgets.QLabel("Sort:"))
        self.cmb_sort = QtWidgets.QComboBox()
        self.cmb_sort.addItems(["Least-confident first", "Most-confident first",
                                "Matcher order"])
        row2.addWidget(self.cmb_sort)
        row2.addWidget(QtWidgets.QLabel("Worst %:"))
        self.spin = QtWidgets.QDoubleSpinBox()
        self.spin.setRange(0, 100); self.spin.setDecimals(0)
        self.spin.setToolTip("0 = full queue; N = only the worst N% matched pairs")
        row2.addWidget(self.spin)
        v.addLayout(row2)

        btns = QtWidgets.QHBoxLayout()
        b_open = QtWidgets.QPushButton("Open QC"); b_open.clicked.connect(self._open)
        b_cancel = QtWidgets.QPushButton("Cancel"); b_cancel.clicked.connect(self.reject)
        btns.addStretch(1); btns.addWidget(b_cancel); btns.addWidget(b_open)
        v.addLayout(btns)

        if self.subjects:
            self.listw.setCurrentRow(0)
        self.resize(620, 440)

    def _open(self):
        i = self.listw.currentRow()
        if i < 0:
            return
        sid, _ = self.subjects[i]
        variant = self.var_edit.text().strip() or DEFAULT_VARIANT
        csv = find_matches_csv(sid, variant)
        if csv is None:
            QtWidgets.QMessageBox.warning(
                self, "No matcher output",
                f"No matches found for {sid} under\n{_config.QC_MATCHES_DIR / variant / sid}\n\n"
                f"Run the pipeline first:\n    autocoreg run {sid} --build-qc")
            return
        self.selected = dict(
            sid=sid, variant=variant, matches_csv=str(csv),
            sort_mode={0: "soma_desc", 1: "soma_asc", 2: "matcher"}[self.cmb_sort.currentIndex()],
            worst_pct=(self.spin.value() or None),
        )
        self.accept()


def main(argv=None):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    dlg = LauncherDialog()
    if not dlg.subjects:
        QtWidgets.QMessageBox.warning(
            None, "No labeled assets",
            f"No HCR_*_HCR-ROI-label_*/*_roi_quality_proba.parquet under "
            f"{_config.DATA_ROOT}.\nAttach a Capsule-1 ROI-quality asset first.")
        return
    if dlg.exec_() != QtWidgets.QDialog.Accepted or not dlg.selected:
        return
    sel = dlg.selected
    from .app import launch
    _app, _win = launch(sel["sid"], variant=sel["variant"],
                        matches_csv=sel["matches_csv"], sort_mode=sel["sort_mode"],
                        worst_pct=sel["worst_pct"])
    sys.exit(_app.exec_())


if __name__ == "__main__":
    main()
