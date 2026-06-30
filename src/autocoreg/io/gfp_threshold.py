"""Scale-only GFP+ threshold via GMM-intersection.

Vendored from ``dev_code/07b_gfp_intersection_threshold.py`` (monorepo).
The module name was changed from ``07b_gfp_intersection_threshold`` (which
starts with a digit and cannot be imported normally) to
``gfp_intersection_threshold``.

Changes vs the original:
* Local sys.path manipulation replaced by package-relative imports.
* ``matplotlib`` import is lazy (guarded inside ``_plot_subject``) so that
  ``analyze_subject`` / ``strict_gfp_df`` — the only paths called in
  production — do NOT require matplotlib.
* ``SESSION_DIR``, ``FIG_DIR``, and the JSON output path are routed through
  ``config.py`` so they resolve to writable locations without hard-coding the
  capsule path.

Public API (matches the original):
    analyze_subject(sid: str) -> GmmIntersection
    strict_gfp_df(sid: str, cutoff_linear: float) -> pd.DataFrame
    GmmIntersection (dataclass)
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from autocoreg import config as _config
from autocoreg.io.subjects import DATA_DIR, BENCHMARK_SUBJECTS, _aggregate_spots_from_hcr, load_subject

# Output paths routed through config so they land in a writable location.
# These are optional outputs (the JSON result and PNG figures); they are NOT
# required as inputs on any production code path.  The GMM cutoff is computed
# live in analyze_subject() every time (seeded random_state=0 for reproducibility).
SESSION_DIR = _config.CACHE_DIR / "sessions" / "07b_scale_clean_gfp"
FIG_DIR = SESSION_DIR / "figures"

SPOT_SUBJECTS = {"788406", "790322", "767018", "782149"}
INTENSITY_SUBJECTS = {"755252", "767022"}


def detect_gfp_class(sid: str) -> str:
    """Return the GFP-feature class for a subject: ``'spot'`` or ``'intensity'``.

    The six benchmark subjects keep their hardcoded class (behaviour unchanged).
    Any other subject (new / pipeline-monitor data) is classified from the
    attached HCR data so the automated path needs no per-subject list edit:

      * ``'spot'`` if a ``*spot_488_counts.csv`` is present in the coreg dir, or
        the HCR processed asset carries 488 spot detection
        (``image_spot_detection/channel_488_spots/spots.csv``);
      * else ``'intensity'`` if a ``cell_data_mean_{sid}_R1.csv`` is present;
      * else ``'spot'`` (the modern-HCR default — every recent HCR run produces
        488 spot detection; if no feature is actually loadable, the downstream
        ``analyze_subject`` / ``strict_gfp_df`` raise a clear error).

    Detection uses cheap path-existence checks only (no large CSV reads).
    """
    if sid in SPOT_SUBJECTS:
        return "spot"
    if sid in INTENSITY_SUBJECTS:
        return "intensity"
    hcr_dir = None
    try:
        s = load_subject(sid)
        if list(Path(s.coreg_dir).glob("*spot_488_counts.csv")):
            return "spot"
        hcr_dir = Path(s.hcr_dir)
    except Exception:
        hcr_dir = None
    if hcr_dir is not None and (
        hcr_dir / "image_spot_detection" / "channel_488_spots" / "spots.csv"
    ).exists():
        return "spot"
    for cand in (Path(DATA_DIR) / f"cell_data_mean_{sid}_R1.csv",
                 Path(_config.DATA_LEVEL) / f"cell_data_mean_{sid}_R1.csv"):
        if cand.exists():
            return "intensity"
    return "spot"


@dataclass
class GmmIntersection:
    subject: str
    feature_name: str          # 'density' | 'mean_minus_bg'
    log_base: str              # 'ln' | 'log10'
    n_components: int          # BIC-selected K
    means_log: list            # ascending by mean
    sigmas_log: list
    weights: list
    intersection_log: float    # in log-domain
    cutoff_linear: float       # linear-domain cutoff
    no_interior_root: bool
    sigma_right_over_next: float
    bic: float                 # BIC of the selected K
    bic_sweep: list            # [{n_components, bic, aic, intersection_log, ...}] over K=2..6
    n_total: int               # rows in the raw feature table before >0 filter
    n_positive: int            # rows with feature > 0 (used for GMM)
    n_strict: int              # rows passing the strict cutoff
    n_v22_reference: int       # rows that the v2.2 loader keeps (from SubjectData)
    cutoff_v22_linear: float   # v2.2 threshold in the same linear feature
    coreg_total: int
    coreg_kept_strict: int
    coreg_coverage_strict: float
    sanity_passed: bool
    sanity_notes: list


# ----------------------------------------------------------------------
# GMM intersection math
# ----------------------------------------------------------------------
def gaussian_intersection(
    mu1: float, s1: float, w1: float,
    mu2: float, s2: float, w2: float,
) -> tuple[float, bool]:
    """Solve w1·N(x; μ1, σ1) == w2·N(x; μ2, σ2) for x; return (x, no_interior_root).

    Preconditions: mu1 < mu2 (caller ensures ascending order). We select the
    root in (mu1, mu2). Returns midpoint and sets ``no_interior_root=True`` if
    no interior root exists (extreme weights/variances).
    """
    if mu1 == mu2:
        return float(mu1), True

    v1 = s1 * s1
    v2 = s2 * s2

    # Degenerate equal-variance → linear equation
    if abs(v1 - v2) < 1e-12:
        denom = mu2 - mu1
        x = 0.5 * (mu1 + mu2) + (v1 / denom) * np.log(w2 / max(w1, 1e-300))
        inside = (mu1 < x) and (x < mu2)
        return float(x), (not inside)

    A = 1.0 / (2.0 * v2) - 1.0 / (2.0 * v1)
    B = mu1 / v1 - mu2 / v2
    C = (
        np.log(max(w1, 1e-300)) - np.log(max(w2, 1e-300))
        + 0.5 * (np.log(v2) - np.log(v1))
        + (mu2 * mu2) / (2.0 * v2)
        - (mu1 * mu1) / (2.0 * v1)
    )
    disc = B * B - 4.0 * A * C
    if disc < 0:
        return 0.5 * (mu1 + mu2), True
    sq = float(np.sqrt(disc))
    r1 = (-B + sq) / (2.0 * A)
    r2 = (-B - sq) / (2.0 * A)
    candidates = [r for r in (r1, r2) if mu1 < r < mu2]
    if not candidates:
        return 0.5 * (mu1 + mu2), True
    # Prefer the root closer to the weighted midpoint (for robustness)
    ref = 0.5 * (mu1 + mu2)
    best = min(candidates, key=lambda r: abs(r - ref))
    return float(best), False


def fit_gmm_intersection(
    log_values: np.ndarray, n_components: int, random_state: int = 0
) -> dict:
    """Fit a K-component GMM on log-values and return intersection between
    the rightmost (K−1) and second-rightmost (K−2) components."""
    X = np.asarray(log_values, dtype=float).reshape(-1, 1)
    gmm = GaussianMixture(
        n_components=n_components, n_init=5, random_state=random_state
    ).fit(X)
    order = np.argsort(gmm.means_.ravel())
    mus = gmm.means_.ravel()[order]
    sigmas = np.sqrt(gmm.covariances_.ravel()[order])
    weights = gmm.weights_[order]
    # Rightmost (index K-1) and second-rightmost (K-2)
    mu1, s1, w1 = mus[-2], sigmas[-2], weights[-2]
    mu2, s2, w2 = mus[-1], sigmas[-1], weights[-1]
    x_int, no_interior = gaussian_intersection(mu1, s1, w1, mu2, s2, w2)
    return {
        "means": mus.tolist(),
        "sigmas": sigmas.tolist(),
        "weights": weights.tolist(),
        "intersection_log": float(x_int),
        "no_interior_root": bool(no_interior),
        "sigma_right": float(sigmas[-1]),
        "sigma_next": float(sigmas[-2]),
        "bic": float(gmm.bic(X)),
        "aic": float(gmm.aic(X)),
        "n_components": int(n_components),
    }


def fit_gmm_sweep(
    log_values: np.ndarray,
    k_min: int = 2,
    k_max: int = 6,
    random_state: int = 0,
) -> dict:
    """Sweep ``n_components`` over ``[k_min, k_max]`` and pick the best K by BIC.

    Returns a dict with:
      * ``best``: the fit for the BIC-winning K (same shape as
        ``fit_gmm_intersection``'s return value).
      * ``sweep``: list of per-K dicts with BIC, AIC, means, intersection_log,
        so the driver can log a full sweep table.
    The selection rule is plain argmin of BIC — no within-1-SE heuristic,
    because we want the strictest density fit the data supports.
    """
    X = np.asarray(log_values, dtype=float)
    sweep = []
    for k in range(k_min, k_max + 1):
        try:
            fit = fit_gmm_intersection(X, n_components=k, random_state=random_state)
            sweep.append(fit)
        except Exception as e:
            sweep.append({"n_components": k, "error": f"{type(e).__name__}: {e}"})
    valid = [f for f in sweep if "bic" in f]
    if not valid:
        raise RuntimeError("GMM sweep produced no valid fits")
    best = min(valid, key=lambda f: f["bic"])
    return {"best": best, "sweep": sweep}


# ----------------------------------------------------------------------
# Feature loaders (no loader modification)
# ----------------------------------------------------------------------
def _load_spot_feature(sid: str) -> pd.DataFrame:
    """Return DataFrame with columns [hcr_id, density] for a spot subject.

    Source priority matches ``benchmark_data_loader._load_gfp``:
      1. ``*spot_488_counts.csv`` in the subject's coreg dir.
      2. ``_aggregate_spots_from_hcr`` fallback.
    """
    s = load_subject(sid)
    # Use SubjectData paths: s.coreg_dir / s.hcr_dir
    coreg_dir = s.coreg_dir
    hcr_dir = s.hcr_dir
    spot_files = list(Path(coreg_dir).glob("*spot_488_counts.csv"))
    if spot_files:
        df = pd.read_csv(spot_files[0])
        if "density" not in df.columns and {"counts", "volume"}.issubset(df.columns):
            df["density"] = df["counts"] / df["volume"]
        return df[["hcr_id", "density"]].dropna()
    agg = _aggregate_spots_from_hcr(Path(hcr_dir))
    if agg is None or len(agg) == 0:
        raise RuntimeError(f"No spot feature available for {sid}")
    if "density" not in agg.columns:
        raise RuntimeError(f"Spot data for {sid} has no density column")
    return agg[["hcr_id", "density"]].dropna()


def _load_intensity_feature(sid: str) -> pd.DataFrame:
    """Return DataFrame with columns [hcr_id, mean_minus_bg] for intensity subject."""
    fname = f"cell_data_mean_{sid}_R1.csv"
    # The two intensity subjects' CSVs were re-attached only under the archived
    # data dir (read-only), not directly under DATA_DIR — search both.
    candidates = [
        Path(DATA_DIR) / fname,
        Path(_config.DATA_LEVEL) / fname,
    ]
    p = next((c for c in candidates if c.exists()), None)
    if p is None:
        raise RuntimeError(
            f"No intensity CSV for {sid}: searched {[str(c) for c in candidates]}"
        )
    df = pd.read_csv(p)
    if "channel" in df.columns:
        df = df[df["channel"] == 488]
    rename_candidates = {"cell_id": "hcr_id", "id": "hcr_id"}
    for k, v in rename_candidates.items():
        if k in df.columns:
            df = df.rename(columns={k: v})
    df = df.copy()
    df["mean_minus_bg"] = df["mean"].astype(float) - df["background"].astype(float)
    return df[["hcr_id", "mean_minus_bg"]].dropna()


# ----------------------------------------------------------------------
# Per-subject analyse + plot
# ----------------------------------------------------------------------
def _plot_subject(
    sid: str, feat_values: np.ndarray, fit: dict, cutoff_linear: float,
    v22_cutoff_linear: float, feature_name: str, log_base: str,
    out_path: Path, sweep_entries: list | None = None,
):
    # Lazy matplotlib import so that analyze_subject / strict_gfp_df do not
    # require matplotlib on the production code path.
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    is_log10 = (log_base == "log10")
    logfn = np.log10 if is_log10 else np.log
    expfn = (lambda x: 10.0 ** x) if is_log10 else np.exp
    base_str = "log10" if is_log10 else "ln"

    pos = feat_values[feat_values > 0]
    log_x = logfn(pos)
    mus = fit["means"]
    sigmas = fit["sigmas"]
    weights = fit["weights"]
    x_int = fit["intersection_log"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.4))
    # linear density panel (log-x)
    ax = axes[0]
    bins = np.linspace(log_x.min() - 0.2, log_x.max() + 0.2, 100)
    ax.hist(log_x, bins=bins, density=True, alpha=0.55, color="#94a3b8",
            label=f"{base_str}({feature_name})")
    xs = np.linspace(bins[0], bins[-1], 400)
    total = np.zeros_like(xs)
    for mu, sig, w in zip(mus, sigmas, weights):
        comp = w * norm.pdf(xs, mu, sig)
        ax.plot(xs, comp, lw=1.0, alpha=0.8)
        total = total + comp
    ax.plot(xs, total, lw=1.5, color="black", label="GMM total")
    ax.axvline(x_int, color="#cc3333", lw=2.0, label=f"strict cutoff ({base_str}): {x_int:.3f}")
    if np.isfinite(v22_cutoff_linear) and v22_cutoff_linear > 0:
        v22_log = logfn(v22_cutoff_linear)
        ax.axvline(v22_log, color="#3b7dd8", lw=1.2, ls="--",
                   label=f"v2.2 cutoff ({base_str}): {v22_log:.3f}")
    ax.set_xlabel(f"{base_str}({feature_name})")
    ax.set_ylabel("density")
    ax.set_title(f"{sid} — {feature_name} histogram + GMM-{len(mus)} + intersection")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right panel: linear-axis histogram zoomed around the two top components
    ax = axes[1]
    if is_log10:
        edges = 10 ** np.linspace(log_x.min() - 0.1, log_x.max() + 0.1, 100)
    else:
        edges = np.exp(np.linspace(log_x.min() - 0.1, log_x.max() + 0.1, 100))
    ax.hist(pos, bins=edges, color="#94a3b8", alpha=0.6, label="all cells")
    ax.axvline(cutoff_linear, color="#cc3333", lw=2.0,
               label=f"strict cutoff: {cutoff_linear:.3g}")
    if np.isfinite(v22_cutoff_linear) and v22_cutoff_linear > 0:
        ax.axvline(v22_cutoff_linear, color="#3b7dd8", lw=1.2, ls="--",
                   label=f"v2.2 cutoff: {v22_cutoff_linear:.3g}")
    ax.set_xscale("log")
    ax.set_xlabel(f"{feature_name} (linear, log-axis)")
    ax.set_ylabel("count")
    ax.set_title(f"{sid} — linear-axis view")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    # BIC sweep panel
    ax = axes[2]
    if sweep_entries:
        ks = [e["n_components"] for e in sweep_entries if "bic" in e]
        bics = [e["bic"] for e in sweep_entries if "bic" in e]
        aics = [e["aic"] for e in sweep_entries if "bic" in e]
        ax.plot(ks, bics, "o-", color="#268bd2", label="BIC")
        ax.plot(ks, aics, "s--", color="#859900", label="AIC", alpha=0.6)
        best_k = fit["n_components"]
        best_bic = fit["bic"]
        ax.axvline(best_k, color="#cc3333", lw=1.5, alpha=0.6,
                   label=f"best K = {best_k}")
        ax.scatter([best_k], [best_bic], color="#cc3333", zorder=5, s=80)
    ax.set_xlabel("n_components")
    ax.set_ylabel("information criterion")
    ax.set_title(f"{sid} — BIC/AIC sweep K=2..6")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def analyze_subject(sid: str) -> GmmIntersection:
    """Run BIC-GMM sweep on the GFP+ feature for ``sid`` and return the
    Bayes-optimal intersection cutoff.

    NOTE: This function is called ``analyze_subject`` to match the original
    ``07b_gfp_intersection_threshold.py`` public API.  It is distinct from
    ``benchmark_analysis.analyze_subject`` (which fits surfaces / computes
    coreg metadata).  Callers must import this module explicitly:
        from . import gfp_intersection_threshold as _gfp
        gi = _gfp.analyze_subject(sid)
    """
    s = load_subject(sid)
    coreg = s.coreg_table
    coreg_hcr_ids = set(int(x) for x in coreg["hcr_id"].values)

    # v2.2 cutoff (linear): back-derived from the loaded gfp_df.
    v22_df = s.hcr_gfp_df
    n_v22 = int(len(v22_df))

    gfp_class = detect_gfp_class(sid)
    if gfp_class == "spot":
        df_feat = _load_spot_feature(sid)
        feat_values = df_feat["density"].values.astype(float)
        log_x = np.log(feat_values[feat_values > 0])
        sweep = fit_gmm_sweep(log_x, k_min=2, k_max=6)
        fit = sweep["best"]
        cutoff = float(np.exp(fit["intersection_log"]))
        if "density" in v22_df.columns and n_v22:
            v22_cut = float(v22_df["density"].min())
        else:
            v22_cut = float("nan")
        feature_name = "density"
        log_base = "ln"
    elif gfp_class == "intensity":
        df_feat = _load_intensity_feature(sid)
        feat_values = df_feat["mean_minus_bg"].values.astype(float)
        log_x = np.log10(feat_values[feat_values > 0])
        sweep = fit_gmm_sweep(log_x, k_min=2, k_max=6)
        fit = sweep["best"]
        cutoff = float(10.0 ** fit["intersection_log"])
        if "mean_minus_bg" in v22_df.columns and n_v22:
            v22_cut = float(v22_df["mean_minus_bg"].min())
        elif {"mean", "background"}.issubset(v22_df.columns) and n_v22:
            v22_cut = float((v22_df["mean"] - v22_df["background"]).min())
        else:
            v22_cut = float("nan")
        feature_name = "mean_minus_bg"
        log_base = "log10"
    else:
        raise ValueError(f"unknown subject class for {sid}")  # detect_gfp_class never returns this

    # Apply strict cutoff and compute coverage
    mask = df_feat[feature_name].values.astype(float) >= cutoff
    df_strict = df_feat[mask].reset_index(drop=True)
    n_strict = int(len(df_strict))
    strict_ids = set(int(x) for x in df_strict["hcr_id"].values)
    coreg_kept = len(coreg_hcr_ids & strict_ids)
    coreg_cov = coreg_kept / max(len(coreg_hcr_ids), 1)

    # Sanity
    notes = []
    mu_right = fit["means"][-1]
    mu_next = fit["means"][-2]
    interior_ok = (mu_next < fit["intersection_log"] < mu_right)
    if not interior_ok or fit["no_interior_root"]:
        notes.append("intersection outside top-two mean bracket")
    if n_strict < 300:
        notes.append(f"n_strict={n_strict} < 300")
    if coreg_cov < 0.80:
        notes.append(f"coreg_coverage_strict={coreg_cov:.3f} < 0.80")
    sanity = (len(notes) == 0)

    # Optional diagnostic plot — only if FIG_DIR is writable; silently skipped
    # on read-only deployments.  Does NOT affect the returned cutoff.
    try:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        out_png = FIG_DIR / f"gmm_threshold_{sid}.png"
        _plot_subject(
            sid, feat_values, fit, cutoff, v22_cut, feature_name, log_base, out_png,
            sweep_entries=sweep["sweep"],
        )
    except Exception:
        pass  # plotting is non-critical; never block the cutoff computation

    # Compact sweep entries for JSON storage
    sweep_entries = []
    for entry in sweep["sweep"]:
        if "error" in entry:
            sweep_entries.append({"n_components": entry["n_components"], "error": entry["error"]})
        else:
            sweep_entries.append({
                "n_components": entry["n_components"],
                "bic": entry["bic"],
                "aic": entry["aic"],
                "intersection_log": entry["intersection_log"],
                "no_interior_root": entry["no_interior_root"],
                "means": entry["means"],
            })

    return GmmIntersection(
        subject=sid,
        feature_name=feature_name,
        log_base=log_base,
        n_components=int(fit["n_components"]),
        means_log=fit["means"],
        sigmas_log=fit["sigmas"],
        weights=fit["weights"],
        intersection_log=float(fit["intersection_log"]),
        cutoff_linear=float(cutoff),
        no_interior_root=bool(fit["no_interior_root"]),
        sigma_right_over_next=float(fit["sigma_right"] / max(fit["sigma_next"], 1e-12)),
        bic=float(fit["bic"]),
        bic_sweep=sweep_entries,
        n_total=int(len(df_feat)),
        n_positive=int((feat_values > 0).sum()),
        n_strict=n_strict,
        n_v22_reference=n_v22,
        cutoff_v22_linear=float(v22_cut) if np.isfinite(v22_cut) else float("nan"),
        coreg_total=int(len(coreg_hcr_ids)),
        coreg_kept_strict=int(coreg_kept),
        coreg_coverage_strict=float(coreg_cov),
        sanity_passed=sanity,
        sanity_notes=notes,
    )


def strict_gfp_df(sid: str, cutoff_linear: float) -> pd.DataFrame:
    """Return the strict GFP+ table for a subject given the linear cutoff.

    Columns: ``hcr_id`` plus the feature column (``density`` or
    ``mean_minus_bg``).  Matches the original ``07b`` public API.
    """
    gfp_class = detect_gfp_class(sid)
    if gfp_class == "spot":
        df_feat = _load_spot_feature(sid)
        return df_feat[df_feat["density"].values.astype(float) >= cutoff_linear].reset_index(drop=True)
    if gfp_class == "intensity":
        df_feat = _load_intensity_feature(sid)
        return df_feat[df_feat["mean_minus_bg"].values.astype(float) >= cutoff_linear].reset_index(drop=True)
    raise ValueError(f"unknown subject class for {sid}")  # detect_gfp_class never returns this


def main():
    from dataclasses import asdict
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for sid in BENCHMARK_SUBJECTS:
        try:
            r = analyze_subject(sid)
            results.append(asdict(r))
            print(
                f"  {sid}  cutoff={r.cutoff_linear:.4g}  n_strict={r.n_strict}  "
                f"coreg_cov={r.coreg_coverage_strict:.3f}  "
                f"sanity={'OK' if r.sanity_passed else '/'.join(r.sanity_notes)}"
            )
        except Exception as e:
            print(f"  {sid} FAILED: {type(e).__name__}: {e}")
            results.append({"subject": sid, "status": f"error: {e}"})
    out = SESSION_DIR / "gmm_threshold_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=lambda v: v.tolist()
                  if hasattr(v, "tolist") else str(v))
    print(f"Wrote {out}")
    print(f"Figures: {FIG_DIR}")


if __name__ == "__main__":
    main()
