from __future__ import annotations
"""
wp_ml_eda.py — Step 3.3: TRAIN-only EDA & feature audit

Reads Step 3.2 artifacts, analyzes TRAIN split (plus optional per-ticker + pre-impute audit),
plots distributions and correlations, and writes a compact JSON summary with
candidate feature drops for v1.0 spec.

Inputs (under base_dir):
  data_int/ml/
    X_train_k{K}.(parquet|csv)
    y_train_k{K}.(parquet|csv)
    X_val_k{K}.(parquet|csv)         # used only for optional drift charts
    y_val_k{K}.(parquet|csv)
    X_test_k{K}.(parquet|csv)
    y_test_k{K}.(parquet|csv)
    features_raw_long.(parquet|csv)  # has date,ticker,features,label_k
    split_meta_k{K}.json             # per-ticker cutoffs
    feature_spec_k{K}.json
  data_raw/ohlc_long.csv             # optional (pre-impute missingness audit)

Outputs (under data_int/ml/):
  eda_k{K}_summary.json
  eda_k{K}_figures/
    hist_label.png
    hist_<feature>.png               # top-N by |spearman|
    corr_topN.png                    # Spearman corr heatmap for top-N
    missingness_pre_impute.png       # if pre-impute audit available
    per_ticker_counts.csv

No new dependencies; uses numpy, pandas, matplotlib, (optional) sklearn if available.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional sklearn for mutual information (won't break if missing)
try:
    from sklearn.feature_selection import mutual_info_regression  # type: ignore
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

# Reuse feature builder to audit pre-impute missingness
try:
    from syslib import wp_ml_features as wp_feat
except Exception:
    wp_feat = None


# ── IO helpers ───────────────────────────────────────────────────────────────
def _read_table(path_parquet: Path, path_csv: Path) -> pd.DataFrame:
    if path_parquet.exists():
        return pd.read_parquet(path_parquet)
    if path_csv.exists():
        return pd.read_csv(path_csv)
    raise FileNotFoundError(f"Missing both {path_parquet} and {path_csv}")


def load_splits(base_dir: Path, k: int) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    ml = Path(base_dir) / "data_int" / "ml"
    Xtr = _read_table(ml / f"X_train_k{k}.parquet", ml / f"X_train_k{k}.csv")
    ytr = _read_table(ml / f"y_train_k{k}.parquet", ml / f"y_train_k{k}.csv")
    Xva = _read_table(ml / f"X_val_k{k}.parquet", ml / f"X_val_k{k}.csv")
    yva = _read_table(ml / f"y_val_k{k}.parquet", ml / f"y_val_k{k}.csv")
    Xte = _read_table(ml / f"X_test_k{k}.parquet", ml / f"X_test_k{k}.csv")
    yte = _read_table(ml / f"y_test_k{k}.parquet", ml / f"y_test_k{k}.csv")
    # y files may come with column 'y'; ensure Series
    if isinstance(ytr, pd.DataFrame) and ytr.shape[1] == 1:
        ytr = ytr.iloc[:,0]
    if isinstance(yva, pd.DataFrame) and yva.shape[1] == 1:
        yva = yva.iloc[:,0]
    if isinstance(yte, pd.DataFrame) and yte.shape[1] == 1:
        yte = yte.iloc[:,0]
    return Xtr, ytr, Xva, yva, Xte, yte


def load_features_raw(base_dir: Path) -> pd.DataFrame:
    ml = Path(base_dir) / "data_int" / "ml"
    return _read_table(ml / "features_raw_long.parquet", ml / "features_raw_long.csv")


# ── EDA primitives ───────────────────────────────────────────────────────────
def spearman_corr(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    # Rank-based corr, robust to monotonic nonlinearity
    df = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    ycol = df.columns[-1]
    corr = df.corr(method="spearman", numeric_only=True)[ycol].drop(ycol)
    return corr.sort_values(key=np.abs, ascending=False)


def maybe_mutual_info(X: pd.DataFrame, y: pd.Series, n_neighbors: int = 3) -> Optional[pd.Series]:
    if not SKLEARN_OK:
        return None
    X_ = X.replace([np.inf, -np.inf], np.nan).dropna()
    y_ = y.loc[X_.index]
    try:
        mi = mutual_info_regression(X_.values, y_.values, n_neighbors=n_neighbors, random_state=42)
        return pd.Series(mi, index=X_.columns).sort_values(ascending=False)
    except Exception:
        return None


def plot_hist(series: pd.Series, title: str, outpath: Path):
    plt.figure()
    s = series.replace([np.inf, -np.inf], np.nan).dropna()
    plt.hist(s.values, bins=50)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_hist_central(series: pd.Series, title: str, outpath: Path, p_lo: float = 1.0, p_hi: float = 99.0):
    """Same data, zoomed x-limits to central percentiles (for interpretability only)."""
    s = series.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return
    lo, hi = np.percentile(s.values, [p_lo, p_hi])
    plt.figure()
    plt.hist(s.values, bins=50)
    plt.xlim(lo, hi)
    plt.title(f"{title} — central {p_lo:.0f}-{p_hi:.0f}%")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_corr_heatmap(corr: pd.DataFrame, title: str, outpath: Path):
    plt.figure()
    # Scale to [-1,1] and show as image
    im = plt.imshow(corr.values, vmin=-1, vmax=1)
    plt.colorbar(im)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


# ── Pre-impute missingness audit (optional) ──────────────────────────────────
def pre_impute_missingness(base_dir: Path, k: int, spec: Optional[wp_feat.FeatureSpec] = None) -> Optional[pd.DataFrame]:
    if wp_feat is None:
        return None
    try:
        ohlc = pd.read_csv(Path(base_dir)/"data_raw"/"ohlc_long.csv")
    except FileNotFoundError:
        return None
    spec = spec or wp_feat.FeatureSpec(k_label=k)
    rows = []
    for tkr, g in ohlc.groupby("ticker"):
        g1 = g[[c for c in g.columns if c.lower() in {"date","ticker","open","high","low","close","adj_close","volume"}]].copy()
        g1.columns = [c.strip().lower() for c in g1.columns]
        g1 = g1.drop_duplicates(subset=["date"]).sort_values("date")
        feat_raw = wp_feat.build_features_for_ticker(g1, spec)  # masked, pre-impute
        miss = feat_raw.isna().mean().rename(tkr)
        rows.append(miss)
    miss_tbl = pd.DataFrame(rows)
    return miss_tbl


# ── Per-ticker split counts using split_meta & features_raw_long ─────────────
def per_ticker_counts(base_dir: Path, k: int) -> pd.DataFrame:
    ml = Path(base_dir)/"data_int"/"ml"
    meta = json.loads((ml/f"split_meta_k{k}.json").read_text())
    frl = load_features_raw(base_dir)
    frl["date"] = pd.to_datetime(frl["date"]).dt.tz_localize(None)
    out_rows = []
    for tkr, rec in meta.get("per_ticker", {}).items():
        g = frl[frl["ticker"]==tkr].sort_values("date")
        te = pd.to_datetime(rec.get("train_end")) if rec.get("train_end") else None
        ve = pd.to_datetime(rec.get("val_end")) if rec.get("val_end") else None
        # infer boundaries
        tr_n = len(g[g["date"]<=te]) if te is not None else 0
        va_n = len(g[(g["date"]>te) & (g["date"]<=ve)]) if te is not None and ve is not None else 0
        ts_n = len(g[g["date"]>ve]) if ve is not None else 0
        out_rows.append({"ticker":tkr, "train":tr_n, "val":va_n, "test":ts_n, "total":len(g)})
    return pd.DataFrame(out_rows).sort_values("ticker")


# ── Main runner ──────────────────────────────────────────────────────────────
def run_step_3_3(base_dir: Path, k_label: int = 10, topN: int = 15) -> Dict[str, Path]:
    """Perform EDA on TRAIN, save figures and a JSON summary with suggestions."""
    base_dir = Path(base_dir)
    ml = base_dir/"data_int"/"ml"
    fig_dir = ml/f"eda_k{k_label}_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    Xtr, ytr, Xva, yva, Xte, yte = load_splits(base_dir, k_label)

    # Basic shapes & dtype checks
    n_rows, n_feats = Xtr.shape
    assert len(ytr) == n_rows, "TRAIN X/y length mismatch"
    assert Xtr.select_dtypes("number").shape[1] == n_feats, "Non-numeric columns in TRAIN features"

    # Label histogram
    plot_hist(ytr, f"label_k{k_label} (TRAIN)", fig_dir/"hist_label.png")

    # Spearman correlations (TRAIN)
    sp = spearman_corr(Xtr, ytr)
    top_feats = sp.index[:min(topN, len(sp))].tolist()

    # Hist for top features
    for f in top_feats:
        plot_hist(Xtr[f], f"{f} (TRAIN)", fig_dir/f"hist_{f}.png")
        # also save a zoomed-in view that preserves data integrity but improves readability
        plot_hist_central(Xtr[f], f"{f} (TRAIN)", fig_dir/f"hist_{f}_central.png", p_lo=1.0, p_hi=99.0)

    # Feature↔feature corr on topN only
    corr_ff = Xtr[top_feats].corr(method="spearman")
    plot_corr_heatmap(corr_ff, f"Feature↔Feature Spearman (top {len(top_feats)})", fig_dir/"corr_topN.png")

    # Redundancy candidates (|ρ| > 0.90)
    redundant_pairs = []
    cols = corr_ff.columns
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            rho = corr_ff.iloc[i,j]
            if abs(rho) >= 0.90:
                redundant_pairs.append((cols[i], cols[j], float(rho)))

    # Optional MI scores
    mi_scores = None
    if SKLEARN_OK:
        mi = maybe_mutual_info(Xtr[top_feats], ytr)
        if mi is not None:
            mi_scores = mi.to_dict()

    # Pre-impute missingness audit (if wp_feat & ohlc available)
    miss_tbl = pre_impute_missingness(base_dir, k_label, getattr(wp_feat, "FeatureSpec", None)() if wp_feat else None)
    miss_png = None
    if isinstance(miss_tbl, pd.DataFrame) and not miss_tbl.empty:
        # average missingness per feature across tickers
        mean_miss = miss_tbl.mean(axis=0).sort_values(ascending=False)
        # plot
        plt.figure(figsize=(8, 4+0.15*len(mean_miss)))
        plt.barh(mean_miss.index, mean_miss.values)
        plt.title("Pre-impute missingness (avg across tickers)")
        plt.tight_layout()
        miss_png = fig_dir/"missingness_pre_impute.png"
        plt.savefig(miss_png)
        plt.close()

    # Per-ticker counts per split
    ptc = per_ticker_counts(base_dir, k_label)
    ptc_path = fig_dir/"per_ticker_counts.csv"
    ptc.to_csv(ptc_path, index=False)

    # Build summary
    summary = {
        "k_label": k_label,
        "n_train_rows": int(n_rows),
        "n_features": int(n_feats),
        "top_features_by_abs_spearman": sp.head(topN).round(4).to_dict(),
        "redundant_pairs_abs_r_ge_0.90": [
            {"f1":a, "f2":b, "rho":round(r,4)} for a,b,r in redundant_pairs
        ],
        "mutual_info_topN": mi_scores,
        "figures": {
            "hist_label": str((fig_dir/"hist_label.png").name),
            "corr_topN": str((fig_dir/"corr_topN.png").name),
            "missingness_pre_impute": str(Path(miss_png).name) if miss_png else None
        },
        "files": {
            "per_ticker_counts_csv": str(ptc_path.name)
        }
    }

    out_json = ml/f"eda_k{k_label}_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))

    # Return key paths
    return {
        "summary_json": out_json,
        "fig_dir": fig_dir,
        "per_ticker_counts_csv": ptc_path,
    }


if __name__ == "__main__":
    from datetime import date
    BASE = Path.cwd() / date.today().strftime("%d-%m-%Y")
    print(run_step_3_2_with_options(BASE, k_label=10, winsorize=True, drop_cols=["rv20"]))
