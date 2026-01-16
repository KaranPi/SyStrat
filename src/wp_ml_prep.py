from __future__ import annotations  # no installation needed
"""
wp_ml_prep.py — Step 3.1: label builder (daily‑equivalent realized volatility)
-------------------------------------------------------------------------------
Build leakage‑safe labels σ_t^(k) = sqrt((1/k) * sum_{i=1..k} r_{t+i}^2) using
**log returns** (close→close), for k in {5, 10, 20} by default.

Inputs:
- log return matrix (ascending index, wide: columns=tickers). If unavailable,
  pass a Close matrix and we'll compute log returns for you.

Outputs (under your chosen base directory):
- data_int/ml/labels_k{K}.csv(.parquet if available)
- a compact sanity DataFrame (per k and ticker) that you can display/log.

Notes:
- Daily‑equivalent means no annualization (units align with your sizing logic).
- Implementation is vectorized and leakage‑safe (uses forward windows only).
- Compatible with artifacts produced by `wp_core.save_ohlc_and_returns`.
"""

# ── Imports (annotated) ────────────────────────────────────────────────────────
from typing import Dict, Iterable, List, Optional  # no installation needed
from pathlib import Path  # no installation needed

import numpy as np  # already in env — no new install
import pandas as pd  # already in env — no new install


# ── Utilities ─────────────────────────────────────────────────────────────────
def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out


def compute_log_returns_from_close(close_asc: pd.DataFrame) -> pd.DataFrame:
    """Compute ascending log returns from ascending Close matrix."""
    close_asc = _ensure_datetime_index(close_asc)
    lr = np.log(close_asc / close_asc.shift(1))
    return lr.dropna(how="all")


def build_labels_from_log_returns(
    logret_asc: pd.DataFrame,
    ks: Iterable[int] = (5, 10, 20),
) -> Dict[int, pd.DataFrame]:
    """Vectorized forward realized‑vol labels for multiple k values.
    Returns dict {k: labels_df} with the same columns as logret_asc.
    """
    lr = _ensure_datetime_index(logret_asc).sort_index()
    r2 = lr ** 2

    out: Dict[int, pd.DataFrame] = {}
    for k in ks:
        # sum of next k r^2 aligned at t: shift(-1), rolling(k).sum(), then shift(-(k-1))
        s = r2.shift(-1)
        sum_next_k = s.rolling(window=k, min_periods=k).sum().shift(-(k - 1))
        lab = (sum_next_k / float(k)) ** 0.5
        lab = lab.dropna(how="all")
        out[int(k)] = lab
    return out


def _try_parquet(df: pd.DataFrame, path: Path) -> Optional[Path]:
    try:
        df.to_parquet(path)
        return path
    except Exception:
        return None


def save_labels(
    labels: Dict[int, pd.DataFrame],
    base_dir: Path,
    prefix: str = "labels_k",
) -> Dict[str, Optional[Path]]:
    """Save each labels[k] to CSV (+ Parquet if available) under data_int/ml."""
    base_dir = Path(base_dir)
    out_dir = base_dir / "data_int" / "ml"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Optional[Path]] = {}
    for k, df in labels.items():
        csv_p = out_dir / f"{prefix}{k}.csv"
        pq_p = out_dir / f"{prefix}{k}.parquet"
        df.to_csv(csv_p)
        paths[f"k{k}_csv"] = csv_p
        paths[f"k{k}_parquet"] = _try_parquet(df, pq_p)
    return paths


def sanity_report(labels: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Return a tidy sanity table with counts & basic stats by k and ticker."""
    rows = []
    for k, df in labels.items():
        for tkr in df.columns:
            s = df[tkr].dropna()
            if s.empty:
                rows.append({
                    "k": k, "ticker": tkr, "n": 0,
                    "start": None, "end": None,
                    "mean": np.nan, "median": np.nan,
                    "std": np.nan, "p10": np.nan, "p90": np.nan,
                })
                continue
            rows.append({
                "k": k,
                "ticker": tkr,
                "n": int(s.shape[0]),
                "start": s.index[0].date(),
                "end": s.index[-1].date(),
                "mean": float(s.mean()),
                "median": float(s.median()),
                "std": float(s.std(ddof=1)) if s.shape[0] > 1 else 0.0,
                "p10": float(s.quantile(0.10)),
                "p90": float(s.quantile(0.90)),
            })
    rep = pd.DataFrame(rows).sort_values(["k", "ticker"]).reset_index(drop=True)
    return rep


# ── Convenience: end‑to‑end from files ────────────────────────────────────────
def build_labels_from_files(
    base_dir: Path,
    ks: Iterable[int] = (5, 10, 20),
    log_returns_csv: Optional[Path] = None,
    close_csv: Optional[Path] = None,
) -> Dict[int, pd.DataFrame]:
    """Load returns (prefer log_returns.csv), fall back to close.csv if needed,
    compute labels and return {k: DataFrame}.
    """
    base_dir = Path(base_dir)
    ml_dir = base_dir / "data_int"

    if log_returns_csv is None:
        log_returns_csv = ml_dir / "log_returns.csv"
    if close_csv is None:
        close_csv = ml_dir / "close.csv"

    if Path(log_returns_csv).exists():
        lr = pd.read_csv(log_returns_csv, index_col=0)
        lr.index = pd.to_datetime(lr.index)
        return build_labels_from_log_returns(lr, ks=ks)

    if Path(close_csv).exists():
        close = pd.read_csv(close_csv, index_col=0)
        close.index = pd.to_datetime(close.index)
        lr = compute_log_returns_from_close(close)
        return build_labels_from_log_returns(lr, ks=ks)

    raise FileNotFoundError(
        f"Neither log_returns.csv nor close.csv found under: {ml_dir}"
    )


if __name__ == "__main__":
    # Example (adjust base_dir as needed):
    BASE = Path.cwd()  # expects data_int/log_returns.csv or data_int/close.csv
    labs = build_labels_from_files(BASE, ks=(5, 10, 20))
    paths = save_labels(labs, BASE)
    print(paths)
    print(sanity_report(labs).head())
