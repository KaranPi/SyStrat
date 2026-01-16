from __future__ import annotations  # no installation needed
"""
wp_ml_features.py — Step 3.2: feature table (Parkinson, EMA9, MACD‑21 lens, slope20, vol/shape/calendar),
leakage‑safe imputation, time‑series splits, and scaling (TRAIN‑fit only).

Input artifacts expected under base_dir:
- data_raw/ohlc_long.csv  (from wp_core.save_ohlc_and_returns)
- data_int/ml/labels_k{K}.csv  (from wp_ml_prep)

Outputs under base_dir/data_int/ml/ :
- features_raw_long.parquet (date,ticker,+features) and CSV twin
- X_train/val/test.parquet + CSV, y_train/val/test.parquet + CSV
- feature_spec.json, split_meta.json, scaler_stats.json, build_report.json

Defaults:
- Label horizon k_default = 10 (we also support 5,20 if you pass other file)
- Imputation: rolling median (past‑only, window=20, min_valid=3) → expanding median fallback (min_expand=5)
- Splits: per‑ticker strict time split 40% / 35% / 25%
- Scaling: StandardScaler fit on TRAIN only, applied to VAL/TEST

All imports are annotated to match your env hygiene rules.
"""

# ── Imports (annotated) ────────────────────────────────────────────────────────
from typing import Dict, List, Tuple, Optional  # no installation needed
from dataclasses import dataclass  # no installation needed
from pathlib import Path  # no installation needed
import json  # no installation needed

import numpy as np  # already in env — no new install
import pandas as pd  # already in env — no new install

# sklearn only for scaling (approved)
from sklearn.preprocessing import StandardScaler  # install: scikit-learn==1.5.x


# ── Helpers: generic time ops ─────────────────────────────────────────────────
def _ensure_dt_index(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    """Ensure a tz-naive DatetimeIndex, robust to tz-aware/naive/strings.
    - If `col` exists, parse it and set as index.
    - Always coerce to UTC first, then drop tz to avoid TypeError on naive values.
    """
    out = df.copy()
    if col in out.columns:
        # Parse as UTC to avoid tz_localize errors on naive vs aware
        out[col] = pd.to_datetime(out[col], errors="coerce", utc=True).dt.tz_convert(None)
        out = out.set_index(col)
    # Ensure index is datetime and tz-naive
    idx = pd.to_datetime(out.index, errors="coerce", utc=True)
    out.index = pd.DatetimeIndex(idx.tz_convert(None))
    return out.sort_index()


def _choose_close_cols(df_long: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Return (open, high, low, close) series from a single‑ticker long frame.
    Prefers adj_close as close if present.
    """
    open_s = df_long["open"] if "open" in df_long.columns else None
    high_s = df_long["high"] if "high" in df_long.columns else None
    low_s  = df_long["low"]  if "low"  in df_long.columns else None
    if "adj_close" in df_long.columns:
        close_s = df_long["adj_close"]
    else:
        close_s = df_long["close"] if "close" in df_long.columns else None
    return open_s, high_s, low_s, close_s


# ── Feature primitives ────────────────────────────────────────────────────────
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def macd_line(price: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    return ema(price, fast) - ema(price, slow)


def macd_signal(macd: pd.Series, signal: int = 9) -> pd.Series:
    return ema(macd, signal)


def macd_21_lens(price: pd.Series) -> pd.Series:
    """A 21‑day lens on MACD: smooth the MACD line with EMA(21)."""
    m = macd_line(price, fast=12, slow=26)
    return ema(m, 21)


def parkinson_daily_sigma(high: pd.Series, low: pd.Series) -> pd.Series:
    """Parkinson's daily volatility estimate (dimensionless, daily).
    sigma_P = sqrt( (1/(4 ln2)) * (ln(H/L))^2 )."""
    c = 1.0 / (4.0 * np.log(2.0))
    hl = np.log(high / low)
    return np.sqrt(c * (hl ** 2))


def realized_vol(lr: pd.Series, k: int) -> pd.Series:
    r2 = (lr ** 2).rolling(window=k, min_periods=k).sum()
    return (r2 / float(k)).pow(0.5)


def ewma_vol(lr: pd.Series, lam: float = 0.94, min_periods: int = 10) -> pd.Series:
    alpha = 1.0 - lam
    return (lr.pow(2).ewm(alpha=alpha, adjust=False, min_periods=min_periods).mean()).pow(0.5)


def rolling_skew(lr: pd.Series, w: int) -> pd.Series:
    return lr.rolling(w, min_periods=w).skew()


def rolling_kurt(lr: pd.Series, w: int) -> pd.Series:
    return lr.rolling(w, min_periods=w).kurt()


def drawdown_rolling(close: pd.Series, w: int = 60) -> pd.Series:
    roll_max = close.rolling(w, min_periods=1).max()
    return (close / roll_max) - 1.0


def log_price(close: pd.Series) -> pd.Series:
    return np.log(close)


def slope_linreg(y: pd.Series, w: int) -> pd.Series:
    """Slope of log price vs time index over rolling window w (per‑day slope).
    Closed form using centered x: slope = cov(x,y)/var(x) with x = 0..w-1.
    """
    x = np.arange(w, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()
    # Rolling means
    y_rolling = y.rolling(w, min_periods=w)
    y_mean = y_rolling.mean()
    # Rolling cov: E[xy] - E[x]E[y] where E[x] is constant over window
    xy = y.rolling(w, min_periods=w).apply(lambda arr: np.dot(x, arr), raw=True) / w
    cov = xy - (x_mean * y_mean)
    slope = cov * (w / x_var)  # scale to per‑step slope
    return slope


# ── Imputation (time‑safe) ────────────────────────────────────────────────────
def impute_rolling_median_past_only(s: pd.Series, window: int = 20, min_valid: int = 3, min_expand: int = 5) -> pd.Series:
    """Impute using past‑only information.
    1) rolling median of s.shift(1) with given window if it has ≥ min_valid non‑NaNs
    2) else expanding median of s.shift(1) if it has ≥ min_expand
    3) else leave NaN
    """
    past = s.shift(1)
    roll_med = past.rolling(window=window, min_periods=min_valid).median()
    # expanding with min periods
    exp_med = past.expanding(min_periods=min_expand).median()
    filled = s.copy()
    mask = s.isna()
    filled.loc[mask] = roll_med.loc[mask]
    # still missing? fallback to expanding
    mask2 = filled.isna()
    filled.loc[mask2] = exp_med.loc[mask2]
    return filled


# ── Feature builder per ticker ────────────────────────────────────────────────
@dataclass
class FeatureSpec:
    k_label: int = 10
    # windows
    w_rv5: int = 5
    w_rv10: int = 10
    w_rv20: int = 20
    w_ema9: int = 9
    w_slope: int = 20
    w_skew: int = 20
    w_kurt: int = 20
    w_dd: int = 60
    lam_ewma: float = 0.94
    # impute
    imp_window: int = 20
    imp_min_valid: int = 3
    imp_min_expand: int = 5


def build_features_for_ticker(df_long: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """df_long: one ticker, columns include: open, high, low, close/adj_close, volume (optional)."""
    df = df_long.copy()
    df = _ensure_dt_index(df, col="date")

    o, h, l, c = _choose_close_cols(df)
    if c is None:
        raise ValueError("No close/adj_close column present for ticker.")

    # Base series
    lp = log_price(c)
    lr = lp.diff()

    # Range‑based
    park_d = parkinson_daily_sigma(h, l)

    # Trend
    ema9 = ema(c, spec.w_ema9)
    macd21 = macd_21_lens(c)
    slope20 = slope_linreg(lp, spec.w_slope)

    # Vol levels/dynamics
    rv5 = realized_vol(lr, spec.w_rv5)
    rv10 = realized_vol(lr, spec.w_rv10)
    rv20 = realized_vol(lr, spec.w_rv20)
    ewma94 = ewma_vol(lr, lam=spec.lam_ewma)
    d_rv10 = rv10.diff()
    vov = rv10.rolling(10, min_periods=10).std()  # vol of vol

    # Shape & risk
    skew20 = rolling_skew(lr, spec.w_skew)
    kurt20 = rolling_kurt(lr, spec.w_kurt)
    dd60 = drawdown_rolling(c, spec.w_dd)

    # Calendar (one‑hots)
    cal = df.index.to_series()
    dow = pd.get_dummies(cal.dt.dayofweek.rename("dow"), prefix="dow", dtype=float)
    mon = pd.get_dummies(cal.dt.month.rename("mon"), prefix="mon", dtype=float)

    feat = pd.concat({
        "park_d": park_d,
        "ema9": ema9,
        "macd21": macd21,
        "slope20": slope20,
        "rv5": rv5,
        "rv10": rv10,
        "rv20": rv20,
        "ewma94": ewma94,
        "d_rv10": d_rv10,
        "vov": vov,
        "skew20": skew20,
        "kurt20": kurt20,
        "dd60": dd60,
    }, axis=1)
    feat = pd.concat([feat, dow, mon], axis=1)
    # Ensure all feature columns are float to allow NaN operations without dtype warnings
    feat = feat.apply(pd.to_numeric, errors="coerce").astype("float32")

    # Enforce feature readiness (min history)
    min_hist = {
        "park_d": 1,
        "ema9": spec.w_ema9,
        "macd21": max(26, 21),
        "slope20": spec.w_slope,
        "rv5": spec.w_rv5,
        "rv10": spec.w_rv10,
        "rv20": spec.w_rv20,
        "ewma94": 10,
        "d_rv10": spec.w_rv10 + 1,
        "vov": 10,
        "skew20": spec.w_skew,
        "kurt20": spec.w_kurt,
        "dd60": spec.w_dd,
    }
    for col, need in min_hist.items():
        if col in feat.columns:
            valid = feat[col].notna() & (feat.index.to_series().rank(method="first").astype(int) >= need)
            # set NaN via mask (keeps dtype float32 and avoids future warnings)
            feat[col] = feat[col].mask(~valid, other=np.nan)

    return feat


# ── Dataset assembly ─────────────────────────────────────────────────────────
def load_ohlc_long(base_dir: Path) -> pd.DataFrame:
    path = Path(base_dir) / "data_raw" / "ohlc_long.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_csv(path)
    # normalize columns
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    required = {"date", "ticker", "high", "low"}
    if not required.issubset(df.columns):
        raise ValueError(f"ohlc_long.csv missing required columns: {required - set(df.columns)}")
    return df


def load_labels(base_dir: Path, k: int = 10) -> pd.DataFrame:
    path = Path(base_dir) / "data_int" / "ml" / f"labels_k{k}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing labels file: {path}")
    lab = pd.read_csv(path, index_col=0)
    lab.index = pd.to_datetime(lab.index)
    return lab


def assemble_features_long(base_dir: Path, k_label: int = 10, spec: Optional[FeatureSpec] = None,
                           impute: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (features_long, labels_long) with columns:
    - features_long: date,ticker,<features>
    - labels_long:  date,ticker,label_k
    """
    spec = spec or FeatureSpec(k_label=k_label)
    ohlc_long = load_ohlc_long(base_dir)

    feats_list = []
    labels = load_labels(base_dir, k=k_label)

    for tkr, g in ohlc_long.groupby("ticker"):
        g1 = g[[c for c in g.columns if c in {"date","ticker","open","high","low","close","adj_close","volume"}]].copy()
        g1 = g1.drop_duplicates(subset=["date"]).sort_values("date")
        feat = build_features_for_ticker(g1, spec)
        feat["ticker"] = tkr
        feats_list.append(feat.reset_index().rename(columns={"index":"date"}))

    feats_long = pd.concat(feats_list, axis=0, ignore_index=True)

    # Join labels: labels is wide (index=date, columns=tickers)
    lab = labels.rename_axis("date").stack().reset_index()
    lab.columns = ["date","ticker",f"label_k{k_label}"]

    # Merge
    Xy = pd.merge(feats_long, lab, on=["date","ticker"], how="inner")

    # Time‑safe imputation on features only, per ticker
    if impute:
        feature_cols = [c for c in Xy.columns if c not in {"date","ticker",f"label_k{k_label}"}]
        Xy = Xy.sort_values(["ticker","date"]).reset_index(drop=True)
        for tkr, idx in Xy.groupby("ticker").groups.items():
            sl = Xy.loc[sorted(idx), feature_cols]
            for col in feature_cols:
                Xy.loc[sl.index, col] = impute_rolling_median_past_only(sl[col])

    # Drop rows with any remaining NA in features or label
    Xy = Xy.dropna(subset=[f"label_k{k_label}"] + [c for c in Xy.columns if c not in {"date","ticker",f"label_k{k_label}"}])

    # Save raw features
    out_dir = Path(base_dir) / "data_int" / "ml"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_pq = out_dir / "features_raw_long.parquet"
    raw_csv = out_dir / "features_raw_long.csv"
    Xy.to_parquet(raw_pq)
    Xy.to_csv(raw_csv, index=False)

    # Split out labels_long for convenience
    labels_long = Xy[["date","ticker",f"label_k{k_label}"]].copy()
    return Xy, labels_long


# ── Splits & scaling ─────────────────────────────────────────────────────────
@dataclass
class SplitResult:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    meta: Dict


def split_per_ticker_time(Xy: pd.DataFrame, label_col: str, frac: Tuple[float,float,float] = (0.4,0.35,0.25)) -> SplitResult:
    """Strict time split per ticker, then concatenate.
    Returns SplitResult and metadata with per‑ticker cutoffs.
    """
    a,b,c = frac
    if abs(a+b+c - 1.0) > 1e-9:
        raise ValueError("Split fractions must sum to 1.0")

    feats = [c for c in Xy.columns if c not in {"date","ticker",label_col}]
    recs = {"per_ticker":{}}

    parts = {"train":[], "val":[], "test":[]}

    for tkr, g in Xy.sort_values(["ticker","date"]).groupby("ticker"):
        n = len(g)
        i1 = int(np.floor(a*n))
        i2 = i1 + int(np.floor(b*n))
        g_tr = g.iloc[:i1]
        g_va = g.iloc[i1:i2]
        g_te = g.iloc[i2:]
        recs["per_ticker"][tkr] = {
            "n": n,
            "train_end": g_tr["date"].max().strftime("%Y-%m-%d") if not g_tr.empty else None,
            "val_end": g_va["date"].max().strftime("%Y-%m-%d") if not g_va.empty else None,
            "test_end": g_te["date"].max().strftime("%Y-%m-%d") if not g_te.empty else None,
        }
        parts["train"].append(g_tr)
        parts["val"].append(g_va)
        parts["test"].append(g_te)

    df_tr = pd.concat(parts["train"], ignore_index=True)
    df_va = pd.concat(parts["val"], ignore_index=True)
    df_te = pd.concat(parts["test"], ignore_index=True)

    X_train = df_tr[feats].copy(); y_train = df_tr[label_col].copy()
    X_val   = df_va[feats].copy(); y_val   = df_va[label_col].copy()
    X_test  = df_te[feats].copy(); y_test  = df_te[label_col].copy()

    return SplitResult(X_train, y_train, X_val, y_val, X_test, y_test, recs)


def scale_with_train(split: SplitResult, base_dir: Path, prefix: str = "k10") -> SplitResult:
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(split.X_train.values)
    Xva = scaler.transform(split.X_val.values)
    Xte = scaler.transform(split.X_test.values)

    # Save scaler stats
    out_dir = Path(base_dir) / "data_int" / "ml"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "feature_order": split.X_train.columns.tolist()
    }
    (out_dir / f"scaler_stats_{prefix}.json").write_text(json.dumps(stats, indent=2))

    # Return as DataFrames preserving column names
    Xtr_df = pd.DataFrame(Xtr, columns=split.X_train.columns, index=split.X_train.index)
    Xva_df = pd.DataFrame(Xva, columns=split.X_train.columns, index=split.X_val.index)
    Xte_df = pd.DataFrame(Xte, columns=split.X_train.columns, index=split.X_test.index)

    return SplitResult(Xtr_df, split.y_train, Xva_df, split.y_val, Xte_df, split.y_test, split.meta)

def save_splits(split: SplitResult, base_dir: Path, prefix: str = "k10") -> Dict[str, Path]:
    out_dir = Path(base_dir) / "data_int" / "ml"
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for name, X, y in (
        ("train", split.X_train, split.y_train),
        ("val",   split.X_val,   split.y_val),
        ("test",  split.X_test,  split.y_test),
    ):
        Xp = out_dir / f"X_{name}_{prefix}.parquet"
        Yp = out_dir / f"y_{name}_{prefix}.parquet"
        X.to_parquet(Xp); y.to_frame("y").to_parquet(Yp)
        X.to_csv(out_dir / f"X_{name}_{prefix}.csv", index=False)
        y.to_frame("y").to_csv(out_dir / f"y_{name}_{prefix}.csv", index=False)
        paths[f"X_{name}"] = Xp
        paths[f"y_{name}"] = Yp

    # Save metadata
    (out_dir / f"split_meta_{prefix}.json").write_text(json.dumps(split.meta, indent=2))
    (out_dir / f"feature_spec_{prefix}.json").write_text(json.dumps(FeatureSpec().__dict__, indent=2))

    return paths


# --- Winsorization helpers (TRAIN-fit caps, applied to all splits) ---

from typing import Optional, List, Tuple, Dict  # already imported at top in your file

def winsorize_with_train(
    split: SplitResult,
    lower_q: float = 0.005,
    upper_q: float = 0.995,
    cols: Optional[List[str]] = None
) -> SplitResult:
    """Clip features to TRAIN quantile caps and apply to VAL/TEST (pre-scaling).
    If cols is None, apply to all numeric columns. Returns a new SplitResult.
    """
    Xtr, Xva, Xte = split.X_train.copy(), split.X_val.copy(), split.X_test.copy()
    feats = cols or Xtr.columns.tolist()
    q_low = Xtr[feats].quantile(lower_q)
    q_hi  = Xtr[feats].quantile(upper_q)
    Xtr[feats] = Xtr[feats].clip(lower=q_low, upper=q_hi, axis=1)
    Xva[feats] = Xva[feats].clip(lower=q_low, upper=q_hi, axis=1)
    Xte[feats] = Xte[feats].clip(lower=q_low, upper=q_hi, axis=1)
    return SplitResult(Xtr, split.y_train, Xva, split.y_val, Xte, split.y_test, split.meta)

def pick_auto_winsor_cols(columns: List[str]) -> List[str]:
    """Heuristic: heavy-tailed vol family only.
    Includes: rv*, ewma*, d_rv*, vov, park_d, skew*, kurt*.
    Leaves price-trend (ema9, macd21, slope20) and calendar dummies alone.
    """
    keep = []
    for c in columns:
        name = c.lower()
        if (
            name.startswith("rv") or name.startswith("ewma") or name.startswith("d_rv")
            or name in {"vov", "park_d"} or name.startswith("skew") or name.startswith("kurt")
        ):
            keep.append(c)
    return keep


# ── One‑shot runner ──────────────────────────────────────────────────────────
def run_step_3_2(
    base_dir: Path,
    k_label: int = 10,
    do_scale: bool = True,
    split_frac: Tuple[float,float,float] = (0.4,0.35,0.25),
) -> Dict[str, Path]:
    """Build features+label, impute, split per ticker, (optionally) scale with TRAIN stats, save artifacts."""
    Xy, labels_long = assemble_features_long(base_dir, k_label=k_label, spec=FeatureSpec(k_label=k_label), impute=True)
    label_col = f"label_k{k_label}"
    split = split_per_ticker_time(Xy, label_col=label_col, frac=split_frac)
    if do_scale:
        split = scale_with_train(split, base_dir, prefix=f"k{k_label}")
    paths = save_splits(split, base_dir, prefix=f"k{k_label}")

    # High‑level build report
    rep = {
        "n_rows_total": int(Xy.shape[0]),
        "n_feats": int(len([c for c in Xy.columns if c not in {"date","ticker",label_col}])),
        "k_label": k_label,
        "split_sizes": {
            "train": int(split.X_train.shape[0]),
            "val":   int(split.X_val.shape[0]),
            "test":  int(split.X_test.shape[0]),
        },
    }
    out_dir = Path(base_dir) / "data_int" / "ml"
    (out_dir / f"build_report_k{k_label}.json").write_text(json.dumps(rep, indent=2))
    return paths


def run_step_3_2_with_options(
    base_dir: Path,
    k_label: int = 10,
    do_scale: bool = True,
    split_frac: Tuple[float,float,float] = (0.4,0.35,0.25),
    winsorize: bool = False,
    winsor_q: Tuple[float, float] = (0.005, 0.995),
    winsor_cols: Optional[List[str]] = None,   # None=auto vol-family; []=no winsor
    drop_cols: Optional[List[str]] = None,     # e.g. ["rv20"] for EN lean spec
) -> Dict[str, Path]:
    """Step 3.2 with optional duplicate trimming and winsorization."""
    Xy, _ = assemble_features_long(
        base_dir, k_label=k_label, spec=FeatureSpec(k_label=k_label), impute=True
    )
    label_col = f"label_k{k_label}"
    split = split_per_ticker_time(Xy, label_col=label_col, frac=split_frac)

    # Trim duplicates (e.g., drop 'rv20' for EN)
    if drop_cols:
        cols_keep = [c for c in split.X_train.columns if c not in set(drop_cols)]
        split = SplitResult(
            split.X_train[cols_keep], split.y_train,
            split.X_val[cols_keep],   split.y_val,
            split.X_test[cols_keep],  split.y_test,
            split.meta
        )

    # Optional winsorization (pre-scaling), TRAIN caps applied to all splits
    if winsorize:
        cols = winsor_cols
        if winsor_cols is None:
            cols = pick_auto_winsor_cols(split.X_train.columns.tolist())
        split = winsorize_with_train(split, lower_q=winsor_q[0], upper_q=winsor_q[1], cols=cols)

    if do_scale:
        split = scale_with_train(split, base_dir, prefix=f"k{k_label}")

    paths = save_splits(split, base_dir, prefix=f"k{k_label}")
    return paths




if __name__ == "__main__":
    from datetime import date
    BASE = Path.cwd() / date.today().strftime("%d-%m-%Y")
    print(run_step_3_2(BASE, k_label=10))
