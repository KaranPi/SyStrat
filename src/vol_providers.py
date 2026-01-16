# src/syslib/vol_providers.py
from __future__ import annotations

from pathlib import Path            # no installation needed
import json                         # no installation needed
from typing import Dict, Iterable   # no installation needed

import pandas as pd                 # already in env — no new install
import numpy as np                  # already in env — no new install

# We import *inside* functions to avoid circular imports with wp_bt_legacy.


def ewma_legacy(prices: pd.DataFrame, *, stdev_window: int, annualize_factor: float,
                blend_weight: float, lookahead_mean_horizon: int,
                tail_policy: str = "truncate") -> Dict[str, pd.Series]:
    from syslib.wp_bt_legacy import _daily_stdev, legacy_pred_vol_from_stdev

    out = {}
    for sym in prices.columns:
        px = prices[sym].dropna()
        st = _daily_stdev(px, stdev_window)
        pred = legacy_pred_vol_from_stdev(
            st,
            annualize_factor=annualize_factor,
            w=blend_weight,
            mean_h=lookahead_mean_horizon,
            tail_policy=tail_policy,
        )
        out[sym] = pred
    return out



def _read_manifest_alpha(manifest_path: Path) -> float:
    """
    Read alpha from model_manifest_k{K}.json if present; default to 1.0.
    Supports a few common key shapes.
    """
    try:
        obj = json.loads(Path(manifest_path).read_text())
    except Exception:
        return 1.0

    # try a few key patterns
    for key in ("alpha", "alpha_calibrator", "calibrator_alpha"):
        if key in obj and isinstance(obj[key], (float, int)):
            return float(obj[key])

    # nested possibilities
    for key in ("calibration", "report", "meta"):
        if key in obj and isinstance(obj[key], dict):
            for k2 in ("alpha", "alpha_calibrator", "calibrator_alpha"):
                if k2 in obj[key] and isinstance(obj[key][k2], (float, int)):
                    return float(obj[key][k2])

    return 1.0


def _find_preds_path(base_dir: Path, k_label: int) -> Path:
    """
    Try a few file names for robustness; prefer parquet.
    """
    candidates = [
        base_dir / f"data_int/ml/preds_test_k{k_label}.parquet",
        base_dir / f"data_int/ml/preds_test_k{k_label}.pq",
        base_dir / f"data_int/ml/preds_TEST_k{k_label}.parquet",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find preds_test_k{k_label} parquet under {base_dir}/data_int/ml/")


# --- replace the existing _guess_cols with this ---
def _guess_cols(df: pd.DataFrame, overrides: dict | None = None) -> tuple[str, str, str] | tuple[str, None, None]:
    """
    Return (date_col, ticker_col, pred_col), or ('__WIDE__', None, None) if df is already wide
    (DatetimeIndex + columns are tickers). 'overrides' may include explicit names.
    """
    # 1) explicit overrides
    if overrides:
        d = overrides.get("date_col")
        t = overrides.get("ticker_col")
        p = overrides.get("pred_col")
        if d and t and p:
            return d, t, p

    # 2) detect already-wide: DatetimeIndex and > 2 columns, no obvious long-form cols
    if isinstance(df.index, pd.DatetimeIndex) and df.shape[1] >= 2:
        likely_long_cols = {"ticker", "symbol", "asset", "name", "secid", "security", "code", "sid"}
        if df.columns.isin(likely_long_cols).sum() == 0:
            return "__WIDE__", None, None

    # 3) try common date names (case-sensitive list)
    for d in ("date","Date","dt","timestamp","ts","time","datetime","asof","asof_date","trade_date","ds","index"):
        if d in df.columns:
            date_col = d
            break
    else:
        # if index looks datetime, use it
        if isinstance(df.index, pd.DatetimeIndex):
            date_col = None  # signal to use index
        else:
            raise ValueError("Could not infer date column in predictions file.")

    # 4) ticker name
    for t in ("ticker","symbol","asset","name","secid","security","code","sid"):
        if t in df.columns:
            ticker_col = t
            break
    else:
        raise ValueError("Could not infer ticker column in predictions file.")

    # 5) prediction column
    for p in ("sigma_hat","pred_vol","y_pred","vol_pred","sigma","pred","yhat","target","y"):
        if p in df.columns:
            pred_col = p
            break
    else:
        # fallback: last numeric column
        numcols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numcols:
            raise ValueError("Could not infer predicted-vol column in predictions file.")
        pred_col = numcols[-1]

    # normalize 'index' case: we will reset_index upstream
    return date_col, ticker_col, pred_col

def ml_manifest(base_dir: Path, symbols: Iterable[str], date_index: pd.Index,
                *, k_label: int = 10, model_tag: str | None = None,
                date_col: str | None = None, ticker_col: str | None = None, pred_col: str | None = None,
                single_symbol: str | None = None, pred_model_col: str = "pred_gbm", per_ticker_dir: Path | None = None,
                x_test_path: Path | None = None, x_date_col: str | None = None, x_ticker_col: str | None = None,
                align: str = "tail") -> Dict[str, pd.Series]:
    base_dir = Path(base_dir)
    p_preds = _find_preds_path(base_dir, k_label)
    p_manifest = base_dir / f"data_int/ml/model_manifest_k{k_label}.json"

    df = pd.read_parquet(p_preds)

    # allow MultiIndex parquet; collapse cleanly
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    
    # alpha
    alpha = _read_manifest_alpha(p_manifest)

    # --- per-ticker directory of simple files (one file per ticker) ---
    date_candidates = ("date","Date","dt","timestamp","ts","datetime")
    if per_ticker_dir is not None:
        per_ticker_dir = Path(per_ticker_dir)
        out: Dict[str, pd.Series] = {}
        for sym in symbols:
            p = per_ticker_dir / f"{sym}.parquet"
            if not p.exists():
                p = per_ticker_dir / f"{sym}.csv"
            if not p.exists():
                out[sym] = pd.Series(index=date_index, dtype="float64")
                continue

            dfi = pd.read_parquet(p) if p.suffix.lower()==".parquet" else pd.read_csv(p)

            # choose prediction column: explicit -> common names -> last numeric
            use_col = pred_model_col if pred_model_col in dfi.columns else None
            if use_col is None:
                for c in ("pred_gbm","pred_en","sigma_hat","pred","y_pred","vol_pred","sigma"):
                    if c in dfi.columns:
                        use_col = c; break
            if use_col is None:
                numcols = [c for c in dfi.columns if pd.api.types.is_numeric_dtype(dfi[c])]
                use_col = numcols[-1] if numcols else None
            if use_col is None:
                out[sym] = pd.Series(index=date_index, dtype="float64")
                continue

            # date-aware alignment (preferred)
            date_col_local = next((c for c in ("date","Date","dt","timestamp","ts","datetime") if c in dfi.columns), None)
            a = _read_manifest_alpha(p_manifest)
            if date_col_local is not None:
                dfi[date_col_local] = pd.to_datetime(dfi[date_col_local], utc=False).dt.tz_localize(None)
                s = dfi.set_index(date_col_local)[use_col].astype(float).sort_index()
                out[sym] = (a * s).reindex(date_index)
            else:
                # vector fallback: map to head/tail of calendar
                n = min(len(dfi), len(date_index))
                vals = dfi[use_col].astype(float).values
                if align == "tail":
                    idx = date_index[-n:]; vals = vals[-n:]
                else:
                    idx = date_index[:n]; vals = vals[:n]
                out[sym] = (a * pd.Series(vals, index=idx)).reindex(date_index)
        return out
    
    # --- (preds + X_test) join path: preds have only numeric cols; X_test carries date/ticker ---
    if set(df.columns) >= {"y_true", "pred_en", "pred_gbm"} and \
       not any(c in df.columns for c in ("date","Date","dt","timestamp","ticker","symbol","asset","name")) and \
       x_test_path is not None:
        xt = pd.read_parquet(x_test_path)

        # infer X_test columns if not provided
        if x_date_col is None:
            for d in ("date","Date","dt","timestamp","ts","time","datetime","asof"):
                if d in xt.columns:
                    x_date_col = d; break
            else:
                raise ValueError("X_test file missing a recognizable date column.")
        if x_ticker_col is None:
            for t in ("ticker","Ticker","symbol","asset","name","code"):
                if t in xt.columns:
                    x_ticker_col = t; break
            else:
                raise ValueError("X_test file missing a recognizable ticker column.")

        if len(xt) != len(df):
            raise ValueError(f"Row count mismatch: X_test({len(xt)}) vs preds({len(df)}).")

        # choose which preds column to use
        use_col = pred_model_col or ("pred_gbm" if (model_tag or "").lower() == "gbm" else "pred_en")
        if use_col not in df.columns:
            raise ValueError(f"Prediction column '{use_col}' not in preds file. Available: {list(df.columns)}")

        merged = pd.DataFrame({
            "date": pd.to_datetime(xt[x_date_col], utc=False).dt.tz_localize(None),
            "ticker": xt[x_ticker_col].astype(str),
            "sigma_hat": df[use_col].astype(float),
        })

        alpha = _read_manifest_alpha(p_manifest)
        wide = merged.pivot(index="date", columns="ticker", values="sigma_hat").sort_index()
        wide = alpha * wide

        out = {}
        for sym in symbols:
            s = wide.get(sym)
            out[sym] = (s.reindex(date_index) if s is not None else pd.Series(index=date_index, dtype="float64"))
        return out
    

    # --- single-file fallback with NO date/ticker cols (your current case) ---
    if set(df.columns).issuperset({"y_true"}) and \
       any(c in df.columns for c in ("pred_gbm","pred_en","pred")) and \
       not any(c in df.columns for c in ("date","Date","dt","timestamp","ticker","symbol")):
        if single_symbol is None or single_symbol not in symbols or pred_model_col not in df.columns:
            raise ValueError("For files without date/ticker, provide single_symbol and a valid pred_model_col.")
        alpha = _read_manifest_alpha(base_dir / f"data_int/ml/model_manifest_k{k_label}.json")
        n = min(len(df), len(date_index))
        s = pd.Series(df[pred_model_col].values[:n], index=date_index[:n])
        return {single_symbol: (alpha * s).reindex(date_index)}

    
    # detect layout (use overrides if provided)
    layout = _guess_cols(df, overrides={"date_col": date_col, "ticker_col": ticker_col, "pred_col": pred_col})



    if layout[0] == "__WIDE__":
        # already wide: rows=dates, cols=tickers
        wide = df.copy()
        if not isinstance(wide.index, pd.DatetimeIndex):
            # try to coerce an 'index' column to datetime
            if "index" in wide.columns:
                wide["index"] = pd.to_datetime(wide["index"], utc=False).dt.tz_localize(None)
                wide = wide.set_index("index")
            else:
                raise ValueError("Wide predictions must have a DatetimeIndex or an 'index' datetime column.")
        wide.index = pd.to_datetime(wide.index, utc=False).tz_localize(None)
        wide = wide.sort_index()
        wide = alpha * wide
    else:
        date_col, ticker_col, pred_col = layout
        if date_col is None:
            # the index is the date
            df = df.reset_index().rename(columns={"index": "index"})
            date_col = "index"
        # basic cleanup
        df = df[[date_col, ticker_col, pred_col]].copy()
        df[date_col] = pd.to_datetime(df[date_col], utc=False).dt.tz_localize(None)
        if model_tag and "model" in df.columns:
            df = df[df["model"].astype(str).str.lower() == str(model_tag).lower()]
        # rows=dates, cols=tickers
        wide = df.pivot(index=date_col, columns=ticker_col, values=pred_col).sort_index()
        wide = alpha * wide

    # align to requested index and return per symbol
    out: Dict[str, pd.Series] = {}
    for sym in symbols:
        s = wide.get(sym)
        if s is None:
            out[sym] = pd.Series(index=date_index, dtype="float64")
            continue
        s = s.reindex(date_index)
        out[sym] = s
    return out

