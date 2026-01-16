# C:\Users\quantbase\Desktop\SyStrat\src\syslib\wp_bt.py
from __future__ import annotations  # no installation needed

from pathlib import Path           # no installation needed
import json                        # no installation needed
from typing import Dict, List      # no installation needed

import numpy as np                 # already in env — no new install
import pandas as pd               # already in env — no new install
import matplotlib.pyplot as plt    # already in env — no new install
import yfinance as yf              # already in env — no new install


# =========================
# Strategy-wide defaults
# =========================
# Legacy vol smoother (kept to match your prior logic)
WINDOW_LEN   = 32
W0           = 0.6
DECAY        = 0.4
SCALE        = 19.0
BLEND_RECENT = 0.7
HIST_SPAN    = 54

# Sizer / evaluation defaults
K_LABEL        = 10
NOTIONAL       = 100_000.0
LEVERAGE       = 2.5
RISK_FRACTION  = 0.2
PRICE_MODE     = "last_close"   # weekly close
WEEKLY_RULE    = "W-FRI"        # weekly bar alignment
MIN_VOL        = 1e-8           # guard against division by ~0


# =========================
# Data loading helpers
# =========================

def _uniq_sorted(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure tz-naive, sorted, unique DatetimeIndex (keep last on duplicates)."""
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    if not out.index.is_unique:
        out = out[~out.index.duplicated(keep="last")]
    return out

def _norm_sym_key(s: str) -> str:
    """Uppercase and normalize separators; strip a trailing -USD if present for matching."""
    u = str(s).upper().replace("_", "-")
    return u[:-4] if u.endswith("-USD") else u

def _map_ml_cols_to_symbols(ml_cols, symbols):
    """
    Build a mapping from whatever ML column labels to our target symbols.
    Accepts BTC ↔ BTC-USD; ETH_USD ↔ ETH-USD, etc. Returns {ml_col: target_symbol}.
    """
    # make a lookup of target variants
    canonical = {}
    for t in symbols:
        tU  = t.upper().replace("_", "-")
        tB  = _norm_sym_key(t)           # strip -USD
        canonical[tU] = t
        canonical[tB] = t
    mapping = {}
    for c in ml_cols:
        cU = str(c).upper().replace("_", "-")
        cB = _norm_sym_key(c)
        target = canonical.get(cU) or canonical.get(cB)
        if target:
            mapping[c] = target
    return mapping


def load_close_df(base_dir: Path | str, symbols: List[str]) -> pd.DataFrame:
    """
    Load ASCENDING daily Close for the requested symbols.
    Prefers saved artifacts at <base_dir>/data_int/close.csv; falls back to yfinance.
    """
    base_dir = Path(base_dir)
    close_path = base_dir / "data_int" / "close.csv"
    if close_path.exists():
        df = pd.read_csv(close_path, parse_dates=True, index_col=0)
        df.index = pd.to_datetime(df.index)
        # keep + order requested columns if present
        got = [s for s in symbols if s in df.columns]
        out = df[got].copy()
        missing = [s for s in symbols if s not in df.columns]
        if missing:
            extra = yf.download(
                missing, period="5y", interval="1d", auto_adjust=True, progress=False
            )["Close"]
            if isinstance(extra, pd.Series):
                extra = extra.to_frame(missing[0])
            out = out.join(extra, how="outer")
        return _uniq_sorted(out.reindex(columns=symbols))
    else:
        data = yf.download(
            symbols, period="5y", interval="1d", auto_adjust=True, progress=False
        )["Close"]
        if isinstance(data, pd.Series):
            data = data.to_frame(symbols[0])
        data.index = pd.to_datetime(data.index)
        return _uniq_sorted(data.reindex(columns=symbols))


def _weighted_std_rolling(x: np.ndarray) -> float:
    w = W0 * (DECAY ** np.arange(WINDOW_LEN)[::-1])
    w = w / w.sum()
    mu = float(np.dot(w, x))
    return float(np.sqrt(np.dot(w, (x - mu) ** 2)))


def legacy_pred_vol_daily(close: pd.Series) -> pd.Series:
    """
    Time-safe daily predicted vol from legacy smoother.
    Input: ASC daily close; Output: ASC daily vol aligned at t (shifted 1).
    """
    r = close.pct_change(fill_method=None)
    stdev = r.rolling(WINDOW_LEN, min_periods=WINDOW_LEN).apply(_weighted_std_rolling, raw=True)
    hist_mean = stdev.rolling(HIST_SPAN, min_periods=HIST_SPAN).mean()
    pred = SCALE * (BLEND_RECENT * stdev + (1.0 - BLEND_RECENT) * hist_mean)
    return pred.shift(1)


def load_ml_vol_daily(
    base_dir: Path | str,
    symbols: List[str],
    k_label: int = K_LABEL,
    model: str = "gbm",            # 'gbm' or 'en'
) -> pd.DataFrame:
    """
    Load ASC daily ML predicted vol for TEST, multiply by alpha from manifest if present.
    Accepts:
      • long/tidy: date, ticker, {pred_gbm|pred_en|pred|y_hat|yhat|prediction|vol_pred|sigma_pred}[, model]
      • wide: date as first col, the rest are tickers (BTC, BTC-USD, ETH_USD, etc.)
    """
    ml_dir = Path(base_dir) / "data_int" / "ml"
    preds_parq = ml_dir / f"preds_test_k{k_label}.parquet"
    preds_csv  = ml_dir / f"preds_test_k{k_label}.csv"
    manifest   = ml_dir / f"model_manifest_k{k_label}.json"

    alpha = 1.0
    if manifest.exists():
        m = json.loads(manifest.read_text())
        alpha = m.get(f"alpha_{model}", 1.0)

    # Read preds
    if preds_parq.exists():
        df = pd.read_parquet(preds_parq)
    else:
        df = pd.read_csv(preds_csv)

    # Flexible schema detection
    cols_lc = {c.lower(): c for c in df.columns}
    # Candidate value columns (first match wins)
    value_candidates = ["pred_gbm", "pred_en", "pred", "y_hat", "yhat", "prediction", "vol_pred", "sigma_pred", "vol_hat"]

    # If there is an explicit 'model' column, select the requested model first
    col_model = cols_lc.get("model")
    if col_model:
        # normalize model labels (gbm/gradientboosting → gbm; en/elasticnet → en)
        model_map = {"gbm": "gbm", "gradientboosting": "gbm", "gbr": "gbm",
                     "en": "en", "elasticnet": "en"}
        df[col_model] = df[col_model].astype(str).str.lower().map(lambda x: model_map.get(x, x))
        df = df[df[col_model] == model]

    col_date   = cols_lc.get("date")
    col_ticker = cols_lc.get("ticker")

    if col_date and col_ticker:
        # long/tidy path
        # choose a value column
        value_col = None
        for cand in value_candidates:
            if cand in cols_lc:
                value_col = cols_lc[cand]
                break
        # fallback to model-specific default if present
        if not value_col:
            want_col = {"gbm": "pred_gbm", "en": "pred_en"}[model]
            value_col = cols_lc.get(want_col)

        if value_col is None:
            # Nothing suitable; return empty frame to trigger legacy fallback
            return pd.DataFrame(index=pd.to_datetime(df[col_date]).sort_values())

        # pivot to wide
        tidy = df[[col_date, col_ticker, value_col]].copy()
        tidy[col_date] = pd.to_datetime(tidy[col_date])
        pivot = tidy.pivot(index=col_date, columns=col_ticker, values=value_col)

    else:
        # wide path: assume first column is date, rest are tickers
        pivot = df.set_index(df.columns[0]).copy()

    # Normalize index and columns, apply alpha
    pivot.index = pd.to_datetime(pivot.index)
    pivot = (pivot.sort_index() * alpha)
    pivot = _uniq_sorted(pivot)

    # Map ML column labels to requested symbols (handles BTC ↔ BTC-USD etc.)
    col_map = _map_ml_cols_to_symbols(pivot.columns, symbols)
    if not col_map:
        # no overlap → return empty to trigger per-ticker legacy fallback
        return pd.DataFrame(index=pivot.index)

    pivot = pivot.rename(columns=col_map)
    # Collapse duplicates if multiple ML columns map to the same target symbol
    pivot = pivot.T.groupby(level=0).mean().T

    # Keep only requested symbols that are present
    keep = [s for s in symbols if s in pivot.columns]
    return pivot[keep]


def auto_pick_ml_model(base_dir: Path | str, k_label: int = K_LABEL) -> str:
    """
    Choose 'gbm' vs 'en' using metrics_k{K}.json if available; fallback 'gbm'.
    Tries 'rmse_test' then 'mae_test'; lower is better.
    """
    metrics_path = Path(base_dir) / "data_int" / "ml" / f"metrics_k{k_label}.json"
    if not metrics_path.exists():
        return "gbm"
    try:
        m = json.loads(metrics_path.read_text())
        # tolerant keys
        gbm_rmse = m.get("gbm", {}).get("rmse_test", m.get("gbm", {}).get("RMSE_TEST"))
        en_rmse  = m.get("en",  {}).get("rmse_test",  m.get("en",  {}).get("RMSE_TEST"))
        if gbm_rmse is not None and en_rmse is not None:
            return "gbm" if gbm_rmse <= en_rmse else "en"
        gbm_mae = m.get("gbm", {}).get("mae_test", m.get("gbm", {}).get("MAE_TEST"))
        en_mae  = m.get("en",  {}).get("mae_test",  m.get("en",  {}).get("MAE_TEST"))
        if gbm_mae is not None and en_mae is not None:
            return "gbm" if gbm_mae <= en_mae else "en"
    except Exception:
        pass
    return "gbm"


# =========================
# Weekly backtest core
# =========================
def weekly_backtest(
    base_dir: Path | str,
    symbols: List[str],
    portw: List[float] | Dict[str, float],
    vol_provider: str = "ewma_blend",   # 'ewma_blend' or 'sklearn_vol_v1'
    ml_model: str | None = None,        # 'gbm' or 'en' (None → auto)
    k_label: int = K_LABEL,
    notional: float = NOTIONAL,
    leverage: float = LEVERAGE,
    risk_fraction: float = RISK_FRACTION,
    week_rule: str = WEEKLY_RULE, start: str | None = None, end: str | None = None
) -> Dict[str, pd.Series]:
    """
    Weekly rebalance backtest, sizing at week t close, realizing PnL over [t, t+1].
    Returns dict with per-week pnl, cumulative pnl, cumulative return, and metadata.
    """
    # Normalize weights
    symbols = list(symbols)
    if isinstance(portw, dict):
        w = pd.Series({s: float(portw[s]) for s in symbols}, dtype=float)
    else:
        w = pd.Series(portw, index=symbols, dtype=float)
    w = w / w.sum()

    # Prices (daily → weekly)
    close_daily = load_close_df(base_dir, symbols)  # ASC daily
    price_w = close_daily.resample(week_rule).last().dropna(how="all")
    if start or end:
        price_w = price_w.loc[start:end]

    # Vols (daily → weekly)
    if vol_provider == "ewma_blend":
        vol_daily = pd.concat(
            {sym: legacy_pred_vol_daily(close_daily[sym]) for sym in symbols}, axis=1
        )
    elif vol_provider == "sklearn_vol_v1":
        if ml_model is None:
            ml_model = auto_pick_ml_model(base_dir, k_label=k_label)
        ml = load_ml_vol_daily(base_dir, symbols, k_label=k_label, model=ml_model)
        # backfill any missing symbols per-day with legacy vol
        missing = [s for s in symbols if s not in ml.columns]
        if missing:
            leg = pd.concat(
                {sym: legacy_pred_vol_daily(close_daily[sym]) for sym in missing}, axis=1
            )
            ml = ml.join(leg, how="outer")
            ml = _uniq_sorted(ml)
        vol_daily = ml.reindex(index=close_daily.index).ffill()
    else:
        raise ValueError("vol_provider must be 'ewma_blend' or 'sklearn_vol_v1'")

    vol_w   = vol_daily.resample(week_rule).last()
    vol_w   = vol_w.reindex(price_w.index).ffill()
    vol_w = vol_w.clip(lower=MIN_VOL)

    # Shares sized at week t close
    sizing_capital = (notional * leverage * risk_fraction)
    shares = sizing_capital * w / (price_w * vol_w)

    # PnL over next week (t→t+1)
    price_next = price_w.shift(-1)
    pnl_by_sym = shares * (price_next - price_w)
    pnl_by_sym = pnl_by_sym.iloc[:-1]  # drop last NaN week

    # Portfolio series
    weekly_pnl = pnl_by_sym.sum(axis=1)
    weekly_cum_pnl = weekly_pnl.cumsum()
    weekly_cum_return = weekly_cum_pnl / sizing_capital

    return {
        "weekly_pnl": weekly_pnl,
        "weekly_cum_pnl": weekly_cum_pnl,
        "weekly_cum_return": weekly_cum_return,
        "shares_weekly": shares.loc[weekly_pnl.index],
        "vol_weekly": vol_w.loc[weekly_pnl.index],
        "price_weekly": price_w.loc[weekly_pnl.index],
        "meta": {
            "vol_provider": vol_provider,
            "ml_model": ml_model,
            "k_label": k_label,
            "notional": notional,
            "leverage": leverage,
            "risk_fraction": risk_fraction,
            "week_rule": week_rule,
        },
    }


def ab_compare(
    base_dir: Path | str,
    symbols: List[str],
    portw: List[float] | Dict[str, float],
    k_label: int = K_LABEL,
    notional: float = NOTIONAL,
    leverage: float = LEVERAGE,
    risk_fraction: float = RISK_FRACTION,
    week_rule: str = WEEKLY_RULE, ml_model: str | None = None, start: str | None = None, end: str | None = None
) -> Dict[str, Dict]:
    """
    Run legacy vs ML backtests (identical sizer; only vol provider flips).
    Returns dict with both runs and a small summary.
    """
    res_L = weekly_backtest(
        base_dir, symbols, portw,
        vol_provider="ewma_blend",
        ml_model=None,
        k_label=k_label,
        notional=notional, leverage=leverage, risk_fraction=risk_fraction,
        week_rule=week_rule, 
        start=start, 
        end=end
    )
    res_M = weekly_backtest(
        base_dir, symbols, portw,
        vol_provider="sklearn_vol_v1",
        ml_model=ml_model,
        k_label=k_label,
        notional=notional, leverage=leverage, risk_fraction=risk_fraction,
        week_rule=week_rule, 
        start=start, 
        end=end
    )

    # Simple summary metrics
    def _summ(s: pd.Series) -> Dict[str, float]:
        ret = s / (notional * leverage * risk_fraction)
        return {
            "weeks": int(s.shape[0]),
            "cum_pnl": float(s.iloc[-1]),
            "cum_return": float(ret.iloc[-1]),
            "mean_weekly_pnl": float(s.mean()),
            "std_weekly_pnl": float(s.std(ddof=1)),
            "sharpe_weekly": float(np.nan if s.std(ddof=1) == 0 else s.mean() / s.std(ddof=1)),
            "max_drawdown": float((s.cummax() - s).max()),
        }

    summary = {
        "legacy": _summ(res_L["weekly_pnl"]),
        "ml":     _summ(res_M["weekly_pnl"]),
    }

    return {"legacy": res_L, "ml": res_M, "summary": summary}


def plot_ab_curves(
    out_dir: Path | str,
    res: Dict[str, Dict],
    title_suffix: str = "",
) -> Dict[str, str]:
    """
    Save weekly cumulative return & PnL overlays.
    Returns file paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p1 = out_dir / f"weekly_cum_return_AB{title_suffix}.pdf"
    plt.figure()
    res["legacy"]["weekly_cum_return"].plot(label="Legacy (ewma_blend)")
    res["ml"]["weekly_cum_return"].plot(label="ML (sklearn_vol_v1)")
    plt.title("Weekly Cumulative Return — A/B")
    plt.xlabel("Week")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(p1)
    plt.close()

    p2 = out_dir / f"weekly_cum_pnl_AB{title_suffix}.pdf"
    plt.figure()
    res["legacy"]["weekly_cum_pnl"].plot(label="Legacy PnL")
    res["ml"]["weekly_cum_pnl"].plot(label="ML PnL")
    plt.title("Weekly Cumulative PnL — A/B")
    plt.xlabel("Week")
    plt.ylabel("PnL (USD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(p2)
    plt.close()

    return {"cum_return_pdf": str(p1), "cum_pnl_pdf": str(p2)}


# =========================
# Example CLI runner
# =========================
if __name__ == "__main__":
    BASE = Path(r"C:\Users\quantbase\Desktop\SyStrat") / "<DD-MM-YYYY>"  # ← set to your run folder
    FIGS = BASE / "figures"
    symbols = ['BTC-USD','ETH-USD','BNB-USD','XRP-USD','ADA-USD','LINK-USD','LTC-USD']
    portw   = [0.23, 0.078, 0.078, 0.078, 0.078, 0.078, 0.078]  # example weights

    results = ab_compare(
        base_dir=BASE, symbols=symbols, portw=portw,
        k_label=10, notional=100_000.0, leverage=2.5, risk_fraction=0.2,
        week_rule="W-FRI", ml_model=None  # None → auto-pick via metrics_k10.json
    )
    files = plot_ab_curves(FIGS, results, title_suffix="")
    print("Saved:", files)
