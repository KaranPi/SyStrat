# bt_ewma_legacy.py
# Minimal helper to mimic the legacy EWMA backtest & cumulative-returns plot.
# Imports annotated as per project convention.

from __future__ import annotations

import math                 # no installation needed
import datetime as dt       # no installation needed
from dataclasses import dataclass  # no installation needed
from typing import Dict, Iterable, Optional  # no installation needed

import numpy as np          # already in env — no new install
import pandas as pd         # already in env — no new install

try:
    import yfinance as yf   # already in env — no new install
except Exception:           # yfinance optional; you can pass preloaded prices
    yf = None

# ---------------------------------------------------------------------
# Config container

@dataclass
class LegacyConfig:
    start: Optional[pd.Timestamp] = None
    end: Optional[pd.Timestamp] = None
    notional: float = 100_000.0
    leverage: float = 2.5
    risk_fraction: float = 0.2
    # EWMA / stdev settings
    stdev_window: int = 55
    annualize_factor: float = 19.0  # ~sqrt(365)
    blend_weight: float = 0.7       # 0.7 * stdev_t + 0.3 * mean(stdev_{t+1:t+54})
    lookahead_mean_horizon: int = 54
    # Mimic quirks from the legacy code
    strict_mimic: bool = True       # reverse series, pad zeros across tickers, daily scale
    legacy_sign: float = -1.0       # legacy used a negative sign on PnL
    normalize_to_zero: bool = True
    warmup_backfill_days: int = 60  # to cover stdev_window warmup

# ---------------------------------------------------------------------
# Data access

def fetch_prices_yf(symbols: Iterable[str], start: Optional[str|pd.Timestamp] = None,
                    end: Optional[str|pd.Timestamp] = None) -> pd.DataFrame:
    """
    Fetch adjusted close prices from yfinance (wide DataFrame with columns=tickers).
    Falls back to auto-adjusted close, no corporate action math here.
    """
    if yf is None:
        raise RuntimeError("yfinance not available. Pass a prepared prices DataFrame instead.")
    if end is None:
        end = pd.Timestamp.today().normalize()
    if start is None:
        start = end - pd.Timedelta(days=365*4)  # 4y default
    data = yf.download(list(symbols), start=start, end=end, progress=False, auto_adjust=True)
    close = data['Close'].copy()
    close = close.dropna(how='all').ffill().dropna(axis=1, how='all')
    return close


# ---------------------------------------------------------------------
# Core legacy math (mimic)

def _daily_stdev(prices: pd.Series, window: int) -> pd.Series:
    rets = prices.pct_change()
    return rets.rolling(window=window, min_periods=window).std()

def legacy_pred_vol_from_stdev(stdev: pd.Series, annualize_factor: float,
                               w: float = 0.7, mean_h: int = 54,
                               tail_policy: str = "truncate") -> pd.Series:
    """
    pred_vol[t] = ann * ( w*stdev[t] + (1-w)*mean(stdev[t+1:t+mean_h]) )
    If tail_policy='truncate' (default), we mimic legacy and drop the last mean_h+1 points.
    If tail_policy='ffill', we forward-fill the last available value to the end of stdev.index.
    """
    st = stdev.values
    n = len(st)
    out_len = max(0, n - (mean_h + 1))
    out = np.full(out_len, np.nan, dtype=float)
    for i in range(out_len):
        if np.isfinite(st[i]):
            tail = st[i+1:i+mean_h]
            m = np.nanmean(tail) if tail.size > 0 else np.nan
            out[i] = annualize_factor * (w * st[i] + (1.0 - w) * m)
        else:
            out[i] = np.nan
    ser = pd.Series(out, index=stdev.index[:out_len])

    if tail_policy == "ffill":
        # reindex to full stdev index and forward-fill the truncated tail
        ser = ser.reindex(stdev.index).ffill()

    return ser


def legacy_position_sizes(prices: pd.Series, pred_vol: pd.Series,
                          notional: float, leverage: float,
                          risk_fraction: float, portw: float) -> pd.Series:
    # Align to common index
    idx = pred_vol.index.intersection(prices.index)
    p = prices.reindex(idx)
    v = pred_vol.reindex(idx)
    shares = (notional * leverage * risk_fraction * portw) / (p * v)
    return shares

def legacy_pnl_series(prices: pd.Series, shares: pd.Series, legacy_sign: float = -1.0) -> pd.Series:
    """
    price_change[t] = price[t] - price[t-1]; return[t] = legacy_sign * price_change[t] * shares[t]
    We drop the first NaN caused by diff().
    """
    idx = shares.index.intersection(prices.index)
    p = prices.reindex(idx)
    px_chg = p.diff()
    pnl = legacy_sign * (px_chg * shares)
    pnl = pnl.iloc[1:]  # drop the first NA
    return pnl

def legacy_reverse(series: pd.Series) -> pd.Series:
    return series.iloc[::-1]

def pad_with_zeros_to(series: pd.Series, index: pd.Index) -> pd.Series:
    s2 = series.reindex(index)
    s2 = s2.fillna(0.0)
    return s2

# ---------------------------------------------------------------------
# Public: total cumulative returns (mimic of original plot)

def run_legacy_total(
    symbol_weights: Dict[str, float],
    prices: Optional[pd.DataFrame] = None,
    cfg: Optional[LegacyConfig] = None,
    save_pdf: Optional[str] = None,
    *,
    baseline: str | None = None,
    vol_provider: str = "ewma_legacy",
    provider_kwargs: dict | None = None,
    return_details: bool = False,                 # NEW
    export_holdings_csv: Optional[str] = None,    # NEW
) -> pd.Series | tuple[pd.Series, dict]:
    """
    Compute legacy-style total cumulative returns (sum of per-ticker PnL, then cumsum),
    reproducing quirks from the original notebook/script when cfg.strict_mimic=True.
    Returns the cumulative series (no plotting here).
    If return_details=True, returns (cum, details) where details includes:
      - 'daily_pnl': total daily PnL (Series)
      - 'equity'   : notional + cum (Series)
      - 'pnl_df'   : per-ticker daily PnL (DataFrame)
      - 'shares_df': per-ticker shares used for that day's PnL (DataFrame)
    If export_holdings_csv is a path, saves a long-form holdings file.
    """

    if cfg is None:
        cfg = LegacyConfig()
    symbols = list(symbol_weights.keys())

    fetch_start = cfg.start
    if prices is None:
        if fetch_start is not None and getattr(cfg, "warmup_backfill_days", 0) > 0:
            fetch_start = fetch_start - pd.Timedelta(days=cfg.warmup_backfill_days)
        prices = fetch_prices_yf(symbols, fetch_start, cfg.end)

    # --- NEW: get predicted vols by provider ---
    if vol_provider == "ewma_legacy":
        from syslib.vol_providers import ewma_legacy
        if provider_kwargs is None:
            provider_kwargs = {}
        pred_dict = ewma_legacy(
            prices,
            stdev_window=cfg.stdev_window,
            annualize_factor=cfg.annualize_factor,
            blend_weight=cfg.blend_weight,
            lookahead_mean_horizon=cfg.lookahead_mean_horizon,
            tail_policy=provider_kwargs.get("tail_policy", "truncate"),  # NEW
        )

    elif vol_provider == "ml_manifest":
        if provider_kwargs is None:
            provider_kwargs = {}
        from syslib.vol_providers import ml_manifest
        pred_dict = ml_manifest(
            provider_kwargs["base_dir"],
            symbols,
            prices.index,
            k_label=provider_kwargs.get("k_label", cfg.k_label if hasattr(cfg, "k_label") else 10),
            model_tag=provider_kwargs.get("model_tag"),
            date_col=provider_kwargs.get("date_col"),
            ticker_col=provider_kwargs.get("ticker_col"),
            pred_col=provider_kwargs.get("pred_col"),
            single_symbol=provider_kwargs.get("single_symbol"),
            pred_model_col=provider_kwargs.get("pred_model_col", "pred_gbm"),
            per_ticker_dir=provider_kwargs.get("per_ticker_dir"),
            x_test_path=provider_kwargs.get("x_test_path"),
            x_date_col=provider_kwargs.get("x_date_col"),
            x_ticker_col=provider_kwargs.get("x_ticker_col"),
        )
    else:
        raise ValueError(f"Unknown vol_provider: {vol_provider!r}")

    per_ticker_pnl = {}
    per_ticker_shares = {}
    for sym in symbols:
        px = prices[sym].dropna()
        pred = pred_dict.get(sym)
        if pred is None or pred.dropna().empty:
            # no predictions for this ticker -> skip it
            continue
        shr_raw = legacy_position_sizes(px, pred, cfg.notional, cfg.leverage, cfg.risk_fraction, portw=symbol_weights[sym])
        pnl = legacy_pnl_series(px, shr_raw, legacy_sign=cfg.legacy_sign)
        if cfg.strict_mimic:
            pnl = legacy_reverse(pnl)
            shr = legacy_reverse(shr_raw).reindex(pnl.index)  # align to daily pnl index
        else:
            shr = shr_raw.reindex(pnl.index)

        per_ticker_pnl[sym] = pnl
        per_ticker_shares[sym] = shr

    # Align/pad like the legacy script (zeros where missing)
    if cfg.strict_mimic:
        # use a sorted union of all indices (more faithful when tickers have different spans)
        all_idx = pd.Index(sorted(set().union(*[s.index for s in per_ticker_pnl.values()])))
        aligned = [pad_with_zeros_to(s, all_idx) for s in per_ticker_pnl.values()]
        total = pd.Series(np.sum(aligned, axis=0), index=all_idx)
    else:
        df = pd.concat(per_ticker_pnl, axis=1).fillna(0.0)
        total = df.sum(axis=1)

    if cfg.start is not None:
        total = total[total.index >= cfg.start]
    cum = total.cumsum()
    
    # zero-normalize so plots start at 0 (or compute equity from this)
    if cfg.normalize_to_zero and len(cum) > 0:
        cum = cum - cum.iloc[0]
    
    details = None
    if return_details or export_holdings_csv:
        # per-ticker frames aligned to total's index
        pnl_df = pd.concat(per_ticker_pnl, axis=1).reindex(cum.index).fillna(0.0)
        shares_df = pd.concat(per_ticker_shares, axis=1).reindex(cum.index).fillna(0.0)
        equity = (cfg.notional + (cum - cum.iloc[0])) if baseline in ("zero", None) else (cfg.notional + cum)
        daily_pnl = total.reindex(cum.index).fillna(0.0)

        if export_holdings_csv:
            # long-form holdings: date, ticker, shares, price, dollar_value, weight, vol_provider
            rows = []
            for sym in shares_df.columns:
                sh = shares_df[sym]
                px = prices[sym].reindex(sh.index)
                rows.append(pd.DataFrame({
                    "date": sh.index,
                    "ticker": sym,
                    "shares": sh.values,
                    "price": px.values,
                    "dollar_value": (sh * px).values,
                    "weight": symbol_weights.get(sym, np.nan),
                    "vol_provider": vol_provider,
                }))
            hold = pd.concat(rows, ignore_index=True)
            hold.to_csv(export_holdings_csv, index=False)

        if return_details:
            details = {
                "daily_pnl": daily_pnl,
                "equity": equity,
                "pnl_df": pnl_df,
                "shares_df": shares_df,
            }
    # ----- Return -----
    if return_details:
        return cum, details
    return cum
    
    if save_pdf:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(9, 4.5))
            ax.plot(cum.index, cum.values, label='Total PNL')
            ax.set_xlabel('Date')
            ax.set_ylabel('Total PNL')
            ax.set_title('Total Cumulative Returns')
            ax.legend()
            fig.autofmt_xdate()
            fig.tight_layout()
            fig.savefig(save_pdf)
            plt.close(fig)
        except Exception:
            pass
    return cum


# --- Summary metrics helper (kept simple) ---
def summarize_backtest(equity: pd.Series, daily_pnl: pd.Series, *, periods_per_year: int = 365) -> dict:
    import numpy as np  # already in env — no new install
    equity = equity.dropna()
    daily_pnl = daily_pnl.reindex(equity.index).fillna(0.0)

    # daily return as PnL over prior equity
    ret = daily_pnl / equity.shift(1)
    ret = ret.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # metrics
    tot_ret = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    n = max(1, len(equity))
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (periods_per_year / n) - 1.0)

    vol_ann = float(ret.std(ddof=1) * np.sqrt(periods_per_year)) if ret.std(ddof=1) > 0 else float("nan")
    sharpe = float(ret.mean() / ret.std(ddof=1) * np.sqrt(periods_per_year)) if ret.std(ddof=1) > 0 else float("nan")

    runup = equity.cummax()
    dd = equity / runup - 1.0
    max_dd = float(dd.min())
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else float("nan")

    hit = float((daily_pnl > 0).mean())
    avg_win = float(daily_pnl[daily_pnl > 0].mean()) if (daily_pnl > 0).any() else 0.0
    avg_loss = float((-daily_pnl[daily_pnl < 0]).mean()) if (daily_pnl < 0).any() else 0.0

    return {
        "total_return": tot_ret,
        "cagr": cagr,
        "vol_ann": vol_ann,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "hit_rate": hit,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "days": int(n),
    }
