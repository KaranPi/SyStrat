# src/syslib/size_legacy.py
from __future__ import annotations

from dataclasses import dataclass  # no installation needed
from datetime import date, timedelta  # no installation needed
from typing import Dict, Iterable  # no installation needed

import numpy as np          # already in env — no new install
import pandas as pd         # already in env — no new install
import yfinance as yf       # already in env — no new install


@dataclass(frozen=True)
class SizeConfig:
    # sizing knobs
    notional: float = 50_000.0
    leverage: float = 2.5
    risk_fraction: float = 0.2
    # legacy EWMA predictor
    window_len: int = 32
    w0: float = 0.6
    decay: float = 0.4
    hist_span: int = 54
    blend_recent: float = 0.7
    scale: float = 19.0
    # data
    lookback_years: int = 4
    interval: str = "1d"
    # NEW: which close to size on
    price_mode: str = "prev_close"  # or "last_close"



def _geom_weights(cfg: SizeConfig) -> np.ndarray:
    i = np.arange(cfg.window_len, dtype=float)
    return cfg.w0 * (cfg.decay ** i)

def _descending_returns(close_desc: pd.Series) -> pd.Series:
    # most-recent first convention (matches legacy script)
    return (close_desc.shift(0) - close_desc.shift(1)) / close_desc.shift(1)

def _predicted_vol_from_returns(r_desc: pd.Series, cfg: SizeConfig) -> float:
    arr = r_desc.dropna().to_numpy()
    need = cfg.window_len + max(1, cfg.hist_span)
    if arr.size < need:
        raise ValueError(f"Need >= {need} return obs; have {arr.size}. Increase lookback or reduce window.")
    w = _geom_weights(cfg)
    ewma = np.array([np.sum(w * arr[i:i+len(w)]) for i in range(len(arr)-len(w)+1)])
    stdev = np.array([np.sqrt(np.sum(w * (ewma[i] - arr[i:i+len(w)])**2)) for i in range(len(ewma))])
    recent = stdev[0]
    rest   = stdev[1:min(len(stdev), cfg.hist_span)]
    blended = cfg.blend_recent * recent + (1.0 - cfg.blend_recent) * (np.mean(rest) if rest.size else recent)
    return float(cfg.scale * blended)

def _latest_price_for_position(close_desc: pd.Series, *, price_mode: str) -> float:
    close_desc = close_desc.dropna()
    if len(close_desc) < 2:
        raise ValueError("Need at least two prices to size.")
    idx = 1 if price_mode == "prev_close" else 0
    return float(close_desc.iloc[idx])


def _size_units(price: float, pred_vol: float, portw: float, mult: float, cfg: SizeConfig) -> float:
    # legacy formula: units = (notional * leverage * risk_fraction * portw) / (price * pred_vol * mult)
    numer = cfg.notional * cfg.leverage * cfg.risk_fraction * portw
    denom = price * pred_vol * float(mult)
    return float(numer / denom)

def fetch_close_yf(tickers: Iterable[str], cfg: SizeConfig) -> pd.DataFrame:
    end = date.today()
    start = end - timedelta(days=365 * cfg.lookback_years)
    raw = yf.download(list(tickers), start=start, end=end, interval=cfg.interval,
                      auto_adjust=True, progress=False)
    # normalize to wide Close
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        close = raw[["Close"]].copy()
        if close.shape[1] == 1:
            close.columns = [list(tickers)[0]]
    # tz-naive, descending (most recent first)
    try: close.index = close.index.tz_localize(None)
    except Exception: pass
    return close.sort_index(ascending=False)

def size_snapshot_ewma(
    ticker_weights: Dict[str, float],
    mult_map: Dict[str, float] | None = None,
    cfg: SizeConfig = SizeConfig(),
    prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if mult_map is None:
        mult_map = {t: 1.0 for t in ticker_weights}  # default multiplier = 1
    if prices is None:
        prices = fetch_close_yf(ticker_weights.keys(), cfg)

    rows = []
    for tkr, portw in ticker_weights.items():
        if tkr not in prices.columns:
            continue
        ser_desc = prices[tkr].dropna()
        r_desc = _descending_returns(ser_desc).dropna()
        pred_vol = _predicted_vol_from_returns(r_desc, cfg)
        px = _latest_price_for_position(ser_desc, price_mode=cfg.price_mode)
        mult = float(mult_map.get(tkr, 1.0))
        units = _size_units(px, pred_vol, portw, mult, cfg)
        rows.append({
            "ticker": tkr,
            "portw": float(portw),
            "mult": mult,
            "last_price": px,
            "pred_vol": pred_vol,
            "units": units,
            "notional_alloc": units * px * mult,
        })
    out = pd.DataFrame(rows).set_index("ticker").sort_index()
    return out
