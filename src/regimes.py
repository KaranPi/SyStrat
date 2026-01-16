# src/syslib/regimes.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# Optional ML σ̂ loader (uses your existing provider)
# You already have this wired in your repo.
from syslib.vol_providers import ml_manifest  # uses model_manifest, preds_test_k10, etc. 


@dataclass(frozen=True)
class RegimeConfig:
    base_dir: Path
    tickers: Tuple[str, ...]
    k_label: int = 10
    model_tag: str = "gbm"
    vwap_lookback: int = 90
    ema_fast: int = 9
    ema_slow: int = 21
    vol_qtiles: Tuple[float, float] = (0.33, 0.66)
    start: Optional[pd.Timestamp] = None
    end: Optional[pd.Timestamp] = None
    ml_provider_kwargs: Optional[dict] = None



def _ensure_dtindex(df: pd.DataFrame) -> pd.DataFrame:
    idx = pd.to_datetime(df.index).tz_localize(None)
    df = df.copy()
    df.index = idx
    return df.sort_index()


def fetch_close_vol_yf(tickers: Iterable[str], start=None, end=None) -> pd.DataFrame:
    """Return wide dataframe with MultiIndex columns: (field, ticker), fields=['Close','Volume']."""
    data = yf.download(list(tickers), start=start, end=end, interval="1d", auto_adjust=True, progress=False)
    if not isinstance(data.columns, pd.MultiIndex):
        # single ticker case: build a MI
        data = pd.concat({"Close": data["Close"], "Volume": data["Volume"]}, axis=1)
        data.columns = pd.MultiIndex.from_product([["Close","Volume"], list(tickers)])
    data = _ensure_dtindex(data)
    return data


def _rolling_vwap(price: pd.Series, vol: pd.Series, win: int) -> pd.Series:
    """Daily VWAP over a rolling window using daily data."""
    v = vol.fillna(0.0)
    p = price
    num = (p * v).rolling(win, min_periods=max(5, win//3)).sum()
    den = v.rolling(win, min_periods=max(5, win//3)).sum()
    vwap = num / den
    # Fallback when volume is all zeros/NaN: use price EMA as a proxy to avoid NaNs
    vwap = vwap.fillna(p.ewm(span=win, adjust=False).mean())
    return vwap


def _expanding_terciles(x: pd.Series, qtiles=(0.33, 0.66)) -> pd.Series:
    """Expanding tercile label (0/1/2) without look-ahead: each t uses quantiles from data[:t)."""
    x = x.astype(float)
    out = np.full(len(x), np.nan)
    xs = x.values
    for i in range(1, len(x)):  # start at 1 so we have some history
        hist = xs[:i]
        q1, q2 = np.nanquantile(hist, qtiles)
        out[i] = 0 if xs[i] <= q1 else (2 if xs[i] >= q2 else 1)
    return pd.Series(out, index=x.index).astype("Int64")


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def build_regime_features(cfg: RegimeConfig) -> pd.DataFrame:
    """
    Output: MultiIndex (date, ticker) → columns:
      ['price','volume','sigma_hat','vol_tercile','ema_fast','ema_slow','ema_slope',
       'vwap_roll','vwap_z']
    """
    # 1) Price/volume
    raw = fetch_close_vol_yf(cfg.tickers, start=cfg.start, end=cfg.end)
    close = raw["Close"]
    vol   = raw["Volume"]

    # Use the calendar from prices
    date_index = close.index

    # 2) ML σ̂ per ticker aligned to date_index (DESC/ASC agnostic inside ml_manifest)
    #    If ML not available, you could later add a fallback σ̂ (e.g., EWMA), but we keep it lean here.
    kw = cfg.ml_provider_kwargs or {}
    sigma_dict = ml_manifest(
        cfg.base_dir, cfg.tickers, date_index,
        k_label=cfg.k_label, model_tag=cfg.model_tag,
        date_col=kw.get("date_col"), ticker_col=kw.get("ticker_col"), pred_col=kw.get("pred_col"),
        single_symbol=kw.get("single_symbol"), pred_model_col=kw.get("pred_model_col", "pred_gbm"),
        per_ticker_dir=kw.get("per_ticker_dir"),
        x_test_path=kw.get("x_test_path"), x_date_col=kw.get("x_date_col"), x_ticker_col=kw.get("x_ticker_col"),
    )

    rows = []
    for tkr in cfg.tickers:
        p = close[tkr].dropna()
        v = vol[tkr].reindex(p.index).fillna(0.0)

        s = sigma_dict.get(tkr)
        if s is None:
            # leave sigma as NaN if not present
            s = pd.Series(index=p.index, dtype="float64")

        # EMAs
        ema_f = _ema(p, cfg.ema_fast)
        ema_s = _ema(p, cfg.ema_slow)
        ema_slope = ema_f - ema_s

        # Rolling VWAP and z-score of (price - VWAP)
        vwap = _rolling_vwap(p, v, cfg.vwap_lookback)
        spread = (p - vwap)
        z = (spread - spread.rolling(cfg.vwap_lookback, min_periods=10).mean()) / (
            spread.rolling(cfg.vwap_lookback, min_periods=10).std()
        )

        # Expanding terciles of sigma_hat (no look-ahead)
        vol_tercile = _expanding_terciles(s, qtiles=cfg.vol_qtiles)

        df = pd.DataFrame({
            "price": p,
            "volume": v,
            "sigma_hat": s.reindex(p.index),
            "vol_tercile": vol_tercile.reindex(p.index),
            "ema_fast": ema_f,
            "ema_slow": ema_s,
            "ema_slope": ema_slope,
            "vwap_roll": vwap,
            "vwap_z": z,
        })
        # make date explicit and stable
        df["date"] = pd.to_datetime(df.index).tz_localize(None)
        df["ticker"] = tkr
        df = df.reset_index(drop=True)
        rows.append(df)

    out = pd.concat(rows, ignore_index=True)
    out = out.set_index(["date", "ticker"]).sort_index()
    return out



def save_regime_features(cfg: RegimeConfig, df: pd.DataFrame) -> Path:
    outdir = Path(cfg.base_dir) / "data_int" / "regimes"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"regime_features_k{cfg.k_label}.parquet"
    df.to_parquet(path)
    return path



# --- TIER-1 LABELS & TRANSITIONS ---

from dataclasses import dataclass
import numpy as np
import pandas as pd

REGIME_ORDER = ("risk_off", "neutral", "risk_on")  # index 0..2

def _apply_hysteresis(state_series: pd.Series, k: int = 3) -> pd.Series:
    """Require k consecutive days in a new state before switching."""
    if k <= 1:
        return state_series
    out = []
    prev = None
    hold = None
    counter = 0
    for s in state_series:
        if prev is None:
            prev = s
            out.append(s)
            continue
        if s == prev:
            hold = s
            counter = 0
            out.append(prev)
        else:
            # candidate change
            if hold != s:
                hold = s
                counter = 1
            else:
                counter += 1
            if counter >= k:
                prev = hold
                counter = 0
            out.append(prev)
    return pd.Series(out, index=state_series.index)

def label_regimes_tier1(feat: pd.DataFrame,
                        ema_thresh: float = 0.0,
                        vwap_hi: float = -0.5,
                        vwap_lo: float = -1.0,
                        hysteresis_days: int = 3) -> tuple[pd.DataFrame, pd.Series]:
    """
    Input: MultiIndex (date,ticker) features with cols: ['sigma_hat','vol_tercile','ema_slope','vwap_z']
    Output:
      labels_per_ticker: same index with column 'state' in {'risk_off','neutral','risk_on'}
      daily_state: portfolio-level state per date (majority vote across tickers), hysteresis-applied
    """
    df = feat.copy()
    # Booleans; treat NaNs as False so they contribute 0 to scores
    lo    = (df['vol_tercile'] == 0).fillna(False).astype(bool)
    hi    = (df['vol_tercile'] == 2).fillna(False).astype(bool)
    up    = (df['ema_slope']   >  ema_thresh).fillna(False).astype(bool)
    vw_hi = (df['vwap_z']      >  vwap_hi).fillna(False).astype(bool)
    vw_lo = (df['vwap_z']      <  vwap_lo).fillna(False).astype(bool)
    
    # Scores
    risk_on_score  = lo.astype(int) + up.astype(int) + vw_hi.astype(int)
    risk_off_score = hi.astype(int) + (~up).astype(int) + vw_lo.astype(int)


    # Argmax → state
    state_pt = np.where(risk_off_score > risk_on_score, "risk_off",
                 np.where(risk_on_score > risk_off_score, "risk_on", "neutral"))
    labels = pd.DataFrame({"state": state_pt}, index=df.index)

    # Aggregate to portfolio-level state by date (majority vote)
    daily = (labels
             .reset_index()
             .pivot_table(index="date", columns="state", values="ticker", aggfunc="count", fill_value=0))
    # ensure all columns exist, then order them
    for s in REGIME_ORDER:
        if s not in daily.columns:
            daily[s] = 0
    daily = daily.reindex(columns=list(REGIME_ORDER), fill_value=0)

    daily_state = daily.idxmax(axis=1)

    # Apply hysteresis on the portfolio state
    daily_state = _apply_hysteresis(daily_state, k=hysteresis_days)
    # back to a clean Series named 'state'
    daily_state.name = "state"

    return labels, daily_state

def estimate_transitions(state_series: pd.Series, smoothing: float = 0.5) -> pd.DataFrame:
    """
    Empirical Markov transition matrix P(i->j) with Laplace smoothing.
    state_series: daily portfolio-level state.
    """
    states = list(REGIME_ORDER)
    counts = pd.DataFrame(0.0, index=states, columns=states)
    prev = None
    for s in state_series:
        if prev is not None:
            counts.loc[prev, s] += 1.0
        prev = s
    # Laplace smoothing
    counts += smoothing
    probs = counts.div(counts.sum(axis=1), axis=0)
    return probs

def save_tier1_labels(cfg: RegimeConfig, labels: pd.DataFrame, daily_state: pd.Series) -> tuple[Path, Path]:
    outdir = Path(cfg.base_dir) / "data_int" / "regimes"
    outdir.mkdir(parents=True, exist_ok=True)
    p1 = outdir / f"tier1_labels_k{cfg.k_label}.parquet"
    p2 = outdir / f"tier1_daily_state_k{cfg.k_label}.parquet"
    labels.to_parquet(p1)
    daily_state.to_frame().to_parquet(p2)
    return p1, p2

def save_transitions(cfg: RegimeConfig, P: pd.DataFrame) -> Path:
    outdir = Path(cfg.base_dir) / "data_int" / "regimes"
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / f"tier1_transitions_k{cfg.k_label}.csv"
    P.to_csv(p, float_format="%.6f")
    return p
