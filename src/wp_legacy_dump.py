# wp_legacy_dump.py
# Recreates the legacy weighted_port math and dumps intermediate artifacts
# (Daily returns, sum-products EWMA, weighted stdev, predicted vol, position size)
# to CSV (and optionally Parquet) for each symbol.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf  # optional: you can pass prices explicitly to avoid this
except Exception:
    yf = None


@dataclass
class LegacyDumpConfig:
    # data window; if None, we’ll fetch ~5y by default when using yfinance
    start: Optional[pd.Timestamp] = None
    end: Optional[pd.Timestamp] = None

    # sizing
    notional: float = 50_000.0
    leverage: float = 2.5
    risk_fraction: float = 0.2

    # legacy math knobs
    weights_len: int = 32                    # length of EWMA weights
    w0: float = 0.6                          # first weight
    w_decay: float = 0.4                     # geometric decay for subsequent weights
    annualize_factor: float = 19.0           # 19x (legacy constant)
    lookahead_h: int = 54                    # stdev[1:54] look-ahead mean window (legacy)

    # output controls
    save_parquet: bool = False               # save parquet alongside CSV
    # when True, we also save a compact "_summary.csv" with the scalar t0 values
    save_summary: bool = True


def _fetch_prices_yf(symbols: Iterable[str], start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance is not available. Pass a prices DataFrame instead.")
    if end is None:
        end = pd.Timestamp.today().normalize()
    if start is None:
        start = end - pd.Timedelta(days=365 * 5)  # ~5 years default
    data = yf.download(list(symbols), start=start, end=end, progress=False, auto_adjust=True)
    close = data["Close"].copy()
    close = close.dropna(how="all").ffill().dropna(axis=1, how="all")
    # Ensure single-column for single symbol pulls
    if isinstance(close, pd.Series):
        close = close.to_frame()
    return close


def _legacy_weights(length: int, w0: float, w_decay: float) -> np.ndarray:
    # weights = [w0 * (w_decay ** i) for i in range(length)]  (exact legacy form)
    w = np.array([w0 * (w_decay ** i) for i in range(length)], dtype=float)
    return w  # no normalization (legacy didn’t normalize)


def _reverse_series(s: pd.Series) -> pd.Series:
    # legacy reverses price array so index 0 is most-recent
    return s.iloc[::-1]


def _daily_returns_legacy(closing_rev: np.ndarray) -> np.ndarray:
    # returns[i] = (p[i] - p[i+1]) / p[i+1], for i in 0..N-2  (legacy)
    p = closing_rev
    return (p[:-1] - p[1:]) / p[1:]


def _ewma_sumproducts(returns: np.ndarray, weights: np.ndarray) -> np.ndarray:
    L = len(weights)
    R = len(returns)
    if R < L:
        return np.array([], dtype=float)
    # ewma[i] = sum(weights * returns[i:i+L])
    out = np.zeros(R - L + 1, dtype=float)
    for i in range(out.size):
        out[i] = np.sum(weights * returns[i:i + L])
    return out


def _weighted_stdev(returns: np.ndarray, ewma: np.ndarray, weights: np.ndarray) -> np.ndarray:
    L = len(weights)
    if returns.size < L or ewma.size == 0:
        return np.array([], dtype=float)
    out = np.zeros(ewma.size, dtype=float)
    for i in range(out.size):
        # sqrt( sum(weights * (ewma[i] - returns[i:i+L])**2) )
        out[i] = np.sqrt(np.sum(weights * (ewma[i] - returns[i:i + L]) ** 2))
    return out


def _pred_vol_series(stdev: np.ndarray, annualize_factor: float, lookahead_h: int) -> np.ndarray:
    """
    Generalize the legacy scalar to a series:
      pred_vol[i] = 19 * (0.7*stdev[i] + 0.3*mean(stdev[i+1:i+54]))
    Output length: max(0, len(stdev) - lookahead_h)
    """
    n = len(stdev)
    out_len = max(0, n - lookahead_h)
    if out_len == 0:
        return np.array([], dtype=float)
    out = np.full(out_len, np.nan, dtype=float)
    for i in range(out_len):
        tail = stdev[i + 1:i + lookahead_h]
        m = np.mean(tail) if tail.size > 0 else np.nan
        out[i] = annualize_factor * (0.7 * stdev[i] + 0.3 * m)
    return out


def _position_size_series(
    closing_rev: np.ndarray,
    pred_vol: np.ndarray,
    *,
    notional: float,
    leverage: float,
    risk_fraction: float,
    portw: float,
    multiplier: float,
) -> np.ndarray:
    """
    shares[i] = (notional * leverage * risk_fraction * portw) / (price_rev[i] * pred_vol[i] * multiplier)
    Computed wherever pred_vol[i] exists; price_rev is aligned at the same index.
    """
    n = min(len(closing_rev), len(pred_vol))
    if n == 0:
        return np.array([], dtype=float)
    price = closing_rev[:n]
    vol = pred_vol[:n]
    # avoid division by zero
    vol = np.where(vol == 0.0, np.nan, vol)
    shares = (notional * leverage * risk_fraction * portw) / (price * vol * multiplier)
    return shares


def compute_legacy_vectors_for_symbol(
    symbol: str,
    portw: float,
    mult: float,
    cfg: LegacyDumpConfig,
    prices: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Returns a wide DataFrame aligned on a descending time index (most-recent first),
    with columns:
      date_rev, close_rev, return_legacy, ewma32, stdev32, pred_vol, position_size
    Note: Different columns start at different offsets; NaNs are expected.
    """
    if prices is None:
        prices = _fetch_prices_yf([symbol], cfg.start, cfg.end)

    px = prices[symbol].dropna()
    # Build descending (legacy) index and values
    px_rev = _reverse_series(px)
    date_rev = px_rev.index

    # Arrays
    closing_rev = px_rev.values.astype(float)
    returns = _daily_returns_legacy(closing_rev)  # len N-1

    w = _legacy_weights(cfg.weights_len, cfg.w0, cfg.w_decay)
    ewma = _ewma_sumproducts(returns, w)          # len N-1 - L + 1
    stdev = _weighted_stdev(returns, ewma, w)     # len = len(ewma)

    pred_vol = _pred_vol_series(stdev, cfg.annualize_factor, cfg.lookahead_h)
    pos_size = _position_size_series(
        closing_rev, pred_vol,
        notional=cfg.notional, leverage=cfg.leverage, risk_fraction=cfg.risk_fraction,
        portw=portw, multiplier=mult,
    )

    # Scalar legacy values at t0 (closest to today), for quick unit checks
    pred_vol_scalar_t0 = (cfg.annualize_factor * (0.7 * stdev[0] + 0.3 * np.mean(stdev[1:cfg.lookahead_h]))
                          if stdev.size > cfg.lookahead_h else np.nan)
    pos_size_scalar_t0 = (
        (cfg.notional * cfg.leverage * cfg.risk_fraction * portw) /
        (closing_rev[0] * pred_vol_scalar_t0 * mult)
        if np.isfinite(pred_vol_scalar_t0) and pred_vol_scalar_t0 > 0 else np.nan
    )

    # Assemble into a wide DataFrame (descending index)
    # We right-align each vector so index 0 lines up across columns.
    max_len = len(closing_rev)
    def pad_right(a, target_len):
        out = np.full(target_len, np.nan, dtype=float)
        out[:len(a)] = a  # right-aligned to the "most recent" start
        return out

    df = pd.DataFrame({
        "date_rev": pd.Series(date_rev, index=date_rev),
        "close_rev": pd.Series(closing_rev, index=date_rev),
    })

    # Create an integer descending index to hold all columns uniformly
    idx = pd.RangeIndex(start=0, stop=max_len, step=1)  # 0 = most recent
    df = df.reset_index(drop=True).set_index(idx)

    df["return_legacy"] = pd.Series(pad_right(returns, max_len))
    df["ewma32"]        = pd.Series(pad_right(ewma,    max_len))
    df["stdev32"]       = pd.Series(pad_right(stdev,   max_len))
    df["pred_vol"]      = pd.Series(pad_right(pred_vol, max_len))
    df["position_size"] = pd.Series(pad_right(pos_size, max_len))

    # Annotate scalars at t0 for quick access
    df.attrs["pred_vol_scalar_t0"] = float(pred_vol_scalar_t0) if np.isfinite(pred_vol_scalar_t0) else None
    df.attrs["pos_size_scalar_t0"] = float(pos_size_scalar_t0) if np.isfinite(pos_size_scalar_t0) else None
    df.attrs["weights32"] = w.tolist()

    return df


def dump_legacy_artifacts(
    out_dir: Path,
    symbol_weights: Dict[str, float],
    multipliers: Optional[Dict[str, float]] = None,
    cfg: Optional[LegacyDumpConfig] = None,
    prices: Optional[pd.DataFrame] = None,
) -> Path:
    """
    Runs compute_legacy_vectors_for_symbol for each symbol and saves:
      <symbol>_legacy_vectors.csv   (and .parquet if cfg.save_parquet)
    Also writes a summary CSV with scalar values at t0 and the 32 weights.
    Returns the output folder.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if cfg is None:
        cfg = LegacyDumpConfig()
    if multipliers is None:
        multipliers = {s: 1.0 for s in symbol_weights}

    summary_rows = []

    for sym, portw in symbol_weights.items():
        mult = float(multipliers.get(sym, 1.0))
        df = compute_legacy_vectors_for_symbol(sym, portw, mult, cfg, prices=prices)

        # Save main table
        csv_path = out_dir / f"{sym}_legacy_vectors.csv"
        df.to_csv(csv_path, index=True)

        if cfg.save_parquet:
            pq_path = out_dir / f"{sym}_legacy_vectors.parquet"
            df.to_parquet(pq_path, index=True)

        # Summary scalars
        if cfg.save_summary:
            summary_rows.append({
                "symbol": sym,
                "pred_vol_scalar_t0": df.attrs.get("pred_vol_scalar_t0"),
                "pos_size_scalar_t0": df.attrs.get("pos_size_scalar_t0"),
                "weights32": "|".join(map(str, df.attrs.get("weights32", []))),
                "notional": cfg.notional,
                "leverage": cfg.leverage,
                "risk_fraction": cfg.risk_fraction,
                "portw": portw,
                "multiplier": mult,
            })

    if cfg.save_summary and summary_rows:
        pd.DataFrame(summary_rows).to_csv(out_dir / "legacy_dump_summary.csv", index=False)

    return out_dir
