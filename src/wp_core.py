from __future__ import annotations  # no installation needed

"""
wp_core.py — Basket-first helpers for the `weighted_port` strategy (Steps 1 & 2)
-------------------------------------------------------------------------------
This module reproduces your working math starting from the point you noted
(`pd.set_option('display.max_columns', None)`) and makes it reusable.

What’s included here (no ML yet):
1) Core helpers for fetching data (yfinance), computing descending returns,
   geometric weights, EWMA-style weighted mean & weighted stdev, predicted_vol
   via the 0.7/0.3 blend and the SCALE=19.0 multiplier, price selection logic,
   and share sizing for a *basket* (no WPConfig involved).
2) Diagnostics helpers: realized volatility, tracking error, and rolling
   versions you’ll later use for tuning/validation.

All imports are annotated to satisfy your environment hygiene rules.
"""

# ── Imports (annotated) ────────────────────────────────────────────────────────
from typing import Dict, Iterable, Tuple, Literal, Optional  # no installation needed
from dataclasses import dataclass  # no installation needed (used only for typed containers; no WPConfig)
from pathlib import Path  # no installation needed
from datetime import date  # no installation needed

import numpy as np  # already in env — no new install
import pandas as pd  # already in env — no new install
import yfinance as yf  # already in env — no new install (install if missing)


# ── Defaults / Constants (match your working script) ───────────────────────────
LOOKBACK_YEARS: int = 4
INTERVAL: str = "1d"  # yfinance interval
WINDOW_LEN: int = 32
W0: float = 0.6
DECAY: float = 0.4
SCALE: float = 19.0
BLEND_RECENT: float = 0.7
HIST_SPAN: int = 54
NOTIONAL: float = 100_000.0
LEVERAGE: float = 2.5
RISK_FRACTION: float = 0.2
PRICE_MODE = Literal["prev_close", "last_close"]
RET_KIND = Literal["simple", "log"]


# ── Small typed container for results (optional) ───────────────────────────────
@dataclass
class TickerSizing:
    ticker: str
    portw: float
    price_used: float
    predicted_vol: float
    shares: float


# ── Fetch & normalize prices ───────────────────────────────────────────────────
def fetch_close_yf(
    tickers: Iterable[str],
    start: Optional[pd.Timestamp | str] = None,
    end: Optional[pd.Timestamp | str] = None,
    period: Optional[str] = None,
    interval: str = INTERVAL,
    strict: bool = True,
) -> pd.DataFrame:
    """Download Close prices for a list of tickers from yfinance as a wide DataFrame.
    Robust to different yfinance column layouts (ticker-first vs field-first) and
    falls back to 'Adj Close' if 'Close' is missing. Returns **ascending** index.
    """
    kwargs = {
        "interval": interval,
        "group_by": "ticker",   # yfinance may still return field-first in some versions
        "auto_adjust": False,
        "progress": False,
        "threads": True,
    }
    # Download
    if period is not None:
        raw = yf.download(list(tickers), period=period, **kwargs)
    else:
        raw = yf.download(list(tickers), start=start, end=end, **kwargs)

    if raw is None or len(raw) == 0:
        raise RuntimeError(
            "yfinance returned no data. Possible causes: bad ticker, interval/start-end out of range, or transient API issue."
        )

    def _select_close(df: pd.DataFrame) -> pd.DataFrame:
        # MultiIndex case (common with multi-ticker)
        if isinstance(df.columns, pd.MultiIndex):
            lv0 = df.columns.get_level_values(0)
            lv1 = df.columns.get_level_values(1)
            if "Close" in lv0:
                close = df.xs("Close", axis=1, level=0)
            elif "Close" in lv1:
                close = df.xs("Close", axis=1, level=1)
            elif "Adj Close" in lv0:
                close = df.xs("Adj Close", axis=1, level=0)
            elif "Adj Close" in lv1:
                close = df.xs("Adj Close", axis=1, level=1)
            else:
                # case-insensitive fallback
                for level in (0, 1):
                    vals = [str(v).lower() for v in df.columns.get_level_values(level)]
                    if "close" in vals:
                        key = df.columns.get_level_values(level)[vals.index("close")]
                        close = df.xs(key, axis=1, level=level)
                        break
                else:
                    raise RuntimeError(
                        "Could not find 'Close' or 'Adj Close' in yfinance result (MultiIndex)."
                    )
            # After xs, columns should be tickers. If still MultiIndex, flatten to last level.
            if isinstance(close.columns, pd.MultiIndex):
                close.columns = close.columns.get_level_values(-1)
            return close
        # Single-index columns (single ticker or some yfinance versions)
        cols_lower = {str(c).lower(): c for c in df.columns}
        if "close" in cols_lower:
            close = df[[cols_lower["close"]]].copy()
        elif "adj close" in cols_lower or "adj_close" in cols_lower:
            key = cols_lower.get("adj close") or cols_lower.get("adj_close")
            close = df[[key]].copy()
        else:
            candidates = [c for c in df.columns if "close" in str(c).lower()]
            if candidates:
                close = df[candidates].copy()
            else:
                raise RuntimeError(
                    f"Could not find a Close-like column in columns: {list(df.columns)[:10]}"
                )
        # If only one ticker requested, name the column as that ticker
        if len(tickers) == 1 and close.shape[1] == 1:
            close.columns = [list(tickers)[0]]
        return close

    close = _select_close(raw)
    # Drop all-NaN columns and check for missing tickers
    close = close.dropna(how="all", axis=1)
    missing = [t for t in tickers if t not in list(close.columns)]
    if missing and strict:
        raise RuntimeError(
            f"Missing data for tickers: {missing}. Check symbols/interval or try again (transient yfinance issue)."
        )

    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close


# ── Returns (descending convention like your script) ───────────────────────────
def to_descending(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy sorted with most-recent first (descending index)."""
    return df.sort_index(ascending=False).copy()


def descending_returns(series_desc: pd.Series, kind: RET_KIND = "simple") -> pd.Series:
    """Compute descending-direction returns. If `series_desc` is most-recent first,
    then r[i] uses price[i] vs price[i+1] (the previous day chronologically).
    """
    s = series_desc
    if kind == "simple":
        r = (s / s.shift(-1)) - 1.0
    elif kind == "log":
        r = np.log(s / s.shift(-1))
    else:
        raise ValueError("RET_KIND must be 'simple' or 'log'.")
    return r.dropna()


# ── Geometric weights & EWMA-style stats ───────────────────────────────────────
def geom_weights(window_len: int = WINDOW_LEN, w0: float = W0, decay: float = DECAY) -> np.ndarray:
    """Geometric weights w_i = w0 * decay**i, i=0..window_len-1.
    With w0=0.6 and decay=0.4, the infinite sum is 1.0, and the 32-term sum is ≈ 1.
    No extra normalization is applied to remain faithful to your script.
    """
    i = np.arange(window_len, dtype=float)
    w = (w0 * (decay ** i)).astype(float)
    return w


def _weighted_mean_window(r_window: np.ndarray, w: np.ndarray) -> float:
    # weights approximately sum to 1 under your default; use dot product directly
    return float(np.dot(w, r_window))


def _weighted_stdev_window(r_window: np.ndarray, w: np.ndarray, mean: float) -> float:
    # same convention as your trial: sqrt(sum(w * (mean - r)^2))
    diff2 = (mean - r_window) ** 2
    return float(np.sqrt(np.dot(w, diff2)))


def rolling_predicted_vol_ewma_blend(
    r_desc: pd.Series,
    window_len: int = WINDOW_LEN,
    w0: float = W0,
    decay: float = DECAY,
    blend_recent: float = BLEND_RECENT,
    hist_span: int = HIST_SPAN,
    scale: float = SCALE,
) -> pd.Series:
    """Compute predicted_vol over time using your EWMA+blend recipe on descending returns.
    Returns a Series aligned to the *start* of each window (i=0 is most-recent window).
    Requires at least `window_len + hist_span + 1` observations.
    """
    r = np.asarray(r_desc.dropna().values, dtype=float)
    n = len(r)
    need = window_len + hist_span + 1
    if n < need:
        raise ValueError(
            f"Not enough observations: have {n}, need >= {need} (window_len={window_len}, hist_span={hist_span})."
        )

    w = geom_weights(window_len, w0, decay)

    # For each i, compute weighted mean & stdev on r[i : i+window_len]
    max_i = n - window_len
    means = np.empty(max_i, dtype=float)
    stdev = np.empty(max_i, dtype=float)
    for i in range(max_i):
        rw = r[i : i + window_len]
        mu = _weighted_mean_window(rw, w)
        means[i] = mu
        stdev[i] = _weighted_stdev_window(rw, w, mu)

    # Blend: pv[i] = scale * (blend_recent*stdev[i] + (1-blend_recent)*mean(stdev[i+1 : i+1+hist_span]))
    pv = np.empty(max_i - hist_span, dtype=float)
    for i in range(max_i - hist_span):
        recent = stdev[i]
        tail = stdev[i + 1 : i + 1 + hist_span]
        pv[i] = scale * (blend_recent * recent + (1.0 - blend_recent) * float(np.mean(tail)))

    idx = r_desc.index[: len(pv)]  # descending index alignment
    return pd.Series(pv, index=idx)


# ── Price selection & sizing ───────────────────────────────────────────────────
def latest_price(series_desc: pd.Series, price_mode: PRICE_MODE = "prev_close") -> float:
    """Choose price from a descending price series.
    - 'prev_close' => element at position 1 (your current behavior)
    - 'last_close' => element at position 0 (most recent)
    """
    if len(series_desc) < 2 and price_mode == "prev_close":
        raise ValueError("Need at least two observations for prev_close price_mode.")
    if price_mode == "prev_close":
        return float(series_desc.iloc[1])
    elif price_mode == "last_close":
        return float(series_desc.iloc[0])
    else:
        raise ValueError("price_mode must be 'prev_close' or 'last_close'.")


def size_shares(
    price: float,
    predicted_vol: float,
    portw: float,
    notional: float = NOTIONAL,
    leverage: float = LEVERAGE,
    risk_fraction: float = RISK_FRACTION,
) -> float:
    numer = notional * leverage * risk_fraction * portw
    denom = price * predicted_vol
    if denom <= 0:
        raise ValueError("Non-positive denominator in size_shares (check price/predicted_vol).")
    return float(numer / denom)


# ── Basket runner (no CLI here; import into notebooks) ────────────────────────
def run_weighted_port_basket(
    ticker_to_portw: Dict[str, float],
    lookback_years: int = LOOKBACK_YEARS,
    interval: str = INTERVAL,
    price_mode: PRICE_MODE = "prev_close",
    ret_kind: RET_KIND = "simple",
    window_len: int = WINDOW_LEN,
    w0: float = W0,
    decay: float = DECAY,
    blend_recent: float = BLEND_RECENT,
    hist_span: int = HIST_SPAN,
    scale: float = SCALE,
) -> pd.DataFrame:
    """Compute per-ticker predicted_vol and position size for a basket.
    Returns a tidy DataFrame with columns: [portw, price_used, predicted_vol, shares].
    """
    if not ticker_to_portw:
        raise ValueError("Empty ticker_to_portw mapping.")

    end = pd.Timestamp(date.today())
    start = end - pd.DateOffset(years=lookback_years)

    close_asc = fetch_close_yf(list(ticker_to_portw.keys()), start=start, end=end, interval=interval)
    close_desc = to_descending(close_asc)

    rows: list[TickerSizing] = []
    for tkr, portw in ticker_to_portw.items():
        s_desc = close_desc[tkr].dropna()
        r_desc = descending_returns(s_desc, kind=ret_kind)
        pv_series = rolling_predicted_vol_ewma_blend(
            r_desc,
            window_len=window_len,
            w0=w0,
            decay=decay,
            blend_recent=blend_recent,
            hist_span=hist_span,
            scale=scale,
        )
        pv0 = float(pv_series.iloc[0])  # most recent predicted vol
        px = latest_price(s_desc, price_mode=price_mode)
        shares = size_shares(px, pv0, portw)
        rows.append(TickerSizing(ticker=tkr, portw=portw, price_used=px, predicted_vol=pv0, shares=shares))

    out = pd.DataFrame([r.__dict__ for r in rows]).set_index("ticker")
    return out


# ── Diagnostics (Step 2) ──────────────────────────────────────────────────────
def realized_vol_desc(r_desc: pd.Series, k: int) -> float:
    """Next-k-day realized volatility in *descending* convention:
    sqrt(sum_{i=0..k-1} r[i]^2). Keep units consistent with predicted_vol.
    """
    if len(r_desc) < k:
        raise ValueError(f"Need at least k={k} returns to compute realized vol.")
    r = np.asarray(r_desc.iloc[:k].values, dtype=float)
    return float(np.sqrt(np.sum(r * r)))


def rolling_realized_vol_desc(r_desc: pd.Series, k: int) -> pd.Series:
    """Rolling realized vol series in descending convention: RV[i] = sqrt(sum r[i:i+k]^2)."""
    r = r_desc.dropna().values.astype(float)
    n = len(r)
    if n < k:
        raise ValueError(f"Need at least k={k} returns to compute rolling RV.")
    rv = np.empty(n - k + 1, dtype=float)
    for i in range(n - k + 1):
        seg = r[i : i + k]
        rv[i] = np.sqrt(np.sum(seg * seg))
    idx = r_desc.index[: len(rv)]  # align to descending index
    return pd.Series(rv, index=idx)


def tracking_error_series(predicted: pd.Series, realized: pd.Series) -> pd.Series:
    """Absolute error |predicted - realized| with inner alignment."""
    aligned = pd.concat([predicted.rename("pred"), realized.rename("real")], axis=1).dropna()
    return (aligned["pred"] - aligned["real"]).abs()


# ── OHLC fetch & artifact helpers (Step 2+: data artifacts) ───────────────────
FIELDS_CANON = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def _standardize_ohlc_columns(raw: pd.DataFrame, tickers: Iterable[str]) -> pd.DataFrame:
    """Return OHLC with a MultiIndex (ticker, field), ascending index, tz-naive."""
    df = raw.copy()
    # Normalize index
    df.index = pd.to_datetime(df.index).tz_localize(None)

    # MultiIndex columns (common for multi-ticker)
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = list(map(str, df.columns.get_level_values(0)))
        lvl1 = list(map(str, df.columns.get_level_values(1)))
        # Detect if level 0 are fields (e.g., "Open")
        fields_in_lvl0 = any(x in set(lvl0) for x in FIELDS_CANON)
        fields_in_lvl1 = any(x in set(lvl1) for x in FIELDS_CANON)
        if fields_in_lvl0 and not fields_in_lvl1:
            df = df.swaplevel(0, 1, axis=1)
        # After this, expect columns as (ticker, field)
        # Drop any unexpected levels beyond 2
        if isinstance(df.columns, pd.MultiIndex) and df.columns.nlevels > 2:
            df.columns = pd.MultiIndex.from_tuples([(a, b) for a, b, *_ in df.columns.to_flat_index()])
        # Ensure consistent order of fields if present
        # (Some tickers may lack Adj Close; keep what's available)
        # Sort columns by ticker then field
        df = df.sort_index(axis=1, level=[0, 1])
        return df

    # Single-index columns (single ticker)
    cols_lower = {str(c).lower(): c for c in df.columns}
    # Try to infer a single ticker name
    tickers = list(tickers)
    tkr = tickers[0] if tickers else "TICKER"
    # Keep any close/open/high/low/volume columns present
    present = [c for c in df.columns if any(k in str(c).lower() for k in ["open", "high", "low", "close", "adj close", "adj_close", "volume"])]
    df = df[present]
    # Build MultiIndex columns
    new_cols = []
    for c in df.columns:
        name = str(c)
        # Canonicalize common variations
        lc = name.lower().replace("adj_close", "adj close")
        if "open" in lc:
            f = "Open"
        elif "high" in lc:
            f = "High"
        elif "low" in lc:
            f = "Low"
        elif "adj close" in lc:
            f = "Adj Close"
        elif "close" in lc:
            f = "Close"
        elif "volume" in lc:
            f = "Volume"
        else:
            f = name
        new_cols.append((tkr, f))
    df.columns = pd.MultiIndex.from_tuples(new_cols, names=["ticker", "field"])
    return df


def fetch_ohlc_yf(
    tickers: Iterable[str],
    start: Optional[pd.Timestamp | str] = None,
    end: Optional[pd.Timestamp | str] = None,
    period: Optional[str] = None,
    interval: str = INTERVAL,
    strict: bool = True,
) -> pd.DataFrame:
    """Download OHLCV for tickers, return MultiIndex columns (ticker, field).
    Ascending index. Robust to yfinance's differing layouts.
    """
    kwargs = {
        "interval": interval,
        "group_by": "ticker",
        "auto_adjust": False,
        "progress": False,
        "threads": True,
    }
    if period is not None:
        raw = yf.download(list(tickers), period=period, **kwargs)
    else:
        raw = yf.download(list(tickers), start=start, end=end, **kwargs)

    if raw is None or len(raw) == 0:
        raise RuntimeError("yfinance returned no data (OHLC). Check symbols/interval or try later.")

    ohlc = _standardize_ohlc_columns(raw, tickers)
    # Drop tickers with all-NaN across fields
    # Determine present tickers from level 0
    present_tkrs = list(dict.fromkeys(ohlc.columns.get_level_values(0)))
    pruned = []
    for t in present_tkrs:
        sub = ohlc[t]
        if sub.dropna(how="all").empty:
            continue
        pruned.append(t)
    ohlc = ohlc.loc[:, ohlc.columns.get_level_values(0).isin(pruned)]

    missing = [t for t in tickers if t not in pruned]
    if missing and strict:
        raise RuntimeError(f"Missing OHLC for tickers: {missing}")

    return ohlc


def ohlc_to_long(ohlc: pd.DataFrame) -> pd.DataFrame:
    """Convert (ticker, field) columns to long format using the *new* pandas
    stack implementation when available (future_stack=True) to silence the
    deprecation warning. Output columns: date, ticker, fields... (lower_snake).
    """
    df = ohlc.copy()
    # Ensure clean index and canonical lowercase field names with underscores
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.columns = pd.MultiIndex.from_tuples(
        [
            (t, str(f).lower().replace(" ", "_"))
            for t, f in df.columns.to_flat_index()
        ],
        names=["ticker", "field"],
    )

    # Use the future stack when available (pandas >= 2.1). Fallback otherwise.
    try:
        long = df.stack(level="ticker", future_stack=True)
    except TypeError:
        long = df.stack(level="ticker")

    # The stacked index is (date, ticker); convert to columns
    long.index.names = ["date", "ticker"]
    long = long.reset_index()

    # Order columns if present
    preferred = [
        "date", "ticker", "open", "high", "low", "close", "adj_close", "volume"
    ]
    ordered = [c for c in preferred if c in long.columns] + [c for c in long.columns if c not in preferred]
    return long[ordered]


def extract_close_from_ohlc(ohlc: pd.DataFrame) -> pd.DataFrame:
    """Wide Close matrix (ascending). Falls back to Adj Close if Close absent for a ticker."""
    tkrs = list(dict.fromkeys(ohlc.columns.get_level_values(0)))
    pieces = []
    for t in tkrs:
        fields = list(dict.fromkeys(ohlc[t].columns))
        if "Close" in fields:
            s = ohlc[(t, "Close")]
        elif "Adj Close" in fields:
            s = ohlc[(t, "Adj Close")]
        else:
            continue
        s = s.rename(t)
        pieces.append(s)
    if not pieces:
        raise RuntimeError("No Close/Adj Close found for any ticker.")
    close = pd.concat(pieces, axis=1)
    return close


def compute_returns_df(close_asc: pd.DataFrame, kind: str = "log") -> pd.DataFrame:
    if kind == "log":
        return np.log(close_asc / close_asc.shift(1)).dropna(how="all")
    elif kind == "simple":
        return close_asc.pct_change().dropna(how="all")
    else:
        raise ValueError("kind must be 'log' or 'simple'")


def _try_parquet(df: pd.DataFrame, path: Path) -> Optional[Path]:
    try:
        df.to_parquet(path)
        return path
    except Exception:
        return None


def save_ohlc_and_returns(
    tickers: Dict[str, float] | Iterable[str],
    base_dir: Path,
    lookback_years: int = LOOKBACK_YEARS,
    interval: str = INTERVAL,
    save_returns: bool = True,
    period: Optional[str] = None,
) -> dict:
    """Fetch OHLC, save CSVs (and Parquet if available) in data_raw/data_int.
    Returns paths dict.
    """
    base_dir = Path(base_dir)
    (base_dir / "data_raw").mkdir(parents=True, exist_ok=True)
    (base_dir / "data_int").mkdir(parents=True, exist_ok=True)

    end = pd.Timestamp(date.today())
    start = end - pd.DateOffset(years=lookback_years)

    tlist = list(tickers.keys()) if isinstance(tickers, dict) else list(tickers)

    ohlc = fetch_ohlc_yf(tlist, start=start, end=end, period=period, interval=interval)
    close_asc = extract_close_from_ohlc(ohlc)

    # Save OHLC wide and long
    ohlc_csv = base_dir / "data_raw" / "ohlc_wide.csv"
    ohlc.to_csv(ohlc_csv)
    ohlc_pq = _try_parquet(ohlc, base_dir / "data_raw" / "ohlc_wide.parquet")

    long = ohlc_to_long(ohlc)
    long_csv = base_dir / "data_raw" / "ohlc_long.csv"
    long.to_csv(long_csv, index=False)
    long_pq = _try_parquet(long, base_dir / "data_raw" / "ohlc_long.parquet")

    close_csv = base_dir / "data_int" / "close.csv"
    close_asc.to_csv(close_csv)
    close_pq = _try_parquet(close_asc, base_dir / "data_int" / "close.parquet")

    paths = {
        "ohlc_wide_csv": ohlc_csv,
        "ohlc_wide_parquet": ohlc_pq,
        "ohlc_long_csv": long_csv,
        "ohlc_long_parquet": long_pq,
        "close_csv": close_csv,
        "close_parquet": close_pq,
    }

    if save_returns:
        logret = compute_returns_df(close_asc, kind="log")
        simp = compute_returns_df(close_asc, kind="simple")
        lr_csv = base_dir / "data_int" / "log_returns.csv"
        sr_csv = base_dir / "data_int" / "simple_returns.csv"
        logret.to_csv(lr_csv)
        simp.to_csv(sr_csv)
        paths.update({
            "log_returns_csv": lr_csv,
            "simple_returns_csv": sr_csv,
            "log_returns_parquet": _try_parquet(logret, base_dir / "data_int" / "log_returns.parquet"),
            "simple_returns_parquet": _try_parquet(simp, base_dir / "data_int" / "simple_returns.parquet"),
        })

    return paths


# ── Simple usage example (call these from your notebook/script) ───────────────
if __name__ == "__main__":
    # Minimal smoke test: SPY and QQQ with tiny weights
    mapping = {"SPY": 0.0167, "QQQ": 0.0167}
    res = run_weighted_port_basket(mapping)
    print(res)

    # Diagnostics example on one ticker
    close_asc = fetch_close_yf(["SPY"], period="5y")
    s_desc = to_descending(close_asc)["SPY"].dropna()
    r_desc = descending_returns(s_desc, kind="simple")
    pv = rolling_predicted_vol_ewma_blend(r_desc)
    rv20 = rolling_realized_vol_desc(r_desc, k=20)
    te = tracking_error_series(pv, rv20)
    print({"rv20_mae_vs_pred": te.mean(), "rv20_median": rv20.median(), "pv_median": pv.median()})

    # OHLC artifacts smoke test
    from pathlib import Path
    from datetime import date
    run_date = date.today().strftime("%d-%m-%Y")
    base = Path.cwd() / run_date
    out_paths = save_ohlc_and_returns(["SPY", "QQQ"], base, lookback_years=4)
    print(out_paths)
