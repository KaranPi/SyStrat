# src/syslib/semi_markov.py
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

REGIME_ORDER = ("risk_off", "neutral", "risk_on")

# ---------- durations / hazards ----------

def runs_with_age(state_series: pd.Series) -> list[tuple[str, int]]:
    """Return (state, run_length) for consecutive runs in a daily state series."""
    vals = state_series.astype(str).values
    out, i, n = [], 0, len(vals)
    while i < n:
        j = i
        while j + 1 < n and vals[j + 1] == vals[i]:
            j += 1
        out.append((vals[i], j - i + 1))
        i = j + 1
    return out

def estimate_hazards(state_series: pd.Series) -> pd.DataFrame:
    """
    Empirical discrete-time hazard h_s(a) = P(exit at age a | survived to age a)
    Returns: DataFrame[state, age, hazard]
    """
    runs = runs_with_age(state_series)
    by_state: dict[str, list[int]] = {}
    for s, L in runs:
        by_state.setdefault(s, []).append(int(L))

    rows = []
    for s, arr in by_state.items():
        arr = np.asarray(arr, dtype=int)
        maxA = int(arr.max())
        surv = np.array([(arr >= a).sum() for a in range(1, maxA + 1)], dtype=float)
        exits = np.array([(arr == a).sum() for a in range(1, maxA + 1)], dtype=float)
        haz = np.divide(exits, surv, out=np.zeros_like(exits), where=surv > 0)
        rows.append(pd.DataFrame({"state": s, "age": np.arange(1, maxA + 1), "hazard": haz}))
    H = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["state","age","hazard"])
    return H

def smooth_hazards(H: pd.DataFrame, window: int | None = 3, cap: float = 0.99) -> pd.DataFrame:
    """Optional smoothing/capping to avoid spikes; returns same shape."""
    if H.empty or window is None or window <= 1:
        out = H.copy()
    else:
        out = (H.sort_values(["state","age"])
                 .groupby("state", group_keys=False)
                 .apply(lambda g: g.assign(hazard=g["hazard"].rolling(window, min_periods=1).mean())))
    out["hazard"] = out["hazard"].clip(0.0, cap)
    return out

def save_hazards_plot(H: pd.DataFrame, outpath: Path, title: str = "Empirical exit hazard vs. state age"):
    plt.figure(figsize=(8,4))
    for s, g in H.groupby("state"):
        plt.plot(g["age"], g["hazard"], label=s)
    plt.title(title)
    plt.xlabel("Age in state (days)"); plt.ylabel("Exit hazard")
    plt.legend(); plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=150); plt.close()

# ---------- state age / nowcast ----------

def state_age(state_series: pd.Series) -> pd.Series:
    """Age counter (1,2,3,...) within the current run of the state."""
    s = state_series.astype(str)
    age = np.ones(len(s), dtype=int)
    for i in range(1, len(s)):
        age[i] = age[i-1] + 1 if s.iloc[i] == s.iloc[i-1] else 1
    return pd.Series(age, index=s.index, name="age")

def _stay_prob_from_hazard(H: pd.DataFrame, state: str, age_val: int) -> float:
    h = H.loc[H["state"].eq(state) & H["age"].eq(age_val), "hazard"]
    if h.empty:
        # If age beyond observed, use last hazard; if no row, default to small exit prob
        h_state = H.loc[H["state"].eq(state), "hazard"]
        hazard = float(h_state.iloc[-1]) if not h_state.empty else 0.05
    else:
        hazard = float(h.iloc[0])
    return float(np.clip(1.0 - hazard, 0.0, 1.0))  # stay prob

def nowcast_soft_probs(current_state: str, current_age: int,
                       P: pd.DataFrame, H: pd.DataFrame) -> pd.Series:
    """
    Semi-Markov one-step soft state probabilities for 'today':
    - Stay probability = 1 - hazard_s(age)
    - Exit mass distributed across other states ∝ Markov row (excluding self)
    Returns Series over REGIME_ORDER.
    """
    states = list(P.index)
    assert list(P.columns) == states, "P must be square with same index/columns."
    if current_state not in states:
        raise ValueError(f"Unknown state {current_state}; expected one of {states}")

    s = current_state
    stay = _stay_prob_from_hazard(H, s, int(current_age))

    # Conditional next-state distribution given exit: normalize row without diagonal
    row = P.loc[s].astype(float).copy()
    offdiag = row.drop(s)
    mass = float(offdiag.sum())
    if mass <= 0:
        # degenerate: if no exits in P, split evenly across others
        q = pd.Series(1.0 / (len(states) - 1), index=offdiag.index)
    else:
        q = offdiag / mass

    probs = pd.Series(0.0, index=states)
    probs[s] = stay
    probs[q.index] += (1.0 - stay) * q
    return probs

# ---------- convenience I/O ----------

def load_P(base_dir: Path, k_label: int) -> pd.DataFrame:
    P = pd.read_csv(base_dir / "data_int" / "regimes" / f"tier1_transitions_k{k_label}.csv", index_col=0)
    return P.reindex(index=REGIME_ORDER, columns=REGIME_ORDER)

def load_daily_state(base_dir: Path, k_label: int) -> pd.Series:
    daily = pd.read_parquet(base_dir / "data_int" / "regimes" / f"tier1_daily_state_k{k_label}.parquet")["state"]
    return daily

def save_hazards_csv(base_dir: Path, k_label: int, H: pd.DataFrame) -> Path:
    out = base_dir / "data_int" / "regimes" / f"semi_markov_hazards_k{k_label}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    H.to_csv(out, index=False)
    return out
