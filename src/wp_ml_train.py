from __future__ import annotations
"""
SyStrat · Step 4 — Model Training & Tuning (ElasticNet + GradientBoosting)

Reads Step 3.2 artifacts (X_train/val/test_k{K}, y_*_k{K}), runs time‑safe CV to tune
ElasticNet and GradientBoostingRegressor, fits final models, evaluates on VAL/TEST,
saves models, metrics, and prediction files under data_int/ml/.

Targets are daily‑equivalent realized volatility (non‑negative). We clip predictions
at a tiny epsilon and optionally apply a robust scalar calibrator α = median(y/pred)
(when enabled) to align risk scale to the legacy model.

No new dependencies; adheres to the user's environment hygiene.
"""

from pathlib import Path  # no installation needed
from dataclasses import dataclass  # no installation needed
from typing import Dict, Tuple, List  # no installation needed
import json  # no installation needed
import math  # no installation needed
import numpy as np  # already in env — no new install
import pandas as pd  # already in env — no new install
from sklearn.linear_model import ElasticNet  # already in env — no new install
from sklearn.ensemble import GradientBoostingRegressor  # already in env — no new install
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # already in env — no new install
import matplotlib.pyplot as plt  # already in env — no new install
from joblib import load  # already in env — no new install
from joblib import dump  # already in env — no new install

# --- Optional XGBoost support (only if installed) ---
try:
    import xgboost as xgb  # NEW DEPENDENCY if you choose to use it
    HAS_XGB = True
except Exception:
    HAS_XGB = False



EPS = 1e-12

# --- Optimized params from last good CV run (no new installs) ---
OPT_EN = dict(alpha=1e-4, l1_ratio=0.95, fit_intercept=True)  # scikit-learn ElasticNet # already in env — no new install
OPT_GBM = dict(                                             # scikit-learn GradientBoostingRegressor # already in env — no new install
    n_estimators=300, learning_rate=0.03, max_depth=3,
    subsample=0.70, min_samples_leaf=20
)



@dataclass
class Step4Config:
    k_label: int = 10
    n_splits: int = 4  # time-series folds on TRAIN only
    gap: int = 0       # optional gap (rows) between train/val in CV
    use_alpha_calibrator: bool = True
    random_state: int = 17
    mode: str = 'cv'  # 'cv' (tune) or 'opt' (use OPT_EN / OPT_GBM)



def _ml_dir(base_dir: Path) -> Path:
    return Path(base_dir) / 'data_int' / 'ml'


def _try_read(path: Path) -> pd.DataFrame:
    """Read Parquet if exists else CSV; preserve dtypes; do not parse dates implicitly."""
    pqt = path.with_suffix('.parquet')
    csv = path.with_suffix('.csv')
    if pqt.exists():
        return pd.read_parquet(pqt)
    elif csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Neither {pqt.name} nor {csv.name} found at {path.parent}")


def load_splits(base_dir: Path, k: int) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load X/y for TRAIN, VAL, TEST for horizon k."""
    mlp = _ml_dir(base_dir)
    Xtr = _try_read(mlp / f'X_train_k{k}')
    ytr = _try_read(mlp / f'y_train_k{k}')
    Xva = _try_read(mlp / f'X_val_k{k}')
    yva = _try_read(mlp / f'y_val_k{k}')
    Xte = _try_read(mlp / f'X_test_k{k}')
    yte = _try_read(mlp / f'y_test_k{k}')

    # y may come as DF; squeeze to Series with a stable name
    if isinstance(ytr, pd.DataFrame):
        ytr = ytr.iloc[:, 0]
    if isinstance(yva, pd.DataFrame):
        yva = yva.iloc[:, 0]
    if isinstance(yte, pd.DataFrame):
        yte = yte.iloc[:, 0]

    return Xtr, ytr, Xva, yva, Xte, yte


def _extract_dates(X: pd.DataFrame) -> pd.Series:
    """Best‑effort to obtain a datetime series aligned to rows for time CV."""
    if 'date' in X.columns:
        dt = pd.to_datetime(X['date'], errors='coerce')
    elif 'Date' in X.columns:
        dt = pd.to_datetime(X['Date'], errors='coerce')
    else:
        # Fall back to a monotone index; still preserves order
        dt = pd.Series(np.arange(len(X)), index=X.index, name='row_order')
    return dt


def make_time_folds(dates: pd.Series, n_splits: int = 4, gap: int = 0) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Construct expanding‑window time folds. Dates may be integers (row order)."""
    # Sort by date while preserving stable order for ties
    order = np.argsort(dates.values)
    N = len(dates)
    fold_sizes = np.linspace(0.5, 1.0, n_splits + 1)[1:]  # last val -> full train
    folds = []
    for frac in fold_sizes[:-1]:
        split = int(frac * N)
        # Train: [0 : split - gap), Val: [split : next_split)
        next_split = int((frac + (1.0 - fold_sizes[0]) / n_splits) * N)
        tr_idx = order[: max(0, split - gap)]
        va_idx = order[split: max(split, next_split)]
        if len(tr_idx) > 0 and len(va_idx) > 0:
            folds.append((tr_idx, va_idx))
    # Ensure at least one fold
    if not folds:
        split = int(0.8 * N)
        folds = [(order[:split], order[split:])]
    return folds


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.clip(np.asarray(y_pred), EPS, None)
    denom = np.maximum(np.abs(y_true), EPS)
    return float(np.mean(np.abs(y_true - y_pred) / denom))


def _spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank corr without SciPy."""
    a = pd.Series(y_true).rank(method='average').to_numpy()
    b = pd.Series(y_pred).rank(method='average').to_numpy()
    if np.std(a) < EPS or np.std(b) < EPS:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_pred = np.clip(y_pred, EPS, None)
    return {
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'RMSE': float(math.sqrt(mean_squared_error(y_true, y_pred))),
        'MAPE': _mape(y_true, y_pred),
        'R2': float(r2_score(y_true, y_pred)),
    }

def robust_alpha(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = np.clip(y_pred, EPS, None)
    ratios = y_true / y_pred
    # Guard against inf/NaN; use median for robustness
    ratios = ratios[np.isfinite(ratios)]
    if ratios.size == 0:
        return 1.0
    return float(np.median(ratios))


def tune_elasticnet(X: pd.DataFrame, y: pd.Series, dates: pd.Series, n_splits: int = 4) -> Dict:
    alphas = np.logspace(-4, 1, 12)
    l1s = [0.05, 0.2, 0.5, 0.8, 0.95]
    folds = make_time_folds(dates, n_splits=n_splits)
    best = None
    for a in alphas:
        for l1 in l1s:
            maes = []
            mapes = []
            for tr, va in folds:
                mdl = ElasticNet(alpha=a, l1_ratio=l1, fit_intercept=True, max_iter=5000, tol=1e-4, random_state=17)
                mdl.fit(X.iloc[tr, :], y.iloc[tr])
                pred = np.clip(mdl.predict(X.iloc[va, :]), EPS, None)
                mapes.append(_mape(y.iloc[va].values, pred))
                maes.append(mean_absolute_error(y.iloc[va].values, pred))
            score = float(np.mean(mapes))  # primary selection metric
            if (best is None) or (score < best['score']):
                best = {'alpha': float(a), 'l1_ratio': float(l1), 'score': score, 'mae': float(np.mean(maes))}
    return best


def tune_gbm(X: pd.DataFrame, y: pd.Series, dates: pd.Series, n_splits: int = 4, random_state: int = 17) -> Dict:
    grid = {
        'n_estimators': [300, 600, 900],
        'learning_rate': [0.03, 0.06, 0.1],
        'max_depth': [2, 3],
        'subsample': [0.7, 1.0],
        'min_samples_leaf': [5, 20],
    }
    keys = list(grid.keys())
    folds = make_time_folds(dates, n_splits=n_splits)
    best = None
    # iterate cartesian grid
    from itertools import product  # no installation needed
    for vals in product(*[grid[k] for k in keys]):
        params = dict(zip(keys, vals))
        maes, mapes = [], []
        for tr, va in folds:
            mdl = GradientBoostingRegressor(random_state=random_state, **params)
            mdl.fit(X.iloc[tr, :], y.iloc[tr])
            pred = np.clip(mdl.predict(X.iloc[va, :]), EPS, None)
            mapes.append(_mape(y.iloc[va].values, pred))
            maes.append(mean_absolute_error(y.iloc[va].values, pred))
        score = float(np.mean(mapes))
        if (best is None) or (score < best['score']):
            best = {'params': params, 'score': score, 'mae': float(np.mean(maes))}
    return best

def tune_xgb(X: pd.DataFrame, y: pd.Series, dates: pd.Series, n_splits: int = 4, random_state: int = 17) -> Dict:
    if not HAS_XGB:
        return {'params': None, 'score': float('inf'), 'mae': float('inf')}
    grid = {
        'n_estimators': [400, 700],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 4],
        'subsample': [0.7, 1.0],
        'colsample_bytree': [0.7, 1.0],
        'min_child_weight': [1, 5],
        'reg_alpha': [0.0, 0.1],    # L1
        'reg_lambda': [1.0, 5.0],   # L2
    }
    keys = list(grid.keys())
    folds = make_time_folds(dates, n_splits=n_splits)
    best = None
    from itertools import product
    for vals in product(*[grid[k] for k in keys]):
        params = dict(zip(keys, vals))
        maes, mapes = [], []
        for tr, va in folds:
            mdl = xgb.XGBRegressor(
                objective='reg:squarederror',
                tree_method='hist',
                n_jobs=-1,
                random_state=random_state,
                **params
            )
            mdl.fit(X.iloc[tr, :], y.iloc[tr])
            pred = np.clip(mdl.predict(X.iloc[va, :]), EPS, None)
            mapes.append(_mape(y.iloc[va].values, pred))
            maes.append(mean_absolute_error(y.iloc[va].values, pred))
        score = float(np.mean(mapes))
        if (best is None) or (score < best['score']):
            best = {'params': params, 'score': score, 'mae': float(np.mean(maes))}
    return best


def fit_final_models(base_dir: Path, cfg: Step4Config) -> Dict:
    """Main orchestration for Step 4. Returns a summary registry."""
    Xtr, ytr, Xva, yva, Xte, yte = load_splits(base_dir, cfg.k_label)

    # Ensure numeric arrays only (drop non‑feature columns if present)
    def _get_feature_cols(df: pd.DataFrame) -> List[str]:
        return [c for c in df.columns if c.lower() not in {'date', 'ticker'}]

    fcols = _get_feature_cols(Xtr)
    XtrN, XvaN, XteN = Xtr[fcols], Xva[fcols], Xte[fcols]

    # Dates for time folds (TRAIN only)
    dates_tr = _extract_dates(Xtr)

    # --- Tune ---
    best_en = tune_elasticnet(XtrN, ytr, dates_tr, n_splits=cfg.n_splits)
    best_gbm = tune_gbm(XtrN, ytr, dates_tr, n_splits=cfg.n_splits, random_state=cfg.random_state)
    best_xgb = tune_xgb(XtrN, ytr, dates_tr, n_splits=cfg.n_splits, random_state=cfg.random_state)

    if cfg.mode == 'opt':
        best_en  = {'alpha': OPT_EN['alpha'], 'l1_ratio': OPT_EN['l1_ratio'], 'score': None, 'mae': None}
        best_gbm = {'params': dict(OPT_GBM), 'score': None, 'mae': None}
        best_xgb = {'params': None, 'score': float('inf'), 'mae': float('inf')}

    # --- Fit on TRAIN+VAL with best params ---
    Xtrval = pd.concat([XtrN, XvaN], axis=0)
    ytrval = pd.concat([ytr, yva], axis=0)

    if cfg.mode == 'opt':
        en = ElasticNet(**{**OPT_EN, 'max_iter': 5000, 'tol': 1e-4, 'random_state': cfg.random_state})
    else:
        en = ElasticNet(alpha=best_en['alpha'], l1_ratio=best_en['l1_ratio'],
                        fit_intercept=True, max_iter=5000, tol=1e-4,
                        random_state=cfg.random_state)
    en.fit(Xtrval, ytrval)

    if cfg.mode == 'opt':
        gbm = GradientBoostingRegressor(random_state=cfg.random_state, **OPT_GBM)
    else:
        gbm = GradientBoostingRegressor(random_state=cfg.random_state, **best_gbm['params'])
    gbm.fit(Xtrval, ytrval)

    if HAS_XGB and best_xgb.get('params') is not None:
        xgbm = xgb.XGBRegressor(
            objective='reg:squarederror',
            tree_method='hist',
            n_jobs=-1,
            random_state=cfg.random_state,
            **best_xgb['params']
        )
        xgbm.fit(Xtrval, ytrval)
    else:
        xgbm = None


    # --- Predict ---
    preds = {}
    preds['val_en'] = np.clip(en.predict(XvaN), EPS, None)
    preds['val_gbm'] = np.clip(gbm.predict(XvaN), EPS, None)
    preds['test_en'] = np.clip(en.predict(XteN), EPS, None)
    preds['test_gbm'] = np.clip(gbm.predict(XteN), EPS, None)
    
    if xgbm is not None:
        preds['val_xgb']  = np.clip(xgbm.predict(XvaN), EPS, None)
        preds['test_xgb'] = np.clip(xgbm.predict(XteN), EPS, None)



    # --- Optional α calibrator (fit on TRAIN+VAL) ---
    alpha_en = alpha_gbm = alpha_xgb = 1.0
    if cfg.use_alpha_calibrator:
        alpha_en  = robust_alpha(ytrval.values, np.clip(en.predict(Xtrval),  EPS, None))
        alpha_gbm = robust_alpha(ytrval.values, np.clip(gbm.predict(Xtrval), EPS, None))
        preds['val_en']  *= alpha_en
        preds['test_en'] *= alpha_en
        preds['val_gbm'] *= alpha_gbm
        preds['test_gbm']*= alpha_gbm
    
        if xgbm is not None:
            alpha_xgb = robust_alpha(ytrval.values, np.clip(xgbm.predict(Xtrval), EPS, None))
            preds['val_xgb']  *= alpha_xgb
            preds['test_xgb'] *= alpha_xgb
    
        

    # --- Metrics ---
    metrics = {
        'VAL_EN': compute_metrics(yva.values, preds['val_en']),
        'VAL_GBM': compute_metrics(yva.values, preds['val_gbm']),
        'TEST_EN': compute_metrics(yte.values, preds['test_en']),
        'TEST_GBM': compute_metrics(yte.values, preds['test_gbm']),
        'CV_BEST_EN': best_en,
        'CV_BEST_GBM': best_gbm,
        'ALPHA_EN': alpha_en,
        'ALPHA_GBM': alpha_gbm,
    }

    if xgbm is not None:
        metrics['VAL_XGB'] = compute_metrics(yva.values, preds['val_xgb'])
        metrics['TEST_XGB'] = compute_metrics(yte.values, preds['test_xgb'])
        metrics['CV_BEST_XGB'] = best_xgb
        metrics['ALPHA_XGB'] = alpha_xgb


    # --- Save outputs ---
    outdir = _ml_dir(base_dir)
    (outdir / 'models').mkdir(parents=True, exist_ok=True)

    dump(en, outdir / f'models/elasticnet_k{cfg.k_label}.pkl')
    dump(gbm, outdir / f'models/gbm_k{cfg.k_label}.pkl')

    if xgbm is not None:
        # save as JSON for portability
        xgb_path = outdir / f'models/xgb_k{cfg.k_label}.json'
        xgbm.save_model(str(xgb_path))

    # Save metrics and a small manifest
    manifest = {
        'k_label': cfg.k_label,
        'feature_columns': fcols,
        'best_en': best_en,
        'best_gbm': best_gbm,
        'alpha_en': alpha_en,
        'alpha_gbm': alpha_gbm,
        'best_xgb': best_xgb if HAS_XGB else None,
        'alpha_xgb': alpha_xgb if HAS_XGB and best_xgb.get('params') is not None else None,
    }
    (outdir / f'metrics_k{cfg.k_label}.json').write_text(json.dumps(metrics, indent=2))
    (outdir / f'model_manifest_k{cfg.k_label}.json').write_text(json.dumps(manifest, indent=2))

    # Save predictions for audit
    cols_val = {'y_true': yva.values, 'pred_en': preds['val_en'], 'pred_gbm': preds['val_gbm']}
    if 'val_xgb' in preds:
        cols_val['pred_xgb'] = preds['val_xgb']
    df_val = pd.DataFrame(cols_val)
    if 'ticker' in Xva.columns: df_val['ticker'] = Xva['ticker'].values
    if 'date' in Xva.columns:   df_val['date']   = Xva['date'].values

    cols_test = {'y_true': yte.values, 'pred_en': preds['test_en'], 'pred_gbm': preds['test_gbm']}
    if 'test_xgb' in preds:
        cols_test['pred_xgb'] = preds['test_xgb']
    df_test = pd.DataFrame(cols_test)
    if 'ticker' in Xte.columns: df_test['ticker'] = Xte['ticker'].values
    if 'date' in Xte.columns:   df_test['date']   = Xte['date'].values

    df_val.to_parquet(outdir / f'preds_val_k{cfg.k_label}.parquet')
    df_test.to_parquet(outdir / f'preds_test_k{cfg.k_label}.parquet')

    return metrics


# --- Reporting helpers ---

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _decile_table(y: np.ndarray, yhat: np.ndarray, n: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({'y_true': y, 'y_pred': yhat})
    # rank predictions to reduce ties bias
    r = df['y_pred'].rank(method='first')
    # try qcut → equal-count bins; fallback to fewer bins if data too small / low variety
    bins = n
    while bins >= 2:
        try:
            df['decile'] = pd.qcut(r, q=bins, labels=list(range(1, bins + 1)))
            break
        except Exception:
            bins -= 1
    if 'decile' not in df.columns:
        # degenerate case: all predictions identical
        df['decile'] = 1

    g = df.groupby('decile', observed=True)
    base = g.agg(count=('y_true', 'size'),
                 mean_y=('y_true', 'mean'),
                 mean_pred=('y_pred', 'mean'))
    errs = g.apply(lambda d: pd.Series({
        'mae': float(np.mean(np.abs(d['y_true'] - d['y_pred']))),
        'mape': float(np.mean(np.abs(d['y_true'] - d['y_pred']) / np.maximum(np.abs(d['y_true']), EPS)))
    }))
    out = base.join(errs)
    return out.reset_index()

def _calibration_slope(y: np.ndarray, yhat: np.ndarray) -> float:
    yhat = np.asarray(yhat)
    y = np.asarray(y)
    denom = float(np.sum(yhat * yhat))
    if denom <= 0:
        return float('nan')
    return float(np.sum(y * yhat) / denom)


def _plot_calibration(deciles: pd.DataFrame, outpath: Path, title: str) -> None:
    plt.figure(figsize=(6, 6))
    x = deciles['mean_pred'].to_numpy()
    y = deciles['mean_y'].to_numpy()
    plt.scatter(x, y)
    lim = [0, max(1e-8, x.max() if x.size else 0.0, y.max() if y.size else 0.0)]
    plt.plot(lim, lim)
    plt.xlabel('Predicted (decile mean)')
    plt.ylabel('Actual (decile mean)')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=140)
    plt.close()


def _plot_residuals(y: np.ndarray, yhat: np.ndarray, outpath: Path, title: str) -> None:
    err = np.asarray(y) - np.asarray(yhat)
    plt.figure(figsize=(7, 4))
    plt.hist(err, bins=40)
    plt.title(title)
    plt.xlabel('Residual (y - yhat)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(outpath, dpi=140)
    plt.close()


def _by_ticker_metrics(df_pred: pd.DataFrame, pred_col: str = 'pred_gbm') -> pd.DataFrame:
    if 'ticker' not in df_pred.columns or pred_col not in df_pred.columns:
        return pd.DataFrame()
    grp = df_pred.groupby('ticker', dropna=False)
    out = grp.apply(lambda d: pd.Series({
        'count': int(len(d)),
        'MAE': float(np.mean(np.abs(d['y_true'] - d[pred_col]))),
        'MAPE': float(np.mean(np.abs(d['y_true'] - d[pred_col]) / np.maximum(np.abs(d['y_true']), EPS))),
        'RMSE': float(np.sqrt(np.mean((d['y_true'] - d[pred_col]) ** 2))),
    }))
    return out.sort_values('MAE', ascending=False).reset_index()


def make_postrun_report(base_dir: Path, k_label: int, model_for_report: str = 'gbm',importance_from: str = None) -> Dict:
    if importance_from is None:
        importance_from = model_for_report.lower()
    mlp = _ml_dir(base_dir)
    rep = mlp / f'reports_k{k_label}'
    _model = model_for_report.lower()
    pred_col = {'gbm': 'pred_gbm', 'en': 'pred_en', 'xgb': 'pred_xgb'}.get(_model, 'pred_gbm')
    _ensure_dir(rep)

    # Load predictions
    df_val = pd.read_parquet(mlp / f'preds_val_k{k_label}.parquet')
    df_test = pd.read_parquet(mlp / f'preds_test_k{k_label}.parquet')

    # Deciles
    dec_val = _decile_table(df_val['y_true'].values, df_val[pred_col].values)
    dec_test = _decile_table(df_test['y_true'].values, df_test[pred_col].values)
    dec_val.to_csv(rep / 'deciles_val.csv', index=False)
    dec_test.to_csv(rep / 'deciles_test.csv', index=False)

    # Plots
    _plot_calibration(dec_val, rep / 'calibration_val.png', f'Calibration (VAL, {model_for_report.upper()})')
    _plot_calibration(dec_test, rep / 'calibration_test.png', f'Calibration (TEST, {model_for_report.upper()})')
    _plot_residuals(df_val['y_true'].values, df_val[pred_col].values, rep / 'residuals_val.png', f'Residuals (VAL, {model_for_report.upper()})')
    _plot_residuals(df_test['y_true'].values, df_test[pred_col].values, rep / 'residuals_test.png', f'Residuals (TEST, {model_for_report.upper()})')

    # Rolling MAPE (TEST) — 60-day window
    def _plot_rolling_mape(y, yhat, outpath: Path, w: int = 60):
        y = np.asarray(y); yhat = np.asarray(yhat)
        denom = np.maximum(np.abs(y), EPS)
        ae = np.abs(y - yhat) / denom
        roll = pd.Series(ae).rolling(w, min_periods=max(5, w//3)).mean()
        plt.figure(figsize=(8, 3.5))
        plt.plot(roll.values)
        plt.title(f'Rolling MAPE (w={w}) — TEST')
        plt.xlabel('Observation')
        plt.ylabel('MAPE')
        plt.tight_layout()
        plt.savefig(outpath, dpi=140); plt.close()

    _plot_rolling_mape(df_test['y_true'].values, df_test[pred_col].values,
                       rep / 'rolling_mape_test.png', w=60)

    # Overlay actual vs predicted (TEST)
    def _plot_overlay(y, yhat, outpath: Path, title: str):
        plt.figure(figsize=(9, 3.5))
        plt.plot(y, label='actual')
        plt.plot(yhat, label='pred', alpha=0.9)
        plt.title(title)
        plt.xlabel('Observation')
        plt.ylabel('Vol (daily-equiv)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(outpath, dpi=140); plt.close()

    _plot_overlay(df_test['y_true'].values, df_test[pred_col].values,
                  rep / 'overlay_test.png', f'Actual vs Predicted (TEST, {model_for_report.upper()})')

    # Lifts & correlations
    lift_val = float(dec_val.loc[dec_val['decile'] == dec_val['decile'].max(), 'mean_y'].iloc[0] / dec_val['mean_y'].mean()) if not dec_val.empty else float('nan')
    lift_test = float(dec_test.loc[dec_test['decile'] == dec_test['decile'].max(), 'mean_y'].iloc[0] / dec_test['mean_y'].mean()) if not dec_test.empty else float('nan')
    spear_val = _spearman(df_val['y_true'].values, df_val[pred_col].values)
    spear_test = _spearman(df_test['y_true'].values, df_test[pred_col].values)
    slope_val = _calibration_slope(df_val['y_true'].values, df_val[pred_col].values)
    slope_test = _calibration_slope(df_test['y_true'].values, df_test[pred_col].values)

    # By-ticker table (GBM shown; EN similar if needed)
    by_ticker_val = _by_ticker_metrics(df_val, pred_col)
    by_ticker_test = _by_ticker_metrics(df_test, pred_col)
    if not by_ticker_val.empty:
        by_ticker_val.to_csv(rep / 'by_ticker_val.csv', index=False)
    if not by_ticker_test.empty:
        by_ticker_test.to_csv(rep / 'by_ticker_test.csv', index=False)

    def _emit_importance(rep: Path, mlp: Path, k: int, which: str, fcols: list):
        which = which.lower()
        out = {}
        if which in ('gbm','all'):
            gbm = load(mlp / f'models/gbm_k{k}.pkl')
            fi = pd.DataFrame({'feature': fcols,
                               'importance': getattr(gbm, 'feature_importances_', np.zeros(len(fcols)))})
            fi.sort_values('importance', ascending=False, inplace=True)
            fi.to_csv(rep / 'feature_importance_gbm.csv', index=False)
            top = fi.head(20)
            plt.figure(figsize=(8,6)); plt.barh(top['feature'][::-1], top['importance'][::-1])
            plt.title('GBM Feature Importance (Top 20)'); plt.tight_layout()
            plt.savefig(rep / 'feature_importance_gbm.png', dpi=140); plt.close()
            out['feature_importance_gbm_csv'] = str(rep / 'feature_importance_gbm.csv')
            out['feature_importance_gbm_png'] = str(rep / 'feature_importance_gbm.png')
    
        if which in ('en','all'):
            en = load(mlp / f'models/elasticnet_k{k}.pkl')
            coefs = pd.DataFrame({'feature': fcols, 'coef': getattr(en, 'coef_', np.zeros(len(fcols)))})
            coefs['abs_coef'] = coefs['coef'].abs()
            coefs.sort_values('abs_coef', ascending=False, inplace=True)
            coefs.to_csv(rep / 'elasticnet_coefs.csv', index=False)
            top = coefs.head(20)
            plt.figure(figsize=(8,6)); plt.barh(top['feature'][::-1], top['abs_coef'][::-1])
            plt.title('ElasticNet |abs(coef)| (Top 20)'); plt.tight_layout()
            plt.savefig(rep / 'elasticnet_coefs.png', dpi=140); plt.close()
            out['elasticnet_coefs_csv'] = str(rep / 'elasticnet_coefs.csv')
            out['elasticnet_coefs_png'] = str(rep / 'elasticnet_coefs.png')
    
        if which in ('xgb','all'):
            try:
                import xgboost as xgb
                booster = xgb.Booster()
                booster.load_model(str(mlp / f'models/xgb_k{k}.json'))
                # ‘gain’ is usually the most informative; you can choose 'weight' or 'total_gain'
                gain = booster.get_score(importance_type='gain')
                # Map f0,f1,.. to column names
                fmap = {f'f{i}': name for i, name in enumerate(fcols)}
                fi = pd.DataFrame([{'feature': fmap.get(k, k), 'importance': v} for k, v in gain.items()])
                if fi.empty:
                    fi = pd.DataFrame({'feature': fcols, 'importance': 0.0})
                fi.sort_values('importance', ascending=False, inplace=True)
                fi.to_csv(rep / 'feature_importance_xgb.csv', index=False)
                top = fi.head(20)
                plt.figure(figsize=(8,6)); plt.barh(top['feature'][::-1], top['importance'][::-1])
                plt.title('XGB Feature Importance (gain, Top 20)'); plt.tight_layout()
                plt.savefig(rep / 'feature_importance_xgb.png', dpi=140); plt.close()
                out['feature_importance_xgb_csv'] = str(rep / 'feature_importance_xgb.csv')
                out['feature_importance_xgb_png'] = str(rep / 'feature_importance_xgb.png')
            except Exception:
                pass
        return out


    # Feature importances (selectable)
    summary_paths = {}
    try:
        manifest = json.loads((mlp / f'model_manifest_k{k_label}.json').read_text())
        fcols = manifest['feature_columns']
        summary_paths.update(_emit_importance(rep, mlp, k_label, importance_from, fcols))
    except Exception:
        pass


    summary = {
        'k_label': k_label,
        'paths': {
            'deciles_val': str(rep / 'deciles_val.csv'),
            'deciles_test': str(rep / 'deciles_test.csv'),
            'calibration_val_png': str(rep / 'calibration_val.png'),
            'calibration_test_png': str(rep / 'calibration_test.png'),
            'residuals_val_png': str(rep / 'residuals_val.png'),
            'residuals_test_png': str(rep / 'residuals_test.png'),
            'by_ticker_val_csv': str(rep / 'by_ticker_val.csv') if (rep / 'by_ticker_val.csv').exists() else None,
            'by_ticker_test_csv': str(rep / 'by_ticker_test.csv') if (rep / 'by_ticker_test.csv').exists() else None,
            **summary_paths,
        },
        'val': {
            'top_decile_lift': lift_val,
            'spearman': spear_val,
            'calibration_slope': slope_val,
        },
        'test': {
            'top_decile_lift': lift_test,
            'spearman': spear_test,
            'calibration_slope': slope_test,
        },
    }
    (rep / 'report_summary.json').write_text(json.dumps(summary, indent=2))
    return summary


# --- Convenience runner ---

def run_step4(base_dir: Path, k_label: int = 10, n_splits: int = 4,
              use_alpha_calibrator: bool = True, mode: str = 'cv') -> Dict:
    cfg = Step4Config(k_label=k_label, n_splits=n_splits,
                      use_alpha_calibrator=use_alpha_calibrator, mode=mode)
    return fit_final_models(Path(base_dir), cfg)

def run_step4_with_report(base_dir: Path, k_label: int = 10, n_splits: int = 4, use_alpha_calibrator: bool = True, model_for_report: str = 'gbm', mode: str = 'cv') -> Dict:
    """Train, evaluate, and emit a compact visual+tabular report under data_int/ml/reports_k{K}
    mode: 'cv' → expanding-CV tune (default), 'opt' → use OPT_EN / OPT_GBM fixed params."""
    metrics = run_step4(base_dir, k_label=k_label, n_splits=n_splits, use_alpha_calibrator=use_alpha_calibrator, mode=mode)
    summary = make_postrun_report(base_dir, k_label, model_for_report=model_for_report)
    out = {'metrics': metrics, 'report': summary}
    # also mirror to a single JSON for convenience
    (_ml_dir(base_dir) / f'reports_k{k_label}' / 'all_results.json').write_text(json.dumps(out, indent=2))
    return out


if __name__ == '__main__':
    # Example:
    # from datetime import date
    # RUN_DATE = date.today().strftime('%d-%m-%Y')
    # BASE = Path(r'C:\\Users\\quantbase\\Desktop\\SyStrat') / RUN_DATE
    # metrics = run_step4(BASE, k_label=10, n_splits=4, use_alpha_calibrator=True)
    # print(json.dumps(metrics, indent=2))
    pass
