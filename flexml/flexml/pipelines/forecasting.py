"""
forecasting.py

- High-level run_forecasting(data, config, target=None, ts_col=None, exog_cols=None, freq=None)
- Supports two model families:
    * "rf"      → RandomForestRegressor on lag/rolling/calendar (unchanged behavior)
    * "sarimax" → statsmodels SARIMAX with robust fitting and exogenous handling
- If DataFrame:
    * Infers ts_col (datetime), target (numeric) when not provided
    * Optional exogenous columns (aligned to target)
- Pipeline: grid search → rolling backtest → residual diagnostics → train-on-full → next-horizon forecast
- Returns JSON-serializable dict.
"""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid

# Optional statsmodels support
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    from statsmodels.stats.diagnostic import acorr_ljungbox

    HAVE_STATSMODELS = True
except Exception:  # pragma: no cover
    SARIMAX = None
    ConvergenceWarning = RuntimeError  # dummy
    HAVE_STATSMODELS = False

# -----------------------------------------------------------------------------
# Global determinism
# -----------------------------------------------------------------------------
np.random.seed(42)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
LOG_FMT = "[%(levelname)s] %(asctime)s %(name)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
log = logging.getLogger("run_forecasting")


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def smape(y_true: Iterable[float], y_pred: Iterable[float], eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) + eps
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))


def mase(y_true: Iterable[float], y_pred: Iterable[float], y_insample: Iterable[float],
         m: int = 1, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_insample = np.asarray(y_insample, dtype=float)
    d = np.abs(np.diff(y_insample, n=m)).mean() + eps  # seasonal naive denom
    return float(np.mean(np.abs(y_true - y_pred)) / d)


def _infer_seasonal_period(y: pd.Series) -> int:
    """Heuristic seasonal period for MASE scaling."""
    try:
        freq = pd.infer_freq(y.index)
        if not freq:
            return 1
        off = pd.tseries.frequencies.to_offset(freq)
        base = off.name  # e.g. 'D','H','W-SUN','M'
        if base.startswith("D"):
            return 7
        if base.startswith("H"):
            return 24
        if base.startswith("W"):
            return 52
        if base.startswith("M"):
            return 12
        return 1
    except Exception:
        return 1


# -----------------------------------------------------------------------------
# Calendar features
# -----------------------------------------------------------------------------
@dataclass
class FeatureConfig:
    lags: List[int]
    roll_windows: List[int]
    add_calendar: bool = True


class SloveniaHolidays(AbstractHolidayCalendar):
    rules = [
        Holiday('New Year', month=1, day=1),
        Holiday('New Year 2', month=1, day=2),
        Holiday('Prešern day', month=2, day=8),
        Holiday('Dan upora proti okupatorju', month=4, day=27),
        Holiday('May Day', month=5, day=1),
        Holiday('May Day 2', month=5, day=2),
        Holiday('Binkošti', month=6, day=8),
        Holiday('Statehood Day', month=6, day=25),
        Holiday('Marijino vnebovzetje', month=8, day=15),
        Holiday('Dan reformacije', month=10, day=31),
        Holiday('Dan spomina na mrtve', month=11, day=1),
        Holiday('Božič', month=12, day=25),
        Holiday('Dan samostojnosti in enotnosti', month=12, day=26),
    ]


def _calendar_df(idx: pd.DatetimeIndex) -> pd.DataFrame:
    cal = pd.DataFrame(index=idx)
    cal["dow"] = idx.dayofweek.astype("int8")
    cal["weekofyear"] = idx.isocalendar().week.astype("int16")
    cal["month"] = idx.month.astype("int8")
    cal["quarter"] = idx.quarter.astype("int8")
    cal["day"] = idx.day.astype("int8")
    cal["is_month_start"] = idx.is_month_start.astype("int8")
    cal["is_month_end"] = idx.is_month_end.astype("int8")
    cal["is_weekend"] = (idx.dayofweek >= 5).astype("int8")
    cal["year"] = idx.year.astype("int16")
    holidays = SloveniaHolidays().holidays(idx.min(), idx.max())
    cal["is_holiday"] = idx.isin(holidays).astype("int8")
    return cal


# -----------------------------------------------------------------------------
# RF feature engineering (unchanged behavior)
# -----------------------------------------------------------------------------
def make_features(
        y: pd.Series,
        cfg: FeatureConfig,
        exog: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    assert isinstance(y.index, pd.DatetimeIndex), "y must have a DateTimeIndex"
    X = pd.DataFrame(index=y.index)
    for lag in cfg.lags:
        X[f"lag_{lag}"] = y.shift(lag)
    for w in cfg.roll_windows:
        X[f"roll_mean_{w}"] = y.shift(1).rolling(window=w, min_periods=w).mean()
    if cfg.add_calendar:
        X = X.join(_calendar_df(y.index))
    if exog is not None:
        X = X.join(exog)
    return X


# -----------------------------------------------------------------------------
# Generic helpers (freq/index/exog)
# -----------------------------------------------------------------------------
def _build_future_index(y_index: pd.DatetimeIndex, horizon: int) -> pd.DatetimeIndex:
    freq = y_index.freq or pd.infer_freq(y_index) or "D"
    start = y_index[-1] + to_offset(freq)
    return pd.date_range(start=start, periods=horizon, freq=freq)


def _coerce_regular_freq(
        y: pd.Series,
        exog: Optional[pd.DataFrame],
        freq: Optional[str] = None,
        y_fill: float = 0.0
) -> Tuple[pd.Series, Optional[pd.DataFrame], str]:
    """Ensure DateTimeIndex with an explicit, regular frequency; reindex & fill."""
    idx = y.index
    if not isinstance(idx, (pd.DatetimeIndex, pd.PeriodIndex)):
        idx = pd.to_datetime(idx, errors="coerce", utc=False)
        y = pd.Series(y.values, index=idx, name=y.name)
    f = freq or (y.index.freqstr or pd.infer_freq(y.index))
    if f is None:
        f = "D"
    full = pd.date_range(y.index.min(), y.index.max(), freq=f)
    y2 = y.reindex(full).astype(float).ffill().fillna(y_fill)
    ex2 = None
    if exog is not None:
        if not isinstance(exog.index, (pd.DatetimeIndex, pd.PeriodIndex)):
            exog = exog.copy()
            exog.index = pd.to_datetime(exog.index, errors="coerce", utc=False)
        ex2 = exog.reindex(full)
    return y2, ex2, f


def _ensure_supported_index(y: pd.Series) -> Tuple[pd.Series, str]:
    """Return y with DateTimeIndex carrying a freq string (coerce if needed)."""
    if not isinstance(y.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        y.index = pd.to_datetime(y.index, errors="coerce", utc=False)
    f = (y.index.freqstr or pd.infer_freq(y.index))
    if f is None:
        y, _, f = _coerce_regular_freq(y, None, freq="D", y_fill=0.0)
    return y, f


def _build_exog_future_if_needed(
        exog_cols: List[str] | None,
        y_index: pd.DatetimeIndex,
        horizon: int,
        hist_df: pd.DataFrame | None = None,
) -> Optional[pd.DataFrame]:
    """Synthesize future exog when not supplied: persist last value (binary→0)."""
    if not exog_cols:
        return None
    future_idx = _build_future_index(y_index, horizon)
    exog_future = pd.DataFrame(index=future_idx)
    hist_df = hist_df if hist_df is not None else pd.DataFrame(index=y_index)
    for c in exog_cols:
        series = hist_df[c] if c in hist_df.columns else pd.Series(index=y_index, dtype="float64")
        last_val = series.dropna().iloc[-1] if not series.dropna().empty else None
        is_binary_like = (
                pd.api.types.is_bool_dtype(series) or
                (not series.dropna().empty and pd.Series(pd.unique(series.dropna())).map(lambda v: v in (0, 1)).all())
        )
        if is_binary_like:
            exog_future[c] = 0
        else:
            exog_future[c] = float(last_val) if last_val is not None else 0.0
    return exog_future


# -----------------------------------------------------------------------------
# SARIMAX-specific exog builder
# -----------------------------------------------------------------------------
def _exog_matrix_for_sarimax(
        y_index: pd.DatetimeIndex,
        X: Optional[pd.DataFrame],
        add_calendar: bool = True,
) -> pd.DataFrame:
    """
    - Align to y_index
    - Replace inf/NaN (ffill→0)
    - Drop non-numeric and constant columns
    - Z-score standardize
    - Optionally append calendar features (also standardized)
    """
    if X is None or (isinstance(X, pd.DataFrame) and X.empty):
        base = pd.DataFrame(index=y_index)
    else:
        base = X.copy()
        if not isinstance(base.index, (pd.DatetimeIndex, pd.PeriodIndex)):
            base.index = pd.to_datetime(base.index, errors="coerce", utc=False)
        base = base.reindex(y_index)

    base = base.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    # Num-cast and drop non-numeric
    for c in list(base.columns):
        if not pd.api.types.is_numeric_dtype(base[c]):
            base[c] = pd.to_numeric(base[c], errors="coerce").fillna(0.0)
        base[c] = base[c].astype(float)
    # Drop constants
    if not base.empty:
        std = base.std(ddof=0)
        keep = std[std > 1e-12].index.tolist()
        base = base[keep]
    # Z-score
    if not base.empty:
        mu = base.mean()
        sigma = base.std(ddof=0).replace(0.0, 1.0)
        base = (base - mu) / sigma
    # Calendar
    if add_calendar:
        cal = _calendar_df(y_index).astype(float)
        cal = (cal - cal.mean()) / cal.std(ddof=0).replace(0.0, 1.0)
        base = pd.concat([base, cal], axis=1)
    base = base.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return base


def _normalize_sarimax_search_space(
        search_space: Dict[str, Any] | List[Dict[str, Any]] | None
) -> List[Dict[str, Any]]:
    """
    Accepts:
      - None / {} -> [{}]
      - [{"order":[1,1,1], "seasonal_order":[1,1,1,7]}]  -> treat as single literal spec
      - {"order":[(1,1,1),(2,1,1)], "seasonal_order":[(1,1,1,7)]} -> proper grid over tuples
      - [{"order":(1,1,1), "seasonal_order":(1,1,1,7)}, ...] -> literal list

    Returns a list of dicts with 'order' and 'seasonal_order' as tuples, ready to fit.
    """
    if search_space in (None, {}, []):
        return [{}]

    # Case A: list of dicts → treat each dict as a literal spec
    if isinstance(search_space, list):
        out: List[Dict[str, Any]] = []
        for d in search_space:
            dd = dict(d)
            if "order" in dd:
                v = dd["order"]
                if isinstance(v, list) and all(isinstance(x, (int, np.integer)) for x in v) and len(v) == 3:
                    dd["order"] = tuple(v)
                elif isinstance(v, tuple):
                    pass
                else:
                    # If it's a list of tuples, keep literal; caller wanted explicit combos
                    dd["order"] = tuple(v) if isinstance(v, list) and len(v) == 3 else v
            if "seasonal_order" in dd:
                v = dd["seasonal_order"]
                if isinstance(v, list) and all(isinstance(x, (int, np.integer)) for x in v) and len(v) == 4:
                    dd["seasonal_order"] = tuple(v)
                elif isinstance(v, tuple):
                    pass
                else:
                    dd["seasonal_order"] = tuple(v) if isinstance(v, list) and len(v) == 4 else v
            out.append(dd)
        return out

    # Case B: dict → only grid when values are lists of tuples
    if isinstance(search_space, dict):
        # Are values lists of tuples? If yes, do a real grid.
        is_grid = False
        for k, v in search_space.items():
            if isinstance(v, list) and v and all(isinstance(t, tuple) for t in v):
                is_grid = True
                break

        if is_grid:
            # Proper grid over tuples
            return [dict(p) for p in ParameterGrid(search_space)]

        # Otherwise treat as literal one spec
        d = dict(search_space)
        if "order" in d and isinstance(d["order"], list) and len(d["order"]) == 3:
            d["order"] = tuple(d["order"])
        if "seasonal_order" in d and isinstance(d["seasonal_order"], list) and len(d["seasonal_order"]) == 4:
            d["seasonal_order"] = tuple(d["seasonal_order"])
        return [d]

    # Fallback
    return [{}]


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
def _detect_nested_multiprocessing() -> bool:
    try:
        import multiprocessing as mp
        if mp.current_process().name != "MainProcess":
            return True
    except Exception:
        pass
    start_method = os.environ.get("JOBLIB_START_METHOD", "").lower()
    if start_method and start_method not in ("threading",):
        pass
    return False


class BaseModel:
    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> "BaseModel":
        raise NotImplementedError

    def predict(self, horizon: int, X_future: Optional[pd.DataFrame] = None) -> np.ndarray:
        raise NotImplementedError


class RFRegressorModel(BaseModel):
    """RandomForest forecaster using lag/rolling/calendar features (unchanged)."""

    def __init__(self, **kwargs: Any):
        if "n_jobs" in kwargs:
            n_jobs = kwargs.pop("n_jobs")
        else:
            n_jobs = 1 if _detect_nested_multiprocessing() else -1
        logging.getLogger("RFRegressorModel").info(
            "Initializing RandomForestRegressor with n_jobs=%s (%s)",
            n_jobs, "nested-mp-safe" if n_jobs == 1 else "max-parallel"
        )
        self.model = RandomForestRegressor(
            random_state=42, n_estimators=400, n_jobs=n_jobs, **kwargs
        )
        self._fitted = False
        self.feature_names_: List[str] | None = None

    def fit(self, y: pd.Series, X: pd.DataFrame) -> "RFRegressorModel":
        mask = X.notna().all(axis=1) & y.notna()
        y_aligned = y[mask]
        X_aligned = X[mask]
        dropped = len(X) - len(X_aligned)
        logging.getLogger("RFRegressorModel").info("Fit rows=%d (dropped %d NaN rows)", len(y_aligned), dropped)
        self.model.fit(X_aligned, y_aligned)
        self._fitted = True
        self.feature_names_ = list(X_aligned.columns)
        return self

    def predict(self, horizon: int, X_future: Optional[pd.DataFrame] = None) -> np.ndarray:
        assert self._fitted, "Model not fitted"
        assert X_future is not None, "X_future required"
        if self.feature_names_ is not None:
            X_future = X_future.reindex(columns=self.feature_names_, fill_value=0)
        return self.model.predict(X_future)

    def predict_iterative(
            self,
            y_hist: pd.Series,
            horizon: int,
            feat_cfg: FeatureConfig,
            exog_future: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        assert self._fitted, "Model not fitted"
        preds: List[float] = []
        if exog_future is not None:
            future_index = exog_future.index
            assert len(future_index) == horizon, "exog_future length must equal horizon"
        else:
            future_index = _build_future_index(y_hist.index, horizon)

        y_aug = y_hist.copy()
        for ts in future_index:
            feat: Dict[str, Any] = {}
            for lag in feat_cfg.lags:
                feat[f"lag_{lag}"] = float(y_aug.iloc[-lag]) if len(y_aug) >= lag else np.nan
            for w in feat_cfg.roll_windows:
                feat[f"roll_mean_{w}"] = float(y_aug.iloc[-w:].mean()) if len(y_aug) >= w else np.nan
            if feat_cfg.add_calendar:
                cal_ts = _calendar_df(pd.DatetimeIndex([ts])).iloc[0].to_dict()
                feat.update({k: float(v) for k, v in cal_ts.items()})
            if exog_future is not None:
                if ts not in exog_future.index:
                    raise ValueError("exog_future must be indexed by future timestamps.")
                for col, val in exog_future.loc[ts].items():
                    feat[col] = float(val) if pd.notna(val) else np.nan
            x_row_df = pd.DataFrame([feat], index=pd.DatetimeIndex([ts])).ffill().bfill()
            if self.feature_names_ is not None:
                x_row_df = x_row_df.reindex(columns=self.feature_names_, fill_value=0)
            y_hat = float(self.model.predict(x_row_df)[0])
            preds.append(y_hat)
            y_aug.loc[ts] = y_hat  # iterative step
        return np.asarray(preds, dtype=float)


class SarimaxModel(BaseModel):
    """Robust SARIMAX with multiple optimizer retries."""

    def __init__(self, order=(1, 1, 1), seasonal_order=(0, 1, 1, 7)):
        if not HAVE_STATSMODELS:
            raise RuntimeError("statsmodels not installed; SARIMAX disabled.")
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)
        self.result = None
        self._train_exog_cols: List[str] = []
        self._train_calendar = True
        self._train_index: Optional[pd.DatetimeIndex] = None
        self._train_freq: Optional[str] = None

    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> "SarimaxModel":
        y2, freq = _ensure_supported_index(y)
        if not freq:
            y2, _, freq = _coerce_regular_freq(y2, None, freq="D", y_fill=0.0)
        X2 = _exog_matrix_for_sarimax(y2.index, X, add_calendar=True)
        self._train_exog_cols = list(X2.columns)
        self._train_calendar = True
        self._train_index = y2.index
        self._train_freq = freq

        def _fit_once(method: str, enforce_stat: bool, enforce_inv: bool, maxiter: int):
            with warnings.catch_warnings(record=True) as wlist:
                warnings.simplefilter("always", ConvergenceWarning)
                res = SARIMAX(
                    endog=y2,
                    exog=(X2 if not X2.empty else None),
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    enforce_stationarity=enforce_stat,
                    enforce_invertibility=enforce_inv,
                    dates=y2.index,
                    freq=freq,
                ).fit(disp=False, method=method, maxiter=maxiter)
                for w in wlist:
                    if issubclass(w.category, ConvergenceWarning):
                        log.warning("SARIMAX %s: %s", method, str(w.message))
                return res

        tried = []
        for attempt in [
            dict(method="lbfgs", enforce_stat=False, enforce_inv=False, maxiter=500),
            dict(method="powell", enforce_stat=False, enforce_inv=False, maxiter=800),
            dict(method="nm", enforce_stat=True, enforce_inv=True, maxiter=1200),
            dict(method="lbfgs", enforce_stat=True, enforce_inv=True, maxiter=700),
        ]:
            tried.append(attempt)
            try:
                res = _fit_once(**attempt)
                self.result = res
                try:
                    log.info("SARIMAX fit OK: method=%s converged=%s niter=%s llf=%.3f",
                             attempt["method"],
                             res.mle_retvals.get("converged", None),
                             res.mle_retvals.get("iterations", None),
                             float(getattr(res, "llf", np.nan)))
                except Exception:
                    pass
                return self
            except Exception as e:
                log.warning("SARIMAX fit failed with %s: %s", attempt, e)

        # Final fallback: simplify seasonal terms
        try:
            simple_seasonal = (0, 1, 0, 7)
            log.warning("Retrying with simplified seasonal_order=%s", simple_seasonal)
            res = SARIMAX(
                endog=y2,
                exog=(X2 if not X2.empty else None),
                order=self.order,
                seasonal_order=simple_seasonal,
                enforce_stationarity=True,
                enforce_invertibility=True,
                dates=y2.index,
                freq=freq,
            ).fit(disp=False, method="lbfgs", maxiter=700)
            self.result = res
            return self
        except Exception as e:
            log.error("SARIMAX final fallback failed: %s", e)
            raise

    def predict(self, horizon: int, X_future: Optional[pd.DataFrame] = None) -> np.ndarray:
        if X_future is None:
            X_future = pd.DataFrame(index=_build_future_index(self._train_index, horizon))
        else:
            if not isinstance(X_future.index, (pd.DatetimeIndex, pd.PeriodIndex)):
                X_future.index = pd.to_datetime(X_future.index, errors="coerce", utc=False)
        Xf = X_future.reindex(_build_future_index(self._train_index, horizon))
        Xf = _exog_matrix_for_sarimax(Xf.index, Xf, add_calendar=self._train_calendar)
        missing_cols = [c for c in self._train_exog_cols if c not in Xf.columns]
        for c in missing_cols:
            Xf[c] = 0.0
        Xf = Xf[self._train_exog_cols]
        fc = self.result.get_forecast(steps=horizon, exog=(Xf if not Xf.empty else None))
        return np.asarray(fc.predicted_mean, dtype=float)


# Registry & allowed params
ModelRegistry: Dict[str, Callable[..., BaseModel]] = {"rf": RFRegressorModel}
if HAVE_STATSMODELS:
    ModelRegistry["sarimax"] = SarimaxModel

ALLOWED_PARAMS = {
    "rf": {"n_estimators", "max_depth", "min_samples_leaf", "n_jobs", "random_state"},
    "sarimax": {"order", "seasonal_order"},
}


# -----------------------------------------------------------------------------
# Backtesting & Search
# -----------------------------------------------------------------------------
@dataclass
class BacktestReport:
    preds: pd.Series
    trues: pd.Series
    metrics: Dict[str, float]
    cutoffs: List[pd.Timestamp]
    model_name: str
    params: Dict[str, Any]
    horizon: int
    step: int
    initial_train: int


def rolling_backtest(
        y: pd.Series,
        horizon: int,
        initial_train: int,
        step: int,
        model_name: str,
        model_params: Dict[str, Any],
        feat_cfg: FeatureConfig,
        exog: Optional[pd.DataFrame] = None,
) -> BacktestReport:
    model_cls = ModelRegistry[model_name]

    if model_name == "rf":
        X_full = make_features(y, feat_cfg, exog)
    else:  # sarimax uses exog only; calendar is appended inside its fit
        X_full = exog.copy() if exog is not None else None

    preds_list: List[pd.Series] = []
    trues_list: List[pd.Series] = []
    cutoffs: List[pd.Timestamp] = []

    i = initial_train
    round_idx = 0
    while i + horizon <= len(y):
        round_idx += 1
        y_train = y.iloc[:i]
        y_test = y.iloc[i:i + horizon]

        if model_name == "rf":
            X_train = X_full.iloc[:i]
            # prepare future exog for iterative prediction
            exog_future = None
            if exog is not None:
                exog_future = exog.iloc[i:i + horizon]
                if exog_future.shape[0] != horizon:
                    exog_future = _build_exog_future_if_needed(
                        exog_cols=list(exog.columns),
                        y_index=y_train.index,
                        horizon=horizon,
                        hist_df=pd.concat([exog.iloc[:i], exog.iloc[i:i + horizon]], axis=0)
                    )
            m = model_cls(**model_params)
            m.fit(y_train, X_train)
            y_hat = m.predict_iterative(y_hist=y_train, horizon=horizon, feat_cfg=feat_cfg, exog_future=exog_future)
        else:
            X_train = X_full.iloc[:i] if X_full is not None else None
            # For forecast window, provide future exog (or synthesize)
            if exog is not None:
                exog_future = exog.iloc[i:i + horizon]
                if exog_future.shape[0] != horizon:
                    exog_future = _build_exog_future_if_needed(
                        exog_cols=list(exog.columns),
                        y_index=y_train.index,
                        horizon=horizon,
                        hist_df=exog
                    )
            else:
                exog_future = None
            m = model_cls(**model_params)
            m.fit(y_train, X_train)
            y_hat = m.predict(horizon=horizon, X_future=exog_future)

        preds_list.append(pd.Series(y_hat, index=y_test.index, name="y_hat"))
        trues_list.append(y_test)
        cutoffs.append(y_train.index[-1])

        log.info("Backtest step %d: train_end=%s → test=[%s..%s]",
                 round_idx, y_train.index[-1].date(), y_test.index[0].date(), y_test.index[-1].date())
        i += step

    preds = pd.concat(preds_list)
    trues = pd.concat(trues_list)

    m_season = _infer_seasonal_period(y)
    metrics = {
        "RMSE": rmse(trues, preds),
        "sMAPE": smape(trues, preds),
        "MASE": mase(trues, preds, y.iloc[:initial_train], m=m_season)
    }
    log.info("Backtest metrics: RMSE=%.4f sMAPE=%.4f MASE=%.4f",
             metrics["RMSE"], metrics["sMAPE"], metrics["MASE"])

    return BacktestReport(
        preds=preds, trues=trues, metrics=metrics, cutoffs=cutoffs,
        model_name=model_name, params=model_params,
        horizon=horizon, step=step, initial_train=initial_train
    )


def _coerce_sarimax_params(p: Dict[str, Any]) -> Dict[str, Any]:
    q = dict(p)
    if "order" in q and isinstance(q["order"], (list, tuple)):
        q["order"] = tuple(q["order"])
    if "seasonal_order" in q and isinstance(q["seasonal_order"], (list, tuple)):
        q["seasonal_order"] = tuple(q["seasonal_order"])
    return q


def grid_search(
        y: pd.Series,
        horizon: int,
        initial_train: int,
        step: int,
        search_space: Dict[str, List[Any]] | List[Dict[str, Any]] | None,
        feat_cfg: FeatureConfig,
        model_name: str,
        exog: Optional[pd.DataFrame] = None,
) -> Tuple[Dict[str, Any], BacktestReport]:
    model_cls = ModelRegistry[model_name]

    if model_name == "sarimax":
        # Normalize so {'order':[1,1,1]} is treated as a *literal* not a grid,
        # and only lists of tuples trigger true grid expansion.
        param_iter = _normalize_sarimax_search_space(search_space)

        # Ensure tuple types
        def _coerce(p: Dict[str, Any]) -> Dict[str, Any]:
            q = dict(p)
            if "order" in q and isinstance(q["order"], list):
                q["order"] = tuple(q["order"])
            if "seasonal_order" in q and isinstance(q["seasonal_order"], list):
                q["seasonal_order"] = tuple(q["seasonal_order"])
            return q

        param_iter = [_coerce(p) for p in param_iter]
    else:
        if search_space in (None, {}):
            param_iter = [{}]
        else:
            param_iter = list(ParameterGrid(search_space))

    best_params = None
    best_report = None
    best_score = np.inf
    log.info("=== Hyperparameter Search (model=%s) ===", model_name)

    for raw_params in param_iter:
        allowed = ALLOWED_PARAMS.get(model_name, set())
        params = {k: v for k, v in raw_params.items() if k in allowed}
        try:
            log.info("Trying params: %s", params)
            report = rolling_backtest(
                y=y, horizon=horizon, initial_train=initial_train, step=step,
                model_name=model_name, model_params=params, feat_cfg=feat_cfg, exog=exog
            )
            score = report.metrics["RMSE"]
            log.info(" -> RMSE=%.4f | sMAPE=%.4f | MASE=%.4f",
                     report.metrics["RMSE"], report.metrics["sMAPE"], report.metrics["MASE"])
            if score < best_score:
                best_score = score
                best_params = params
                best_report = report
        except Exception as e:
            log.warning("Param set %s failed: %s", params, e)
            continue

    if best_params is None or best_report is None:
        raise RuntimeError(f"All parameter candidates failed for model={model_name}.")
    log.info("Best params: %s with RMSE=%.4f", best_params, best_score)
    return best_params, best_report


# -----------------------------------------------------------------------------
# Diagnostics & Final Train/Forecast
# -----------------------------------------------------------------------------
def residual_diagnostics(report: BacktestReport) -> Dict[str, Any]:
    res = (report.trues - report.preds).dropna()
    summary = res.describe().to_dict()
    out = {"summary": summary, "ljung_box": None}
    if HAVE_STATSMODELS:
        try:
            lb = acorr_ljungbox(res, lags=[10], return_df=True)
            out["ljung_box"] = lb.to_dict(orient="list")
            log.info("Ljung–Box(10):\n%s", lb.to_string(index=False))
        except Exception as e:
            log.warning("Ljung–Box unavailable: %s", e)
    else:
        log.info("statsmodels not installed; skipping Ljung–Box.")
    log.info("Residual summary:\n%s", pd.Series(summary).to_string())
    return out


@dataclass
class FinalForecast:
    forecast: pd.Series
    model_name: str
    params: Dict[str, Any]


def train_full_and_forecast(
        y: pd.Series,
        horizon: int,
        model_name: str,
        params: Dict[str, Any],
        feat_cfg: FeatureConfig,
        exog: Optional[pd.DataFrame] = None
) -> FinalForecast:
    log.info("=== Train on Full History & Forecast Next %d ===", horizon)

    if model_name == "rf":
        X_train = make_features(y, feat_cfg, exog)
        exog_future = None
        if exog is not None:
            exog_future = _build_exog_future_if_needed(
                exog_cols=list(exog.columns),
                y_index=y.index,
                horizon=horizon,
                hist_df=exog
            )
        m = ModelRegistry[model_name](**params)
        m.fit(y, X_train)
        y_hat = m.predict_iterative(y_hist=y, horizon=horizon, feat_cfg=feat_cfg, exog_future=exog_future)
    else:
        # SARIMAX uses exog only; calendar/standardization handled in model
        X_train = exog.copy() if exog is not None else None
        exog_future = None
        if exog is not None:
            exog_future = _build_exog_future_if_needed(
                exog_cols=list(exog.columns),
                y_index=y.index,
                horizon=horizon,
                hist_df=exog
            )
        m = ModelRegistry[model_name](**params)
        m.fit(y, X_train)
        y_hat = m.predict(horizon=horizon, X_future=exog_future)

    future_index = _build_future_index(y.index, horizon)
    fc = pd.Series(y_hat, index=future_index, name="y_hat")
    log.info("Forecast preview:\n%s", fc.head(10).to_string())
    return FinalForecast(forecast=fc, model_name=model_name, params=params)


# -----------------------------------------------------------------------------
# Public Entry Point + Config
# -----------------------------------------------------------------------------
@dataclass
class RunConfig:
    # Backtest
    horizon: int = 14
    initial_train: int = 360
    step: int = 14
    # Model & search
    model_name: str = "rf"
    search_space: Dict[str, List[Any]] | List[Dict[str, Any]] | None = None
    # Features (RF only)
    feature_config: FeatureConfig | None = None
    # Exogenous
    use_exog: bool = False
    # Logging
    log_level: int = logging.INFO
    # Serialization
    max_serialize_points: Optional[int] = None

    def __post_init__(self):
        if self.feature_config is None:
            self.feature_config = FeatureConfig(
                lags=[1, 7, 14, 28],
                roll_windows=[7, 14, 28],
                add_calendar=True
            )
        if self.search_space is None:
            if self.model_name == "sarimax":
                # Small, stable space for daily data with weekly seasonality
                self.search_space = [
                    {"order": (1, 1, 1), "seasonal_order": (1, 1, 1, 7)},
                    {"order": (2, 1, 1), "seasonal_order": (1, 1, 1, 7)},
                    {"order": (1, 1, 2), "seasonal_order": (1, 1, 1, 7)},
                ]
            else:
                self.search_space = {
                    "max_depth": [6, 10],
                    "min_samples_leaf": [1, 4],
                    "n_estimators": [300, 600],
                }


# ---- Data preparation helpers -----------------------------------------------
def _ensure_datetime_index(df: pd.DataFrame, ts_col: Optional[str]) -> Tuple[pd.DataFrame, str]:
    if ts_col is None:
        # first datetime-like column or try parsing columns
        datetime_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]
        if not datetime_cols:
            for c in df.columns:
                try:
                    parsed = pd.to_datetime(df[c], errors="raise", utc=False)
                    df = df.copy()
                    df[c] = parsed
                    datetime_cols = [c]
                    break
                except Exception:
                    continue
        if not datetime_cols:
            raise ValueError("Could not infer time column. Provide ts_col.")
        ts_col = datetime_cols[0]

    df = df.copy()
    if not np.issubdtype(df[ts_col].dtype, np.datetime64):
        df[ts_col] = pd.to_datetime(df[ts_col], errors="raise", utc=False)

    df = df.sort_values(ts_col).set_index(ts_col)
    return df, ts_col


def _infer_target(df: pd.DataFrame, target: Optional[str], exog_cols: Optional[List[str]]) -> str:
    if target is not None:
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not in DataFrame.")
        return target
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if exog_cols:
        numeric_cols = [c for c in numeric_cols if c not in exog_cols]
    if len(numeric_cols) == 1:
        return numeric_cols[0]
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns to use as target. Provide target=...")
    raise ValueError(f"Ambiguous target. Candidates: {numeric_cols}. Provide target=...")


def _prepare_inputs(
        data: pd.Series | pd.DataFrame,
        target: Optional[str],
        ts_col: Optional[str],
        exog_cols: Optional[List[str]],
        use_exog: bool
) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
    """Return y (Series with DateTimeIndex and regular freq) and exog aligned to y (or None)."""
    if isinstance(data, pd.Series):
        y = data.copy()
        if not isinstance(y.index, (pd.DatetimeIndex, pd.PeriodIndex)):
            y.index = pd.to_datetime(y.index, errors="raise", utc=False)
        y = y.sort_index()
        y, _, _ = _coerce_regular_freq(y, None, freq=None, y_fill=0.0)
        return y, None

    if not isinstance(data, pd.DataFrame):
        raise TypeError("run_forecasting data must be pandas Series or DataFrame")

    df = data.copy()
    df, ts_col = _ensure_datetime_index(df, ts_col)
    exog = None
    if use_exog and exog_cols:
        missing = [c for c in exog_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Exogenous columns not in DataFrame: {missing}")
        exog = df[exog_cols].copy()

    y_col = _infer_target(df, target=target, exog_cols=exog_cols if exog_cols else None)
    y = df[y_col].astype(float).copy()

    # Ensure regular frequency on y (and exog aligned)
    y, exog, _ = _coerce_regular_freq(y, exog, freq=None, y_fill=0.0)

    if exog is not None and y_col in exog.columns:
        exog = exog.drop(columns=[y_col])
    return y, exog


# -----------------------------------------------------------------------------
# Public Entry Point
# -----------------------------------------------------------------------------
def run_forecasting(
        data: pd.Series | pd.DataFrame,
        config: RunConfig,
        target: Optional[str] = None,
        ts_col: Optional[str] = None,
        exog_cols: Optional[List[str]] = None,
        freq: Optional[str] = None
) -> Dict[str, Any]:
    """
    Orchestrates the full pipeline and returns JSON-serializable results.

    Parameters:
      - data: Series (DateTimeIndex) or DataFrame with a time column
      - config: RunConfig (model_name: "rf" or "sarimax")
      - target, ts_col, exog_cols: hints for DataFrame input
      - freq: optional explicit frequency to enforce (if provided)
    """
    logging.getLogger().setLevel(config.log_level)

    # Prepare inputs
    y, exog = _prepare_inputs(
        data=data,
        target=target,
        ts_col=ts_col,
        exog_cols=exog_cols,
        use_exog=config.use_exog
    )

    # Early validation
    min_len = config.initial_train + config.horizon
    if len(y) < min_len:
        raise ValueError(
            f"Series too short: len(y)={len(y)} < initial_train+horizon={min_len}"
        )

    # Optionally enforce explicit frequency
    if freq is not None:
        y = y.asfreq(freq)
        if exog is not None:
            exog = exog.asfreq(freq)
        if y.isna().any():
            log.warning("Frequency coercion introduced NaNs in target; consider imputing before run.")

    log.info("=== run_forecasting ===")
    log.info("Series length=%d | range=[%s .. %s]", len(y), y.index.min(), y.index.max())
    log.info("Config: %s", asdict(config))
    if exog is not None:
        log.info("Using exogenous columns: %s", list(exog.columns))

    # Search + backtest
    best_params, best_report = grid_search(
        y=y,
        horizon=config.horizon,
        initial_train=config.initial_train,
        step=config.step,
        search_space=config.search_space,
        feat_cfg=config.feature_config,
        model_name=config.model_name,
        exog=exog if config.use_exog else None,
    )

    # Diagnostics
    diag = residual_diagnostics(best_report)

    # Final train & forecast
    final = train_full_and_forecast(
        y=y,
        horizon=config.horizon,
        model_name=config.model_name,
        params=best_params,
        feat_cfg=config.feature_config,
        exog=exog if config.use_exog else None
    )

    # Optional thinning for payload size
    def _thin(series: pd.Series, k: Optional[int]) -> pd.Series:
        if not k or len(series) <= k:
            return series
        step = max(1, len(series) // k)
        return series.iloc[::step]

    bt_preds_ser = _thin(best_report.preds, config.max_serialize_points)
    bt_trues_ser = _thin(best_report.trues, config.max_serialize_points)
    backtest_preds = [{"ts": ts.isoformat(), "y_hat": float(val)} for ts, val in bt_preds_ser.items()]
    backtest_trues = [{"ts": ts.isoformat(), "y": float(val)} for ts, val in bt_trues_ser.items()]
    final_forecast = [{"ts": ts.isoformat(), "y_hat": float(val)} for ts, val in final.forecast.items()]

    result: Dict[str, Any] = {
        "best_params": best_params,
        "backtest": {
            "metrics": best_report.metrics,
            "preds": backtest_preds,
            "trues": backtest_trues,
            "horizon": best_report.horizon,
            "step": best_report.step,
            "initial_train": best_report.initial_train,
        },
        "diagnostics": {"residuals": diag},
        "final_forecast": final_forecast,
        "model_name": final.model_name,
    }

    log.info("=== run_forecasting completed ===")
    log.info("Selected params: %s", best_params)
    log.info("Backtest metrics: %s", best_report.metrics)
    log.info("Forecast horizon: %d | First ts: %s", config.horizon, final.forecast.index[0])
    return result
