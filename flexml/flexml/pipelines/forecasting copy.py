"""
- High-level run_forecasting(data, config, target=None, ts_col=None, exog_cols=None, freq=None)
- Accepts pandas Series or DataFrame; works with call style: run_forecasting(df, cfg)
- If DataFrame:
    * Infers ts_col (datetime), target (numeric) when not provided
    * Optional exogenous columns
- Performs: grid search -> rolling backtest -> residual diagnostics -> train-on-full -> next-horizon forecast
- Returns JSON-friendly dict; logs rich details to terminal.
"""

from __future__ import annotations

import logging
import os
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

    HAVE_STATSMODELS = True
except Exception:
    SARIMAX = None
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
    # seasonal naive denominator
    d = np.abs(np.diff(y_insample, n=m)).mean() + eps
    return float(np.mean(np.abs(y_true - y_pred)) / d)


def _infer_seasonal_period(y: pd.Series) -> int:
    """
    Heuristic seasonal period used for MASE scaling.
    """
    try:
        freq = pd.infer_freq(y.index)
        if not freq:
            return 1
        off = pd.tseries.frequencies.to_offset(freq)
        base = off.name  # e.g., 'D', 'H', 'M', 'W-SUN'
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
# Feature Engineering
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


def make_features(y: pd.Series,
                  cfg: FeatureConfig,
                  exog: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    assert isinstance(y.index, pd.DatetimeIndex), "y must have a DateTimeIndex"
    X = pd.DataFrame(index=y.index)

    for lag in cfg.lags:
        X[f"lag_{lag}"] = y.shift(lag)

    for w in cfg.roll_windows:
        # rolling mean using only past values; require full window
        X[f"roll_mean_{w}"] = y.shift(1).rolling(window=w, min_periods=w).mean()

    if cfg.add_calendar:
        X = X.join(_calendar_df(y.index))

    if exog is not None:
        X = X.join(exog)

    return X


# -----------------------------------------------------------------------------
# Exogenous helpers
# -----------------------------------------------------------------------------

def _build_future_index(y_index: pd.DatetimeIndex, horizon: int) -> pd.DatetimeIndex:
    freq = y_index.freq or pd.infer_freq(y_index)
    if freq is None:
        freq = "D"
    start = y_index[-1] + to_offset(freq)
    return pd.date_range(start=start, periods=horizon, freq=freq)


def _build_exog_future_if_needed(
        exog_cols: List[str] | None,
        y_index: pd.DatetimeIndex,
        horizon: int,
        hist_df: pd.DataFrame | None = None,
) -> Optional[pd.DataFrame]:
    """
    If exog columns were used during fit but the caller did not provide future
    values, synthesize a reasonable default:

      - numeric floats/ints → forward-fill last known value; if none, 0
      - binary-like (0/1 or bool) → fill with 0 (e.g., no promo by default)
      - missing columns → 0

    Returns a DataFrame indexed by future dates, or None if no exog_cols.
    """
    if not exog_cols:
        return None

    future_idx = _build_future_index(y_index, horizon)
    exog_future = pd.DataFrame(index=future_idx)

    hist_df = hist_df if hist_df is not None else pd.DataFrame(index=y_index)

    for c in exog_cols:
        series = hist_df[c] if c in hist_df.columns else pd.Series(index=y_index, dtype="float64")

        # last known value (if any)
        last_val = None
        if not series.dropna().empty:
            last_val = series.dropna().iloc[-1]

        # detect binary-like
        is_binary_like = False
        if pd.api.types.is_bool_dtype(series):
            is_binary_like = True
        else:
            uniq = pd.Series(pd.unique(series.dropna()))
            if not uniq.empty and uniq.map(lambda v: v in (0, 1)).all():
                is_binary_like = True

        if is_binary_like:
            exog_future[c] = 0
        else:
            fill_val = float(last_val) if last_val is not None else 0.0
            exog_future[c] = fill_val

    return exog_future


def _ensure_supported_index(y: pd.Series) -> tuple[pd.Series, Optional[str]]:
    """
    Ensure y has a DatetimeIndex (or PeriodIndex) with an attached freq if possible.
    Returns (possibly modified y, freq_str_or_None).
    """
    if not isinstance(y.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        y = y.copy()
        y.index = pd.to_datetime(y.index, errors="raise", utc=False)

    # Try to get (or infer) frequency
    freq = y.index.freqstr or pd.infer_freq(y.index)
    if freq:
        try:
            # Attach freq to index; if gaps prevent setting, keep index but still return freq
            y = y.copy()
            y.index = pd.DatetimeIndex(y.index, freq=to_offset(freq))
        except Exception:
            pass
    return y, freq


def _align_exog_to_y_index(y_idx: pd.DatetimeIndex, exog: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Align exogenous to y's index; forward-fill, then 0-fill; remove inf; coerce to numeric.
    """
    if exog is None:
        return None
    X = exog.copy()
    X = X.reindex(y_idx).ffill().fillna(0.0)
    X = X.replace([np.inf, -np.inf], 0.0)
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
    return X


def _exog_matrix_for_sarimax(y_index: pd.DatetimeIndex,
                             exog: Optional[pd.DataFrame],
                             add_calendar: bool = True) -> pd.DataFrame:
    """
    Build SARIMAX design matrix: calendar features (+ optional user exog), aligned to y_index.
    """
    parts = []
    if add_calendar:
        parts.append(_calendar_df(y_index))
    if exog is not None:
        parts.append(_align_exog_to_y_index(y_index, exog))
    if not parts:
        return pd.DataFrame(index=y_index)
    X = pd.concat(parts, axis=1)
    X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    X = X.reindex(y_index, fill_value=0.0)
    return X


def _exog_future_for_sarimax(y_index: pd.DatetimeIndex,
                             horizon: int,
                             exog_hist: Optional[pd.DataFrame],
                             add_calendar: bool,
                             columns_like: Optional[list[str]]) -> pd.DataFrame:
    """
    Build exog for the forecast horizon:
      - calendar features for future timestamps
      - last-observed values for user exog (if given)
      - same columns/order as 'columns_like'
    """
    future_index = _build_future_index(y_index, horizon)
    parts = []
    if add_calendar:
        parts.append(_calendar_df(future_index))

    if exog_hist is not None and not exog_hist.empty:
        last_vals = exog_hist.ffill().iloc[-1:].copy()
        exog_future = pd.DataFrame(
            np.repeat(last_vals.values, len(future_index), axis=0),
            index=future_index,
            columns=last_vals.columns
        )
        parts.append(exog_future)

    Xf = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=future_index)
    Xf = Xf.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    if columns_like is not None:
        Xf = Xf.reindex(columns=columns_like, fill_value=0.0)
    return Xf


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

def _detect_nested_multiprocessing() -> bool:
    """
    Returns True if we're likely inside a multiprocessing worker where
    nested process-based parallelism (loky) would cause warnings.
    """
    try:
        import multiprocessing as mp  # local import to avoid hard dep at import time
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
    """RandomForest forecaster using tabular lag/rolling features (iterative multi-step)."""

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
            random_state=42,
            n_estimators=400,
            n_jobs=n_jobs,
            **kwargs
        )
        self._fitted = False
        self.feature_names_: List[str] | None = None  # NEW: persist fitted feature columns

    def fit(self, y: pd.Series, X: pd.DataFrame) -> "RFRegressorModel":
        mask = X.notna().all(axis=1) & y.notna()
        y_aligned = y[mask]
        X_aligned = X[mask]
        dropped = len(X) - len(X_aligned)
        logging.getLogger("RFRegressorModel").info(
            "Fit rows=%d (dropped %d NaN rows)", len(y_aligned), dropped
        )
        self.model.fit(X_aligned, y_aligned)
        self._fitted = True
        # Persist exact feature set seen at fit time (NEW)
        self.feature_names_ = list(X_aligned.columns)
        return self

    def predict(self, horizon: int, X_future: Optional[pd.DataFrame] = None) -> np.ndarray:
        assert self._fitted, "Model not fitted"
        assert X_future is not None, "X_future required for RF direct predict"
        # Enforce same columns as fit (NEW)
        if self.feature_names_ is not None:
            X_future = X_future.reindex(columns=self.feature_names_, fill_value=0)
        return self.model.predict(X_future)

    def predict_iterative(self,
                          y_hist: pd.Series,
                          horizon: int,
                          feat_cfg: FeatureConfig,
                          exog_future: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Leakage-safe iterative multi-step prediction with O(1) per-step feature construction."""
        assert self._fitted, "Model not fitted"
        preds: List[float] = []

        if exog_future is not None:
            future_index = exog_future.index
            assert len(future_index) == horizon, "exog_future length must equal horizon"
        else:
            future_index = _build_future_index(y_hist.index, horizon)

        y_aug = y_hist.copy()

        for ts in future_index:
            # Build a single-row feature vector for timestamp ts
            feat: Dict[str, Any] = {}

            # Lags: values at t - lag
            for lag in feat_cfg.lags:
                feat[f"lag_{lag}"] = float(y_aug.iloc[-lag]) if len(y_aug) >= lag else np.nan

            # Rolling means over the last w observed values (up to t-1)
            for w in feat_cfg.roll_windows:
                if len(y_aug) >= w:
                    feat[f"roll_mean_{w}"] = float(y_aug.iloc[-w:].mean())
                else:
                    feat[f"roll_mean_{w}"] = np.nan

            # Calendar
            if feat_cfg.add_calendar:
                cal_ts = _calendar_df(pd.DatetimeIndex([ts])).iloc[0].to_dict()
                feat.update({k: float(v) for k, v in cal_ts.items()})

            # Exogenous future row (if provided)
            if exog_future is not None:
                if ts not in exog_future.index:
                    raise ValueError("exog_future must be indexed by the exact future timestamps.")
                for col, val in exog_future.loc[ts].items():
                    feat[col] = float(val) if pd.notna(val) else np.nan

            x_row_df = pd.DataFrame([feat], index=pd.DatetimeIndex([ts])).ffill().bfill()

            # Enforce feature names seen at fit time (NEW)
            if self.feature_names_ is not None:
                x_row_df = x_row_df.reindex(columns=self.feature_names_, fill_value=0)

            y_hat = float(self.model.predict(x_row_df)[0])
            preds.append(y_hat)
            # Append prediction to history for next-step features
            y_aug.loc[ts] = y_hat

        return np.asarray(preds, dtype=float)


class SarimaxModel(BaseModel):
    """Seasonal ARIMA (if statsmodels available)."""

    def __init__(self, order=(1, 1, 1), seasonal_order=(0, 1, 1, 7)):
        if not HAVE_STATSMODELS:
            raise RuntimeError("statsmodels not installed; SARIMAX disabled.")
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)
        self.result = None
        self._train_exog_cols: list[str] | None = None
        self._train_calendar: bool = True
        self._train_index: Optional[pd.DatetimeIndex] = None
        self._train_freq: Optional[str] = None

    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> "SarimaxModel":
        # Ensure supported index on y
        y2, freq = _ensure_supported_index(y)
        self._train_index = y2.index
        self._train_freq = freq

        # Build design matrix for SARIMAX (calendar + exog aligned)
        X2 = _exog_matrix_for_sarimax(y2.index, X, add_calendar=True)
        self._train_exog_cols = list(X2.columns)
        self._train_calendar = True

        self.result = SARIMAX(
            endog=y2,
            exog=(X2 if not X2.empty else None),
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
            dates=y2.index,     # explicit dates
            freq=freq           # explicit freq (may be None if cannot attach)
        ).fit(disp=False)
        return self

    def predict(self, horizon: int, X_future: Optional[pd.DataFrame] = None) -> np.ndarray:
        # Build future exog using same columns and calendar setting as training
        Xf = _exog_future_for_sarimax(
            y_index=self._train_index,
            horizon=horizon,
            exog_hist=X_future,                     # if caller provided, it is the *hist* exog to copy
            add_calendar=self._train_calendar,
            columns_like=self._train_exog_cols
        )
        fc = self.result.get_forecast(
            steps=horizon,
            exog=(Xf if not Xf.empty else None)
        )
        return np.asarray(fc.predicted_mean, dtype=float)


# Registry
ModelRegistry: Dict[str, Callable[..., BaseModel]] = {
    "rf": RFRegressorModel,
}
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

    preds_list: List[pd.Series] = []
    trues_list: List[pd.Series] = []
    cutoffs: List[pd.Timestamp] = []

    # Precompute feature matrices
    if model_name == "sarimax":
        # For SARIMAX we use a single design matrix: calendar (+ exog)
        X_full = _exog_matrix_for_sarimax(y.index, exog, add_calendar=feat_cfg.add_calendar)
    else:
        # RF-style: classical lag/rolling features + calendar + user exog
        X_full = make_features(y, feat_cfg, exog)

    i = initial_train
    round_idx = 0
    while i + horizon <= len(y):
        round_idx += 1
        y_train = y.iloc[:i]
        y_test = y.iloc[i:i + horizon]

        if model_name == "sarimax":
            X_train = X_full.iloc[:i]
            # Build exog for forecast window consistent with training design
            exog_future = _exog_future_for_sarimax(
                y_index=y_train.index,
                horizon=horizon,
                exog_hist=exog.iloc[:i] if exog is not None else None,
                add_calendar=feat_cfg.add_calendar,
                columns_like=list(X_full.columns)
            )
        else:
            X_train = X_full.iloc[:i]
            # For RF iterative, pass the *future* exog slice if available; synthesize if short
            if exog is not None:
                exog_future = exog.iloc[i:i + horizon]
                if exog_future.shape[0] != horizon:
                    exog_future = _build_exog_future_if_needed(
                        exog_cols=list(exog.columns),
                        y_index=y_train.index,
                        horizon=horizon,
                        hist_df=pd.concat([exog.iloc[:i], exog.iloc[i:i + horizon]], axis=0)
                    )
            else:
                exog_future = None

        log.info("Init model=%s with params=%s", model_name, model_params)
        m = model_cls(**model_params)

        if model_name == "sarimax":
            m.fit(y_train, X_train)
            y_hat = m.predict(horizon=horizon, X_future=exog if exog is not None else None)
        else:
            m.fit(y_train, X_train)
            y_hat = m.predict_iterative(y_hist=y_train, horizon=horizon, feat_cfg=feat_cfg, exog_future=exog_future)

        preds_list.append(pd.Series(y_hat, index=y_test.index, name="y_hat"))
        trues_list.append(y_test)
        cutoffs.append(y_train.index[-1])

        log.info("Backtest step %d: train_end=%s → test=[%s..%s]",
                 round_idx, y_train.index[-1], y_test.index[0], y_test.index[-1])

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
    """Ensure SARIMAX params are proper tuples with correct lengths; fall back to sane defaults."""
    q = dict(p)

    def _order(v):
        if isinstance(v, (list, tuple)) and len(v) == 3:
            return tuple(int(x) for x in v)
        if isinstance(v, dict) and {"p", "d", "q"} <= set(v):
            return (int(v["p"]), int(v["d"]), int(v["q"]))
        return (1, 1, 1)

    def _seasonal(v):
        if isinstance(v, (list, tuple)) and len(v) == 4:
            return tuple(int(x) for x in v)
        if isinstance(v, dict) and {"P", "D", "Q", "m"} <= set(v):
            return (int(v["P"]), int(v["D"]), int(v["Q"]), int(v["m"]))
        return (0, 1, 1, 7)

    q["order"] = _order(q.get("order", (1, 1, 1)))
    q["seasonal_order"] = _seasonal(q.get("seasonal_order", (0, 1, 1, 7)))
    return q


def _normalize_sarimax_space(space: Any) -> List[Dict[str, Any]]:
    """
    Normalize user-provided SARIMAX search_space into a list of param dicts with proper tuples.
    Accepts:
      - None / {} / [] -> default single config
      - dict with single configs (order=[1,1,1], seasonal_order=[1,1,1,7])
      - dict with lists of tuple-like configs (grid)
      - list of dicts (each a single config)
    """
    default = [{"order": (1, 1, 1), "seasonal_order": (1, 1, 1, 7)}]

    if space in (None, {}, []):
        return default

    # list of dicts → normalize each
    if isinstance(space, list):
        out: List[Dict[str, Any]] = []
        for p in space:
            out.append(_coerce_sarimax_params(p))
        return out or default

    # dict case
    if isinstance(space, dict):
        o = space.get("order", (1, 1, 1))
        s = space.get("seasonal_order", (1, 1, 1, 7))

        def as_candidates(v, n, fallback):
            # list of tuple-likes → many candidates
            if isinstance(v, list) and v and isinstance(v[0], (list, tuple)):
                return [tuple(int(x) for x in vv) for vv in v if isinstance(vv, (list, tuple)) and len(vv) == n]
            # single tuple/list of correct length → one candidate
            if isinstance(v, (list, tuple)) and len(v) == n:
                return [tuple(int(x) for x in v)]
            # dict form → one candidate
            if isinstance(v, dict) and (
                    (n == 3 and {"p", "d", "q"} <= set(v)) or (n == 4 and {"P", "D", "Q", "m"} <= set(v))):
                if n == 3:
                    return [(int(v["p"]), int(v["d"]), int(v["q"]))]
                else:
                    return [(int(v["P"]), int(v["D"]), int(v["Q"]), int(v["m"]))]
            # anything else → fallback
            return [fallback]

        orders = as_candidates(o, 3, (1, 1, 1))
        seasonals = as_candidates(s, 4, (0, 1, 1, 7))

        out = [{"order": oo, "seasonal_order": ss} for oo in orders for ss in seasonals]
        return out or default

    # unknown shape → default
    return default


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

    # Build parameter iterator per model
    if model_name == "sarimax":
        param_iter = _normalize_sarimax_space(search_space)
    else:
        # RF and other sklearn models keep ParameterGrid behavior
        if search_space in (None, {}):
            param_iter = [{}]
        else:
            param_iter = list(ParameterGrid(search_space))

    best_params = None
    best_report = None
    best_score = np.inf
    log.info("=== Hyperparameter Search (model=%s) ===", model_name)

    for raw_params in param_iter:
        # Filter to supported keys
        allowed = ALLOWED_PARAMS.get(model_name, set())
        params = {k: v for k, v in raw_params.items() if k in allowed}

        # For SARIMAX ensure tuples (idempotent if already correct)
        if model_name == "sarimax":
            params = _coerce_sarimax_params(params)

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

    assert best_params is not None and best_report is not None
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
            from statsmodels.stats.diagnostic import acorr_ljungbox
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

    if model_name == "sarimax":
        # Calendar + exog design for training
        X_train = _exog_matrix_for_sarimax(y.index, exog, add_calendar=feat_cfg.add_calendar)
        m = ModelRegistry[model_name](**params)
        m.fit(y, X_train)
        # Build future exog consistent with training columns
        X_future = _exog_future_for_sarimax(
            y_index=y.index,
            horizon=horizon,
            exog_hist=exog,
            add_calendar=feat_cfg.add_calendar,
            columns_like=list(X_train.columns)
        )
        y_hat = m.predict(horizon=horizon, X_future=X_future)
    else:
        # RF iterative
        X_train = make_features(y, feat_cfg, exog)
        if exog is not None:
            exog_future = _build_exog_future_if_needed(
                exog_cols=list(exog.columns),
                y_index=y.index,
                horizon=horizon,
                hist_df=exog
            )
        else:
            exog_future = None
        m = ModelRegistry[model_name](**params)
        m.fit(y, X_train)
        y_hat = m.predict_iterative(y_hist=y, horizon=horizon, feat_cfg=feat_cfg, exog_future=exog_future)

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
    search_space: Dict[str, List[Any]] = None  # set default in __post_init__

    # Features
    feature_config: FeatureConfig = None  # set default in __post_init__

    # Exogenous
    use_exog: bool = False  # when True, caller should pass aligned exog DataFrame or exog_cols

    # Logging
    log_level: int = logging.INFO

    # Serialization options
    max_serialize_points: Optional[int] = None  # serialize all if None

    def __post_init__(self):
        if self.search_space is None:
            self.search_space = {
                "max_depth": [6, 10],
                "min_samples_leaf": [1, 4],
                "n_estimators": [300, 600],  # optional; default remains 400 if not provided
            }
        if self.feature_config is None:
            self.feature_config = FeatureConfig(
                lags=[1, 7, 14, 28],
                roll_windows=[7, 14, 28],
                add_calendar=True
            )


# ---- Data preparation helpers ------------------------------------------------

def _ensure_datetime_index(df: pd.DataFrame, ts_col: Optional[str]) -> Tuple[pd.DataFrame, str]:
    if ts_col is None:
        # Try to infer: first datetime-like column
        datetime_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]
        if not datetime_cols:
            # attempt to parse any column that looks like datetime
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

    df = df.sort_values(ts_col)
    df = df.set_index(ts_col)
    return df, ts_col


def _infer_target(df: pd.DataFrame, target: Optional[str], exog_cols: Optional[List[str]]) -> str:
    if target is not None:
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not in DataFrame.")
        return target
    # infer: choose a single numeric column that is not in exog
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
    """
    Returns y (Series with DateTimeIndex) and exog (DataFrame aligned to y) or None.
    """
    if isinstance(data, pd.Series):
        y = data.copy()
        if not isinstance(y.index, pd.DatetimeIndex):
            y.index = pd.to_datetime(y.index, errors="raise", utc=False)
        y = y.sort_index()
        return y, None

    if not isinstance(data, pd.DataFrame):
        raise TypeError("run_forecasting data must be pandas Series or DataFrame")

    df = data.copy()
    # Ensure datetime index
    df, ts_col = _ensure_datetime_index(df, ts_col)

    # Select exogenous columns if requested
    exog = None
    if use_exog and exog_cols:
        missing = [c for c in exog_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Exogenous columns not in DataFrame: {missing}")
        exog = df[exog_cols].copy()

    # Infer or validate target
    y_col = _infer_target(df, target=target, exog_cols=exog_cols if exog_cols else None)
    y = df[y_col].astype(float).copy()

    # Remove target column from exog if accidentally included
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
    Orchestrates the full pipeline and returns a JSON-serializable dict.

    Parameters:
      - data: Series (DateTimeIndex) or DataFrame with a time column
      - config: RunConfig
      - target: optional target column name (when data is a DataFrame)
      - ts_col: optional time column name (when data is a DataFrame)
      - exog_cols: optional list of exogenous column names
      - freq: optional explicit frequency (if not inferable)

    Works with existing call style: run_forecasting(df, cfg)
    """
    logging.getLogger().setLevel(config.log_level)

    # Prepare inputs (supports both Series and DataFrame)
    y, exog = _prepare_inputs(
        data=data,
        target=target,
        ts_col=ts_col,
        exog_cols=exog_cols,
        use_exog=config.use_exog
    )

    # Ensure supported index and align exog once at the top-level
    y, _ = _ensure_supported_index(y)
    if exog is not None:
        exog = _align_exog_to_y_index(y.index, exog)

    # Early validation: ensure we can run at least one fold
    min_len = config.initial_train + config.horizon
    if len(y) < min_len:
        raise ValueError(
            f"Series too short: len(y)={len(y)} < initial_train+horizon={min_len}"
        )

    # Optionally enforce frequency (if provided)
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

    # Hyperparameter search + backtest
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

    # Residual diagnostics
    diag = residual_diagnostics(best_report)

    # Final train and forecast
    final = train_full_and_forecast(
        y=y,
        horizon=config.horizon,
        model_name=config.model_name,
        params=best_params,
        feat_cfg=config.feature_config,
        exog=exog if config.use_exog else None
    )

    # Optional thinning of output payload size
    def _thin(series: pd.Series, k: Optional[int]) -> pd.Series:
        if not k or len(series) <= k:
            return series
        step = max(1, len(series) // k)
        return series.iloc[::step]

    bt_preds_ser = _thin(best_report.preds, config.max_serialize_points)
    bt_trues_ser = _thin(best_report.trues, config.max_serialize_points)

    # Build JSON-friendly payload
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
        "diagnostics": {
            "residuals": diag
        },
        "final_forecast": final_forecast,
        "model_name": final.model_name,
    }

    log.info("=== run_forecasting completed ===")
    log.info("Selected params: %s", best_params)
    log.info("Backtest metrics: %s", best_report.metrics)
    log.info("Forecast horizon: %d | First ts: %s", config.horizon, final.forecast.index[0])
    return result
