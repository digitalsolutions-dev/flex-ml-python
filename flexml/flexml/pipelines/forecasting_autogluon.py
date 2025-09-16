"""
AutoGluon TimeSeries forecasting (AG >= 1.1.0)

This module provides a self-contained path to run forecasting using AutoGluon
without disturbing your existing RF / SARIMAX logic.

Key features:
- Robust construction of TimeSeriesDataFrame for single or multi-series
- Optional known covariates (a.k.a. exogenous / regressors)
- Frequency normalization (convert_frequency) and index alignment
- Simple synthesis of future covariates if you don't have them
- JSON-friendly output compatible with your current results structure

Typical usage (from tasks.py):
    result = run_autogluon_forecasting(
        df,                               # pandas DataFrame
        target="request_cnt",
        ts_col="dt",
        exog_cols=["total_color_kg", "distinct_user_creators", "distinct_colors"],
        horizon=21,
        time_limit=600,
        hyperparameters={"SeasonalNaive": {}, "AutoETS": {}},
        prediction_item_id="series_0",
        freq="D",
        save_dir=None,                    # or a path to save the predictor
    )
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

    HAVE_AG = True
except Exception:  # pragma: no cover
    TimeSeriesDataFrame = object  # type: ignore
    TimeSeriesPredictor = object  # type: ignore
    HAVE_AG = False

log = logging.getLogger(__name__)


# ---------------------------- Dataclasses ------------------------------------


@dataclass
class AGConfig:
    """Config for AutoGluon run. Use sensible defaults if not provided."""
    horizon: int = 14
    time_limit: int = 600
    freq: str = "D"
    hyperparameters: Optional[dict] = None
    eval_metric: str = "MAPE"
    verbosity: int = 2
    # If you pass a directory, we'll save the trained predictor
    save_dir: Optional[str] = None


# ---------------------------- Helpers ----------------------------------------


def _ensure_datetime(series: pd.Series) -> pd.Series:
    """Coerce a column to datetime (raise if totally invalid)."""
    s = pd.to_datetime(series, errors="raise")
    return s


def _ensure_item_id(df: pd.DataFrame, item_id_col: Optional[str]) -> Tuple[pd.DataFrame, str]:
    """
    Ensure an item_id column exists. If missing, create a single-series id 'series_0'.
    Returns (df_copy, item_id_col_name).
    """
    df = df.copy()
    if item_id_col is None or item_id_col not in df.columns:
        df["__item_id__"] = "series_0"
        return df, "__item_id__"
    return df, item_id_col


def _to_tsdf(
        df: pd.DataFrame,
        item_id_col: str,
        ts_col: str,
        value_cols: List[str],
        freq: str,
) -> TimeSeriesDataFrame:
    """
    Convert a long-form DataFrame into a TimeSeriesDataFrame with given value columns.
    For target, pass value_cols=[target].
    For covariates, pass value_cols=exog_cols.

    Notes:
        - We call convert_frequency(freq) to normalize to a regular grid.
        - We forward/back fill within each item_id to handle occasional gaps.
    """
    # Prepare a narrow DF: item_id | timestamp | <values...>
    keep = [item_id_col, ts_col] + value_cols
    miss = [c for c in keep if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns for TSDF: {miss}")

    narrow = df[keep].copy()
    narrow = narrow.rename(columns={item_id_col: "item_id", ts_col: "timestamp"})
    narrow["timestamp"] = _ensure_datetime(narrow["timestamp"])
    narrow = narrow.sort_values(["item_id", "timestamp"])

    ts = TimeSeriesDataFrame.from_data_frame(
        df=narrow,
        id_column="item_id",
        timestamp_column="timestamp",
    )

    # Normalize to regular frequency
    ts = ts.convert_frequency(freq=freq, agg_numeric="mean")

    # ffill/bfill per item to remove NaNs introduced by regularization (safe for covariates)
    ts = ts.groupby(level=0).apply(lambda g: g.ffill().bfill())
    return ts


def _as_two_level_index(idx: pd.Index) -> pd.MultiIndex:
    """
    Normalize a TimeSeriesDataFrame index to a 2-level MultiIndex
    [item_id, timestamp] with consistent names. Works across AG versions.

    - If already 2-level MI -> just ensure names.
    - If MI with >2 levels   -> keep the first two (item_id, timestamp).
    - If not MI (unlikely)   -> build from tuples.
    """
    if isinstance(idx, pd.MultiIndex):
        if idx.nlevels == 2:
            return idx.set_names(["item_id", "timestamp"])
        # fallback: keep the first two levels
        lvl0 = idx.get_level_values(0)
        lvl1 = idx.get_level_values(1)
        return pd.MultiIndex.from_arrays([lvl0, lvl1], names=["item_id", "timestamp"])
    else:
        # Some older conversions might return a sequence of tuples
        # or an Index of tuple-like values.
        try:
            tuples = list(idx)
            return pd.MultiIndex.from_tuples(tuples, names=["item_id", "timestamp"])
        except Exception as e:
            raise ValueError(f"Unexpected index type/shape for TimeSeriesDataFrame: {type(idx)}") from e


def _normalize_ts_index(idx: pd.Index) -> pd.MultiIndex:
    """
    Return a 2-level MultiIndex [item_id, timestamp] with the timestamp level
    converted to tz-naive datetime64[ns]. Works across AG versions.
    """
    # Build a 2-level MI [item, ts]
    if isinstance(idx, pd.MultiIndex):
        # take first two levels if there are more
        lvl_item = idx.get_level_values(0)
        lvl_ts = idx.get_level_values(1)
    else:
        # idx might be an Index of tuples
        tuples = list(idx)
        if not tuples or not isinstance(tuples[0], tuple) or len(tuples[0]) < 2:
            raise ValueError(f"Unexpected index shape/type: {type(idx)}")
        lvl_item = pd.Index([t[0] for t in tuples])
        lvl_ts = pd.Index([t[1] for t in tuples])

    # Coerce timestamp to tz-naive datetime64[ns]
    ts = pd.to_datetime(lvl_ts, errors="coerce", utc=False)

    # If tz-aware slipped in, strip tz to make it naive
    if getattr(ts.dtype, "tz", None) is not None:
        # pandas Series with tz-aware dtype
        ts = ts.tz_convert(None)

    # Some pandas versions need .dt.tz_localize(None) instead:
    # (guarded to avoid raising when tz is already None)
    try:
        if hasattr(ts.dt, "tz") and ts.dt.tz is not None:
            ts = ts.dt.tz_localize(None)
    except Exception:
        pass

    if ts.isna().any():
        bad = int(ts.isna().sum())
        raise ValueError(f"{bad} timestamp values could not be parsed to datetime64[ns].")

    return pd.MultiIndex.from_arrays([pd.Index(lvl_item), pd.Index(ts)], names=["item_id", "timestamp"])


import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame


def _flatten_tsdf(ts: "TimeSeriesDataFrame") -> pd.DataFrame:
    """
    Flatten a TimeSeriesDataFrame to a pandas.DataFrame with columns:
        ['item_id', 'timestamp', 'target', ...]
    Robust to:
      - Single-level DateTimeIndex
      - MultiIndex with != 2 levels (uses first as item_id, last as timestamp)
      - Existing 'item_id'/'timestamp' index level names (we drop the index before assigning columns)
    """
    df = pd.DataFrame(ts).copy()

    # --- Extract index values before we drop the index
    idx = df.index
    nlevels = getattr(idx, "nlevels", 1)

    if nlevels == 1:
        # Single index: interpret as timestamp (coerce to datetime), synthesize item_id=0
        ts_vals = idx
        if not isinstance(ts_vals, pd.DatetimeIndex):
            ts_vals = pd.to_datetime(ts_vals, errors="coerce", utc=False)
        item_vals = pd.Series(0, index=df.index)
    else:
        # MultiIndex: first level -> item_id, last level -> timestamp
        item_vals = idx.get_level_values(0)
        ts_vals = idx.get_level_values(nlevels - 1)
        ts_vals = pd.to_datetime(ts_vals, errors="coerce", utc=False)

    # --- Drop the index entirely to avoid name collisions
    df.reset_index(drop=True, inplace=True)

    # --- Assign canonical columns
    df["item_id"] = item_vals.astype(str)  # string ids are robust
    df["timestamp"] = ts_vals
    if isinstance(df["timestamp"].dtype, pd.DatetimeTZDtype):  # older pandas compatibility
        df["timestamp"] = df["timestamp"].dt.tz_convert(None)

    # Ensure we have a 'target' column (AutoGluon standard)
    if "target" not in df.columns:
        for cand in ("y", "value", "request_cnt"):
            if cand in df.columns:
                df = df.rename(columns={cand: "target"})
                break
    if "target" not in df.columns:
        raise ValueError("Expected a 'target' column in the data.")

    # Clean bad timestamps
    df = df.dropna(subset=["timestamp"])

    # Order columns and sort
    front = ["item_id", "timestamp"]
    rest = [c for c in df.columns if c not in front]
    df = df[front + rest].sort_values(["item_id", "timestamp"])
    return df


def _split_train_val_last_n(ts: "TimeSeriesDataFrame", holdout: int):
    """
    Split last `holdout` points per series into validation.
    Returns (train_ts, val_ts) as TimeSeriesDataFrame objects.
    """
    df = _flatten_tsdf(ts)  # columns: item_id, timestamp, target, ...

    train_rows, val_rows = [], []
    for _, g in df.groupby("item_id", sort=False):
        if len(g) <= holdout:
            train_rows.append(g)
        else:
            train_rows.append(g.iloc[:-holdout, :])
            val_rows.append(g.iloc[-holdout:, :])

    train_df = pd.concat(train_rows, axis=0) if train_rows else pd.DataFrame(columns=df.columns)
    val_df = pd.concat(val_rows, axis=0) if val_rows else pd.DataFrame(columns=df.columns)

    def _to_tsdf(pdf: pd.DataFrame) -> "TimeSeriesDataFrame":
        if pdf.empty:
            empty = (pd.DataFrame(columns=["item_id", "timestamp", "target"])
                     .set_index(["item_id", "timestamp"]))
            return TimeSeriesDataFrame(empty)
        pdf = pdf.copy()
        pdf["item_id"] = pdf["item_id"].astype(str)
        pdf["timestamp"] = pd.to_datetime(pdf["timestamp"], errors="coerce", utc=False)
        if isinstance(pdf["timestamp"].dtype, pd.DatetimeTZDtype):
            pdf["timestamp"] = pdf["timestamp"].dt.tz_convert(None)
        pdf = pdf.dropna(subset=["timestamp"])
        pdf = pdf.set_index(["item_id", "timestamp"]).sort_index()
        return TimeSeriesDataFrame(pdf)

    train_ts = _to_tsdf(train_df)
    val_ts = _to_tsdf(val_df)
    return train_ts, val_ts


def _synthesize_future_covariates(
        last_cov: pd.DataFrame,
        horizon: int,
        freq: str,
) -> TimeSeriesDataFrame:
    """
    Build a simple future covariates frame by carrying forward the last observed covariate values.
    last_cov: pandas DataFrame with MultiIndex [item_id, timestamp] and covariate columns.
    """
    if last_cov.empty:
        return TimeSeriesDataFrame()

    # Get the last timestamp per item
    last_cov = last_cov.copy()
    last_cov.index = pd.MultiIndex.from_tuples(last_cov.index, names=["item_id", "timestamp"])

    future_parts = []
    for item_id, g in last_cov.groupby(level=0):
        if g.empty:
            continue
        last_ts = g.index.get_level_values("timestamp").max()
        future_index = pd.date_range(last_ts, periods=horizon + 1, freq=freq, inclusive="neither")
        # Carry forward last row's values
        last_vals = g.iloc[-1].to_frame().T
        # Create repeated rows with future index
        repeat = pd.concat([last_vals] * len(future_index), ignore_index=True)
        repeat.index = pd.MultiIndex.from_product([[item_id], future_index], names=["item_id", "timestamp"])
        future_parts.append(repeat)

    fut = pd.concat(future_parts) if future_parts else pd.DataFrame()
    cov_future = TimeSeriesDataFrame(fut)
    cov_future = cov_future.groupby(level=0).apply(lambda g: g.ffill().bfill())
    return cov_future


# ---------------------------- Public API -------------------------------------


def run_autogluon_forecasting(
        df: pd.DataFrame,
        target: str,
        ts_col: str,
        exog_cols: Optional[List[str]],
        horizon: int,
        time_limit: int = 600,
        hyperparameters: Optional[dict] = None,
        eval_metric: str = "MAPE",
        verbosity: int = 2,
        freq: str = "D",
        item_id_col: Optional[str] = None,
        prediction_item_id: str = "series_0",
        save_dir: Optional[str] = None,
) -> Dict:
    """
    Train an AutoGluon TimeSeriesPredictor with optional known covariates and
    forecast the next horizon. Returns a JSON-friendly dict similar to other pipelines.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data with at least [ts_col, target] and optionally exog_cols and item_id_col.
    target : str
        Target column name (numeric).
    ts_col : str
        Timestamp column name (datetime-like or coercible).
    exog_cols : list[str] or None
        Known covariates (regressors). Can be empty/None.
    horizon : int
        Prediction length / forecast horizon.
    time_limit : int
        Seconds budget for AutoGluon.
    hyperparameters : dict or None
        Model set. Example: {"SeasonalNaive": {}, "AutoETS": {}, "NPTS": {}}
    eval_metric : str
        E.g. "MAPE", "MASE", "sMAPE", "RMSE".
    verbosity : int
        AutoGluon verbosity level.
    freq : str
        Pandas frequency string ("D", "W", "MS", ...).
    item_id_col : str or None
        Series identifier column. If None or missing, we create a single-series "series_0".
    prediction_item_id : str
        The id to use when we synthesize single-series inputs.
    save_dir : str or None
        If provided, predictor is saved to this directory.

    Returns
    -------
    dict
        {
          "model_name": "autogluon",
          "best_model": "<name>",
          "leaderboard_key": "results/<dataset_id>_ag_leaderboard.json" (if you persist it),
          "final_forecast": [{"ts": "...", "y_hat": float}, ...],
          "ag_summary": {...},   # best model, metric
        }
    """
    if not HAVE_AG:
        raise RuntimeError(
            "AutoGluon is not installed. Please add 'autogluon.timeseries>=1.1.0' to requirements."
        )

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")

    # Ensure item_id existence
    df, item_id_col = _ensure_item_id(df, item_id_col)
    if df[item_id_col].nunique() == 1:
        df[item_id_col] = prediction_item_id  # normalize single id name

    # --- Build target TSDF
    ts_target = _to_tsdf(
        df=df,
        item_id_col=item_id_col,
        ts_col=ts_col,
        value_cols=[target],
        freq=freq,
    )

    # --- Train/test split (holdout last=horizon for quick internal eval)
    train_ts, test_ts = _split_train_val_last_n(ts_target, holdout=horizon)

    # --- Known covariates (optional)
    cov_train = None
    cov_test = None
    cov_future = None
    if exog_cols:
        # Full covariate TSDF (aligned & regularized)
        ts_cov = _to_tsdf(
            df=df,
            item_id_col=item_id_col,
            ts_col=ts_col,
            value_cols=exog_cols,
            freq=freq,
        )
        # Align to train/test windows
        cov_train = ts_cov.reindex(train_ts.index).groupby(level=0).apply(lambda g: g.ffill().bfill())

        if len(test_ts) > 0:
            cov_test = ts_cov.reindex(test_ts.index).groupby(level=0).apply(lambda g: g.ffill().bfill())

        # Also prepare simple future covariates for production (carry-forward last values)
        cov_future = _synthesize_future_covariates(pd.DataFrame(ts_cov), horizon=horizon, freq=freq)

    # --- Fit
    hp = hyperparameters or {
        "SeasonalNaive": {},
        "AutoETS": {},
        "NPTS": {},
        # Add lightweight DL models if desired:
        # "PatchTST": {"epochs": 5},
        # "TiDE": {"epochs": 5},
        # "DeepAR": {"epochs": 5},
    }

    predictor = TimeSeriesPredictor(
        prediction_length=horizon,
        freq=freq,
        eval_metric=eval_metric,
        verbosity=verbosity,
    )

    log.info("Fitting AutoGluon (time_limit=%s)", time_limit)
    predictor.fit(
        train_data=train_ts,
        known_covariates=cov_train if cov_train is not None and not cov_train.empty else None,
        known_covariates_names=exog_cols if exog_cols else None,
        hyperparameters=hp,
        time_limit=int(time_limit),
    )

    # --- Predict next horizon
    # If we have a test split, predict on test; otherwise create a future frame from the last train point
    if len(test_ts) > 0:
        forecast_ts = predictor.predict(
            data=test_ts,
            known_covariates=cov_test if cov_test is not None and not cov_test.empty else None,
        )
    else:
        # Build minimal future frame by slicing last horizon-length context (AG can project forward)
        # Known covariates for future are passed if available
        forecast_ts = predictor.predict(
            data=train_ts,  # AG will forecast next horizon per series
            known_covariates=cov_future if cov_future is not None and not cov_future.empty else None,
        )

    # Serialize forecast to JSON-friendly list
    # forecast_ts is a TSDF with target column named 'mean' or model-dependent; convert to pandas
    fpdf = pd.DataFrame(forecast_ts)
    fpdf.index = pd.MultiIndex.from_tuples(forecast_ts.index, names=["item_id", "timestamp"])
    # Expect one (or many) series; if many, we return all
    final_out: List[Dict] = []
    for (item_id, ts), row in fpdf.sort_index().iterrows():
        # Some models expose 'mean', others use the target name — choose first numeric
        yhat = None
        for c in fpdf.columns:
            if pd.api.types.is_numeric_dtype(fpdf[c]):
                yhat = float(row[c])
                break
        if yhat is None:
            continue
        final_out.append({"item_id": str(item_id), "ts": ts.isoformat(), "y_hat": yhat})

    # Optional save
    if save_dir:
        try:
            predictor.save(save_dir)  # AG 1.1.0+ returns the path; older versions ignore the arg
        except TypeError:
            # Older AG expects no argument to save; ignore save_dir or use predictor.save()
            predictor.save()

    # Leaderboard (best model) — not persisted here, but summarized
    try:
        lb = predictor.leaderboard(silent=True)
        # get best row
        best_row = lb.iloc[0].to_dict() if len(lb) else {}
    except Exception:
        best_row = {}

    out = {
        "model_name": "autogluon",
        "best_model": best_row.get("model", None),
        "best_score": best_row.get("score_val", None),
        "final_forecast": final_out,
        "ag_summary": {
            "eval_metric": eval_metric,
            "time_limit": time_limit,
            "hyperparameters": hp,
        },
    }
    return out
