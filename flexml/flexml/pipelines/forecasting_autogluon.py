"""
AutoGluon TimeSeries forecasting (compatible with AG versions with/without known_covariates)

Key features:
- Robust construction of TimeSeriesDataFrame for single or multi-series
- Optional known covariates; gracefully disabled if unsupported by installed AG version
- Frequency normalization and index alignment (tz-naive Timestamps)
- JSON-friendly output aligned with other pipelines
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional AutoGluon support
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
    save_dir: Optional[str] = None


# ---------------------------- Helpers ----------------------------------------

def _ensure_tsdf_target_name(ts: "TimeSeriesDataFrame", orig_target_name: str) -> "TimeSeriesDataFrame":
    """
    Ensure the TimeSeriesDataFrame has a single numeric column named 'target'.
    If it currently uses the original CSV column name (e.g., 'request_cnt'),
    rename it to 'target'.
    """
    pdf = pd.DataFrame(ts).copy()

    # If already correct
    if list(pdf.columns) == ["target"]:
        return TimeSeriesDataFrame(pdf)

    # If the original target name exists, rename that
    if orig_target_name in pdf.columns:
        pdf = pdf.rename(columns={orig_target_name: "target"})
        return TimeSeriesDataFrame(pdf)

    # If there's exactly one column, rename it to 'target'
    num_cols = [c for c in pdf.columns if pd.api.types.is_numeric_dtype(pdf[c])]
    if len(pdf.columns) == 1:
        pdf = pdf.rename(columns={pdf.columns[0]: "target"})
        return TimeSeriesDataFrame(pdf)

    # As a fallback, if there is exactly one numeric column, use it
    if len(num_cols) == 1:
        pdf = pdf.rename(columns={num_cols[0]: "target"})
        return TimeSeriesDataFrame(pdf)

    raise ValueError(
        f"Could not identify the target column to rename to 'target'. "
        f"Columns present: {list(pdf.columns)} (expected '{orig_target_name}' or a single numeric column)."
    )


def _ensure_datetime(series: pd.Series) -> pd.Series:
    """Coerce a column to datetime (raise if totally invalid)."""
    s = pd.to_datetime(series, errors="raise", utc=False)
    # Strip timezone if present
    try:
        if getattr(s.dt, "tz", None) is not None:
            s = s.dt.tz_convert(None)
    except Exception:
        pass
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


def _parse_datetimes_safely(values) -> pd.DatetimeIndex:
    """
    Parse to tz-naive Timestamps without emitting the pandas 'Could not infer format' warning.
    Strategy:
      1) Normalize Period -> Timestamp.
      2) Try to detect a uniform format across a small sample and use vectorized parse with format=...
      3) If not reliable, fall back to element-wise parsing (no warning).
    """
    # Fast paths for already-datetime index types
    if isinstance(values, pd.DatetimeIndex):
        ts = values
    elif isinstance(values, pd.PeriodIndex):
        ts = values.to_timestamp()
    else:
        # Flatten into a list and normalize Period elements
        try:
            seq = list(values)
        except TypeError:
            seq = [values]
        flat = []
        for v in seq:
            if isinstance(v, pd.Period):
                flat.append(v.to_timestamp())
            else:
                flat.append(v)

        # Try to detect a uniform string format on a small sample
        str_vals = [v for v in flat if isinstance(v, str)]
        fmt = None
        if str_vals:
            sample = [s.strip() for s in str_vals[:50] if s and s.strip()]

            def all_match(p):
                return len(sample) > 0 and all(re.match(p, s) for s in sample)

            # Pure date patterns
            if all_match(r"^\d{4}-\d{2}-\d{2}$"):
                fmt = "%Y-%m-%d"
            elif all_match(r"^\d{4}/\d{2}/\d{2}$"):
                fmt = "%Y/%m/%d"
            elif all_match(r"^\d{2}\.\d{2}\.\d{4}$"):
                fmt = "%d.%m.%Y"
            elif all_match(r"^\d{2}/\d{2}/\d{4}$"):
                # Disambiguate dd/mm vs mm/dd by looking for any day>12
                parts = [s.split("/") for s in sample]
                if any(p[0].isdigit() and int(p[0]) > 12 for p in parts if len(p) == 3):
                    fmt = "%d/%m/%Y"
                else:
                    fmt = "%m/%d/%Y"
            # Datetime (seconds)
            elif all_match(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}$"):
                fmt = "%Y-%m-%dT%H:%M:%S" if "T" in sample[0] else "%Y-%m-%d %H:%M:%S"
            # Datetime (minutes)
            elif all_match(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}$"):
                fmt = "%Y-%m-%dT%H:%M" if "T" in sample[0] else "%Y-%m-%d %H:%M"

        if fmt:
            # Vectorized parse with explicit format (fast, no warning)
            ts = pd.to_datetime(flat, errors="coerce", utc=False, format=fmt)
        else:
            # Element-wise parse to avoid the global-warning path
            parsed = [pd.to_datetime(v, errors="coerce", utc=False) for v in flat]
            ts = pd.DatetimeIndex(parsed)

    # Strip timezone if present
    try:
        if isinstance(ts, pd.DatetimeIndex) and ts.tz is not None:
            ts = ts.tz_convert(None)
    except Exception:
        pass
    try:
        if isinstance(ts, pd.DatetimeIndex) and ts.tz is not None:
            ts = ts.tz_localize(None)
    except Exception:
        pass

    return ts


def _ensure_two_level_index(idx: pd.Index) -> pd.MultiIndex:
    """
    Normalize any index to a 2-level MultiIndex [item_id, timestamp] with tz-naive
    pandas Timestamps at level 1. Robust to:
      - Mixed level ordering (detect timestamp level automatically)
      - PeriodIndex or arrays of Period (converted via .to_timestamp())
      - tz-aware datetimes (tz removed)
      - Index of tuples (item, timestamp) in any order
      - Single DatetimeIndex (treated as single-series with item_id="0")
    """

    def _is_timestamp_like(level_vals) -> bool:
        # Quick heuristics: true for Datetime/Period index; else try parsing a small sample
        if isinstance(level_vals, (pd.DatetimeIndex, pd.PeriodIndex)):
            return True
        try:
            # Parse a tiny sample to avoid O(n) cost
            seq = list(level_vals[:20]) if hasattr(level_vals, "__getitem__") else list(level_vals)
            ts = _parse_datetimes_safely(seq)
            if len(ts) == 0:
                return False
            nat_ratio = float(ts.isna().mean())
            return nat_ratio <= 0.05
        except Exception:
            return False

    def _build_mi(item_vals, ts_vals) -> pd.MultiIndex:
        item_idx = pd.Index([str(x) for x in item_vals], dtype=str)
        ts_idx = _parse_datetimes_safely(ts_vals)
        if ts_idx.isna().any():
            bad = int(ts_idx.isna().sum())
            raise ValueError(f"Timestamp level contains unparseable values (NaT count={bad}).")
        return pd.MultiIndex.from_arrays([item_idx, ts_idx], names=["item_id", "timestamp"])

    # Case A: MultiIndex — auto-detect which level is timestamp-like
    if isinstance(idx, pd.MultiIndex):
        n = idx.nlevels
        if n >= 2:
            lvl_vals = [idx.get_level_values(i) for i in range(n)]
            ts_candidates = [i for i in range(n) if _is_timestamp_like(lvl_vals[i])]
            ts_level = ts_candidates[-1] if ts_candidates else (n - 1)
            item_level = 0 if ts_level != 0 else (1 if n > 1 else 0)
            return _build_mi(lvl_vals[item_level], lvl_vals[ts_level])

        # Rare: MI with 1 level. Try rebuilding from tuples.
        tuples = list(idx)
        if not tuples:
            return pd.MultiIndex.from_arrays(
                [pd.Index([], dtype=str), pd.DatetimeIndex([])],
                names=["item_id", "timestamp"],
            )
        cand0 = [t[0] for t in tuples]
        cand1 = [t[1] for t in tuples]
        if _is_timestamp_like(cand0) and not _is_timestamp_like(cand1):
            return _build_mi(cand1, cand0)
        if _is_timestamp_like(cand1):
            return _build_mi(cand0, cand1)
        return _build_mi(cand0, cand1)

    # Case B: Non-MI: maybe Index of tuples (item, timestamp) in either order
    try:
        tuples = list(idx)
        if tuples and isinstance(tuples[0], tuple) and len(tuples[0]) >= 2:
            cand0 = [t[0] for t in tuples]
            cand1 = [t[1] for t in tuples]
            if _is_timestamp_like(cand0) and not _is_timestamp_like(cand1):
                return _build_mi(cand1, cand0)
            if _is_timestamp_like(cand1):
                return _build_mi(cand0, cand1)
            return _build_mi(cand0, cand1)
    except Exception:
        pass

    # Case C: Single index ⇒ treat as single-series timestamps
    ts = _parse_datetimes_safely(idx)
    if ts.isna().any():
        bad = int(ts.isna().sum())
        raise ValueError(f"Timestamp level contains unparseable values (NaT count={bad}).")
    items = pd.Index(["0"] * len(ts), dtype=str)
    return pd.MultiIndex.from_arrays([items, ts], names=["item_id", "timestamp"])


def _to_tsdf(
        df: pd.DataFrame,
        item_id_col: str,
        ts_col: str,
        value_cols: List[str],
        freq: str,
        fill_na: bool,
        agg_numeric: str = "mean",
) -> "TimeSeriesDataFrame":
    """
    Convert a long-form DataFrame into a TimeSeriesDataFrame with given value columns.
    For target, pass value_cols=[target] and fill_na=False.
    For covariates, pass value_cols=exog_cols and fill_na=True.
    """
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
    ts = ts.convert_frequency(freq=freq, agg_numeric=agg_numeric)

    # ffill/bfill per item for covariates only
    if fill_na:
        ts = ts.groupby(level=0).apply(lambda g: g.ffill().bfill())

    # Normalize index for downstream ops
    ts.index = _ensure_two_level_index(ts.index)
    return ts


def _flatten_tsdf(ts: "TimeSeriesDataFrame") -> pd.DataFrame:
    """
    Flatten a TimeSeriesDataFrame to a pandas.DataFrame with columns:
        ['item_id', 'timestamp', 'target', ...]
    """
    df = pd.DataFrame(ts).copy()
    idx = _ensure_two_level_index(df.index)
    item_vals = idx.get_level_values(0).astype(str)
    ts_vals = pd.to_datetime(idx.get_level_values(1), errors="coerce", utc=False)

    df.reset_index(drop=True, inplace=True)
    df["item_id"] = item_vals
    df["timestamp"] = ts_vals

    if "target" not in df.columns:
        for cand in ("y", "value", "request_cnt"):
            if cand in df.columns:
                df = df.rename(columns={cand: "target"})
                break
    if "target" not in df.columns:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(num_cols) == 1:
            df = df.rename(columns={num_cols[0]: "target"})
        else:
            raise ValueError("Expected a 'target' column in the data.")

    df = df.dropna(subset=["timestamp"])
    df = df.sort_values(["item_id", "timestamp"])
    return df


def _split_train_val_last_n(ts: "TimeSeriesDataFrame", holdout: int):
    """
    Split last `holdout` points per series into validation.
    Returns (train_ts, val_ts) as TimeSeriesDataFrame objects.
    """
    df = _flatten_tsdf(ts)

    train_rows, val_rows = [], []
    for _, g in df.groupby("item_id", sort=False):
        if len(g) <= holdout:
            train_rows.append(g)
        else:
            train_rows.append(g.iloc[:-holdout, :])
            val_rows.append(g.iloc[-holdout:, :])

    train_df = pd.concat(train_rows, axis=0) if train_rows else pd.DataFrame(columns=df.columns)
    val_df = pd.concat(val_rows, axis=0) if val_rows else pd.DataFrame(columns=df.columns)

    def _to_tsdf_back(pdf: pd.DataFrame) -> "TimeSeriesDataFrame":
        if pdf.empty:
            empty = (pd.DataFrame(columns=["item_id", "timestamp", "target"])
                     .set_index(["item_id", "timestamp"]))
            return TimeSeriesDataFrame(empty)
        pdf = pdf.copy()
        pdf["item_id"] = pdf["item_id"].astype(str)
        pdf["timestamp"] = _ensure_datetime(pdf["timestamp"])
        pdf = pdf.dropna(subset=["timestamp"])
        pdf = pdf.set_index(["item_id", "timestamp"]).sort_index()
        pdf.index = _ensure_two_level_index(pdf.index)
        return TimeSeriesDataFrame(pdf)

    train_ts = _to_tsdf_back(train_df)
    val_ts = _to_tsdf_back(val_df)
    return train_ts, val_ts


def _synthesize_future_covariates(
        last_cov: pd.DataFrame,
        horizon: int,
        freq: str,
) -> "TimeSeriesDataFrame":
    """
    Build a simple future covariates frame by carrying forward the last observed covariate values.
    last_cov: pandas DataFrame with MultiIndex [item_id, timestamp] and covariate columns.
    """
    if last_cov.empty:
        return TimeSeriesDataFrame()

    last_cov = last_cov.copy()
    last_cov.index = _ensure_two_level_index(last_cov.index)

    future_parts = []
    for item_id, g in last_cov.groupby(level=0, sort=False):
        if g.empty:
            continue
        last_ts = g.index.get_level_values("timestamp").max()
        future_index = pd.date_range(last_ts, periods=horizon + 1, freq=freq, inclusive="neither")
        last_vals = g.iloc[-1].to_frame().T
        repeat = pd.concat([last_vals] * len(future_index), ignore_index=True)
        repeat.index = pd.MultiIndex.from_product([[item_id], future_index], names=["item_id", "timestamp"])
        future_parts.append(repeat)

    fut = pd.concat(future_parts) if future_parts else pd.DataFrame()
    cov_future = TimeSeriesDataFrame(fut)
    cov_future.index = _ensure_two_level_index(cov_future.index)
    cov_future = cov_future.groupby(level=0).apply(lambda g: g.ffill().bfill())
    return cov_future


def _first_numeric_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    raise ValueError("No numeric column found to use as forecast output.")


def _ag_supports_known_covariates() -> bool:
    """Return True if the installed AutoGluon exposes `known_covariates` in fit()/predict."""
    try:
        fit_sig = inspect.signature(TimeSeriesPredictor.fit)
        pred_sig = inspect.signature(TimeSeriesPredictor.predict)
        return ("known_covariates" in fit_sig.parameters) and ("known_covariates" in pred_sig.parameters)
    except Exception:
        return False


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
        df[item_id_col] = prediction_item_id

    # --- Build target TSDF (no fill on target)
    ts_target = _to_tsdf(
        df=df,
        item_id_col=item_id_col,
        ts_col=ts_col,
        value_cols=[target],
        freq=freq,
        fill_na=False,
        agg_numeric="mean",
    )

    # Ensure target column is named 'target'
    ts_target = _ensure_tsdf_target_name(ts_target, orig_target_name=target)

    # --- Train/test split (holdout last=horizon)
    train_ts, test_ts = _split_train_val_last_n(ts_target, holdout=horizon)

    # --- Known covariates (optional; may be disabled by AG version)
    cov_train = None
    cov_test = None
    cov_future = None
    covariates_supported = _ag_supports_known_covariates()
    covariates_used = False

    if exog_cols:
        if covariates_supported:
            ts_cov = _to_tsdf(
                df=df,
                item_id_col=item_id_col,
                ts_col=ts_col,
                value_cols=exog_cols,
                freq=freq,
                fill_na=True,  # safe for covariates
                agg_numeric="mean",
            )
            ts_cov.index = _ensure_two_level_index(ts_cov.index)
            train_idx = _ensure_two_level_index(train_ts.index)
            test_idx = _ensure_two_level_index(test_ts.index) if len(test_ts) > 0 else None

            cov_train = ts_cov.reindex(train_idx).groupby(level=0).apply(lambda g: g.ffill().bfill())
            if test_idx is not None and len(test_idx) > 0:
                cov_test = ts_cov.reindex(test_idx).groupby(level=0).apply(lambda g: g.ffill().bfill())

            cov_future = _synthesize_future_covariates(pd.DataFrame(ts_cov), horizon=horizon, freq=freq)
            covariates_used = True
        else:
            log.warning(
                "AutoGluon version does not support known_covariates in fit/predict; "
                "proceeding without exogenous regressors."
            )

    # --- Fit AutoGluon
    hp = hyperparameters or {
        "SeasonalNaive": {},
        "AutoETS": {},
        "NPTS": {},
        # enable lightweight DL models if desired:
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

    if covariates_used:
        predictor.fit(
            train_data=train_ts,
            known_covariates=cov_train if cov_train is not None and not getattr(cov_train, "empty", False) else None,
            known_covariates_names=exog_cols if exog_cols else None,
            hyperparameters=hp,
            time_limit=int(time_limit),
        )
    else:
        predictor.fit(
            train_data=train_ts,
            hyperparameters=hp,
            time_limit=int(time_limit),
        )

    # --- Backtest over the held-out window (if any)
    backtest = None
    if len(test_ts) > 0:
        # IMPORTANT:
        # Predict the *validation window* (next horizon after train end),
        # so we must pass data=train_ts here (NOT test_ts).
        if covariates_used:
            # cov_test is indexed on the validation window timestamps -> correct for future steps
            test_fc = predictor.predict(
                data=train_ts,
                known_covariates=cov_test if cov_test is not None and not getattr(cov_test, "empty", False) else None,
            )
        else:
            test_fc = predictor.predict(data=train_ts)

        # Align predictions to the true validation window and compute metrics
        test_df = pd.DataFrame(test_ts).rename(columns={test_ts.columns[0]: "y"})
        pred_df = pd.DataFrame(test_fc)

        test_df.index = _ensure_two_level_index(test_df.index)
        pred_df.index = _ensure_two_level_index(pred_df.index)

        # Inner-join on the common validation timestamps to avoid NaNs
        both = (
            pred_df.join(test_df[["y"]], how="inner")
            .sort_index()
        )
        fcol = _first_numeric_column(both)

        # Drop any remaining NaNs for fair metric calculation
        both = both[np.isfinite(both[fcol]) & np.isfinite(both["y"])]

        y_true = both["y"].astype(float).values
        y_hat = both[fcol].astype(float).values

        def _rmse(y, yhat):
            return float(np.sqrt(np.mean((y - yhat) ** 2))) if len(y) else None

        def _smape(y, yhat, eps=1e-8):
            if not len(y): return None
            y, yhat = np.asarray(y, float), np.asarray(yhat, float)
            denom = (np.abs(y) + np.abs(yhat)) + eps
            return float(np.mean(2.0 * np.abs(yhat - y) / denom))

        def _mase(y, yhat, y_insample, m=7, eps=1e-8):
            if not len(y): return None
            y_insample = np.asarray(y_insample, float)
            d = np.abs(np.diff(y_insample, n=m)).mean() + eps
            return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))) / d)

        insample_df = pd.DataFrame(train_ts)
        insample_vals = insample_df.iloc[:, 0].astype(float).values if len(insample_df) else np.array([])

        backtest = {
            "metrics": {
                "RMSE": _rmse(y_true, y_hat),
                "sMAPE": _smape(y_true, y_hat),
                "MASE": _mase(y_true, y_hat, insample_vals, m=7),
            },
            "preds": [{"ts": ts.isoformat(), "y_hat": float(v)} for (item, ts), v in both[fcol].items()],
            "trues": [{"ts": ts.isoformat(), "y": float(v)} for (item, ts), v in both["y"].items()],
            "horizon": int(horizon),
            "step": int(horizon),
            "initial_train": int(len(insample_df) // max(1, train_ts.index.get_level_values(0).nunique())),
        }

    # --- True future forecast (after the end of the full series) ---
    if covariates_used:
        future_fc = predictor.predict(
            data=ts_target,  # NOTE: use the full series here
            known_covariates=cov_future if cov_future is not None and not getattr(cov_future, "empty", False) else None,
        )
    else:
        future_fc = predictor.predict(data=ts_target)

    fpdf = pd.DataFrame(future_fc)
    fpdf.index = _ensure_two_level_index(fpdf.index)
    fpdf = fpdf.sort_index()

    fcol = _first_numeric_column(fpdf)

    multi_series = df[item_id_col].nunique() > 1
    if not multi_series:
        final_out = [{"ts": ts.isoformat(), "y_hat": float(val)}
                     for (_, ts), val in fpdf[fcol].items()]
    else:
        final_out = [{"item_id": str(item), "ts": ts.isoformat(), "y_hat": float(val)}
                     for (item, ts), val in fpdf[fcol].items()]

    # Optional save
    if save_dir:
        try:
            predictor.save(save_dir)
        except TypeError:
            predictor.save()

    # Leaderboard (best model) — summarized only
    try:
        lb = predictor.leaderboard(silent=True)
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
            "time_limit": int(time_limit),
            "hyperparameters": hyperparameters or {"SeasonalNaive": {}, "AutoETS": {}, "NPTS": {}},
            "multi_series": bool(multi_series),
            "covariates_supported": bool(covariates_supported),
            "covariates_used": bool(covariates_used),
        },
    }
    if backtest is not None:
        out["backtest"] = backtest

    return out
