"""
Celery task definitions for running ML pipelines triggered from FastAPI.

Key improvements:
- Added _pick_ts_col and _pick_numeric_target helpers for AutoGluon branch.
- Robust error handling with consistent artifact writing and clear messages.
- Harmonized forecasting defaults and search space.
- Output schema consistency across forecasting families (RF/SARIMAX/AutoGluon).
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import numpy as np

from .celery_app import celery
from .adapters import ensure_pandas

from flexml.io_s3 import get_bytes, put_bytes
from flexml.io_df import load_tabular, quick_profile
from flexml.pipelines import (
    run_forecasting,
    RunConfig,
    FeatureConfig,
    run_classification,
    predict_classification_df,
    run_clustering,
    run_anomaly,
    AGConfig,
    run_autogluon_forecasting,
)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

LOG_FMT = "[%(levelname)s] %(asctime)s %(name)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
log = logging.getLogger("tasks")


# -----------------------------------------------------------------------------
# JSON serializer helpers
# -----------------------------------------------------------------------------

def _json_default(o):
    # Datetime-like
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    # NumPy scalars (np.float64, np.int64, etc.)
    if isinstance(o, np.generic):
        return o.item()
    # Fallback: stringify (safe for Decimal, Timedelta, etc.)
    return str(o)


def _write_artifact(dataset_id: int, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Persist JSON artifact to object storage and return a standardized envelope.
    """
    meta_key = f"results/{dataset_id}.json"
    try:
        put_bytes(
            meta_key,
            json.dumps(payload, ensure_ascii=False, default=_json_default).encode("utf-8"),
            "application/json",
        )
        return {"dataset_id": dataset_id, "artifacts_key": meta_key, **payload}
    except Exception as e:
        log.exception("Failed to write result artifact")
        # Return what we have with an explicit write error
        return {"dataset_id": dataset_id, "artifact_write_error": str(e), **payload}


# -----------------------------------------------------------------------------
# Light heuristics for AutoGluon inference (when target/ts_col not provided)
# -----------------------------------------------------------------------------

def _pick_ts_col(df) -> str:
    """
    First datetime-like column or any column parsable to datetime.
    Raises on failure.
    """
    import pandas as pd

    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.datetime64):
            return c
    # Try to parse columns lazily (non-destructive)
    for c in df.columns:
        try:
            pd.to_datetime(df[c], errors="raise", utc=False)
            return c
        except Exception:
            continue
    raise ValueError("Could not infer time column; please provide ts_col.")


def _pick_numeric_target(df, exclude: Optional[List[str]] = None) -> str:
    """
    Prefer a single numeric column not in exclude; if multiple candidates,
    try 'y' or 'target', otherwise take the first numeric column.
    """
    import pandas as pd

    exclude_set = set(exclude or [])
    numeric_candidates = [
        c for c in df.columns
        if c not in exclude_set and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not numeric_candidates:
        raise ValueError("Could not infer numeric target; please provide target.")
    for preferred in ("y", "target"):
        if preferred in numeric_candidates:
            return preferred
    return numeric_candidates[0]


# -----------------------------------------------------------------------------
# Dataset profiling task
# -----------------------------------------------------------------------------

@celery.task(name="profile_dataset")
def profile_dataset_task(dataset_id: int, s3_key: str, filename: str):
    try:
        raw = get_bytes(s3_key)
        df = load_tabular(raw, filename)
        prof = quick_profile(df, limit_html=2_000_000)
        metrics = prof["metrics"]
        if prof.get("html"):
            report_key = f"profiles/{dataset_id}.html"
            put_bytes(report_key, prof["html"], "text/html")
            metrics["report_key"] = report_key
        meta_key = f"profiles/{dataset_id}.json"
        put_bytes(
            meta_key,
            json.dumps(metrics, ensure_ascii=False, default=_json_default).encode("utf-8"),
            "application/json",
        )
        return {"dataset_id": dataset_id, **metrics}
    except Exception as e:
        log.exception("profile_dataset failed")
        return {"dataset_id": dataset_id, "error": str(e)}


# -----------------------------------------------------------------------------
# Main pipeline task
# -----------------------------------------------------------------------------

@celery.task(name="run_pipeline")
def run_pipeline_task(dataset_id: int, s3_key: str, filename: str, config: dict):
    """
    Executes the requested pipeline on the uploaded dataset.

    Classification supports:
      - mode="train"   : trains & exports a model (pipeline) + meta (schema & threshold)
      - mode="predict" : loads clf_model_key, scores the input CSV, returns JSON preview
                         or stores a scored CSV when return_file=true.

    Always writes a JSON artifact to results/{dataset_id}.json.
    """

    log.info("Pipeline config: %s", config)

    # -------- 1) Load dataset --------
    try:
        raw = get_bytes(s3_key)
    except Exception as e:
        out = {"error": f"Failed to read input object: {e}"}
        return _write_artifact(dataset_id, out)

    try:
        df_in = load_tabular(raw, filename)
    except Exception as e:
        out = {"error": f"Failed to parse file '{filename}': {e}"}
        return _write_artifact(dataset_id, out)

    # Normalize any backend to pandas
    try:
        df = ensure_pandas(df_in)
    except Exception as e:
        out = {"error": f"Failed to convert input to pandas DataFrame: {e}"}
        return _write_artifact(dataset_id, out)

    # -------- 2) Route by pipeline kind --------
    kind = (config or {}).get("type")
    log.info("Pipeline kind: %s", kind)
    out: Dict[str, Any] = {}

    try:
        # ----------------- Forecasting -----------------
        if kind == "forecasting":
            log.info("Entering forecasting path for dataset %s", dataset_id)

            horizon = int(config.get("horizon", 14))
            initial_train = int(config.get("initial_train", 360))
            step = int(config.get("step", 14))
            raw_model = config.get("model_name")
            model_name = (raw_model or "rf")
            if not isinstance(model_name, str):
                model_name = "rf"
            model_name = model_name.lower()
            if model_name not in {"rf", "sarimax", "autogluon"}:
                log.warning("Unknown model_name %r; defaulting to 'rf'", model_name)
                model_name = "rf"
            ts_col = config.get("ts_col")  # may be None
            target = config.get("target")  # may be None
            exog_cols = list(config.get("exog_cols") or [])
            freq = config.get("freq")  # optional, ex: "D"
            add_calendar = bool(config.get("add_calendar", True))

            # If the caller selects AutoGluon, use AG path and keep output schema parity
            if model_name == "autogluon":
                # Defaults for a quick run; override via config if desired
                hp = config.get("ag_hyperparameters") or {
                    "SeasonalNaive": {},
                    "AutoETS": {},
                    "NPTS": {},
                    # Lightweight DL models can be enabled if desired:
                    # "PatchTST": {"epochs": 5},
                    # "TiDE": {"epochs": 5},
                }
                time_limit = int(config.get("time_limit", 600))
                freq_eff = freq or "D"

                # Heuristics if not provided explicitly
                try:
                    target_eff = target if target else _pick_numeric_target(df, exclude=exog_cols)
                except Exception as e:
                    raise ValueError(f"AutoGluon: target inference failed: {e}")
                try:
                    ts_col_eff = ts_col if ts_col else _pick_ts_col(df)
                except Exception as e:
                    raise ValueError(f"AutoGluon: timestamp inference failed: {e}")

                # Optional multi-series support via item_id_col
                item_id_col = config.get("item_id_col")  # may be None
                prediction_item_id = config.get("item_id_fallback", "series_0")
                save_dir = config.get("ag_save_dir")  # may be None

                out = run_autogluon_forecasting(
                    df=df,
                    target=target_eff,
                    ts_col=ts_col_eff,
                    exog_cols=exog_cols,
                    horizon=horizon,
                    time_limit=time_limit,
                    hyperparameters=hp,
                    eval_metric=config.get("eval_metric", "MAPE"),
                    verbosity=int(config.get("verbosity", 2)),
                    freq=freq_eff,
                    item_id_col=item_id_col,
                    prediction_item_id=prediction_item_id,
                    save_dir=save_dir,
                )

            else:
                # --- RF / SARIMAX path ---
                # Ensure the search space includes n_estimators for RF consistency
                search_space = config.get("search_space")
                if search_space is None:
                    if model_name == "sarimax":
                        # Small, stable space for daily data with weekly seasonality
                        search_space = [
                            {"order": (1, 1, 1), "seasonal_order": (1, 1, 1, 7)},
                            {"order": (2, 1, 1), "seasonal_order": (1, 1, 1, 7)},
                            {"order": (1, 1, 2), "seasonal_order": (1, 1, 1, 7)},
                        ]
                    else:
                        # RF defaults
                        search_space = {
                            "max_depth": [8, 12],
                            "min_samples_leaf": [1, 3],
                            "n_estimators": [300, 600],
                        }

                cfg = RunConfig(
                    horizon=horizon,
                    initial_train=initial_train,
                    step=step,
                    model_name=model_name,  # "rf" or "sarimax"
                    search_space=search_space,
                    feature_config=FeatureConfig(
                        lags=config.get("lags", [1, 7, 14, 28]),
                        roll_windows=config.get("roll_windows", [7, 14, 28]),
                        add_calendar=add_calendar,
                    ),
                    use_exog=bool(exog_cols),  # auto-enable if provided
                    log_level=logging.INFO,
                    # Advanced knobs (optional)
                    max_serialize_points=config.get("max_serialize_points"),
                    mase_m=config.get("mase_m"),
                    max_ffill_steps=config.get("max_ffill_steps"),
                )

                out = run_forecasting(
                    data=df,
                    config=cfg,
                    target=target,
                    ts_col=ts_col,
                    exog_cols=exog_cols,
                    freq=freq,
                )

        # ----------------- Classification -----------------
        elif kind == "classification":
            log.info("Entering classification path for dataset %s", dataset_id)
            mode = (config.get("mode") or "train").lower()

            if mode == "train":
                target = config.get("target")  # optional; auto-infer if None inside pipeline

                # Allow caller to pass a desired model key; else default based on dataset_id
                raw_key = config.get("clf_model_key")
                if isinstance(raw_key, str):
                    raw_key = raw_key.strip()
                clf_model_key = raw_key or f"models/dataset_{dataset_id}_classification.joblib"

                # Train and export model + metadata
                out = run_classification(
                    df,
                    target=target,
                    export_model=True,
                    clf_model_key=clf_model_key,
                    put_bytes_fn=lambda key, data, ctype: put_bytes(key, data, ctype),
                )
                # Keep clf_model_key in the response for the UI to use for prediction calls
                if "export" in out and "clf_model_key" in out["export"]:
                    out["clf_model_key"] = out["export"]["clf_model_key"]

            elif mode == "predict":
                clf_model_key = config.get("clf_model_key")
                if not clf_model_key:
                    raise ValueError("mode='predict' requires 'clf_model_key' to be provided in config.")
                meta_key = config.get("meta_key")  # optional override
                return_file = bool(config.get("return_file", False))

                # Run inference; returns a DataFrame with proba/pred appended
                df_scored = predict_classification_df(
                    df,
                    get_bytes_fn=lambda key: get_bytes(key),
                    clf_model_key=clf_model_key,
                    meta_key=meta_key,
                )

                if return_file:
                    # Persist a scored CSV to results/
                    scored_key = f"results/{dataset_id}_scored.csv"
                    csv_bytes = df_scored.to_csv(index=False).encode("utf-8")
                    put_bytes(scored_key, csv_bytes, "text/csv")
                    out = {
                        "ok": True,
                        "mode": "predict",
                        "clf_model_key": clf_model_key,
                        "scored_key": scored_key,
                        "rows": int(len(df_scored)),
                        "note": "Scored CSV stored to object storage.",
                    }
                else:
                    # Return a compact JSON preview to avoid huge payloads
                    out = {
                        "ok": True,
                        "mode": "predict",
                        "clf_model_key": clf_model_key,
                        "rows": int(len(df_scored)),
                        "preview": df_scored.head(20).to_dict(orient="records"),
                    }

            else:
                raise ValueError(f"Unsupported classification mode: {mode}")

        # ----------------- Clustering -----------------
        elif kind == "clustering":
            out = run_clustering(df, k=int(config.get("k", 5)))

        # ----------------- Anomaly Detection -----------------
        elif kind == "anomaly":
            out = run_anomaly(df)

        else:
            out = {"error": f"Unknown pipeline type: {kind}"}

    except Exception as e:
        log.exception("Pipeline execution failed")
        out = {"error": str(e)}

    # -------- 3) Persist the JSON artifact --------
    return _write_artifact(dataset_id, out)
