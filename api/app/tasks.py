import json
from datetime import date, datetime
import numpy as np
from .celery_app import celery
from flexml.io_s3 import get_bytes, put_bytes
from flexml.io_df import load_tabular, quick_profile
from flexml.pipelines import run_forecasting, RunConfig, FeatureConfig, run_classification, predict_classification_df, \
    run_clustering, run_anomaly, AGConfig, run_autogluon_forecasting
from .adapters import ensure_pandas
import logging

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

LOG_FMT = "[%(levelname)s] %(asctime)s %(name)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
log = logging.getLogger("tasks")


def _json_default(o):
    # Datetime-like
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    # NumPy scalars (np.float64, np.int64, etc.)
    if isinstance(o, np.generic):
        return o.item()
    # Fallback: stringify (safe for Decimal, Timedelta, etc.)
    return str(o)


@celery.task(name="profile_dataset")
def profile_dataset_task(dataset_id: int, s3_key: str, filename: str):
    raw = get_bytes(s3_key)
    df = load_tabular(raw, filename)
    prof = quick_profile(df, limit_html=2000000)
    metrics = prof["metrics"]
    if prof.get("html"):
        report_key = f"profiles/{dataset_id}.html"
        put_bytes(report_key, prof["html"], "text/html")
        metrics["report_key"] = report_key
    meta_key = f"profiles/{dataset_id}.json"
    put_bytes(meta_key, json.dumps(metrics, ensure_ascii=False, default=_json_default).encode("utf-8"),
              "application/json")
    return {"dataset_id": dataset_id, **metrics}


@celery.task(name="run_pipeline")
def run_pipeline_task(dataset_id: int, s3_key: str, filename: str, config: dict):
    """
    Executes the requested pipeline on the uploaded dataset.

    Classification supports:
      - mode="train"   : trains & exports a model (pipeline) + meta (schema & threshold)
      - mode="predict" : loads clf_model_key, scores the input CSV, returns JSON preview or stores a scored CSV

    Returns a JSON artifact written to results/{dataset_id}.json (always),
    plus (optionally) a scored CSV written to results/{dataset_id}_scored.csv when return_file=true.
    """

    # Print out the config
    log.info(f"Pipeline config: {config}")

    # -------- 1) Load dataset --------
    try:
        raw = get_bytes(s3_key)
    except Exception as e:
        out = {"error": f"Failed to read input object: {e}"}
        meta_key = f"results/{dataset_id}.json"
        put_bytes(meta_key, json.dumps(out, ensure_ascii=False, default=_json_default).encode("utf-8"),
                  "application/json")
        return {"dataset_id": dataset_id, "artifacts_key": meta_key, **out}

    try:
        df_in = load_tabular(raw, filename)
    except Exception as e:
        out = {"error": f"Failed to parse file '{filename}': {e}"}
        meta_key = f"results/{dataset_id}.json"
        put_bytes(meta_key, json.dumps(out, ensure_ascii=False, default=_json_default).encode("utf-8"),
                  "application/json")
        return {"dataset_id": dataset_id, "artifacts_key": meta_key, **out}

    # Normalize any backend to pandas
    try:
        df = ensure_pandas(df_in)
    except Exception as e:
        out = {"error": f"Failed to convert input to pandas DataFrame: {e}"}
        meta_key = f"results/{dataset_id}.json"
        put_bytes(meta_key, json.dumps(out, ensure_ascii=False, default=_json_default).encode("utf-8"),
                  "application/json")
        return {"dataset_id": dataset_id, "artifacts_key": meta_key, **out}

    # -------- 2) Route by pipeline kind --------
    kind = (config or {}).get("type")
    try:
        if kind == "forecasting":
            # Common fields from caller
            horizon = int(config.get("horizon", 14))
            initial_train = int(config.get("initial_train", 360))
            step = int(config.get("step", 14))
            model_name = config.get("model_name", "rf")
            ts_col = config.get("ts_col")
            target = config.get("target")
            exog_cols = config.get("exog_cols") or []
            freq = config.get("freq")  # optional, ex: "D"
            log.info("Pipeline config: %s", config)

            if model_name == "autogluon":
                # Defaults for a quick run; override via config if desired
                hp = config.get("ag_hyperparameters") or {
                    "SeasonalNaive": {},
                    "AutoETS": {},
                    "NPTS": {},
                    # uncomment to add light DL models:
                    # "PatchTST": {"epochs": 5},
                    # "TiDE": {"epochs": 5},
                }
                time_limit = int(config.get("time_limit", 600))
                freq_eff = freq or "D"

                out = run_autogluon_forecasting(
                    df=df,
                    target=target if target else _pick_numeric_target(df, exclude=exog_cols),
                    ts_col=ts_col if ts_col else _pick_ts_col(df),
                    exog_cols=exog_cols,
                    horizon=horizon,
                    time_limit=time_limit,
                    hyperparameters=hp,
                    eval_metric=config.get("eval_metric", "MAPE"),
                    verbosity=int(config.get("verbosity", 2)),
                    freq=freq_eff,
                    item_id_col=config.get("item_id_col"),  # optional
                    prediction_item_id=config.get("item_id_fallback", "series_0"),
                    save_dir=config.get("ag_save_dir"),  # optional
                )

        elif kind == "classification":
            mode = (config.get("mode") or "train").lower()
            # Print out the mode
            log.info(f"Classification mode: {mode}")

            if mode == "train":
                log.info(f"We are into training mode for the dataset {dataset_id}!")
                # Train and export model + metadata
                target = config.get("target")  # optional; auto-infer if None
                # Allow caller to pass a desired model key; else default based on dataset_id
                raw_key = config.get("clf_model_key")  # could be None / "" / "  ...  "
                log.info(f"Passed model key (raw): {raw_key!r}")

                if isinstance(raw_key, str):
                    raw_key = raw_key.strip()

                clf_model_key = raw_key or f"models/dataset_{dataset_id}_classification.joblib"
                log.info(f"Using model key: {clf_model_key}")

                # Run training; returns dict with clf_model_key, meta_key, cv_results, etc.
                out = run_classification(
                    df,
                    target=target,
                    export_model=True,
                    clf_model_key=clf_model_key,
                    put_bytes_fn=lambda key, data, ctype: put_bytes(key, data, ctype)
                )
                # Keep clf_model_key in the response for the UI to use for prediction calls
                if "export" in out and "clf_model_key" in out["export"]:
                    out["clf_model_key"] = out["export"]["clf_model_key"]

            elif mode == "predict":
                log.info(f"We are into predicting mode!")
                # Score the uploaded CSV using an existing model
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
                    meta_key=meta_key
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
                        "note": "Scored CSV stored to object storage."
                    }
                else:
                    log.info(f"We are not returning file we just want a JSON preview!")
                    # Return a compact JSON preview to avoid huge payloads
                    out = {
                        "ok": True,
                        "mode": "predict",
                        "clf_model_key": clf_model_key,
                        "rows": int(len(df_scored)),
                        "preview": df_scored.head(20).to_dict(orient="records")
                    }

            else:
                raise ValueError(f"Unsupported classification mode: {mode}")

        elif kind == "clustering":
            out = run_clustering(df, k=int(config.get("k", 5)))

        elif kind == "anomaly":
            out = run_anomaly(df)

        else:
            out = {"error": f"Unknown pipeline type: {kind}"}

    except Exception as e:
        log.exception("Pipeline execution failed")
        out = {"error": str(e)}

    # -------- 3) Persist the JSON artifact --------
    meta_key = f"results/{dataset_id}.json"
    try:
        put_bytes(meta_key, json.dumps(out, ensure_ascii=False, default=_json_default).encode("utf-8"),
                  "application/json")
    except Exception as e:
        # In the unlikely event artifact write fails, still return what we have
        log.exception("Failed to write result artifact")
        out = {**out, "artifact_write_error": str(e)}

    return {"dataset_id": dataset_id, "artifacts_key": meta_key, **out}
