from pydantic import BaseModel, Field
from typing import Any, Dict, List
import duckdb, polars as pl
from flexml.io_s3 import get_bytes
from flexml.pipelines import run_forecasting, RunConfig, FeatureConfig, run_classification, run_clustering, run_anomaly


class SQLSelectReq(BaseModel):
    dataset_key: str
    sql: str
    limit: int = Field(default=500, ge=1, le=5000)


class ForecastReq(BaseModel):
    dataset_key: str
    time_col: str
    target: str
    horizon: int = 28


class ClassifyReq(BaseModel):
    dataset_key: str
    target: str = "target"


class ClusterReq(BaseModel):
    dataset_key: str
    k: int = 5


class AnomalyReq(BaseModel):
    dataset_key: str


TOOLS_SCHEMA: List[Dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "sql_select", "description": "Run read-only SELECT on table t.",
        "parameters": {"type": "object", "properties": {
            "dataset_key": {"type": "string"},
            "sql": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 5000, "default": 500}
        }, "required": ["dataset_key", "sql"]}}},
    {"type": "function", "function": {
        "name": "ml_forecast", "description": "Prophet baseline forecasting.",
        "parameters": {"type": "object", "properties": {
            "dataset_key": {"type": "string"},
            "time_col": {"type": "string"},
            "target": {"type": "string"},
            "horizon": {"type": "integer", "default": 28}
        }, "required": ["dataset_key", "time_col", "target"]}}},
    {"type": "function", "function": {
        "name": "ml_classify", "description": "XGBoost classification (AUROC).",
        "parameters": {"type": "object", "properties": {
            "dataset_key": {"type": "string"},
            "target": {"type": "string", "default": "target"}
        }, "required": ["dataset_key"]}}},
    {"type": "function", "function": {
        "name": "ml_cluster", "description": "KMeans clustering (silhouette).",
        "parameters": {"type": "object", "properties": {
            "dataset_key": {"type": "string"},
            "k": {"type": "integer", "default": 5}
        }, "required": ["dataset_key"]}}},
    {"type": "function", "function": {
        "name": "ml_anomaly", "description": "IsolationForest anomaly detection.",
        "parameters": {"type": "object", "properties": {
            "dataset_key": {"type": "string"}
        }, "required": ["dataset_key"]}}},
]


def _load_polars(dataset_key: str) -> pl.DataFrame:
    raw = get_bytes(dataset_key)
    try:
        return pl.read_csv(raw)
    except Exception:
        import io, pyarrow.parquet as pq
        try:
            table = pq.read_table(io.BytesIO(raw))
            return pl.from_arrow(table)
        except Exception:
            import pandas as pd, io as _io
            return pl.from_pandas(pd.read_csv(_io.BytesIO(raw)))


def tool_sql_select(args: Dict[str, Any]) -> Dict[str, Any]:
    req = SQLSelectReq(**args)
    df = _load_polars(req.dataset_key)
    con = duckdb.connect()
    con.register("t", df.to_arrow())
    sql = req.sql.strip().rstrip(";")
    low = sql.lower()
    if not low.startswith("select "): raise ValueError("Only SELECT is allowed")
    banned = ["insert ", "update ", "delete ", "drop ", "alter ", "create ", "replace ", "grant ", "revoke "]
    if any(b in low for b in banned): raise ValueError("Read-only SELECT only")
    res = con.execute(sql).df()
    truncated = False
    if req.limit and len(res) > req.limit:
        res = res.head(req.limit);
        truncated = True
    return {"rows": res.to_dict(orient="records"), "truncated": truncated, "rowcount": len(res)}


def tool_ml_forecast(args: Dict[str, Any]) -> Dict[str, Any]:
    req = ForecastReq(**args)
    df = _load_polars(req.dataset_key)
    out = run_forecasting(df, time_col=req.time_col, target=req.target, horizon=req.horizon)
    return {"metrics": out.get("metrics", {}), "forecasts_sample": out.get("forecasts", [])[:50]}


def tool_ml_classify(args: Dict[str, Any]) -> Dict[str, Any]:
    req = ClassifyReq(**args)
    df = _load_polars(req.dataset_key)
    out = run_classification(df, target=req.target)
    return {"metrics": out.get("metrics", {})}


def tool_ml_cluster(args: Dict[str, Any]) -> Dict[str, Any]:
    req = ClusterReq(**args)
    df = _load_polars(req.dataset_key)
    out = run_clustering(df, k=req.k)
    return {"metrics": out.get("metrics", {})}


def tool_ml_anomaly(args: Dict[str, Any]) -> Dict[str, Any]:
    req = AnomalyReq(**args)
    df = _load_polars(req.dataset_key)
    out = run_anomaly(df)
    return {"metrics": out.get("metrics", {})}


TOOLS_IMPL = {
    "sql_select": tool_sql_select,
    "ml_forecast": tool_ml_forecast,
    "ml_classify": tool_ml_classify,
    "ml_cluster": tool_ml_cluster,
    "ml_anomaly": tool_ml_anomaly,
}
