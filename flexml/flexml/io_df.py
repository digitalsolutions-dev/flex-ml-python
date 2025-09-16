import io
import polars as pl
import pandas as pd
import duckdb as ddb


def load_tabular(file_bytes: bytes, filename: str):
    name = filename.lower()
    if name.endswith(".csv"):
        return pl.read_csv(io.BytesIO(file_bytes), infer_schema_length=1000)
    elif name.endswith((".xlsx", ".xls")):
        pdf = pd.read_excel(io.BytesIO(file_bytes))
        return pl.from_pandas(pdf)
    else:
        # Fallback: try CSV
        return pl.read_csv(io.BytesIO(file_bytes), infer_schema_length=1000)


# ---- New: version-agnostic dtype classifier ----
_NUMERIC_PREFIXES = ("Int", "UInt", "Float", "Decimal")
_DATETIME_TOKENS = ("Datetime", "Date", "Time", "Duration")


def _role_from_dtype(dtype: pl.DataType) -> str:
    # Polars DataType prints like 'Int64', 'Float32', 'Utf8', 'Datetime[µs]', 'Decimal(10,2)', etc.
    s = str(dtype)
    if s.startswith(_NUMERIC_PREFIXES):
        return "numeric"
    if any(tok in s for tok in _DATETIME_TOKENS):
        return "datetime"
    return "categorical"


def quick_profile(df: pl.DataFrame, limit_html: int = 0):
    rows, cols = df.height, df.width
    schema = []
    for c in df.columns:
        s = df[c]
        role = _role_from_dtype(s.dtype)
        try:
            missing_rate = float(s.null_count()) / max(1, rows)
        except Exception:
            missing_rate = 0.0
        try:
            nunique = int(s.n_unique())
        except Exception:
            nunique = 0
        schema.append({
            "name": c,
            "dtype": str(s.dtype),
            "role": role,
            "missing_rate": missing_rate,
            "nunique": nunique
        })
    metrics = {"rows": rows, "cols": cols, "schema": schema}

    html = None
    if limit_html > 0 and rows <= 200000:
        con = ddb.connect()
        con.register("t", df.to_arrow())
        preview = con.execute("SELECT * FROM t LIMIT 50").df().to_html(index=False)
        html = f"<html><body><h3>Preview (first 50 rows)</h3>{preview}</body></html>".encode("utf-8")
    return {"metrics": metrics, "html": html}


def sample_for_suggest(s3_key: str, filename: str):
    # Keep simple; suggestions use filename heuristics today
    return {"filename": filename}
