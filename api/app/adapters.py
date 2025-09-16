from __future__ import annotations
import pandas as pd


def ensure_pandas(df) -> pd.DataFrame:
    """
    Convert various tabular backends to a clean pandas.DataFrame
    with standard NumPy/pandas dtypes (no Arrow extension arrays).
    Also tries to auto-convert likely datetime columns.
    """
    pdf = None

    # Already pandas
    if isinstance(df, pd.DataFrame):
        pdf = df.copy()

    # Polars
    if pdf is None:
        try:
            import polars as pl  # type: ignore
            if isinstance(df, pl.DataFrame):
                pdf = df.to_pandas(use_pyarrow_extension_array=False)
            elif hasattr(pl, "LazyFrame") and isinstance(df, pl.LazyFrame):
                pdf = df.collect().to_pandas(use_pyarrow_extension_array=False)
        except Exception:
            pass

    # PyArrow
    if pdf is None:
        try:
            import pyarrow as pa  # type: ignore
            if isinstance(df, pa.Table):
                pdf = df.to_pandas(types_mapper=None, use_threads=True)
        except Exception:
            pass

    # DuckDB
    if pdf is None:
        try:
            if hasattr(df, "df") and callable(getattr(df, "df")):
                pdf = df.df()
        except Exception:
            pass

    # Dask
    if pdf is None:
        try:
            import dask.dataframe as dd  # type: ignore
            if isinstance(df, dd.DataFrame):
                pdf = df.compute()
        except Exception:
            pass

    # Vaex
    if pdf is None:
        try:
            import vaex  # type: ignore
            if isinstance(df, vaex.dataframe.DataFrame):
                pdf = df.to_pandas_df()
        except Exception:
            pass

    # Modin
    if pdf is None:
        try:
            import modin.pandas as mpd  # type: ignore
            if isinstance(df, mpd.DataFrame):
                pdf = df._to_pandas()
        except Exception:
            pass

    # Generic to_pandas
    if pdf is None and hasattr(df, "to_pandas") and callable(getattr(df, "to_pandas")):
        try:
            pdf = df.to_pandas()
        except Exception:
            pass

    if pdf is None:
        raise TypeError(
            f"Unsupported DataFrame type {type(df)}. Please convert to pandas before calling the pipeline."
        )

    # 🔑 Normalize datetimes
    for col in pdf.columns:
        if "date" in col.lower() or "time" in col.lower():
            try:
                pdf[col] = pd.to_datetime(pdf[col])
            except (ValueError, TypeError):
                pass

    return pdf
