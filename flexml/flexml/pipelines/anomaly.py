import pandas as pd
from sklearn.ensemble import IsolationForest


def run_anomaly(df):
    pdf = df.to_pandas() if hasattr(df, "to_pandas") else df
    df_num = pdf.select_dtypes(include=["number"]).dropna()
    if df_num.empty:
        return {"metrics": {"note": "no numeric columns"}}
    X = df_num.values
    iso = IsolationForest(contamination=0.01, random_state=42).fit(X)
    scores = iso.decision_function(X).tolist()
    return {"metrics": {"model": "isolation_forest", "n": len(scores),
                        "p01_threshold": sorted(scores)[int(0.01 * len(scores))]}}
