import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def run_clustering(df, k: int = 5):
    pdf = df.to_pandas() if hasattr(df, "to_pandas") else df
    df_num = pdf.select_dtypes(include=["number"]).dropna()
    if len(df_num) < k:
        return {"metrics": {"note": "insufficient rows for chosen k"}}
    X = StandardScaler().fit_transform(df_num.values)
    km = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X)
    sil = float(silhouette_score(X, km.labels_)) if len(df_num) > k else 0.0
    return {"metrics": {"model": "kmeans", "k": k, "silhouette": sil}}
