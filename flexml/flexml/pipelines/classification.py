"""
Robust binary classification pipeline for heterogeneous tabular datasets.

Features
--------
- Target inference (name-based and heuristic mapping to 0/1)
- Column pruning with safe heuristics (missingness, IDs)
- Normalization:
  * Numeric-like coercion (currency, thousand sep, %, k/m/b suffixes)
  * Datetime parsing with derived calendar features
  * Text detection (TF-IDF) vs. categorical (OneHot)
- Stratified split with guard to keep both classes in test
- Imbalance-aware XGBoost (scale_pos_weight)
- Optional probability calibration (isotonic/sigmoid) on a valid holdout
- CV fallback when a proper test evaluation is not feasible
- Model export (joblib) & metadata (JSON)
"""

from __future__ import annotations

# ======================= Standard Library & Third-Party =======================

import io
import json
import logging
import re
import warnings
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

# ================================ Logging =====================================

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# ========================= Public Configuration ===============================

@dataclass(frozen=True)
class Config:
    # Column pruning
    MISSING_DROP_THRESHOLD: float = 0.98
    ID_UNIQUE_RATIO: float = 0.95
    DATETIME_SUCCESS_RATIO: float = 0.70
    NUMERIC_LIKE_SUCCESS_RATIO: float = 0.85

    # Text detection
    TEXT_MIN_AVG_LEN: int = 15
    TEXT_MIN_UNIQUE: int = 30

    # Encoders
    OHE_MAX_CATEGORIES: int = 50
    OHE_MIN_FREQ: float = 0.01
    TFIDF_MAX_FEATURES: int = 3000
    TFIDF_MIN_DF: int = 2
    TFIDF_MAX_DF: float = 0.98

    # Class imbalance & splitting
    IMBALANCE_RATIO: float = 3.0
    CV_ON_SINGLE_CLASS: bool = True
    CV_MAX_SPLITS: int = 5

    # Thresholding
    TOP_FEATURES: int = 30  # exposed for possible future feature reporting

    # Calibration
    USE_CALIBRATION: bool = True
    CALIBRATION_METHOD: str = "isotonic"  # "isotonic" or "sigmoid"
    CALIBRATION_MIN_VALID: int = 100  # minimum validation samples to calibrate


# Defaults to match original constants
DEFAULT_CONFIG = Config()

# ========================== Domain Dictionaries ===============================

LIKELY_TARGET_NAMES: List[str] = [
    "target", "label", "churn", "is_churn", "converted", "is_converted",
    "won", "is_won", "default", "is_default", "fraud", "is_fraud",
    "cancelled", "is_cancelled", "returned", "is_returned", "active", "is_active",
    "spam", "is_spam", "approved", "is_approved", "success", "is_success",
]

YES_SET: set[str] = {
    "1", "true", "t", "y", "yes", "on", "positive", "won", "converted", "churn", "fraud", "default",
    "cancelled", "returned", "active", "ja", "sí", "oui", "vero", "sim", "是", "はい", "да", "ano",
}

NO_SET: set[str] = {
    "0", "false", "f", "n", "no", "off", "negative", "lost", "not_converted", "no_churn", "not_fraud",
    "no_default", "not_cancelled", "not_returned", "inactive", "nein", "não", "non", "нет", "いいえ", "否", "nej",
}

# ============================== Regex Helpers =================================

_CURRENCY_RE = re.compile(r"[€$£¥₽₪₹₩₺]+")
_THOUSAND_SEP_RE = re.compile(r"(?<=\d)[,_](?=\d{3}\b)")
_PCT_RE = re.compile(r"%$")
_DATE_PATTERNS: List[Tuple[re.Pattern, Optional[str]]] = [
    (re.compile(r"^\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(:\d{2})?)?$"), None),
    (re.compile(r"^\d{2}/\d{2}/\d{4}$"), "%d/%m/%Y"),  # 05/01/2024
    (re.compile(r"^\d{2}-\d{2}-\d{4}$"), "%d-%m-%Y"),  # 05-01-2024
    (re.compile(r"^\d{4}/\d{2}/\d{2}$"), "%Y/%m/%d"),  # 2024/01/05
]
_EPOCH_SECS = re.compile(r"^\d{10}$")
_EPOCH_MS = re.compile(r"^\d{13}$")


# ==============================================================================
#                           Serialization Utilities
# ==============================================================================

def _serialize_model(pipe: Pipeline) -> bytes:
    """
    Serialize a fitted sklearn Pipeline to bytes with joblib.
    """
    buf = io.BytesIO()
    joblib.dump(pipe, buf)
    return buf.getvalue()


def _deserialize_model(blob: bytes) -> Pipeline:
    """
    Deserialize a sklearn Pipeline from bytes produced by _serialize_model.
    """
    buf = io.BytesIO(blob)
    return joblib.load(buf)


# ==============================================================================
#                           String → Number / Datetime
# ==============================================================================

def _to_datetime_quiet(series: pd.Series, fmt: Optional[str] = None) -> pd.Series:
    """
    Parse to datetime while suppressing pandas format inference warnings.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        if fmt:
            return pd.to_datetime(series, errors="coerce", utc=False, format=fmt)
        return pd.to_datetime(series, errors="coerce", utc=False)


def _looks_like_datetime(series: pd.Series, sample_size: int = 50, min_ratio: float = 0.6) -> Tuple[
    bool, Optional[str]]:
    """
    Heuristically decide whether a string/object series looks datetime-like.
    Returns (is_datetime, preferred_format).
    """
    s = series.dropna()
    if s.empty:
        return False, None
    if not (pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s)):
        return False, None

    sample = s.astype(str).sample(min(len(s), sample_size), random_state=42)

    # Epoch detection
    matches = sample.str.fullmatch(_EPOCH_SECS.pattern).sum() + sample.str.fullmatch(_EPOCH_MS.pattern).sum()
    if matches / len(sample) >= min_ratio:
        return True, None

    # Pattern match
    best_fmt, best_ratio = None, 0.0
    for pat, fmt in _DATE_PATTERNS:
        r = sample.str.match(pat).mean()
        if r > best_ratio:
            best_ratio, best_fmt = r, fmt
    return (best_ratio >= min_ratio), best_fmt


def _detect_datetime(col: pd.Series, cfg: Config = DEFAULT_CONFIG) -> Optional[pd.Series]:
    """
    Attempt to parse an object/string column as datetime and return the parsed series
    iff the success ratio exceeds cfg.DATETIME_SUCCESS_RATIO.
    """
    is_dt, fmt = _looks_like_datetime(col)
    parsed = _to_datetime_quiet(col, fmt) if is_dt else pd.Series([pd.NaT] * len(col), index=col.index)
    ok_ratio = parsed.notna().mean()
    return parsed if ok_ratio >= cfg.DATETIME_SUCCESS_RATIO else None


def _coerce_numeric_like(obj: pd.Series) -> pd.Series:
    """
    Convert common numeric-like strings into numbers:
    - Currency symbols removed
    - Thousand separators handled
    - Percent suffix → fraction
    - k/m/b suffixes → scientific notation
    """
    s = obj.astype(str).str.strip()
    s = s.str.replace(_CURRENCY_RE, "", regex=True)
    s = s.str.replace(_THOUSAND_SEP_RE, "", regex=True)
    pct = s.str.contains(_PCT_RE)
    s = s.str.replace(_PCT_RE, "", regex=True)
    s = s.str.replace(r"[kK]$", "e3", regex=True)
    s = s.str.replace(r"[mM]$", "e6", regex=True)
    s = s.str.replace(r"[bB]$", "e9", regex=True)
    out = pd.to_numeric(s, errors="coerce")
    out[pct] = out[pct] / 100.0
    return out


def _is_textual(col: pd.Series, cfg: Config = DEFAULT_CONFIG) -> bool:
    """
    Decide if a series should be treated as free text (TF-IDF) rather than categorical.
    """
    if not (pd.api.types.is_object_dtype(col) or pd.api.types.is_string_dtype(col)):
        return False
    nunique = col.nunique(dropna=True)
    if nunique < cfg.TEXT_MIN_UNIQUE:
        return False
    avg_len = col.dropna().astype(str).str.len().mean()
    return (avg_len or 0) >= cfg.TEXT_MIN_AVG_LEN


# ==============================================================================
#                                Target Inference
# ==============================================================================

def _map_strings_to_binary(s: pd.Series) -> Tuple[Optional[pd.Series], bool, Dict[str, Any]]:
    """
    Map string values to {0,1} using YES/NO sets; fall back to arbitrary 2-class mapping.
    """
    norm = s.astype(str).str.strip().str.lower()
    mapping: Dict[str, int] = {}
    for v in norm.unique():
        if v in YES_SET:
            mapping[v] = 1
        elif v in NO_SET:
            mapping[v] = 0
    mapped = norm.map(mapping)
    if set(pd.unique(mapped.dropna())) == {0, 1}:
        return mapped.astype(int), True, {"mapping": "string->binary(heuristic)"}
    uniq = sorted(norm.dropna().unique().tolist())
    if len(uniq) == 2:
        fallback = {uniq[0]: 0, uniq[1]: 1}
        return norm.map(fallback).astype(int), True, {"mapping": f"string->binary({fallback})"}
    return None, False, {}


def _coerce_binary(s: pd.Series) -> Tuple[Optional[pd.Series], bool, Dict[str, Any]]:
    """
    Coerce a column to binary {0,1} if possible (bool, numeric 0/1, or two-string classes).
    """
    if pd.api.types.is_bool_dtype(s):
        return s.astype(int), True, {"mapping": "bool->int"}
    if pd.api.types.is_numeric_dtype(s):
        vals = set(pd.unique(s.dropna()))
        if vals.issubset({0, 1}) and len(vals) == 2:
            return s.astype(int), True, {"mapping": "numeric(0/1)"}
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return _map_strings_to_binary(s)
    return None, False, {}


def _infer_binary_target(pdf: pd.DataFrame) -> Tuple[Optional[str], Optional[pd.Series], Dict[str, Any]]:
    """
    Infer a binary target column from a DataFrame using name matching and content heuristics.
    """
    info: Dict[str, Any] = {"inference": None, "candidates_checked": []}
    lower_cols = {c.lower(): c for c in pdf.columns}

    # 1) Name-based
    for name in LIKELY_TARGET_NAMES:
        if name in lower_cols:
            col = lower_cols[name]
            y, ok, map_info = _coerce_binary(pdf[col])
            info["candidates_checked"].append({"column": col, "reason": "name_match", "ok": ok})
            if ok:
                info["inference"] = {"column": col, "reason": "name_match", **map_info}
                return col, y, info

    # 2) Bool dtype
    for col in pdf.columns:
        if pd.api.types.is_bool_dtype(pdf[col]):
            y = pdf[col].astype(int)
            info["inference"] = {"column": col, "reason": "bool_dtype"}
            return col, y, info
        info["candidates_checked"].append({"column": col, "reason": "scan_bool", "ok": False})

    # 3) Numeric 0/1
    for col in pdf.columns:
        s = pdf[col]
        if pd.api.types.is_numeric_dtype(s):
            vals = set(pd.unique(s.dropna()))
            if vals.issubset({0, 1}) and len(vals) == 2:
                y = s.astype(int)
                info["inference"] = {"column": col, "reason": "numeric_01"}
                return col, y, info
        info["candidates_checked"].append({"column": col, "reason": "scan_numeric_01", "ok": False})

    # 4) Two unique strings
    for col in pdf.columns:
        s = pdf[col]
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            uniq = pd.unique(s.dropna().astype(str).str.strip().str.lower())
            if len(uniq) == 2:
                y, ok, map_info = _map_strings_to_binary(s)
                info["candidates_checked"].append({"column": col, "reason": "object_2_unique", "ok": ok})
                if ok:
                    info["inference"] = {"column": col, "reason": "object_2_unique", **map_info}
                    return col, y, info

    return None, None, info


# ==============================================================================
#                          Column Curation & Normalization
# ==============================================================================

def _drop_bad_columns(pdf: pd.DataFrame, target: Optional[str], cfg: Config = DEFAULT_CONFIG) -> Tuple[
    pd.DataFrame, Dict[str, Any]]:
    """
    Drop only harmful columns:
    - Too missing
    - Constant
    - Likely identifiers (near-unique) unless they can be parsed as datetime or numeric-like
    """
    report: Dict[str, Any] = {"dropped": []}
    keep: List[str] = []
    n = len(pdf)

    for c in pdf.columns:
        if target and c == target:
            keep.append(c)
            continue

        s = pdf[c]

        # Too missing
        if s.isna().mean() > cfg.MISSING_DROP_THRESHOLD:
            report["dropped"].append({"column": c, "reason": f"missing>{int(cfg.MISSING_DROP_THRESHOLD * 100)}%"})
            continue

        # Constant
        if s.nunique(dropna=True) <= 1:
            report["dropped"].append({"column": c, "reason": "constant"})
            continue

        # Small-card booleans / two-level categoricals → keep
        if s.nunique(dropna=True) <= 2:
            keep.append(c)
            continue

        # Near-unique identifiers?
        if s.nunique(dropna=True) / max(1, n) >= cfg.ID_UNIQUE_RATIO:
            # Try datetime
            if _detect_datetime(s, cfg) is not None:
                keep.append(c)
                continue
            # Try numeric-like coercion
            maybe_num = _coerce_numeric_like(s)
            if maybe_num.notna().mean() >= cfg.NUMERIC_LIKE_SUCCESS_RATIO:
                keep.append(c)
                continue
            report["dropped"].append({"column": c, "reason": "likely_id"})
            continue

        keep.append(c)

    return pdf[keep], report


def _normalize_features(pdf: pd.DataFrame, target_col: str, cfg: Config = DEFAULT_CONFIG) -> Tuple[
    pd.DataFrame, Dict[str, List[str]]]:
    """
    Normalize features:
    - Parse datetimes and add calendar features
    - Coerce numeric-like strings to numeric
    - Split into numeric / categorical / text groups
    """
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    text_cols: List[str] = []
    datetime_cols: List[str] = []

    work = pdf.copy()

    for c in list(work.columns):
        if c == target_col:
            continue
        s = work[c]

        # Numeric already
        if pd.api.types.is_numeric_dtype(s):
            numeric_cols.append(c)
            continue

        # Attempt datetime first for string-like
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            dt = _detect_datetime(s, cfg)
            if dt is not None:
                work[c + "__year"] = dt.dt.year
                work[c + "__month"] = dt.dt.month
                work[c + "__dow"] = dt.dt.dayofweek
                hour = getattr(dt.dt, "hour", None)
                work[c + "__hour"] = (hour.fillna(0).astype(int) if hour is not None else 0)
                work[c + "__month_start"] = dt.dt.is_month_start.astype(int)
                work[c + "__month_end"] = dt.dt.is_month_end.astype(int)
                datetime_cols.append(c)
                work.drop(columns=[c], inplace=True)
                continue

        # Numeric-like coercion
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            coerced = _coerce_numeric_like(s)
            if coerced.notna().mean() >= cfg.NUMERIC_LIKE_SUCCESS_RATIO:
                work[c] = coerced
                numeric_cols.append(c)
                continue

        # Text vs categorical
        if _is_textual(s, cfg):
            text_cols.append(c)
        else:
            categorical_cols.append(c)

    return work, {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "text": text_cols,
        "datetime_original": datetime_cols,
    }


# ==============================================================================
#                             Preprocessor Builder
# ==============================================================================

def _sanitize_name(name: str) -> str:
    """Sanitize transformer names (ColumnTransformer dislikes some characters)."""
    return re.sub(r"[^0-9a-zA-Z_]+", "_", name).replace("__", "_")


def _build_preprocessor(cat_cols: List[str], num_cols: List[str], text_cols: List[str],
                        cfg: Config = DEFAULT_CONFIG) -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
    - Imputes and one-hot encodes categorical features
    - Imputes numeric features
    - Applies per-text TF-IDF with ngrams
    """
    transformers: List[Tuple[str, Pipeline, List[str]]] = []

    # OneHotEncoder configuration across sklearn versions
    ohe_kwargs = dict(handle_unknown="ignore", min_frequency=cfg.OHE_MIN_FREQ, max_categories=cfg.OHE_MAX_CATEGORIES,
                      drop="if_binary")
    try:
        ohe = OneHotEncoder(sparse_output=True, **ohe_kwargs)  # sklearn ≥1.2
    except TypeError:  # Backward compat
        ohe = OneHotEncoder(sparse=True, **ohe_kwargs)

    if cat_cols:
        cat_pipe = Pipeline(steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", ohe),
        ])
        transformers.append(("cat", cat_pipe, cat_cols))

    if num_cols:
        num_pipe = Pipeline(steps=[
            ("impute", SimpleImputer(strategy="median")),
        ])
        transformers.append(("num", num_pipe, num_cols))

    for c in text_cols:
        transformers.append((
            _sanitize_name(f"text_{c}"),
            Pipeline(steps=[
                ("impute", SimpleImputer(strategy="constant", fill_value="")),
                ("squeeze", FunctionTransformer(lambda x: np.asarray(x).ravel(), accept_sparse=False)),
                ("tfidf", TfidfVectorizer(
                    max_features=cfg.TFIDF_MAX_FEATURES,
                    min_df=cfg.TFIDF_MIN_DF,
                    max_df=cfg.TFIDF_MAX_DF,
                    ngram_range=(1, 2),
                    lowercase=True,
                )),
            ]),
            [c],
        ))

    return ColumnTransformer(
        transformers=transformers,
        sparse_threshold=0.3,
        remainder="drop",
        verbose_feature_names_out=False,
    )


# ==============================================================================
#                           Splitting & Evaluation
# ==============================================================================

def _force_test_has_both_classes(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, seed: int = 42) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Ensure the test set contains at least one sample of each class when possible.
    """
    y = y.astype(int)
    classes = np.unique(y)
    if len(classes) < 2:
        return X, X.iloc[:0], y, y.iloc[:0]

    # Ensure test size at least two samples
    test_size = max(test_size, 2 / len(y))

    # Try stratified first
    try:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    except ValueError:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed)
    if len(np.unique(yte)) == 2:
        return Xtr, Xte, ytr, yte

    # Try alternative seeds
    for alt_seed in (101, 202, 303, 404, 505):
        try:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=alt_seed)
        except ValueError:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=alt_seed)
        if len(np.unique(yte)) == 2:
            return Xtr, Xte, ytr, yte

    return Xtr, Xte, ytr, yte  # best effort


def _eval_binary(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    """
    Compute standard binary classification metrics with a fixed 0.5 threshold.
    """
    pred = (proba >= 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }


def _choose_threshold(y_true: np.ndarray, proba: np.ndarray, metric: str = "f1") -> float:
    """
    Choose a decision threshold over a coarse grid to maximize the given metric.
    Currently supports 'f1'.
    """
    thr_grid = np.linspace(0.05, 0.95, 19)
    best_thr, best_score = 0.5, -1.0
    for thr in thr_grid:
        pred = (proba >= thr).astype(int)
        if metric == "f1":
            score = f1_score(y_true, pred, zero_division=0)
        else:
            score = f1_score(y_true, pred, zero_division=0)
        if score > best_score:
            best_thr, best_score = float(thr), float(score)
    return best_thr


# ==============================================================================
#                                 Inference Utils
# ==============================================================================

def _align_columns_for_inference(norm_df: pd.DataFrame, training_columns: List[str]) -> pd.DataFrame:
    """
    Align normalized inference schema to the training column order/space.
    """
    X = norm_df.copy()
    extra = [c for c in X.columns if c not in training_columns and c != "__rowid__"]
    if extra:
        X.drop(columns=extra, inplace=True, errors="ignore")
    for c in training_columns:
        if c not in X.columns:
            X[c] = np.nan
    return X[training_columns]


# ==============================================================================
#                                   Main API
# ==============================================================================

def run_classification(
        df: Any,
        target: Optional[str] = None,
        *,
        export_model: bool = False,
        clf_model_key: Optional[str] = None,
        put_bytes_fn: Optional[Callable[[str, bytes, str], None]] = None,
        cfg: Config = DEFAULT_CONFIG,
) -> Dict[str, Any]:
    """
    Train a robust binary classifier on a heterogeneous DataFrame-like input.

    Parameters
    ----------
    df : Any
        Pandas DataFrame or an object exposing .to_pandas().
    target : Optional[str]
        Target column name; if None, target is inferred heuristically.
    export_model : bool
        If True, serialize pipeline & metadata using `put_bytes_fn`.
    clf_model_key : Optional[str]
        Storage key for the serialized model; defaults to 'models/classification_<target>.joblib'.
    put_bytes_fn : Optional[Callable[[str, bytes, str], None]]
        Storage callback: (key, data, content_type) -> None.
    cfg : Config
        Configuration knobs (defaults provided).

    Returns
    -------
    dict
        {
          "metrics": {...},
          "export": {"clf_model_key": "...", "meta_key": "..."}  # when export_model=True
        }
    """
    # -------- Load dataframe --------
    pdf = df.to_pandas() if hasattr(df, "to_pandas") else df
    if not isinstance(pdf, pd.DataFrame):
        raise TypeError("`df` must be a pandas.DataFrame or expose .to_pandas().")

    info: Dict[str, Any] = {"target": target, "used_target": None, "inference": None, "config": asdict(cfg)}

    # -------- Target selection/inference --------
    y: Optional[pd.Series] = None
    tcol: Optional[str] = None

    if target and target in pdf.columns:
        y, ok, map_info = _coerce_binary(pdf[target])
        if ok:
            tcol = target
            info["inference"] = {"column": target, "reason": "user_provided", **map_info}
        else:
            info["inference"] = {"column": target, "reason": "user_provided_invalid"}

    if y is None:
        tcol, y, infer_info = _infer_binary_target(pdf)
        info["inference"] = infer_info

    if y is None:
        return {
            "metrics": {
                "error": "No binary target found or provided.",
                "hint": "Pass a target column with 2 classes (0/1, bool, or two unique strings).",
                "inference": info,
            }
        }

    info["used_target"] = tcol

    # -------- Prune & normalize --------
    pdf2, drop_report = _drop_bad_columns(pdf, target=tcol, cfg=cfg)
    info["dropped_columns"] = drop_report.get("dropped", [])
    norm_df, groups = _normalize_features(pdf2, target_col=tcol, cfg=cfg)
    info["feature_groups"] = groups

    X = norm_df.drop(columns=[tcol]).copy()
    if X.shape[1] == 0:
        return {"metrics": {"error": "No usable feature columns after dropping target.", "used_target": tcol,
                            "inference": info}}

    cat_cols = [c for c in groups["categorical"] if c in X.columns]
    num_cols = [c for c in groups["numeric"] if c in X.columns]
    text_cols = [c for c in groups["text"] if c in X.columns]

    pre = _build_preprocessor(cat_cols, num_cols, text_cols, cfg=cfg)

    # -------- Split with guard --------
    y = y.astype(int)
    class_counts = {int(k): int(v) for k, v in pd.Series(y).value_counts().to_dict().items()}
    split_stats: Dict[str, Any] = {"class_counts": class_counts}

    if len(class_counts) < 2:
        Xtr, Xte, ytr, yte = X, X.iloc[:0], y, y.iloc[:0]
        split_stats["note"] = "Only one class present; metrics limited."
    else:
        Xtr, Xte, ytr, yte = _force_test_has_both_classes(X, y, test_size=0.2, seed=42)

    # -------- Imbalance handling --------
    spw = 1.0
    if len(class_counts) == 2 and class_counts.get(1, 0) > 0:
        ratio = class_counts.get(0, 0) / max(1, class_counts.get(1, 0))
        if ratio >= cfg.IMBALANCE_RATIO:
            spw = max(1.0, ratio)

    # -------- Model (XGBoost) --------
    clf = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        reg_alpha=0.0,
        eval_metric="auc",
        n_jobs=-1,
        random_state=42,
        scale_pos_weight=spw,
        tree_method="hist",
    )

    pipe = Pipeline([("prep", pre), ("xgb", clf)])

    # Enable early stopping inside XGBoost
    pipe.named_steps["xgb"].set_params(early_stopping_rounds=50)

    # Pre-fit a clone of the preprocessor to create eval_set in the XGB space
    fit_params: Dict[str, Any] = {}
    if len(yte) > 0 and len(np.unique(yte)) == 2:
        prep_clone = clone(pre)
        prep_clone.fit(Xtr, ytr)
        Xtr_t = prep_clone.transform(Xtr)
        Xte_t = prep_clone.transform(Xte)
        fit_params = {"xgb__eval_set": [(Xtr_t, ytr), (Xte_t, yte)], "xgb__verbose": False}

    # -------- Fit --------
    pipe.fit(Xtr, ytr, **fit_params)

    # -------- Optional calibration --------
    calibrated = False
    if (
            cfg.USE_CALIBRATION
            and len(yte) >= cfg.CALIBRATION_MIN_VALID
            and len(np.unique(yte)) == 2
    ):
        try:
            # Calibrate only the classifier; the preprocessor is already fitted via the pipeline.
            # We must transform Xte with the preprocessor to get the model input space.
            Xte_t_for_calib = pipe.named_steps["prep"].transform(Xte)
            base_est = pipe.named_steps["xgb"]
            calib = CalibratedClassifierCV(
                base_estimator=base_est,
                method=cfg.CALIBRATION_METHOD,
                cv="prefit",
            )
            calib.fit(Xte_t_for_calib, yte)
            # Replace the classifier in the pipeline with the calibrated wrapper.
            pipe.named_steps["xgb"] = calib  # type: ignore[assignment]
            calibrated = True
            logger.info("Applied probability calibration (%s).", cfg.CALIBRATION_METHOD)
        except Exception as e:
            logger.warning("Calibration skipped due to error: %s", e)

    # -------- Metrics --------
    metrics: Dict[str, Any] = {
        "model": "xgb",
        "used_target": tcol,
        "class_weight": spw,
        **split_stats,
        "inference": info,
    }
    evaluated = False
    best_threshold = 0.5

    if len(yte) > 0 and len(np.unique(yte)) == 2:
        proba = pipe.predict_proba(Xte)[:, 1]
        metrics.update(_eval_binary(yte.values, proba))
        best_threshold = _choose_threshold(yte.values, proba, metric="f1")
        metrics["threshold"] = best_threshold
        evaluated = True
        metrics["calibrated"] = bool(calibrated)
    else:
        metrics.update({"note": "Test set missing one class; attempting CV fallback."})

    # -------- CV fallback --------
    if (not evaluated) and cfg.CV_ON_SINGLE_CLASS and len(np.unique(y)) == 2:
        min_class = min(class_counts.get(0, 0), class_counts.get(1, 0))
        if min_class >= 2:
            splits = min(cfg.CV_MAX_SPLITS, class_counts.get(1, 0), class_counts.get(0, 0), 5)
            splits = max(2, splits)
            skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)

            agg = {"auroc": [], "accuracy": [], "precision": [], "recall": [], "f1": []}
            last_fold: Optional[Tuple[np.ndarray, np.ndarray]] = None

            for tr, te in skf.split(X, y):
                fold_clf = XGBClassifier(**clf.get_params())
                fold_pipe = Pipeline([("prep", pre), ("xgb", fold_clf)])
                fold_pipe.named_steps["xgb"].set_params(early_stopping_rounds=30)
                # We can pass raw X/y to eval_set, since fold_pipe will transform internally during fit
                fold_pipe.fit(
                    X.iloc[tr], y.iloc[tr],
                    xgb__eval_set=[(X.iloc[tr], y.iloc[tr]), (X.iloc[te], y.iloc[te])],
                    xgb__verbose=False,
                )
                proba_cv = fold_pipe.predict_proba(X.iloc[te])[:, 1]
                fold_metrics = _eval_binary(y.iloc[te].values, proba_cv)
                for k, v in fold_metrics.items():
                    agg[k].append(v)
                last_fold = (y.iloc[te].values, proba_cv)

            for k, arr in agg.items():
                if arr:
                    metrics[k] = float(np.mean(arr))
            metrics["cv"] = {"n_splits": int(splits), "strategy": "StratifiedKFold", "averaged": True}

            try:
                if last_fold is not None:
                    y_last, p_last = last_fold
                    best_threshold = _choose_threshold(y_last, p_last, metric="f1")
                    metrics["threshold"] = best_threshold
                else:
                    metrics["threshold"] = 0.5
            except Exception:
                metrics["threshold"] = 0.5
        else:
            metrics["note"] = "CV not feasible (minority class <2)."

    # -------- Optional export --------
    export_info: Dict[str, str] = {}
    if export_model:
        if put_bytes_fn is None or not callable(put_bytes_fn):
            metrics["export_warning"] = "export_model=True but put_bytes_fn not provided."
        else:
            if not clf_model_key:
                clf_model_key = f"models/classification_{info['used_target']}.joblib"

            # 1) model blob
            blob = _serialize_model(pipe)
            put_bytes_fn(clf_model_key, blob, "application/octet-stream")

            # 2) metadata (schema & threshold)
            training_columns = list(X.columns)
            meta = {
                "type": "classification",
                "target": tcol,
                "threshold": float(best_threshold),
                "feature_groups": info.get("feature_groups", {}),
                "training_columns": training_columns,
            }
            meta_key = clf_model_key + ".meta.json"
            put_bytes_fn(meta_key, json.dumps(meta, ensure_ascii=False).encode("utf-8"), "application/json")

            export_info = {"clf_model_key": clf_model_key, "meta_key": meta_key}

    return {"metrics": metrics, **({"export": export_info} if export_info else {})}


# ------------------------------------------------------------------------------
#                                Public Inference
# ------------------------------------------------------------------------------

def predict_classification(
        df_or_records: Any,
        *,
        get_bytes_fn: Callable[[str], bytes],
        clf_model_key: str,
        meta_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Predict on JSON (list[dict]) or DataFrame-like input.

    Parameters
    ----------
    df_or_records : Any
        Pandas DataFrame, object with .to_pandas(), or list[dict] records.
    get_bytes_fn : Callable[[str], bytes]
        Storage loader: given a key, return bytes.
    clf_model_key : str
        Storage key for the serialized model.
    meta_key : Optional[str]
        Storage key for the metadata (JSON). Defaults to '<clf_model_key>.meta.json'.

    Returns
    -------
    dict
        {"threshold": float, "probability": list[float], "prediction": list[int]}
    """
    blob = get_bytes_fn(clf_model_key)
    pipe: Pipeline = _deserialize_model(blob)

    if meta_key is None:
        meta_key = clf_model_key + ".meta.json"
    meta = json.loads(get_bytes_fn(meta_key).decode("utf-8"))

    target_col = meta["target"]
    training_columns = meta["training_columns"]
    threshold = float(meta.get("threshold", 0.5))

    # Accept list[dict] or DataFrame
    if isinstance(df_or_records, list):
        pdf_raw = pd.DataFrame(df_or_records)
    else:
        pdf_raw = df_or_records.to_pandas() if hasattr(df_or_records, "to_pandas") else df_or_records
        if not isinstance(pdf_raw, pd.DataFrame):
            raise TypeError("`df_or_records` must be a DataFrame, list[dict], or expose .to_pandas().")

    # Target is not required at inference
    if target_col in pdf_raw.columns:
        pdf_raw = pdf_raw.drop(columns=[target_col], errors="ignore")

    # Use same pruning & normalization as training (lenient)
    pdf2, _ = _drop_bad_columns(pdf_raw, target=None, cfg=DEFAULT_CONFIG)
    norm_df, _ = _normalize_features(pdf2, target_col="__no_target__", cfg=DEFAULT_CONFIG)

    # Align schema
    X = _align_columns_for_inference(norm_df, training_columns)

    # Predict
    proba = pipe.predict_proba(X)[:, 1]
    labels = (proba >= threshold).astype(int).tolist()

    return {"threshold": threshold, "probability": [float(p) for p in proba], "prediction": labels}


def predict_classification_df(
        pdf: pd.DataFrame,
        *,
        get_bytes_fn: Callable[[str], bytes],
        clf_model_key: str,
        meta_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convenience wrapper returning a DataFrame with appended columns:
    - proba (float), pred (int)
    """
    out = predict_classification(pdf, get_bytes_fn=get_bytes_fn, clf_model_key=clf_model_key, meta_key=meta_key)
    result = pdf.copy()
    result["proba"] = out["probability"]
    result["pred"] = out["prediction"]
    return result
