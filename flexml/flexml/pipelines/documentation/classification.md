# Classification Pipeline — Practical Guide

This document explains the output produced by `run_classification(...)`, how to interpret each metric and diagnostic,
and gives a pragmatic, step-by-step improvement playbook to iteratively raise classification quality.

---

## 1) What the Pipeline Returns

`run_classification(...)` orchestrates:

1. **Input prep** → infers/validates a **binary target**, prunes harmful columns (too-missing, constants, likely IDs),
   and normalizes features (numeric-like coercion, datetime expansion, text vs categorical detection).
2. **Preprocessing graph** → builds a single `ColumnTransformer` with:
   - **Numeric:** median imputation.
   - **Categorical:** most-frequent imputation + One-Hot (capped cardinality; rare levels grouped).
   - **Text:** per-column TF–IDF with unigrams+bigrams (capped features).
3. **Modeling** → trains an **XGBoost** classifier with imbalance awareness (`scale_pos_weight` when needed) and
   early stopping on a held-out validation slice.
4. **Evaluation** → guarantees the test set contains **both classes** when possible; otherwise
   runs a **Stratified K-Fold CV fallback**.
5. **(Optional) Calibration** → if enough validation samples are available, applies **isotonic** (default)
   or **sigmoid** probability calibration.
6. **(Optional) Export** → persists the full pipeline (preprocess + model) and a JSON metadata file
   (schema, threshold, feature groups).

The function returns a JSON-friendly object like:

```json
{
  "metrics": {
    "model": "xgb",
    "used_target": "is_churn",
    "class_weight": 3.5,
    "class_counts": { "0": 1400, "1": 400 },
    "auroc": 0.873,
    "pr_auc": 0.624,
    "accuracy": 0.842,
    "precision": 0.611,
    "recall": 0.563,
    "f1": 0.586,
    "threshold": 0.45,
    "calibrated": true,
    "note": "…",
    "cv": {
      "n_splits": 5,
      "strategy": "StratifiedKFold",
      "averaged": true
    },
    "inference": {
      "used_target": "is_churn",
      "feature_groups": {
        "numeric": ["age", "balance"],
        "categorical": ["country", "segment"],
        "text": ["ticket_text"],
        "datetime_original": ["signup_date"]
      },
      "dropped_columns": [
        {"column": "session_id", "reason": "likely_id"}
      ]
    }
  },
  "export": {
    "clf_model_key": "models/classification_is_churn.joblib",
    "meta_key": "models/classification_is_churn.joblib.meta.json"
  }
}
```

> Notes
> - Some fields (e.g., `calibrated`, `cv`) appear only when applicable.
> - If the dataset contains only **one class**, the pipeline still trains, but metrics are limited and CV may be skipped.

---

## 2) Metrics — Definitions and Practical Meaning

Let \(y \in \{0,1\}\) be the true label and \(\hat{p} \in [0,1]\) the predicted probability for class 1.

### 2.1 AUROC — Area Under ROC Curve
- **What it is:** Probability that a random positive ranks above a random negative.
- **Range:** 0.5 (random) → 1.0 (perfect). *Higher is better.*
- **Use when:** Class balance varies; you care about ranking quality across thresholds.
- **Caveat:** Can look optimistic on highly imbalanced data.

### 2.2 PR AUC — Area Under Precision–Recall Curve
- **What it is:** Area under Precision vs Recall curve for the positive class.
- **Why it matters:** More informative than AUROC when the **positive class is rare**.
- **Interpretation:** Closer to 1 means strong performance on positives without sacrificing precision.

### 2.3 Accuracy
- **Definition:** Share of correct hard predictions at a fixed threshold (default 0.5 unless optimized).
- **Caution:** Misleading under class imbalance; always read with Precision/Recall and PR AUC.

### 2.4 Precision (Positive Predictive Value)
- Among predicted positives, how many are truly positive?  
- **Optimize when:** False positives are costly (e.g., manual reviews, customer impact).

### 2.5 Recall (Sensitivity)
- Among true positives, how many did we catch?  
- **Optimize when:** False negatives are costly (e.g., fraud leakage, churn you fail to prevent).

### 2.6 F1 Score
- **Harmonic mean** of Precision and Recall.  
- Used as the **default objective for threshold tuning** in this pipeline.

---

## 3) Thresholds — From Scores to Decisions

- The model outputs **probabilities**; decisions require a **threshold** \(\tau\).
- By default the pipeline:
  1. Evaluates on the held-out test set (if both classes present).
  2. Selects \(\tau\) that **maximizes F1** over a coarse grid (0.05 … 0.95).  
     The chosen \(\tau\) is reported as `metrics.threshold` and stored in the export metadata.
- **Business alignment:** If your cost function differs, recompute \(\tau\) offline using your preferred utility
  (e.g., Precision@k, expected cost, recall constraint).

**Tip:** Treat \(\tau\) as a **deployment knob**—safe to change post-training without retraining the model.

---

## 4) Calibration — Making Probabilities Honest

- If `USE_CALIBRATION=True` and the validation slice has at least `CALIBRATION_MIN_VALID` samples
  **with both classes**, the pipeline calibrates probabilities using:
  - **Isotonic** (default, non-parametric; needs >100 samples), or
  - **Sigmoid/Platt scaling** (parametric; works with fewer samples).
- Calibrated probabilities improve **decision thresholds**, **alert volumes**, and **downstream cost models**.  
- The `metrics.calibrated` flag indicates whether calibration was applied.

---

## 5) Class Imbalance — What the Pipeline Does

- Computes the ratio \(|\mathcal{N}_0| / |\mathcal{N}_1|\). If the majority/minority ratio ≥ `IMBALANCE_RATIO`
  (default **3.0**), it sets XGBoost’s `scale_pos_weight ≈ ratio`.
- **Effect:** Penalizes mistakes on the minority class more, improving recall without naively oversampling.
- **Still recommended:** Monitor **PR AUC** and **Recall**; consider cost-sensitive thresholding or domain sampling
  if the minority class is extremely rare (e.g., <1%).

---

## 6) Features — How to Read `feature_groups`

- **numeric** — Columns already numeric or successfully coerced (e.g., currency strings, `12k`, `5%`).
- **categorical** — Low/medium-cardinality strings; imputed and one-hot encoded (rare levels binned).
- **text** — Free-text columns with sufficient variability and average length; modeled via **TF–IDF** per column.
- **datetime_original** — Original string/object datetime columns that were parsed; the pipeline **adds**
  derived features (`__year`, `__month`, `__dow`, `__hour`, `__month_start`, `__month_end`) and drops the raw string.

**Dropped columns** (in `metrics.inference.dropped_columns`) list items removed as **constants**, **too-missing**,
or **likely IDs** that were neither parseable datetimes nor numeric-like.

---

## 7) Train/Test Split and CV Fallback

- The splitter **tries hard** to ensure the **test set contains both classes** (changes random seed if needed).
- If the test set still lacks one class or the dataset is too small:
  - The pipeline performs a **Stratified K-Fold** CV (up to `CV_MAX_SPLITS`, limited by minority class count).
  - Metrics are then **averages across folds** and reported under `metrics.cv`.

**Interpretation:** CV metrics reflect expected generalization when a clean hold-out is impractical.

---

## 8) Exported Artifacts — What Gets Saved (if `export_model=True`)

Two files are stored via your `put_bytes_fn`:

1. **Model blob** (`.joblib`): the **full** sklearn pipeline (`ColumnTransformer` + classifier [+ calibrator, if any]).
2. **Metadata** (`.meta.json`):
   ```json
   {
     "type": "classification",
     "target": "<target_name>",
     "threshold": <float>,
     "feature_groups": { ... },
     "training_columns": ["<col1>", "<col2>", "..."]
   }
   ```

- `training_columns` is the exact **post-normalization schema** used during training.  
- The stored `threshold` is the **recommended operating point** (F1-optimal by default).

---

## 9) Predict-Time Outputs

When using `predict_classification(...)`:

```json
{
  "threshold": 0.45,
  "probability": [0.12, 0.81, ...],
  "prediction": [0, 1, ...]
}
```

- **`probability`** — Calibrated (if applied) probability for class 1.
- **`prediction`** — Hard labels from applying the stored threshold to `probability`.
- You can **override the threshold** downstream if your business trade-offs change.

`predict_classification_df(...)` returns the input DataFrame with two extra columns:
- `proba` (float), `pred` (int).

---

## 10) Practical Improvement Playbook

### Step 0 — Sanity checks
- Target is genuinely binary (two classes) and correctly mapped.
- No severe label noise; handle duplicates and obvious data issues.
- Enough minority samples (ideally ≥ 100) for meaningful validation/calibration.
- Leakage avoided (no future info in features).

### Step 1 — Pick your business objective
- **High Precision** use-cases: expensive manual checks, customer-facing actions.
- **High Recall** use-cases: fraud detection, safety, critical incident alerts.
- Decide the **primary metric** (e.g., PR AUC for rarity, Precision@k, Recall@τ).

### Step 2 — Threshold tuning
- Recompute \(\tau\) to align with your objective (e.g., maximize Precision@k or set Recall ≥ 0.8).
- Validate alert volumes and error costs with stakeholders.

### Step 3 — Feature enrichment
- **Datetime:** add domain calendar flags (holidays, fiscal periods).
- **Categorical:** group rare categories meaningfully (domain-driven), avoid exploding cardinality.
- **Text:** consider cleaning (lowercasing, stopwords) and domain tokenization if noisy.
- **Numeric:** engineer ratios, counts, or trend proxies; normalize units.
- Remove/replace **leaky** fields (IDs that encode time or status).

### Step 4 — Handle extreme imbalance
- Try **cost-sensitive thresholds** before changing sampling.
- If necessary, experiment with **SMOTE/undersampling** off-pipeline (ensure no leakage) and compare PR AUC/Recall.

### Step 5 — Model tuning
- XGBoost starters:
  - `n_estimators`: 300–800
  - `max_depth`: 4–8
  - `learning_rate`: 0.05–0.2
  - `subsample` / `colsample_bytree`: 0.6–1.0
  - `reg_lambda`: 1–10; `reg_alpha`: 0–2
- Use a **fixed validation protocol** while iterating to keep comparisons fair.

### Step 6 — Calibration check
- If operating on probabilities (risk scores, triage), ensure calibration remains strong
  after any major changes; recalibrate on a fresh holdout if needed.

---

## 11) Interpreting Mixed Signals

- **AUROC↑, PR AUC↔/↓** — Ranking improved globally, but minority detection (precision/recall trade-off) may not have.
- **Precision↑, Recall↓** — You tightened the threshold; revisit target recall or use Precision@k.
- **Recall↑, Precision↓** — You loosened the threshold; ensure downstream capacity for additional alerts.
- **Accuracy↑, PR AUC↓** — Likely over-weighting the majority class; refocus on PR AUC and thresholding.

---

## 12) Data Preparation Checklist

- Correct binary target and class mapping; remove ambiguous labels.
- Reasonable missingness handling; understand why features are missing.
- Stable categorical encodings across train/inference (watch for new categories).
- Consistent units/scales for numeric fields; avoid mixed units.
- Text fields: clean obvious boilerplate/noise when beneficial.
- No data leakage via IDs, timestamps, or future-derived features.

---

## 13) Reproducibility and Logging

- Deterministic seeds (`random_state=42`) and stable preprocessing.
- Log/track:
  - Data sizes and positive ratio.
  - Final hyperparameters and `scale_pos_weight` used.
  - Whether calibration was applied and with which method.
  - Validation protocol (holdout vs CV) and resulting metrics.
- Store the **exact training schema** (`training_columns`) and version your artifacts.

---

## 14) FAQ

**Q:** The pipeline says *“No binary target found or provided.”*  
**A:** Pass a target with 2 classes (0/1, bool, or two consistent strings). Alternatively, rename the column to a
likely name (e.g., `target`, `label`, `is_churn`) for heuristic inference.

**Q:** Can I use a custom threshold after deployment?  
**A:** Yes. The stored threshold is a recommendation. You can overwrite it based on current capacity/costs.

**Q:** Why are some columns dropped as *likely_id*?  
**A:** Near-unique identifiers (e.g., UUIDs) usually harm generalization. They’re kept only if they parse as datetime
or numeric-like in a stable way.

**Q:** How do I add domain features?  
**A:** Enrich the input DataFrame before calling `run_classification(...)`. The pipeline will safely process them under
the correct modality (numeric/categorical/text/datetime).

**Q:** My data is tiny; metrics look unstable.  
**A:** Use the CV fallback results, widen confidence by repeating with different seeds, and prioritize calibration only
when you have enough validation samples (>100).

---

**Bottom line:**  
Use **PR AUC** and **Recall/Precision** to steer decisions under imbalance, pick a **business-aligned threshold**, and
iterate with a fixed validation protocol. Calibrate probabilities when you act on scores, and version the exported
artifacts (model + metadata) for safe, repeatable deployment.
