# Forecasting Pipeline — Practical Guide

This document explains the output produced by `run_forecasting(...)`, how to interpret each metric, and gives a
pragmatic, step-by-step improvement playbook to iteratively raise forecast quality.

---

## 1) What the Pipeline Returns

`run_forecasting(...)` orchestrates:

1. **Input prep** → infers/validates time and target columns, aligns optional exogenous (exog) variables.
2. **Feature engineering** → lags, rolling means, optional calendar features.
3. **Grid search** → evaluates hyperparameter candidates via **rolling-origin backtests**.
4. **Diagnostics** → summarizes residuals; optionally tests residual autocorrelation (Ljung–Box) if `statsmodels` is
   available.
5. **Final model** → trains on **full history** with the **best hyperparameters**, then produces an **out-of-sample
   forecast** for the next horizon.

The function returns a JSON-friendly object:

```json
{
  "best_params": {
    ...
  },
  "backtest": {
    "metrics": {
      "RMSE": <float>,
      "sMAPE": <float>,
      "MASE": <float>
    },
    "preds": [
      {
        "ts": "...",
        "y_hat": <float>
      },
      ...
    ],
    "trues": [
      {
        "ts": "...",
        "y": <float>
      },
      ...
    ],
    "horizon": <int>,
    "step": <int>,
    "initial_train": <int>
  },
  "diagnostics": {
    "residuals": {
      "summary": {
        "count": ...,
        "mean": ...,
        "std": ...,
        "min": ...,
        "25%": ...,
        "50%": ...,
        "75%": ...,
        "max": ...
      },
      "ljung_box": {
        ...
      }
      |
      null
    }
  },
  "final_forecast": [
    {
      "ts": "...",
      "y_hat": <float>
    },
    ...
  ],
  "model_name": "rf"
  |
  "sarimax"
}
```

### Key fields

- **`best_params`** — Hyperparameters that minimized the backtest objective (RMSE). This is the configuration the final
  model is trained with.
- **`backtest.metrics`**
    - **RMSE** (Root Mean Squared Error): scale-dependent; penalizes large errors more than small ones. *Lower is
      better.*
    - **sMAPE** (Symmetric Mean Absolute Percentage Error): scale-free (≈ average % error); robust to scale changes.
      *Lower is better.*
    - **MASE** (Mean Absolute Scaled Error): compares your model to a seasonal naive baseline. **MASE < 1** means your
      model beats the naive benchmark.
- **`backtest.preds` / `backtest.trues`** — Time-aligned predicted vs actual values across all backtest folds.
- **`backtest.horizon` / `step` / `initial_train`** — Backtest configuration used (forecast length per fold, how far the
  window advances, and initial training size).
- **`diagnostics.residuals.summary`** — Descriptive stats of backtest residuals (`y - y_hat`).
- **`diagnostics.residuals.ljung_box`** — Autocorrelation test of residuals. Significant p-values indicate remaining
  temporal structure not captured by the model.
- **`final_forecast`** — The forward-looking predictions after refitting on the **entire** history with `best_params`.
- **`model_name`** — Which estimator was used (`rf` = RandomForest, `sarimax` when available).

---

## 2) Metrics — Definitions and Practical Meaning

Let \(y_t\) be the actual series and \(\hat{y}_t\) the prediction.

### 2.1 RMSE — Root Mean Squared Error

$ \text{RMSE} = \sqrt{ \frac{1}{n} \sum_{t=1}^{n} (y_t - \hat{y}_t)^2 } $

- **Pros:** Strongly penalizes outliers/large misses; aligned with many operational cost functions.
- **Cons:** Scale-dependent (harder to compare across series), sensitive to heteroskedasticity.

**Improve RMSE by:**

- Increasing model capacity or ensembling (e.g., more trees, deeper trees for RF).
- Adding informative features (seasonal lags, rolling aggregates, calendar, domain exog).
- Transformations (e.g., log for multiplicative variance; forecast in log-space and invert).

### 2.2 sMAPE — Symmetric Mean Absolute Percentage Error

$
\text{sMAPE} = \frac{2}{n} \sum_{t=1}^{n} \frac{|\hat{y}_t - y_t|}{|y_t| + |\hat{y}_t| + \epsilon}
$

- **Pros:** Unitless and comparable across series.
- **Cons:** Can be unstable when values are very small.

**Improve sMAPE by:**

- Reducing relative errors on low volumes (e.g., zero-inflated handling, winsorization, modeling a floor).
- Using transformations that stabilize scale (log, Box–Cox) and appropriate back-transform corrections.

### 2.3 MASE — Mean Absolute Scaled Error

$
\text{MASE} = \frac{\tfrac{1}{n}\sum_{t=1}^{n} |\hat{y}_t - y_t|}{\tfrac{1}{n-m}\sum_{t=m+1}^{n} |y_t - y_{t-m}|}
$

- **Interpretation:** Compares your model to a **seasonal naive** baseline with period \(m\) (heuristically inferred
  from the series frequency).
- **Threshold:** **MASE < 1** ⇒ your model outperforms the naive seasonal model.

**Improve MASE by:**

- Ensuring the model captures seasonality: include weekly/monthly lags, or use SARIMAX with seasonal terms.
- Increasing training data length or adding exogenous drivers aligned to seasonality (e.g., promotions, holidays,
  weather).

---

## 3) Diagnostics — What Problems Are Hiding?

### 3.1 Residual Summary

- **Mean close to 0** → No systematic bias.
- **Large std / long tails** → Volatile errors; consider robust loss proxies (feature engineering) or
  variance-stabilizing transforms.
- **Skewed residuals** → Asymmetric errors; consider quantile models or transform target.

### 3.2 Ljung–Box Test (if available)

- **Significant autocorrelation** in residuals means the model missed temporal structure.
- **Actions:** Add more lags/rolling features; increase seasonal context; try models designed for autocorrelation (
  SARIMAX).

---

## 4) Exogenous Variables (Exog) — Using Them Safely

- **Training:** Exog variables can boost signal (e.g., price, marketing spend, holiday flags).
- **Forecasting:** You need **future values** of exog at prediction timestamps.
    - If exog are **calendar-only** (dow, month, month-start/end) the pipeline can synthesize these automatically.
    - If exog depend on business plans (promos, prices), you must supply a **future exog frame** aligned to the forecast
      horizon.

**Gotcha:** Training with exog but **not** providing future exog leads to forecasts that ignore those drivers beyond the
training window.

---

## 5) Improvement Playbook — A Decision Tree

Use this practical flow whenever metrics or diagnostics look unsatisfactory.

### Step 0 — Sanity checks

- ✅ Enough history? `len(y) ≥ initial_train + horizon`
- ✅ Reasonable frequency? No large gaps or duplicated timestamps
- ✅ Target clean? Handle outliers, obvious data errors, and missing values
- ✅ Exog aligned? No leakage (future information) and properly indexed

---

### Step 1 — Are you beating the naive baseline?

- **If `MASE ≥ 1`:** Model is not beating seasonal naive.
    - ➜ **Add seasonal features:** include lags at seasonal periods (e.g., 7, 14, 28 for daily; 12 for monthly).
    - ➜ **Try SARIMAX** with seasonal order (if available) or expand RF feature set.
    - ➜ **Extend training window** if possible to cover multiple seasonal cycles.
    - ➜ **Check data quality:** missing periods, misaligned timestamps, or regime changes.

If **`MASE < 1`**, continue.

---

### Step 2 — Which error type dominates? (Scale vs Relative)

- **High `RMSE` but moderate `sMAPE`:** Big absolute misses (outliers).
    - ➜ Add robust features (longer rolling means/medians), try transformations (log), clamp extreme targets during
      training.
    - ➜ Increase model capacity (e.g., `n_estimators: 600`, tune `max_depth`, `min_samples_leaf`).
- **High `sMAPE` but acceptable `RMSE`:** Large relative errors, often at low volumes.
    - ➜ Model zero-inflation or floors, consider log/Box–Cox transform.
    - ➜ Segment series (e.g., forecast different regimes separately) if behavior differs at low levels.

---

### Step 3 — Residuals suggest structure?

- **Residual mean ≠ 0 (bias):**
    - ➜ Add missing drivers (trend proxies, exog like price/promo), revisit target transform.
- **Ljung–Box significant (autocorrelation remains):**
    - ➜ Add more **lags** and **seasonal lags**; try **SARIMAX** for explicit AR/MA structure.
    - ➜ Increase **rolling windows** to capture longer memory.

---

### Step 4 — Feature and model enrichment

- **Feature engineering:**
    - Lags: `[1, 7, 14, 28]` (daily) or `[1, 12, 24]` (monthly), plus domain lags.
    - Rolling means: `[7, 14, 28]` with `min_periods=window` to avoid leakage.
    - Calendar: day-of-week, month, month-start/end, holidays (domain-specific).
    - Exog: promotions, prices, marketing, weather; **must** provide future values for forecasting.
- **Model tuning (RF):**
    - `n_estimators`: 300–1000
    - `max_depth`: 6–20 (or None)
    - `min_samples_leaf`: 1–10 (larger can reduce overfit)
    - Always set `n_jobs=-1` for speed.
- **Model alternatives:**
    - SARIMAX for strong seasonality/autocorrelation; tune `(p,d,q)` and seasonal `(P,D,Q,m)`.

---

### Step 5 — Re-run backtest, compare, and lock-in

- Keep the backtest configuration fixed while iterating features/params to enable **apples-to-apples** comparisons.
- Track changes in **RMSE**, **sMAPE**, and **MASE**. Prefer configurations that **consistently** improve all, or that
  improve the metric most aligned with your business objective.

---

## 6) Interpreting Mixed Signals

- **RMSE↓, sMAPE↑:** Absolute errors improved, but relative errors at small values got worse. Consider a hybrid approach
  or separate modeling for low-volume segments.
- **MASE < 1 but RMSE high:** You beat naive seasonal but still have large absolute errors; add capacity/features.
- **Good metrics overall, but residual autocorrelation:** The model forecasts well on average but misses some cyclic
  dynamics; add seasonal lags or try SARIMAX terms.

---

## 7) Data Preparation Checklist

- Uniform frequency (use `asfreq` if needed; inspect NaNs introduced).
- No duplicated timestamps after indexing/sorting.
- Outliers addressed (cap/winsorize or model with robust features).
- Clear handling of holidays and special events (binary flags/countdown windows).
- When using exog:
    - Train and **future** exog have identical columns.
    - Future exog covers **exact forecast timestamps**.

---

## 8) Reproducibility and Logging

- Randomness controlled via fixed seeds; RF uses `random_state=42`.
- Use logs to capture:
    - Data span, counts, and backtest folds.
    - Chosen hyperparameters and backtest metrics.
    - Residual diagnostics (means/stats; Ljung–Box if available).

---

## 9) FAQ

**Q:** Why do I need future exog?  
**A:** If the model learned to rely on exogenous drivers, it needs their **future** values to forecast. Calendar-only
exog can be auto-generated; business drivers must be supplied.

**Q:** Is sMAPE better than RMSE?  
**A:** They measure different things. sMAPE is relative (%-like), RMSE is absolute. Choose based on your cost function;
ideally watch both.

**Q:** What does MASE < 1 really mean?  
**A:** Your model beats a simple seasonal naive forecast. It’s a strong sanity check across series with different
scales.

**Q:** My residuals show autocorrelation; what next?  
**A:** Add (seasonal) lags/rolling features or move to SARIMAX; ensure backtest horizon/step reflect your operational
forecast cadence.

---

## 10) Quick Reference — Tuning Knobs

- **Backtest:** choose `horizon` to match decision interval; `step` often equals `horizon`; `initial_train` should
  cover ≥ 2–3 seasonal cycles.
- **RF search space (starter):**
    - `n_estimators`: 300–600–1000
    - `max_depth`: 6–10–16–None
    - `min_samples_leaf`: 1–2–4–8
- **Seasonality hints:** daily data → weekly seasonality (`m=7`); hourly → `m=24`; monthly → `m=12`.

---

**Bottom line:**  
Use **MASE** to ensure you beat naive seasonal. Balance **RMSE** (absolute cost) and **sMAPE** (relative error). Let *
*residual diagnostics** guide feature and model changes. Iterate with a fixed backtest plan to converge on a reliable
configuration.
