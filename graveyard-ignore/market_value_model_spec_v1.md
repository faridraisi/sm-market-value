# Market Value Model v1.0 — Model Specification

**Document Version:** 1.0  
**Date:** 2024-12-14  
**Status:** DRAFT  
**Depends On:** Feature Freeze v1.0  

---

## 1. Objective

Predict the **expected market value** of an Australian yearling at sale, expressed as:
- **P50** (median expected price)
- **P25** (low estimate)
- **P75** (high estimate)

Output = `session_median_price × exp(predicted_log_price_index)`

---

## 2. Target Variable

| Attribute | Value |
|-----------|-------|
| Field | `log_price_index` |
| Definition | `LOG(hammer_price / session_median_price)` |
| Distribution | Mean ≈ 0, StdDev ≈ 1.0 |
| Training Rows | 28,768 (sold with price > 0) |

**Why log-index?**
- Normalizes across sale tiers
- Handles price skewness
- Makes errors interpretable as % deviation

---

## 3. Model Architecture

### 3.1 Primary Model: Quantile GBDT

| Attribute | Value |
|-----------|-------|
| Framework | LightGBM (preferred) or XGBoost |
| Task | Regression with quantile loss |
| Quantiles | 0.25, 0.50, 0.75 |
| Trees | 500–1000 (tuned) |
| Max Depth | 6–8 |
| Learning Rate | 0.05–0.1 |
| Min Child Samples | 20 |

**Three models trained:**
1. `model_p25` — quantile=0.25
2. `model_p50` — quantile=0.50 (primary)
3. `model_p75` — quantile=0.75

### 3.2 Baseline Model: Elastic Net

For sanity check before GBDT:

| Attribute | Value |
|-----------|-------|
| Framework | scikit-learn ElasticNet |
| Alpha | Cross-validated |
| L1 Ratio | 0.5 |

If Elastic Net R² < 0.3, investigate features before GBDT.

---

## 4. Feature Engineering (Model Input)

### 4.1 Numeric Features (use as-is)
```
session_median_price (log-transformed)
sire_median_price_36m (log-transformed)
sire_median_price_12m (log-transformed)
sire_momentum
sire_clearance_rate_36m
sire_clearance_rate_12m
sire_sold_count_36m
sire_sold_count_12m
dam_progeny_median_price (log-transformed)
dam_progeny_sold_count
vendor_median_price_36m (log-transformed)
vendor_clearance_rate_36m
vendor_sold_count_36m
```

### 4.2 Categorical Features
```
sale_company (2 levels)
sex (Colt/Filly/other)
vendor_volume_bucket (new/low/mid/high)
```

### 4.3 Binary Flags
```
sire_sample_flag_36m
dam_first_foal_flag
vendor_first_seen_flag
```

### 4.4 NULL Handling
| Feature | Fallback |
|---------|----------|
| Sire median | Session median |
| Dam median | Session median |
| Vendor median | Session median |
| Counts | 0 |
| Rates | NULL → -1 (explicit missing indicator) |

---

## 5. Validation Strategy

### 5.1 Time-Based Split (Primary)

| Split | Years | Purpose |
|-------|-------|---------|
| Train | 2020–2022 | Model fitting |
| Validate | 2023 | Hyperparameter tuning |
| Test | 2024–2025 | Final evaluation |

**No random split** — must respect temporal ordering.

### 5.2 Row Counts (Approximate)

| Split | Sold Rows |
|-------|-----------|
| Train (2020-2022) | ~13,270 |
| Validate (2023) | ~5,570 |
| Test (2024-2025) | ~9,930 |

---

## 6. Evaluation Metrics

### 6.1 Primary Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| MAE (log_price_index) | < 0.6 | Mean absolute error on log scale |
| Median AE | < 0.5 | More robust to outliers |
| R² | > 0.35 | Variance explained |

### 6.2 Secondary Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| MAPE (on AUD) | < 40% | Mean absolute % error on prices |
| Coverage (P25-P75) | ~50% | % of actuals within predicted range |
| Calibration | P50 unbiased | Mean residual ≈ 0 |

### 6.3 Segment Analysis

Evaluate separately on:
- Sale tier (Premier / Regional)
- Sex (Colt / Filly)
- First foal vs repeat dam
- Thin sire sample vs robust

---

## 7. Output Contract

### 7.1 Prediction Columns

| Column | Type | Description |
|--------|------|-------------|
| `predicted_log_index_p25` | DECIMAL | 25th percentile |
| `predicted_log_index_p50` | DECIMAL | 50th percentile (primary) |
| `predicted_log_index_p75` | DECIMAL | 75th percentile |
| `marketValueLow` | DECIMAL | `session_median × exp(p25)` |
| `marketValue` | DECIMAL | `session_median × exp(p50)` |
| `marketValueHigh` | DECIMAL | `session_median × exp(p75)` |

### 7.2 Destination Table

```sql
UPDATE tblHorseAnalytics
SET marketValue = @marketValue,
    marketValueLow = @marketValueLow,
    marketValueHigh = @marketValueHigh
WHERE horseId = @horseId
```

---

## 8. Training Pipeline

```
1. Extract features from mv_yearling_lot_features_v1
2. Filter to sold rows (isPassedIn = 0, hammer_price > 0)
3. Apply log transforms to price features
4. Handle NULLs per fallback rules
5. Split by sale_year
6. Train 3 quantile models
7. Evaluate on test set
8. Export model artifacts
9. Score inference population
10. Write to tblHorseAnalytics
```

---

## 9. Model Artifacts

| Artifact | Location | Format |
|----------|----------|--------|
| Model P25 | `/models/mv_v1_p25.lgb` | LightGBM binary |
| Model P50 | `/models/mv_v1_p50.lgb` | LightGBM binary |
| Model P75 | `/models/mv_v1_p75.lgb` | LightGBM binary |
| Feature list | `/models/mv_v1_features.json` | JSON |
| Scaler | `/models/mv_v1_scaler.pkl` | Pickle (if used) |

---

## 10. Monitoring (Post-Deployment)

| Check | Frequency | Alert If |
|-------|-----------|----------|
| Prediction drift | Weekly | Mean shift > 0.2 |
| Feature drift | Weekly | Distribution shift |
| Coverage rate | Per sale | < 40% or > 60% |
| Residual bias | Per sale | Mean residual > 0.1 |

---

## 11. Limitations & Assumptions

1. **No physical assessment** — Model cannot see conformation
2. **No buyer behavior** — Cannot predict bidding dynamics
3. **Session median required** — Inference needs sale context
4. **Thin sire/dam data** — Predictions less reliable for first-crop sires
5. **AU-only** — Not transferable to other markets without retraining

---

## 12. Next Steps

1. [ ] Extract training data to Python/CSV
2. [ ] Baseline Elastic Net
3. [ ] Tune LightGBM hyperparameters
4. [ ] Train quantile models
5. [ ] Evaluate on 2024-2025 test set
6. [ ] Deploy scoring pipeline

---

*Model Spec v1.0 — Ready for implementation.*
