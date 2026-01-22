# Market Value Model v1.0 — Training & Modelling Implementation Runbook

**Document Version:** 1.0  
**Date:** 2024-12-14  
**Status:** READY FOR IMPLEMENTATION  

---

## 1. Objective

This document defines the end-to-end implementation steps to:
- Extract training data from the Golden Table
- Validate via a baseline model
- Train production quantile models
- Calibrate prediction bands
- Persist outputs for product use

This follows the locked:
- Feature Freeze v1.0
- Prediction Bands & Output Contract v1.0

---

## 2. Training Population Definition

### 2.1 Included rows (training)

Training uses sold yearlings only:
- `isWithdrawn = 0`
- `isPassedIn = 0`
- `hammer_price > 0`
- `session_median_price > 0`
- `log_price_index IS NOT NULL`

From QA:
- Expected training rows ≈ 28,768

### 2.2 Excluded from training (but scored later)
- Passed-in lots
- Sold lots with zero price
- Withdrawn lots (never scored)

---

## 3. Data Extraction

### 3.1 Training CSV (sold-only)

```sql
SELECT
  lot_id,
  salesId,
  asOfDate,
  sale_company,
  sale_name,
  sale_year,
  book_number,
  day_number,
  lot_number,
  sex,
  sireId,
  damId,
  vendorId,
  session_median_price,

  -- Sire features
  sire_sold_count_36m, sire_passedin_count_36m, sire_total_offered_36m, sire_clearance_rate_36m, sire_median_price_36m,
  sire_sold_count_12m, sire_passedin_count_12m, sire_total_offered_12m, sire_clearance_rate_12m, sire_median_price_12m,
  sire_momentum, sire_sample_flag_36m,

  -- Dam features
  dam_progeny_sold_count, dam_progeny_passedin_count, dam_progeny_total_offered_count, dam_progeny_median_price, dam_first_foal_flag,

  -- Vendor features
  vendor_sold_count_36m, vendor_passedin_count_36m, vendor_total_offered_36m, vendor_clearance_rate_36m, vendor_median_price_36m,
  vendor_volume_bucket, vendor_first_seen_flag,

  -- Optional explainability
  sire_vs_sale_median_delta,
  dam_vs_sale_median_delta,
  vendor_vs_sale_median_delta,

  -- Target
  log_price_index
FROM dbo.mv_yearling_lot_features_v1
WHERE isWithdrawn = 0
  AND isPassedIn = 0
  AND hammer_price > 0
  AND session_median_price > 0
  AND log_price_index IS NOT NULL;
```

### 3.2 Inference / scoring dataset

```sql
SELECT *
FROM dbo.mv_yearling_lot_features_v1
WHERE isWithdrawn = 0;
```

---

## 4. Baseline Model (Sanity Check)

### Purpose

Ensure data quality and that signal exists beyond a trivial predictor.

### Model
- Elastic Net regression
- Target: `log_price_index`

### Validation split
- Train: `sale_year <= 2023`
- Holdout: `sale_year IN (2024, 2025)`

### Acceptance criteria
- MAE(log_price_index) significantly < 1.0
- Should outperform predicting 0 for all rows

Failure indicates feature or extraction issues.

---

## 5. Production Models — Quantile GBDT

### 5.1 Model family
- LightGBM Quantile Regression

### 5.2 Quantiles

| Model | Alpha | Output              |
|-------|-------|---------------------|
| Q25   | 0.25  | log_price_index_p25 |
| Q50   | 0.50  | log_price_index_p50 |
| Q75   | 0.75  | log_price_index_p75 |

### 5.3 Recommended v1 parameters
- `num_leaves`: 63
- `min_data_in_leaf`: 50–200
- `learning_rate`: 0.05
- `n_estimators`: up to 2000 with early stopping

Categorical features:
- `sex`
- `vendor_volume_bucket`
- optionally `sale_company`

---

## 6. Conversion to Dollar Prices

Predictions are converted using the sale anchor:

```
price_q = session_median_price × EXP(log_price_index_q)
```

Final outputs:
- `mv_low_price` (P25)
- `mv_expected_price` (P50)
- `mv_high_price` (P75)

Clamp:
- `mv_low_price >= 0`

---

## 7. Calibration (Required)

### 7.1 Calibration dataset
- Sold lots in 2024–2025 only

### 7.2 Coverage targets

| Check            | Target |
|------------------|--------|
| % actual ≤ Low   | ~25%   |
| % actual ≤ High  | ~75%   |

### 7.3 Calibration method

Offsets applied in log space:

```
offset_25 = quantile(actual_log - pred_log_p25, 0.25)
offset_75 = quantile(actual_log - pred_log_p75, 0.75)

cal_log_p25 = pred_log_p25 + offset_25
cal_log_p75 = pred_log_p75 + offset_75
```

Offsets are fixed per model version.

---

## 8. Confidence Tier Logic

| Tier   | Rule                                                                                      |
|--------|-------------------------------------------------------------------------------------------|
| High   | No data flags present                                                                     |
| Medium | Exactly one of: `sire_sample_flag_36m`, `dam_first_foal_flag`, `vendor_first_seen_flag`   |
| Low    | Two or more flags present                                                                 |

Confidence tier is stored and surfaced in UI.

---

## 9. Persistence to Analytics Table

### Minimum required fields
- `horseId`
- `salesId`
- `asOfDate`
- `mv_expected_price`
- `mv_low_price`
- `mv_high_price`
- `mv_model_version`
- `mv_generated_at`
- `mv_confidence_tier`

### Optional debug fields
- `mv_expected_index`
- `mv_low_index`
- `mv_high_index`
- `mv_band_width_pct`

---

## 10. Implementation Order

1. Extract training CSV
2. Train baseline Elastic Net
3. Train Q25 / Q50 / Q75 LightGBM models
4. Perform calibration on holdout
5. Persist predictions to database
6. Integrate UI display

---

*This runbook is implementation-ready. Changes require version increment.*
