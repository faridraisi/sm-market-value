# Market Value Model v1.0 — Prediction Bands & Output Contract

**Document Version:** 1.0  
**Date:** 2024-12-14  
**Status:** LOCKED  

---

## 1. Purpose

This document defines how **Market Value prediction bands** are generated, calibrated, stored, and presented for the AU Yearling Market Value model v1.0.

The objective is to provide:

* A single **Expected Market Value**
* A realistic **Low–High range** representing uncertainty
* Outputs that are commercially defensible, explainable, and consistent across sales

---

## 2. Conceptual Definition

Prediction bands represent **uncertainty**, not guarantees.

| Band               | Meaning                                                            |
| ------------------ | ------------------------------------------------------------------ |
| **Low (P25)**      | Conservative downside estimate; ~25% of comparable lots fall below |
| **Expected (P50)** | Model's best estimate for this lot in this sale context            |
| **High (P75)**     | Reasonable upside estimate; ~25% of comparable lots exceed         |

Bands widen naturally when data quality is weaker (thin sire/dam/vendor history).

---

## 3. Modeling Methodology

### 3.1 Target Variable

All prediction bands are generated in **log index space**:

```
log_price_index = log(hammer_price / session_median_price)
```

This ensures predictions are **sale-relative**, not raw-dollar dependent.

---

### 3.2 Model Type (v1)

**Quantile Gradient Boosted Decision Trees (GBDT)**

Three quantile models are trained:

| Model | Quantile | Output              |
| ----- | -------- | ------------------- |
| Q25   | 0.25     | log_price_index_p25 |
| Q50   | 0.50     | log_price_index_p50 |
| Q75   | 0.75     | log_price_index_p75 |

Models are trained on the frozen feature set defined in:
`Market Value Model v1.0 — Feature Freeze`

---

## 4. Conversion to Dollar Values

Predictions are converted back to dollar values using the **sale session median**:

```
price_q = session_median_price × EXP(log_price_index_q)
```

### Final price outputs:

* `mv_low_price`  = P25 price
* `mv_expected_price` = P50 price
* `mv_high_price` = P75 price

All prices are non-negative and rounded per UI rules.

---

## 5. Calibration (Required for v1)

Raw quantile outputs must be calibrated to ensure correct empirical coverage.

### 5.1 Calibration Dataset

* Time-based holdout (recommended): **2024–2025 sales**
* Compare predicted bands vs actual hammer prices

### 5.2 Calibration Objective

| Check        | Target |
| ------------ | ------ |
| % below Low  | ~25%   |
| % below High | ~75%   |

### 5.3 Calibration Method

Apply additive offsets in **log space**:

```
log_low_cal  = log_low_pred  + offset_low
log_high_cal = log_high_pred + offset_high
```

Offsets are derived once per model version and fixed in production.

---

## 6. Edge Case Handling

### 6.1 Passed-In Lots

* Passed-in lots still receive prediction bands
* Bands represent **expected market value**, not outcome

### 6.2 Thin Data / High Uncertainty

If multiple flags are present:

* `sire_sample_flag_36m = 1`
* `dam_first_foal_flag = 1`
* `vendor_first_seen_flag = 1`

Then:

* Confidence tier = **Low**
* UI may display a "Limited data" badge

Optional minimum band width enforcement:

```
low ≤ expected × 0.85
high ≥ expected × 1.20
```

(Apply only if bands are deemed too narrow in practice.)

---

## 7. Output Contract

### 7.1 Core Fields (Persisted)

| Field               | Type    | Description           |
| ------------------- | ------- | --------------------- |
| `mv_expected_price` | DECIMAL | Expected market value |
| `mv_low_price`      | DECIMAL | Downside estimate     |
| `mv_high_price`     | DECIMAL | Upside estimate       |

### 7.2 Explainability / Debug (Optional)

| Field               | Description              |
| ------------------- | ------------------------ |
| `mv_expected_index` | EXP(log_price_index_p50) |
| `mv_low_index`      | EXP(log_price_index_p25) |
| `mv_high_index`     | EXP(log_price_index_p75) |
| `mv_band_width_pct` | (high − low) / expected  |

### 7.3 Metadata

| Field                | Description         |
| -------------------- | ------------------- |
| `mv_model_version`   | Model identifier    |
| `mv_generated_at`    | Timestamp           |
| `mv_confidence_tier` | high / medium / low |

---

## 8. UI Presentation Guidelines

### 8.1 Display Format

**Market Value:** **$X**  
Range: **$L – $H**

Small caption:

> Based on comparable AU yearling sales and current sale context.

---

### 8.2 Tooltip / Help Text

> **Expected** is the model's best estimate for this lot at this sale.  
> **Low–High** shows a typical range where comparable lots often fall.  
> Actual results may vary due to inspection, veterinary reports, buyer competition, and market conditions.

---

## 9. Versioning & Governance

| Attribute        | Value             |
| ---------------- | ----------------- |
| Model            | Market Value v1.0 |
| Feature Contract | v1.0              |
| Prediction Spec  | v1.0              |

**Change Policy:**

* Any band logic change requires a new prediction spec version
* Any feature change requires a new Golden Table and feature freeze
* Historical predictions are not retroactively altered

---

## 10. Next Steps

1. Implement quantile GBDT training (Q25 / Q50 / Q75)
2. Perform holdout calibration
3. Persist outputs to analytics table
4. Integrate UI display and confidence badges

---

*Document locked. Changes require version increment.*
