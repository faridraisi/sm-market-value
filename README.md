# Market Value Model V2

Predict expected yearling sale prices with confidence ranges.

**Supported Countries:** Australia (AUS), New Zealand (NZL)

---

## Quick Start: Score a Sale

### 1. Activate Environment

```bash
cd ~/Projects/sm-market-value
source .venv/bin/activate
```

### 2. Run the Pipeline

```bash
# Output to CSV (default)
python score_sale.py --sale-id 2094

# Output directly to database (tblHorseAnalytics)
python score_sale.py --sale-id 2094 --output db
```

This runs two steps automatically:
1. **Feature rebuild** (`run_rebuild.py`) — Fetches data from database, computes features, outputs `csv/sale_2094_inference.csv`
2. **Scoring** (`score_lots.py`) — Loads the model, scores lots, outputs to CSV or database

The country is auto-detected from the database, and the correct model is loaded based on `.env` configuration.

### 3. Configuration via .env

Model selection and database output settings in `.env`:

```bash
# Model selection
AUS_MODEL=aus   # Use models/aus/ for Australian sales
NZL_MODEL=nzl   # Use models/nzl/ for NZ sales
USA_MODEL=usa   # Use models/usa/ for USA sales

# Database output settings
AUDIT_USER_ID=2  # User ID for createdBy/modifiedBy fields (default: 2)
```

To use the AUS model for NZL sales (e.g., for cross-country testing):
```bash
NZL_MODEL=aus
```

### 4. View Results

Output saved to `csv/sale_{sale_id}_scored.csv` with columns:

| Column | Description |
|--------|-------------|
| `lot_id` | Sales lot ID |
| `horseId` | Horse ID |
| `lot_number` | Catalogue lot number |
| `sire_name` | Sire name |
| `session_median_price` | Sale median used for calculation |
| `mv_expected_price` | **Expected sale price (P50)** |
| `mv_low_price` | Low estimate (P25) |
| `mv_high_price` | High estimate (P75) |
| `mv_confidence_tier` | high / medium / low |

---

## What is Market Value?

Market Value predicts the **expected hammer price** for a yearling at auction based on:

- **Sire performance** — Historical sale results (median price, clearance rate, momentum)
- **Dam production** — Previous progeny sale prices
- **Vendor track record** — Vendor's historical sale performance
- **Sale context** — Which sale, which book, session median

The model outputs three price points:

| Output | Meaning |
|--------|---------|
| **Expected (P50)** | The median predicted price — 50% chance of selling above/below |
| **Low (P25)** | Conservative estimate — 25% chance of selling below this |
| **High (P75)** | Optimistic estimate — 25% chance of selling above this |

### Confidence Tiers

| Tier | Meaning | Flags |
|------|---------|-------|
| **High** | Strong data support | No flags |
| **Medium** | Some uncertainty | 1 flag (thin sire sample, first foal, or new vendor) |
| **Low** | Limited data | 2+ flags |

---

## Model Details

### Algorithm

- **Model Type:** LightGBM Quantile Gradient Boosted Decision Trees
- **Target Variable:** `log(hammer_price / session_median_price)`
- **Quantiles:** P25, P50, P75 trained separately
- **Calibration:** Offsets applied to achieve exact 25%/75% coverage on holdout data

### Training Data

| Parameter | Value |
|-----------|-------|
| Source | Australian Yearling Sales 2020-2025 |
| Training Set | 18,839 lots (2020-2023) |
| Test Set | 9,929 lots (2024-2025) |
| Total Features | 28 |

### Performance Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| MAE | 0.70 | Mean absolute error in log-index space |
| R² | 0.16 | Variance explained |
| P25 Coverage | 25.0% | Perfectly calibrated |
| P75 Coverage | 75.0% | Perfectly calibrated |

> **Note:** R² of 0.16 is typical for yearling markets. Prices are heavily influenced by physical inspection, veterinary reports, and buyer competition — factors not captured in pedigree data.

### Top Features (by importance)

1. `vendor_clearance_rate_36m` — Vendor's sell-through rate
2. `dam_progeny_median_price` — Dam's previous foal prices
3. `vendor_median_price_36m` — Vendor's median sale price
4. `sire_median_price_12m` — Sire's recent median
5. `sire_clearance_rate_36m` — Sire's sell-through rate

---

## Feature Definitions

### Sire Features (rolling windows)

| Feature | Description |
|---------|-------------|
| `sire_sold_count_36m` | Yearlings sold in last 36 months |
| `sire_total_offered_36m` | Yearlings offered (sold + passed in) |
| `sire_clearance_rate_36m` | Sold ÷ Offered × 100 |
| `sire_median_price_36m` | Median hammer price of sold yearlings |
| `sire_sold_count_12m` | Same metrics for 12-month window |
| `sire_median_price_12m` | |
| `sire_momentum` | 12m median − 36m median (price trend) |
| `sire_sample_flag_36m` | 1 if < 10 sold in 36m (thin sample) |

### Dam Features (lifetime)

| Feature | Description |
|---------|-------------|
| `dam_progeny_sold_count` | Total progeny sold at auction |
| `dam_progeny_total_offered_count` | Total progeny offered |
| `dam_progeny_median_price` | Median price of sold progeny |
| `dam_first_foal_flag` | 1 if no prior progeny sold |

### Vendor Features (rolling 36m)

| Feature | Description |
|---------|-------------|
| `vendor_sold_count_36m` | Yearlings sold by vendor |
| `vendor_total_offered_36m` | Yearlings offered by vendor |
| `vendor_clearance_rate_36m` | Vendor sell-through rate |
| `vendor_median_price_36m` | Vendor's median price |
| `vendor_volume_bucket` | new / low / mid / high |
| `vendor_first_seen_flag` | 1 if vendor has no 36m history |

### Sale Context

| Feature | Description |
|---------|-------------|
| `session_median_price` | Median price for the sale/book |
| `sale_company` | Inglis / Magic Millions |
| `book_number` | Book 1, 2, 3 etc. |
| `sex` | Colt / Filly |

---

## Database Tables

### Golden Tables

| Country | Table | Description |
|---------|-------|-------------|
| **AUS** | `mv_yearling_lot_features_v1` | AU yearling lots 2020-2026 |
| **NZL** | `mv_yearling_lot_features_nz_v1` | NZ yearling lots 2020-2026 |

```sql
-- Check available AUS sales
SELECT DISTINCT salesId, sale_name, sale_year, COUNT(*) AS lots
FROM dbo.mv_yearling_lot_features_v1
GROUP BY salesId, sale_name, sale_year
ORDER BY sale_year DESC;

-- Check available NZL sales
SELECT DISTINCT salesId, sale_name, sale_year, COUNT(*) AS lots
FROM dbo.mv_yearling_lot_features_nz_v1
GROUP BY salesId, sale_name, sale_year
ORDER BY sale_year DESC;
```

### Staging Tables

| Country | Tables |
|---------|--------|
| **AUS** | `stg_mv_sale_median_v1`, `stg_mv_sire_metrics_v1`, `stg_mv_dam_stats_v1`, `stg_mv_vendor_metrics_v1` |
| **NZL** | `stg_mv_sale_median_nz_v1`, `stg_mv_sire_metrics_nz_v1`, `stg_mv_dam_stats_nz_v1`, `stg_mv_vendor_metrics_nz_v1` |

### Output Table: `tblHorseAnalytics`

Scored predictions are persisted here for UI display. Use `--output db` to write directly.

| Column | Source |
|--------|--------|
| `horseId` | From inference CSV |
| `salesId` | From inference CSV |
| `marketValue` | `mv_expected_price` (P50) |
| `marketValueLow` | `mv_low_price` (P25) |
| `marketValueHigh` | `mv_high_price` (P75) |
| `marketValueMultiplier` | `mv_expected_index` |
| `marketValueConfidence` | `mv_confidence_tier` (high/medium/low) |
| `sessionMedianPrice` | `session_median_price` |
| `currencyId` | Auto-set: AUS=1, NZL=6, USA=7 |
| `modifiedBy` | From `AUDIT_USER_ID` in .env |

---

## Setting Up a New Sale

For future sales (e.g., 2027), you need to:

### 1. Check if sale exists in Golden Table

```sql
SELECT COUNT(*) FROM dbo.mv_yearling_lot_features_v1 WHERE salesId = [NEW_SALE_ID];
```

### 2. If not present, rebuild Golden Table

Update the stored procedure year range and re-run `proc_BuildMvYearlingFeaturesV1`.

### 3. Set book numbers (if applicable)

```sql
UPDATE dbo.mv_yearling_lot_features_v1
SET book_number = 1
WHERE salesId = [SALE_ID] AND lot_number BETWEEN 1 AND 980;

UPDATE dbo.mv_yearling_lot_features_v1
SET book_number = 2
WHERE salesId = [SALE_ID] AND lot_number >= 981;
```

### 4. Set session median (for future sales)

Use historical median for that sale type:

```sql
-- Check historical medians
SELECT sale_year, sale_name, session_median_price
FROM dbo.mv_yearling_lot_features_v1
WHERE sale_name LIKE '%Gold Coast%Book 1%'
GROUP BY sale_year, sale_name, session_median_price
ORDER BY sale_year DESC;

-- Set for future sale
UPDATE dbo.mv_yearling_lot_features_v1
SET session_median_price = 200000  -- Based on historical
WHERE salesId = [SALE_ID] AND book_number = 1;
```

### 5. Export and score

Follow the Quick Start steps above.

---

## File Structure

```
sm-market-value/
├── .venv/                          # Python virtual environment
├── .env                            # Database credentials & model config
├── models/
│   ├── aus/                        # Australia models
│   │   ├── mv_v1_q25.txt
│   │   ├── mv_v1_q50.txt
│   │   ├── mv_v1_q75.txt
│   │   ├── calibration_offsets.json
│   │   └── feature_cols.json
│   └── nzl/                        # New Zealand models
│       ├── mv_v1_q25.txt
│       ├── mv_v1_q50.txt
│       ├── mv_v1_q75.txt
│       ├── calibration_offsets.json
│       └── feature_cols.json
├── csv/                            # CSV outputs
│   ├── sale_{id}_inference.csv     # Inference data (step 1)
│   ├── sale_{id}_scored.csv        # Output predictions (step 2)
│   ├── feature_importance_aus.csv
│   └── feature_importance_nzl.csv
├── archive/                        # Legacy scripts
│   └── score_lots.py               # Old standalone scoring script
├── src/                            # Training scripts
│   └── train_market_value_model.py
├── score_sale.py                   # Pipeline orchestrator (run this)
├── run_rebuild.py                  # Feature rebuild (step 1)
├── score_lots.py                   # Lot scoring (step 2)
├── requirements.txt
└── README.md
```

---

## Retraining the Model

If you need to retrain (e.g., with new data or features):

### 1. Export training data

**For AUS:**
```sql
SELECT
    lot_id, salesId, sale_company, sale_year, book_number, day_number, sex,
    session_median_price,
    sire_sold_count_36m, sire_total_offered_36m, sire_clearance_rate_36m, sire_median_price_36m,
    sire_sold_count_12m, sire_total_offered_12m, sire_clearance_rate_12m, sire_median_price_12m,
    sire_momentum, sire_sample_flag_36m,
    dam_progeny_sold_count, dam_progeny_total_offered_count, dam_progeny_median_price, dam_first_foal_flag,
    vendor_sold_count_36m, vendor_total_offered_36m, vendor_clearance_rate_36m, vendor_median_price_36m,
    vendor_volume_bucket, vendor_first_seen_flag,
    log_price_index, hammer_price
FROM dbo.mv_yearling_lot_features_v1
WHERE isWithdrawn = 0
  AND isPassedIn = 0
  AND hammer_price > 0
  AND session_median_price > 0
  AND log_price_index IS NOT NULL;
```

Save as `training_data_aus.csv`.

**For NZL:**
```sql
-- Same query but from NZ table:
FROM dbo.mv_yearling_lot_features_nz_v1
```

Save as `training_data_nzl.csv`.

### 2. Update training script

```python
COUNTRY = "aus"  # or "nzl"
USE_CSV = True
CSV_PATH = "training_data_aus.csv"  # or "training_data_nzl.csv"
```

### 3. Run training

```bash
source venv/bin/activate
python3 train_market_value_model.py
```

### 4. Review results

Check `reports/feature_importance_{country}.csv` and console output for metrics.

---

## Troubleshooting

### "No module named 'pandas'"

Activate the virtual environment:
```bash
source venv/bin/activate
```

### "libomp.dylib not found" (macOS)

Install OpenMP:
```bash
brew install libomp
```

### Session median is NULL

For future sales, manually set the session median based on historical averages:
```sql
UPDATE dbo.mv_yearling_lot_features_v1
SET session_median_price = 70000
WHERE salesId = [SALE_ID];
```

### Prices showing as NaN

Check that `session_median_price` is set in your inference CSV. Future sales require manual assignment.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| V2.1 | Jan 2025 | Added `--output db` option to write predictions directly to `tblHorseAnalytics`. Elite scaling for predictions >= $300k. Price-aware confidence tiers. |
| V2.0 | Dec 2024 | Initial production release. LightGBM quantile models with calibrated P25/P50/P75 bands. |

---

## Contact

For questions or issues, contact the G1 Goldmine analytics team.
