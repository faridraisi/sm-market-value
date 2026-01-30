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
python src/score_sale.py --sale-id 2094

# Output directly to database (tblHorseAnalytics)
python src/score_sale.py --sale-id 2094 --output db
```

This runs two steps automatically:
1. **Feature rebuild** (`src/run_rebuild.py`) — Fetches data from database, computes features, outputs `csv/sale_2094_inference.csv`
2. **Scoring** — Loads the model, scores lots, outputs to CSV or database

The country is auto-detected from the database, and the correct model is loaded based on `.env` configuration.

### 3. Configuration

Configuration is split between two files:

**`.env`** — Credentials (secrets, not committed to git):
```bash
# Database credentials
DB_SERVER=127.0.0.1,1433
DB_NAME=G1StallionMatchProductionV5
DB_USER=your_user
DB_PASSWORD=your_password

# API authentication
API_KEY=your-secret-api-key
```

**`config.json`** — App settings (runtime-editable, committed to git):
```json
{
  "models": {
    "aus": "aus",
    "nzl": "nzl",
    "usa": "usa"
  },
  "year_start": 2020,
  "year_end": null,
  "model_test_last_years": 2,
  "audit_user_id": 2,
  "hist_countries": {
    "NZL": ["NZL", "AUS"]
  },
  "currency_map": {
    "AUS": 1,
    "NZL": 6,
    "USA": 7
  }
}
```

**Note:** `year_end: null` uses the current year automatically. Set to a specific year (e.g., `2027`) for future sales.
```

To use the AUS model for NZL sales (e.g., for cross-country testing), update `config.json`:
```json
"models": {
  "nzl": "aus"
}
```

Or via API:
```bash
curl -X PUT "http://localhost:8000/api/config/models/nzl?model=aus" -H "X-API-Key: $API_KEY"
```

### 4. Run via API (Optional)

Start the API server:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Score a sale via HTTP:
```bash
# Score and return JSON
curl -X POST "http://localhost:8000/api/score/2094" \
  -H "X-API-Key: your-api-key"

# Score and write to database
curl -X POST "http://localhost:8000/api/score/2094?output=db" \
  -H "X-API-Key: your-api-key"

# Health check (no auth required)
curl http://localhost:8000/health
```

#### Additional API Endpoints

**Training:**
```bash
# Train new model (runs in background)
curl -X POST "http://localhost:8000/api/train/aus" -H "X-API-Key: $API_KEY"

# Train with specific version
curl -X POST "http://localhost:8000/api/train/aus?version=v5" -H "X-API-Key: $API_KEY"
```

**Model Management:**
```bash
# List all models for a country (with metrics)
curl "http://localhost:8000/api/models/aus" -H "X-API-Key: $API_KEY"

# Get active models (all countries)
curl "http://localhost:8000/api/config/models" -H "X-API-Key: $API_KEY"

# Set active model for a country
curl -X PUT "http://localhost:8000/api/config/models/aus?model=aus_v5" -H "X-API-Key: $API_KEY"
```

**Configuration:**
```bash
# Get/set year range (year_end optional - omit to use current year)
curl "http://localhost:8000/api/config/years" -H "X-API-Key: $API_KEY"
curl -X PUT "http://localhost:8000/api/config/years?year_start=2020" -H "X-API-Key: $API_KEY"
curl -X PUT "http://localhost:8000/api/config/years?year_start=2020&year_end=2027" -H "X-API-Key: $API_KEY"

# Get/set test years (for train/test split)
curl "http://localhost:8000/api/config/test-years" -H "X-API-Key: $API_KEY"
curl -X PUT "http://localhost:8000/api/config/test-years?model_test_last_years=2" -H "X-API-Key: $API_KEY"

# Get/set historical countries
curl "http://localhost:8000/api/config/hist-countries" -H "X-API-Key: $API_KEY"
curl -X PUT "http://localhost:8000/api/config/hist-countries/NZL?hist_countries=NZL&hist_countries=AUS" -H "X-API-Key: $API_KEY"
```

### 5. View Results

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
├── .env                            # Database credentials (secrets)
├── .env.example                    # Template for .env
├── config.json                     # App configuration (runtime-editable)
├── api.py                          # FastAPI server (main entry point)
├── src/                            # Core modules
│   ├── __init__.py
│   ├── config.py                   # Configuration loader
│   ├── run_rebuild.py              # Feature rebuild
│   ├── score_sale.py               # Score sale pipeline (rebuild + scoring)
│   └── train_model.py              # Model training with auto-versioning
├── models/
│   ├── aus/                        # Australia models (base)
│   │   ├── mv_v1_q25.txt
│   │   ├── mv_v1_q50.txt
│   │   ├── mv_v1_q75.txt
│   │   ├── calibration_offsets.json
│   │   └── feature_cols.json
│   ├── aus_v2/                     # Australia models (versioned)
│   │   ├── mv_v1_q25.txt
│   │   ├── mv_v1_q50.txt
│   │   ├── mv_v1_q75.txt
│   │   ├── calibration_offsets.json
│   │   ├── feature_cols.json
│   │   ├── feature_importance_aus_v2.json
│   │   └── training_report.txt     # Training metrics report
│   └── nzl/                        # New Zealand models
│       └── ...
├── csv/                            # CSV outputs
│   ├── sale_{id}_inference.csv     # Inference data (step 1)
│   └── sale_{id}_scored.csv        # Output predictions (step 2)
├── archive/                        # Legacy scripts
├── requirements.txt
└── README.md
```

---

## Retraining the Model

Use `src/train_model.py` to retrain models with auto-versioning and comprehensive training reports.

**Database access is read-only** - no writes to database. Features are computed in Python/pandas.

### Quick Start

```bash
# Retrain from database (recommended)
python src/train_model.py --country aus

# Or use existing CSV
python src/train_model.py --country aus --csv training_data.csv

# Force specific version
python src/train_model.py --country nzl --version v3
```

### What It Does

1. **Fetches data** via read-only SQL queries (no golden table dependency)
2. **Computes features** in Python/pandas (sire/dam/vendor metrics, point-in-time)
3. **Splits data** time-based: 2020-2023 train, 2024+ test
4. **Trains baseline** (Elastic Net) for sanity check
5. **Trains quantile models** (Q25, Q50, Q75) with early stopping
6. **Evaluates** with MAE, RMSE, R², MAPE, coverage metrics
7. **Calibrates** P25/P75 for exact coverage targets
8. **Saves artifacts** to versioned directory with training report

### Output Structure

```
models/aus_v3/
├── mv_v1_q25.txt              # LightGBM Q25 model
├── mv_v1_q50.txt              # LightGBM Q50 model
├── mv_v1_q75.txt              # LightGBM Q75 model
├── calibration_offsets.json   # Calibration offsets + metadata
├── feature_cols.json          # Feature column list
├── feature_importance_aus_v3.json
└── training_report.txt        # Comprehensive metrics report
```

### Training Report Contents

The `training_report.txt` includes:
- **Data summary**: Sample counts, train/val/test splits
- **Baseline model**: Elastic Net MAE/R² vs naive predictor
- **Quantile models**: Trees per model (early stopping iterations)
- **Evaluation**: MAE, RMSE, R², raw coverage, dollar-space MAPE
- **Calibration**: Offsets and calibrated coverage
- **Feature importance**: Top 15 features by gain

### Activating New Model

After training, update `config.json`:
```json
"models": {
  "aus": "aus_v3"
}
```

Or via API:
```bash
curl -X PUT "http://localhost:8000/api/config/models/aus?model=aus_v3" -H "X-API-Key: $API_KEY"
```

### Manual CSV Export (Alternative)

If you prefer to export data manually:

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

Then run with `--csv` flag.

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
| V2.4 | Jan 2025 | Centralized config (`config.json` + `.env`). Added API endpoints for training, model listing, and runtime config management. |
| V2.3 | Jan 2025 | Added `src/train_model.py` with auto-versioning, time-based splits, baseline model comparison, comprehensive training report (`training_report.txt`), and evaluation metrics (MAE/RMSE/R²/MAPE). |
| V2.2 | Jan 2025 | Added FastAPI endpoint (`POST /api/score/{sale_id}`) for HTTP-based scoring. |
| V2.1 | Jan 2025 | Added `--output db` option to write predictions directly to `tblHorseAnalytics`. Elite scaling for predictions >= $300k. Price-aware confidence tiers. |
| V2.0 | Dec 2024 | Initial production release. LightGBM quantile models with calibrated P25/P50/P75 bands. |

---

## Contact

For questions or issues, contact the G1 Goldmine analytics team.
