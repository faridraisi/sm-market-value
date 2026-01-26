# Changelog

All notable changes to this project will be documented in this file.

---

## [2.1.1] - 2025-01-27

### Added
- **FastAPI Endpoint** - REST API for scoring sales
  - `POST /api/score/{sale_id}` - Score a sale with optional DB write
  - `GET /health` - Health check endpoint
  - `GET /docs` - Swagger documentation
  - API key authentication via `X-API-Key` header
- **Configurable Country Lookback** - Pool historical data from multiple countries
  - `COUNTRY_*_LOOKBACK` env var to specify which countries to include
  - `COUNTRY_*_MODEL` env var to specify which model to use
  - NZL sales can now use AUS+NZL pooled data for better sire/dam/vendor metrics
- **USA Support** - Added USA to supported countries (currencyId=7)
  - Requires USA model files to be added to `models/usa/`
- **Postman Collection** - `postman_collection.json` with pre-configured requests
- **Deployment Guide** - `DEPLOYMENT.md` with setup and production deployment instructions

### Changed
- `load_features()` now accepts `lookback_countries` list instead of single country code
- `load_models()` now accepts model directory name directly

### Technical Notes
- Currency conversion for multi-country lookback is not yet implemented
  - AUD/NZL rates are close (~0.90-0.95), impact is minimal
  - TODO: Convert using `tblCurrencyrate` for more accurate metrics

---

## [2.1.0] - 2025-01-15

### Added
- **Database-driven scoring** - `score_sale.py` queries base tables directly
- **Elite scaling** - Additional boost for predictions >= $300k
- **Auto country detection** - Determines country from sale ID
- **Confidence tiers** - high/medium/low based on data quality flags
- **MERGE upsert** - Updates existing records in `tblHorseAnalytics`

### Changed
- Target variable changed to `log(hammer_price / session_median_price)`
- Separate models per country (AUS, NZL)

---

## [2.0.0] - 2024-12-01

### Added
- Initial production release
- LightGBM quantile gradient boosted decision trees
- P25/P50/P75 predictions with calibrated coverage
- CSV-based scoring workflow (`03.score_lots.py`)

### Features
- Sire metrics (36m and 12m lookback)
- Dam production stats
- Vendor track record
- Sale context features
