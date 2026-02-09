# Changelog

All notable changes to this project will be documented in this file.

## [2.13.0] - 2026-02-10

### Added
- **Training status endpoint** `GET /api/train/status` - Monitor training progress with phase tracking (`exporting_data`, `preparing_features`, `splitting_data`, `training_models`, `evaluating`, `saving_artifacts`)
- **Concurrency lock** - `POST /api/train/{country}` now returns `409 Conflict` if training is already in progress
- **Training state tracking** - In-memory state with status (`idle`, `training`, `completed`, `failed`), timestamps, and error messages

## [2.12.0] - 2026-02-06

### Added
- **Sale history** in sale detail endpoint - `history` array with yearly stats (sold_count, gross, avg_price, median_price, clearance_rate)
- **Config: `sale_history_years`** - Number of years of history to include (default 5, set to 0 to disable)
- **Config endpoints** `GET/PUT /api/config/sale-history-years` for runtime configuration
- **Prior year dates** - `start_date` and `end_date` fields in `prior_year` object
- **Currency in search** - `currency_code` field added to sale search results

## [2.11.0] - 2026-02-06

### Added
- **Sale Search Endpoint** `GET /api/sales/search` for typeahead search in frontend:
  - Search by sale name or company name
  - Returns sale_id, sale_name, sale_date, country_code, lot_count, sale_company, status
  - Status is "upcoming" for future sales, "past" for past sales
  - Query param `q` (min 3 chars), optional `limit` (default 20, max 100)
- **Sale Detail Endpoint** `GET /api/sales/{sale_id}` for detailed sale information:
  - Basic info: sale_code, name, dates, type, status, company, country, currency
  - Lot stats: total, sold, passed in, withdrawn, clearance rate
  - **Enhanced price stats** (past sales):
    - Basic: gross, avg, median, min, max
    - Quartiles: q1_price (25th percentile), q3_price (75th percentile)
    - Advanced: top10_avg, std_dev
    - Price bands: under_50k, 50k-100k, 100k-200k, 200k-500k, 500k-1m, over_1m
  - **Prior year comparison** (matched by company + type + month):
    - Prior year sale_ids and names
    - Prior year lot_stats and price_stats
    - YoY changes: gross_pct, avg_price_pct, median_price_pct, sold_count_change, clearance_rate_change
  - Queue stats: report generation status (completed, in_queue, failed, postponed)
  - Book/session breakdown with lot counts per day

## [2.10.0] - 2026-02-05

### Added
- **EKS Deployment** - Kubernetes deployment to `data-feed` cluster:
  - `Dockerfile` with MSSQL ODBC drivers and LightGBM support
  - `deployment.yaml` and `service.yaml` for Kubernetes
  - `create-k8s-secret.sh` for secret management
  - LoadBalancer with SSL termination (port 443)
  - Production URL: `https://smmarketvalue.stallionmatch.horse`
  - Route53 CNAME record configured
- **Session median fallback** - Auto-lookup previous year's median for pre-sale scoring when no sold lots exist

### Changed
- `api.py` moved from `src/` to project root
- Dockerfile CMD updated to `uvicorn api:app`

## [2.9.0] - 2026-02-04

### Added
- **Email OTP authentication** - Alternative to API key for user-based access:
  - `POST /auth/request-otp` - Request OTP for whitelisted email
  - `POST /auth/verify-otp` - Verify OTP and receive JWT token
  - JWT tokens valid for 24 hours (configurable via `JWT_EXPIRY_HOURS`)
  - OTPs expire after 10 minutes and are single-use
- **Dual authentication support** - API accepts both `X-API-Key` header and `Authorization: Bearer <jwt>` header
- **Dev mode** - Set `AUTH_DEV_MODE=true` to log OTPs to console instead of sending emails
- New environment variables: `EMAIL_SERVICE_URL`, `EMAIL_SERVICE_API_KEY`, `AUTH_EMAIL_WHITELIST`, `JWT_SECRET`, `JWT_EXPIRY_HOURS`, `AUTH_DEV_MODE`

### Dependencies
- Added `PyJWT` for JWT token handling
- Added `httpx` for async HTTP requests to email service

## [2.8.0] - 2026-02-04

### Added
- **Model upload/download/delete endpoints** for CI/CD and backup workflows:
  - `GET /api/models/{model_name}/download` - Download model as ZIP
  - `POST /api/models/{model_name}` - Upload new model from ZIP
  - `DELETE /api/models/{model_name}` - Delete model (protected if active)
- Upload validates required files (calibration_offsets.json, feature_cols.json, mv_v1_q*.txt)
- Delete blocked if model is active for any region in config

## [2.7.1] - 2026-02-03

### Added
- **Session median override** - Optional `session_median` query parameter for scoring endpoints:
  - `POST /api/score/{sale_id}?session_median=20000` - Override automatic prior year lookup
  - `POST /api/score/{sale_id}/compare?session_median=20000` - Compare using custom median
  - Useful for future sales where prior year median may not be representative

## [2.7.0] - 2026-02-02

### Changed
- **Simplified Config API** - Replaced granular config endpoints with REST-style region management:
  - `GET /api/config` - Get full configuration including all regions
  - `GET /api/config/{country}` - Get region config
  - `POST /api/config/{country}` - Partial update (supports nested updates like `{"elite_scaling": {"threshold": 600000}}`)
  - `PUT /api/config/{country}` - Create new region (full config required)
  - `DELETE /api/config/{country}` - Remove region
- Kept `GET/PUT /api/config/years` and `GET/PUT /api/config/test-years` as separate endpoints

### Removed
- `/api/config/models` and `/api/config/models/{country}`
- `/api/config/hist-countries` and `/api/config/hist-countries/{country}`
- `/api/config/regions` and `/api/config/regions/{country}`
- `/api/config/elite-scaling/{country}`
- `/api/config/confidence-tiers/{country}`
- `/api/config/sire-sample-min-count/{country}`

## [2.6.0] - 2026-02-02

### Added
- **Selective lot commit endpoint** `POST /api/score/{sale_id}/commit`
  - Two-step workflow: score sale → select lots in UI → commit only selected to DB
  - Returns insert/update counts for feedback
  - Validates `sales_id` in each lot matches path parameter
- `mv_expected_index` field added to `LotScore` response (price multiplier vs session median)

### Changed
- `upsert_to_database()` now returns `tuple[int, int]` (inserted, updated) counts

## [2.5.0] - 2025-01-30

### Added
- **Summary statistics** in `POST /api/score/{sale_id}` response:
  - `model_dir` - Active model directory used for scoring
  - `summary.gross` - Sum of low/expected/high prices across all lots
  - `summary.median_prices` - Median low/expected/high price per lot
  - `summary.confidence_tiers` - Count of lots by confidence tier (high/medium/low)
  - `summary.elite_scaling_count` - Number of lots above elite threshold
  - `summary.elite_scaling_percent` - Percentage of lots above elite threshold

## [2.4.0] - 2025-01-30

### Added
- **Centralized configuration** - App settings now in `config.json` (runtime-editable), credentials remain in `.env`
- `src/config.py` - Configuration loader with `Config` singleton and `reload()` method
- `.env.example` - Template file for credentials

### Added API Endpoints
- `POST /api/train/{country}` - Train new model (background task)
- `GET /api/models/{country}` - List all models with training metrics and top features
- `GET /api/config/models` - Get active models for all countries
- `PUT /api/config/models/{country}` - Set active model for country
- `GET /api/config/years` - Get year range (year_start, year_end)
- `PUT /api/config/years` - Set year range (year_end optional, null = current year)
- `GET /api/config/test-years` - Get model_test_last_years
- `PUT /api/config/test-years` - Set model_test_last_years
- `GET /api/config/hist-countries` - Get all historical country mappings
- `GET /api/config/hist-countries/{country}` - Get historical countries for specific country
- `PUT /api/config/hist-countries/{country}` - Set historical countries
- `DELETE /api/config/hist-countries/{country}` - Remove historical country override

### Changed
- Moved app config from `.env` to `config.json`:
  - `models` (was `AUS_MODEL`, `NZL_MODEL`, `USA_MODEL`)
  - `year_start`, `year_end` (was `YEAR_START`, `YEAR_END`)
  - `audit_user_id` (was `AUDIT_USER_ID`)
  - `hist_countries` (was `HIST_COUNTRIES_*`)
  - `currency_map` (new, was hardcoded)
- `.env` now only contains credentials: `DB_SERVER`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `API_KEY`
- Updated `run_rebuild.py`, `score_sale.py`, `train_model.py` to use `config` module
- **Dynamic year_end** - `year_end: null` in config uses current year automatically; set to specific year for future sales
- **Configurable train/test split** - Added `model_test_last_years` to control test set size (e.g., `2` = last 2 years as test)
- Train/test split now derived from config instead of hardcoded 2024 threshold

## [2.3.0] - 2025-01-29

### Added
- `src/train_model.py` - Automated model training script with:
  - **Read-only database access** - no writes, no golden table dependency
  - Feature computation in Python/pandas (same as `run_rebuild.py`)
  - Auto-versioning (scans `models/` for existing versions, increments automatically)
  - Time-based data splits (2020-2023 train, 2024+ test)
  - Elastic Net baseline model for sanity check comparison
  - Comprehensive evaluation metrics: MAE, RMSE, R², MAPE, coverage
  - `training_report.txt` saved to model directory with full metrics
  - Support for both database export and CSV input
- Training report includes: data summary, baseline comparison, model iterations, evaluation metrics, calibration offsets, feature importance (top 15)

### Changed
- Model directories now use underscore format (`aus_v2`) instead of hyphen (`aus-v2`)
- Version detection supports both formats for backward compatibility
- Merged `score_lots.py` into `score_sale.py` - single script now handles full pipeline
- Removed `score_lots.py` (scoring logic now in `score_sale.py`)

## [2.2.0] - 2025-01-27

### Added
- FastAPI server (`api.py`) for HTTP-based scoring
- `POST /api/score/{sale_id}` endpoint with API key authentication
- `GET /health` endpoint for health checks
- `rebuild_sale_features()` wrapper function in `run_rebuild.py`
- `score_sale_lots()` wrapper function in `score_lots.py`
- `API_KEY` environment variable for API authentication
- `src/__init__.py` for package imports

### Changed
- Moved core modules to `src/` directory (`run_rebuild.py`, `score_lots.py`, `score_sale.py`)
- `api.py` remains in root as main entry point
- Updated all documentation with new paths

## [2.1.0] - 2025-01-15

### Added
- `--output db` option to write predictions directly to `tblHorseAnalytics`
- Elite scaling for predictions >= $300k to fix undervaluation at premium tiers
- Price-aware confidence tiers (predicted price factors into tier calculation)
- Currency mapping by country code (AUS=1, NZL=6, USA=7)
- `AUDIT_USER_ID` environment variable for database audit fields

### Changed
- Confidence tier logic now considers predicted price tier
- Predictions >= $300k automatically get "low" confidence

## [2.0.0] - 2024-12-01

### Added
- Initial production release
- LightGBM quantile gradient boosted models (P25, P50, P75)
- Calibrated prediction bands
- Support for AUS and NZL markets
- CLI pipeline: `score_sale.py`, `run_rebuild.py`, `score_lots.py`
- Feature engineering for sire, dam, and vendor metrics
