# Changelog

All notable changes to this project will be documented in this file.

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
