# Handover — SM Market Value

**Date:** 2026-04-10
**Prepared by:** Farid Raisi
**Repository:** https://github.com/faridraisi/sm-market-value
**Production URL:** https://smmarketvalue.stallionmatch.horse

---

## What This App Does

SM Market Value predicts **expected hammer prices** for thoroughbred yearlings at auction sales worldwide. It produces three price points per lot (P25 low, P50 expected, P75 high) with confidence tiers, using a LightGBM quantile regression model trained on historical sale data.

The system is used by the **Stallion Match** platform to display market value estimates alongside catalogue lots before and after sales.

### Key Capabilities

- **Scoring** — Score all lots in a sale with predicted price ranges
- **Training** — Retrain models per country/region with auto-versioning
- **Sale Analytics** — Search sales, view detailed stats, prior year comparisons, sale history
- **Model Management** — Upload, download, delete, and switch between model versions
- **Two-step Commit** — Score a sale, review in UI, commit selected lots to database
- **Activity Logging** — Audit trail for all mutating operations

---

## Architecture

```
                   ┌──────────────┐
                   │   Frontend   │
                   │ (Stallion    │
                   │  Match UI)   │
                   └──────┬───────┘
                          │ HTTPS
                          ▼
              ┌───────────────────────┐
              │   FastAPI (api.py)    │
              │   uvicorn :8000      │
              │   ──────────────      │
              │   Auth: API Key +    │
              │         JWT/OTP      │
              └───────┬───────┬──────┘
                      │       │
         ┌────────────┘       └────────────┐
         ▼                                 ▼
┌─────────────────┐              ┌──────────────────┐
│  SQL Server DB  │              │  Local Filesystem │
│  (AWS RDS)      │              │  ─────────────── │
│  ─────────────  │              │  models/          │
│  G1Stallion-    │              │  config.json      │
│  MatchProd V5   │              │  logs/            │
└─────────────────┘              │  csv/             │
                                 └──────────────────┘
```

### Infrastructure

| Component | Detail |
|-----------|--------|
| **Runtime** | Python 3.11, FastAPI + Uvicorn |
| **ML** | LightGBM quantile gradient boosted trees |
| **Database** | SQL Server on AWS RDS (`G1StallionMatchProductionV5`) |
| **Hosting** | EKS (`data-feed` cluster), single replica |
| **Container** | Docker, ECR (`398646198502.dkr.ecr.ap-southeast-2.amazonaws.com/smmarketvalue`) |
| **SSL** | AWS LoadBalancer with ACM certificate |
| **DNS** | Route53 CNAME → LoadBalancer |
| **Email** | AWS Lambda + SES (shared service, API Gateway + API key) |

---

## Project Structure

```
sm-market-value/
├── api.py                    # FastAPI server — all endpoints
├── config.json               # Runtime-editable app config (regions, models, thresholds)
├── .env                      # Credentials (DB, API key, JWT, email — NOT in git)
├── .env.example              # Template for .env
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker build (linux/amd64, ODBC 18, LightGBM)
├── deployment.yaml           # K8s Deployment (env from secret)
├── service.yaml              # K8s Service (LoadBalancer, SSL on 443)
├── create-k8s-secret.sh      # K8s secret creation script (NOT in git)
├── src/
│   ├── __init__.py
│   ├── config.py             # Config singleton — loads config.json
│   ├── run_rebuild.py        # Feature engineering pipeline (DB → features CSV)
│   ├── score_sale.py         # Full scoring pipeline (rebuild + model inference)
│   └── train_model.py        # Model training with auto-versioning
├── models/                   # Trained model directories
│   ├── aus_v66/              # Active AUS model
│   ├── nzl_v2/               # Active NZL model
│   ├── usa_v1/               # USA model
│   ├── gbr_v1/, ire_v1/, fra_v1/  # Other region models
│   └── aus_v4/, aus_v6/, aus_v7/   # Historical AUS versions
├── docs/
│   ├── API_DOCUMENTATION.md  # Full API reference
│   ├── LOT_TYPE_EXPANSION.md # Future expansion planning
│   ├── EMAIL_SERVICE_INTEGRATION_GUIDE.md
│   └── EMAIL_LAMBDA_BUILD_SPEC.md
├── logs/                     # Activity log (JSONL, auto-rotating)
├── csv/                      # Scoring output files
├── archive/                  # Legacy V1 scripts (ignore)
├── CHANGELOG.md              # Full version history
├── DEPLOYMENT.md             # Deployment & operations guide
└── README.md                 # Quick start & model documentation
```

---

## Configuration

Configuration is split into two files:

### `.env` — Secrets (not in git)

| Variable | Purpose |
|----------|---------|
| `DB_SERVER` | SQL Server host:port (e.g. `hostname,1433`) |
| `DB_NAME` | Database name (`G1StallionMatchProductionV5`) |
| `DB_USER` / `DB_PASSWORD` | Database credentials |
| `API_KEY` | API key for server-to-server auth |
| `EMAIL_SERVICE_URL` | Lambda email endpoint |
| `EMAIL_SERVICE_API_KEY` | Email service API key |
| `AUTH_EMAIL_WHITELIST` | Comma-separated emails allowed to login |
| `JWT_SECRET` | Secret for signing JWT tokens |
| `JWT_EXPIRY_HOURS` | JWT token lifetime (default 24) |
| `AUTH_DEV_MODE` | `true` = log OTP to console instead of emailing |
| `DOCS_USERNAME` / `DOCS_PASSWORD` | HTTP Basic Auth for `/docs` and `/redoc` |
| `ACTIVITY_LOG_MAX_MB` | Max log file size before rotation (default 20) |
| `ACTIVITY_LOG_MAX_DAYS` | Max log age before rotation (default 26) |

### `config.json` — App Settings (in git, runtime-editable via API)

| Key | Purpose |
|-----|---------|
| `year_start` | Training data start year (2020) |
| `year_end` | Training data end year (`null` = current year) |
| `model_test_last_years` | Years held out for test set (2) |
| `sale_history_years` | Years of sale history in detail endpoint (5) |
| `audit_user_id` | User ID for database audit fields |
| `regions` | Per-country config (model, currency, thresholds, etc.) |

Each region in `regions` has: `model`, `currency_id`, `hist_countries`, `elite_scaling`, `confidence_tiers`, `sire_sample_min_count`.

**Currently configured regions:** AUS, NZL, USA, GBR, IRE, FRA, GER, ZAF, JPN, CAN, HKG

---

## Authentication

The API supports two auth methods (both work on all endpoints):

1. **API Key** — `X-API-Key` header. For server-to-server and CI/CD.
2. **JWT via Email OTP** — User requests OTP to whitelisted email → verifies → gets JWT token. For UI/human users.

API docs (`/docs`, `/redoc`) are protected by separate HTTP Basic Auth credentials.

The `/health` endpoint requires no auth.

---

## Core Pipeline

### How Scoring Works

```
1. Sale ID → fetch lots + metadata from DB
2. Fetch historical data (sire/dam/vendor metrics, rolling windows)
3. Compute features (run_rebuild.py)
4. Load trained model for sale's country
5. Predict P25/P50/P75 quantiles
6. Apply calibration offsets
7. Apply elite scaling for high-value lots
8. Assign confidence tiers
9. Return results (JSON) or write to tblHorseAnalytics (DB)
```

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Session Median** | Median hammer price for sold lots in a sale/book. Used as baseline — model predicts `log(price / session_median)`. For future sales, falls back to prior year's median via fuzzy name matching. |
| **Elite Scaling** | Lots predicted above a threshold (e.g. $500k AUS) get an upward adjustment because the model tends to underestimate premium lots. Configurable per region. |
| **Confidence Tiers** | High (no flags), Medium (1 flag), Low (2+ flags). Flags: thin sire sample, first foal, new vendor. |
| **Calibration Offsets** | Per-model offsets applied to P25/P75 predictions so that exactly 25%/75% of actuals fall within bands on holdout data. |
| **Fuzzy Sale Matching** | Prior year and history lookups use `SequenceMatcher` (threshold 0.75) to match sales across years despite name/type variations. |

---

## Database

### Connection

SQL Server on AWS RDS: `g1stallion.cdiyauaa0gdp.ap-southeast-2.rds.amazonaws.com,1433`

Database: `G1StallionMatchProductionV5`

Uses `pyodbc` with ODBC Driver 17/18 for SQL Server.

### Key Tables

| Table | Purpose |
|-------|---------|
| `tblSalesLot` | All auction lots (source data) |
| `tblSales` | Sale events (dates, companies, countries) |
| `tblSalesLotType` | Lot type lookup (Yearling, Weanling, etc.) |
| `tblHorseAnalytics` | **Output table** — scored predictions written here |
| `tblHorse` | Horse master data |
| `tblSalesCompany` | Sale companies |

The pipeline reads directly from the main tables — no golden tables or stored procedures are required for scoring. Feature engineering is done in Python.

### Write Operations

The only table the app writes to is `tblHorseAnalytics` (via the `/api/score/{sale_id}?output=db` or `/api/score/{sale_id}/commit` endpoints).

---

## API Endpoints Summary

Full documentation: `docs/API_DOCUMENTATION.md` or live at `/docs` (with Basic Auth).

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Health check (no auth) |
| `POST` | `/auth/request-otp` | Request email OTP |
| `POST` | `/auth/verify-otp` | Verify OTP, get JWT |
| `GET` | `/api/sales/search?q=...` | Typeahead sale search |
| `GET` | `/api/sales/{sale_id}` | Sale detail (stats, history, prior year) |
| `POST` | `/api/score/{sale_id}` | Score all lots in a sale |
| `POST` | `/api/score/{sale_id}/compare` | Score with comparison |
| `POST` | `/api/score/{sale_id}/commit` | Commit selected lots to DB |
| `POST` | `/api/train/{country}` | Train new model (background) |
| `GET` | `/api/train/status` | Training progress |
| `GET` | `/api/models/{country}` | List models with metrics |
| `GET` | `/api/models/{model_name}/download` | Download model ZIP |
| `POST` | `/api/models/{model_name}` | Upload model ZIP |
| `DELETE` | `/api/models/{model_name}` | Delete model |
| `GET` | `/api/config` | Full config |
| `GET/POST` | `/api/config/{country}` | Get/update region config |
| `PUT/DELETE` | `/api/config/{country}` | Create/delete region |
| `GET/PUT` | `/api/config/years` | Year range |
| `GET/PUT` | `/api/config/test-years` | Test years setting |
| `GET/PUT` | `/api/config/sale-history-years` | Sale history years |
| `GET` | `/api/activity` | Activity log (filterable) |

---

## Deployment

### Production Deployment Steps

```bash
# 1. Set AWS profile and cluster context
export AWS_PROFILE=default
aws eks update-kubeconfig --name data-feed --region ap-southeast-2

# 2. Build and push Docker image
cd /path/to/sm-market-value
docker build -t smmarketvalue .
docker tag smmarketvalue:latest 398646198502.dkr.ecr.ap-southeast-2.amazonaws.com/smmarketvalue:latest
aws ecr get-login-password --region ap-southeast-2 | docker login --username AWS --password-stdin 398646198502.dkr.ecr.ap-southeast-2.amazonaws.com
docker push 398646198502.dkr.ecr.ap-southeast-2.amazonaws.com/smmarketvalue:latest

# 3. Restart deployment
kubectl rollout restart deployment smmarketvalue
kubectl rollout status deployment smmarketvalue
```

### Updating Secrets

```bash
# Edit create-k8s-secret.sh with new values, then:
./create-k8s-secret.sh
kubectl rollout restart deployment/smmarketvalue
```

### Kubernetes Resources

| Resource | Name | Namespace |
|----------|------|-----------|
| Deployment | `smmarketvalue` | default |
| Service | `smmarketvalue-service` | default |
| Secret | `aws-secret-smmarketvalue` | default |
| ECR Repo | `smmarketvalue` | ap-southeast-2 |

See `DEPLOYMENT.md` for full deployment guide including first-time ECR setup, troubleshooting, and domain configuration.

---

## Local Development Setup

### Prerequisites

- Python 3.9+ (3.11 recommended)
- ODBC Driver 17 or 18 for SQL Server
- Access to the production database (or SSH tunnel to RDS)
- `libomp` on macOS (`brew install libomp`) for LightGBM

### Setup

```bash
git clone https://github.com/faridraisi/sm-market-value.git
cd sm-market-value
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Copy and edit credentials
cp .env.example .env
# Edit .env with your database credentials and API key

# Start the server
uvicorn api:app --host 0.0.0.0 --port 8000
```

### CLI Usage (without API server)

```bash
# Score a sale to CSV
python src/score_sale.py --sale-id 2094

# Score and write to database
python src/score_sale.py --sale-id 2094 --output db

# Train a new model
python src/train_model.py --country aus
```

---

## Trained Models

Models are stored in `models/` as LightGBM text files with calibration offsets.

### Active Models (per `config.json`)

| Region | Model Dir | Notes |
|--------|-----------|-------|
| AUS | `aus_v66` | Primary market, most data (~28k sold) |
| NZL | `nzl_v2` | Uses AUS+NZL historical data |
| USA | `usa` (base) | Second largest market |
| GBR | `gbr` | Combined GBR+IRE history |
| IRE | `ire` | Combined IRE+GBR history |
| FRA | `fra_v1` | France |

### Model Directory Contents

Each model directory contains:
- `mv_v1_q25.txt` / `mv_v1_q50.txt` / `mv_v1_q75.txt` — LightGBM models
- `calibration_offsets.json` — Calibration data and metadata
- `feature_cols.json` — Feature column order
- `feature_importance_*.json` — Feature importance rankings
- `training_report.txt` — Training metrics and evaluation

### Retraining

```bash
# Via API (background task)
curl -X POST "https://smmarketvalue.stallionmatch.horse/api/train/aus" \
  -H "X-API-Key: $API_KEY"

# Monitor progress
curl "https://smmarketvalue.stallionmatch.horse/api/train/status" \
  -H "X-API-Key: $API_KEY"

# Activate new model after training
curl -X POST "https://smmarketvalue.stallionmatch.horse/api/config/AUS" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "aus_v67"}'
```

---

## Known Limitations and Gotchas

1. **Yearling only** — The model currently only supports Yearling lot types. See `docs/LOT_TYPE_EXPANSION.md` for the planned expansion to Weanlings and other lot types.

2. **Session median dependency** — Scoring requires a valid session median. For future (unsold) sales, the system falls back to prior year's median. If no match is found, scoring returns a 422 error.

3. **Single replica** — The EKS deployment runs a single pod. Training locks the process (one training job at a time). If the pod restarts during training, the job is lost.

4. **Models are local** — Trained models are stored on the container filesystem. If the pod is replaced, models trained since the last Docker image build are lost. Use the model download/upload endpoints for backup.

5. **No automated retraining** — Models are retrained manually when needed. There is no scheduled retraining pipeline.

6. **Database user** — The app uses `dev_farid` for database access. You should create a dedicated service account.

7. **Countries without data** — Some configured regions (ZAF, GER, CAN, HKG) have no sold yearling lots with prices. Training for these will fail with a clear error message.

8. **Elite scaling** — Thresholds are region-specific and manually tuned. May need adjustment as market conditions change.

---

## External Dependencies

| Service | Purpose | Owner |
|---------|---------|-------|
| AWS RDS SQL Server | Primary database | G1 Goldmine infra team |
| AWS EKS `data-feed` cluster | Kubernetes hosting | G1 Goldmine infra team |
| AWS ECR | Docker image registry | Same AWS account (398646198502) |
| AWS Lambda email service | OTP email delivery | Shared service |
| Route53 | DNS for `stallionmatch.horse` | G1 Goldmine |
| ACM | SSL certificate | Auto-renewed by AWS |

---

## Documentation Index

| Document | Location | Description |
|----------|----------|-------------|
| Quick Start & Model Info | `README.md` | How to run, features, model details |
| Deployment Guide | `DEPLOYMENT.md` | Full EKS deployment steps, troubleshooting |
| API Reference | `docs/API_DOCUMENTATION.md` | All endpoints, auth, request/response formats |
| Changelog | `CHANGELOG.md` | Full version history (v2.0 → v2.16) |
| Lot Type Expansion | `docs/LOT_TYPE_EXPANSION.md` | Planning for adding Weanling and other lot types |
| Email Service | `docs/EMAIL_SERVICE_INTEGRATION_GUIDE.md` | How to use the shared email Lambda |
| Email Lambda Spec | `docs/EMAIL_LAMBDA_BUILD_SPEC.md` | Build spec for the email Lambda function |

---

## Immediate Actions for New Team

1. **Create a dedicated DB user** — Replace `dev_farid` with a service account
2. **Update `AUTH_EMAIL_WHITELIST`** — Add your team's email addresses
3. **Review K8s secret** — Run `create-k8s-secret.sh` after updating credentials
4. **Rotate `JWT_SECRET` and `API_KEY`** — Generate new secrets for your team
5. **Test locally** — Clone repo, set up `.env`, run `uvicorn api:app` and verify `/health`
6. **Review `config.json`** — Understand region configs, active models, and thresholds
7. **Check `CHANGELOG.md`** — Understand recent changes and current version (2.16.0)
