# Deployment Guide

This guide covers deploying the Market Value Scoring system.

---

## Prerequisites

- **Python 3.9+**
- **SQL Server** with ODBC Driver 18
- **macOS/Linux** (Windows untested)

### macOS Setup

```bash
# Install ODBC Driver
brew tap microsoft/mssql-release https://github.com/Microsoft/homebrew-mssql-release
brew update
brew install msodbcsql18

# Install OpenMP (for LightGBM)
brew install libomp
```

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/faridraisi/sm-market-value.git
cd sm-market-value
```

### 2. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```env
# Database connection
DB_SERVER=127.0.0.1,1433
DB_NAME=G1StallionMatchProductionV5
DB_USER=your_username
DB_PASSWORD=your_password

# Audit
AUDIT_USER_ID=2

# Country-specific configuration
COUNTRY_NZL_LOOKBACK=AUS,NZL
COUNTRY_NZL_MODEL=nzl

COUNTRY_AUS_LOOKBACK=AUS
COUNTRY_AUS_MODEL=aus

COUNTRY_USA_LOOKBACK=USA
COUNTRY_USA_MODEL=usa

# API Configuration
API_KEY=your-secret-api-key
```

---

## Running the CLI

### Score a Sale (Post-Sale)

```bash
source .venv/bin/activate

# Preview to CSV
python3 score_sale.py --sale-id 2096 --dry-run

# Write to database
python3 score_sale.py --sale-id 2096
```

### Score a Sale (Pre-Sale)

```bash
# Provide expected session median
python3 score_sale.py --sale-id 2098 --session-median 360000 --dry-run
```

---

## Running the API

### Development

```bash
source .venv/bin/activate
uvicorn api:app --reload --port 8000
```

### Production

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
```

### With Process Manager (PM2)

```bash
# Install PM2
npm install -g pm2

# Start API
pm2 start "uvicorn api:app --host 0.0.0.0 --port 8000" --name mv-api

# View logs
pm2 logs mv-api

# Restart
pm2 restart mv-api
```

### With systemd (Linux)

Create `/etc/systemd/system/mv-api.service`:

```ini
[Unit]
Description=Market Value API
After=network.target

[Service]
User=www-data
WorkingDirectory=/path/to/sm-market-value
Environment="PATH=/path/to/sm-market-value/.venv/bin"
ExecStart=/path/to/sm-market-value/.venv/bin/uvicorn api:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable mv-api
sudo systemctl start mv-api
```

---

## API Usage

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/api/score/{sale_id}` | Score a sale |
| GET | `/docs` | Swagger documentation |

### Authentication

All `/api/*` endpoints require the `X-API-Key` header:

```bash
curl -X POST "http://localhost:8000/api/score/2096" \
  -H "X-API-Key: your-api-key"
```

### Score a Sale

```bash
# Score and write to database
curl -X POST "http://localhost:8000/api/score/2096?write_to_db=true" \
  -H "X-API-Key: your-api-key"

# Score without DB write (JSON only)
curl -X POST "http://localhost:8000/api/score/2096?write_to_db=false" \
  -H "X-API-Key: your-api-key"

# Pre-sale with session median override
curl -X POST "http://localhost:8000/api/score/2098?write_to_db=false&session_median=360000" \
  -H "X-API-Key: your-api-key"
```

### Response Format

```json
{
  "sale_id": 2096,
  "sale_name": "Karaka Summer Sale",
  "country": "NZL",
  "lookback_countries": ["AUS", "NZL"],
  "model": "nzl",
  "model_version": "v2.1",
  "session_median_price": 50000.0,
  "total_lots": 150,
  "written_to_db": true,
  "predictions": [
    {
      "lot_id": 123,
      "lot_number": 1,
      "horse_name": "Example Horse",
      "sire_name": "Example Sire",
      "sex": "Colt",
      "mv_expected_price": 85000.0,
      "mv_low_price": 65000.0,
      "mv_high_price": 120000.0,
      "mv_confidence_tier": "high"
    }
  ]
}
```

---

## Postman Collection

Import `postman_collection.json` into Postman for pre-configured requests.

**Collection Variables:**
- `base_url`: `http://localhost:8000`
- `api_key`: Your API key

---

## Health Check

### CLI

```bash
python3 score_sale.py --sale-id 2096 --dry-run
```

### API

```bash
curl http://localhost:8000/health
# {"status":"healthy","version":"v2.1"}
```

---

## Troubleshooting

### "No module named 'pyodbc'"

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### "Login failed for user"

Check `.env` has correct database credentials.

### "libomp.dylib not found" (macOS)

```bash
brew install libomp
```

### "Model directory not found"

Ensure model files exist in `models/{country}/`:
- `mv_v1_q25.txt`
- `mv_v1_q50.txt`
- `mv_v1_q75.txt`
- `calibration_offsets.json`
- `feature_cols.json`

### API returns 401 Unauthorized

Check `X-API-Key` header matches `API_KEY` in `.env`.

---

## Security Considerations

1. **Never commit `.env`** - Contains credentials and API key
2. **Use strong API key** - Generate with `python3 -c "import secrets; print(secrets.token_urlsafe(32))"`
3. **Run behind reverse proxy** - Use nginx/Apache for SSL termination
4. **Restrict network access** - Firewall API to trusted IPs only
