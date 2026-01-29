# Running & Deploying Market Value Model

## Running the Pipeline

### Score a Sale (Full Pipeline)

```bash
# Activate environment
source .venv/bin/activate

# Output to CSV (default)
python src/score_sale.py --sale-id 2094

# Output directly to database
python src/score_sale.py --sale-id 2094 --output db
```

### Run via API Server

```bash
# Activate environment
source .venv/bin/activate

# Start the API server
uvicorn api:app --host 0.0.0.0 --port 8000

# Score a sale (JSON response)
curl -X POST "http://localhost:8000/api/score/2094" \
  -H "X-API-Key: your-api-key"

# Score and write to database
curl -X POST "http://localhost:8000/api/score/2094?output=db" \
  -H "X-API-Key: your-api-key"

# Health check (no auth)
curl http://localhost:8000/health
```

### Run Steps Individually

```bash
# Rebuild features only (no scoring)
python src/run_rebuild.py --sale-id 2094
# Creates: csv/sale_2094_inference.csv
```

Note: `score_sale.py` handles both feature rebuild and scoring in a single command.

## Setup

### Prerequisites

- Python 3.9+
- ODBC Driver 17 for SQL Server
- Access to SM production database

### 1. Clone and Create Virtual Environment

```bash
cd ~/Projects/sm-market-value
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment

**Create `.env`** file with database credentials (secrets):

```bash
# Database credentials
DB_SERVER=your-server
DB_NAME=G1StallionMatchProductionV5
DB_USER=your-user
DB_PASSWORD=your-password

# API authentication (required for API server)
API_KEY=your-secret-api-key
```

**Review `config.json`** for app settings (already in repo):

```json
{
  "models": {
    "aus": "aus",
    "nzl": "nzl",
    "usa": "usa"
  },
  "year_start": 2020,
  "year_end": 2026,
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

Config can be modified at runtime via API or by editing the file directly.

## Output Files

| File | Description |
|------|-------------|
| `csv/sale_{id}_inference.csv` | Raw features for model input |
| `csv/sale_{id}_scored.csv` | Scored predictions |

## Database Output

When using `--output db`, the script upserts to `tblHorseAnalytics`:

| Column | Source |
|--------|--------|
| horseId | From lot data |
| salesId | From lot data |
| marketValue | Expected price (P50) |
| marketValueLow | Low price (P25) |
| marketValueHigh | High price (P75) |
| marketValueMultiplier | Price index |
| marketValueConfidence | high / medium / low |
| sessionMedianPrice | Sale session median |
| currencyId | Auto: AUS=1, NZL=6, USA=7 |
| modifiedBy | From AUDIT_USER_ID |
| modifiedOn | Current timestamp |

## Verification

### Check CSV Output

```bash
head -5 csv/sale_2094_scored.csv
```

### Verify Database Records

```sql
SELECT TOP 10
    horseId, marketValue, marketValueConfidence, modifiedOn
FROM tblHorseAnalytics
WHERE salesId = 2094
ORDER BY modifiedOn DESC
```

## Troubleshooting

### "No module named 'pandas'"

```bash
source .venv/bin/activate
```

### "ODBC Driver 17 not found"

macOS:
```bash
brew install msodbcsql17
```

Linux:
```bash
curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
apt-get update && apt-get install -y msodbcsql17
```

### "libomp.dylib not found" (macOS)

```bash
brew install libomp
```

### "Sale not found" or "No lots"

Verify the sale exists and has yearling lots:
```sql
SELECT COUNT(*)
FROM tblSalesLot LT
JOIN tblSales SL ON LT.salesId = SL.Id
JOIN tblSalesLotType LTP ON LT.lotType = LTP.Id
WHERE LT.salesId = 2094
  AND LTP.salesLotTypeName = 'Yearling'
```

### Session median is NULL

For future sales without sold lots, the script falls back to prior year's median. If still NULL, manually set it:

```sql
UPDATE dbo.mv_yearling_lot_features_v1
SET session_median_price = 80000
WHERE salesId = 2094;
```

## Production Deployment Checklist

1. [ ] Verify `.env` has production credentials
2. [ ] Verify `AUDIT_USER_ID` is set correctly
3. [ ] Run with `--output csv` first to review predictions
4. [ ] Spot check a few horses manually
5. [ ] Run with `--output db` to deploy
6. [ ] Verify records in `tblHorseAnalytics`
