# Running & Deploying Market Value Model

## Running the Pipeline

### Score a Sale (Full Pipeline)

```bash
# Activate environment
source .venv/bin/activate

# Output to CSV (default)
python score_sale.py --sale-id 2094

# Output directly to database
python score_sale.py --sale-id 2094 --output db
```

### Run Steps Individually

```bash
# Step 1: Rebuild features only
python run_rebuild.py --sale-id 2094
# Creates: csv/sale_2094_inference.csv

# Step 2: Score lots only
python score_lots.py --sale-id 2094
# Creates: csv/sale_2094_scored.csv

# Step 2 (alternative): Score and write to database
python score_lots.py --sale-id 2094 --output db
# Updates: tblHorseAnalytics
```

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

Create `.env` file with database credentials and settings:

```bash
# Database connection
DB_SERVER=your-server
DB_NAME=G1StallionMatchProductionV5
DB_USER=your-user
DB_PASSWORD=your-password

# Model selection (optional - defaults to country code)
AUS_MODEL=aus
NZL_MODEL=nzl
USA_MODEL=usa

# Historical lookback countries (optional)
HIST_COUNTRIES_NZL=NZL,AUS

# Audit user ID for database writes (default: 2)
AUDIT_USER_ID=2
```

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
