# Running & Deploying Market Value Model

**Production URL:** https://smmarketvalue.stallionmatch.horse

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

# Search sales by name or company (typeahead)
curl "http://localhost:8000/api/sales/search?q=Magic" \
  -H "X-API-Key: your-api-key"

# Get sale details
curl "http://localhost:8000/api/sales/2002" \
  -H "X-API-Key: your-api-key"

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

---

## EKS Deployment

Deploy the API to the `data-feed` EKS cluster.

### Prerequisites

- AWS CLI configured with appropriate credentials
- `kubectl` configured for the `data-feed` EKS cluster
- Docker installed

```bash
# Verify cluster access
kubectl get nodes
```

### 1. Create ECR Repository (first time only)

```bash
aws ecr create-repository \
    --repository-name smmarketvalue \
    --region ap-southeast-2
```

### 2. Build and Push Docker Image

```bash
# Navigate to project root
cd /path/to/sm-market-value

# Login to ECR
aws ecr get-login-password --region ap-southeast-2 | \
    docker login --username AWS --password-stdin 398646198502.dkr.ecr.ap-southeast-2.amazonaws.com

# Build image (from project root, using Dockerfile)
docker build -t smmarketvalue:latest .

# Tag for ECR
docker tag smmarketvalue:latest \
    398646198502.dkr.ecr.ap-southeast-2.amazonaws.com/smmarketvalue:latest

# Push to ECR
docker push 398646198502.dkr.ecr.ap-southeast-2.amazonaws.com/smmarketvalue:latest
```

### 3. Create Kubernetes Secret

```bash
./create-k8s-secret.sh
```

Or manually (replace with actual values):

```bash
kubectl create secret generic aws-secret-smmarketvalue \
    --from-literal=DB_SERVER='your-db-server' \
    --from-literal=DB_NAME='G1StallionMatchProductionV5' \
    --from-literal=DB_USER='your-username' \
    --from-literal=DB_PASSWORD='your-password' \
    --from-literal=AUDIT_USER_ID='2' \
    --from-literal=COUNTRY_NZL_LOOKBACK='AUS,NZL' \
    --from-literal=COUNTRY_NZL_MODEL='nzl' \
    --from-literal=COUNTRY_AUS_LOOKBACK='AUS' \
    --from-literal=COUNTRY_AUS_MODEL='aus' \
    --from-literal=COUNTRY_USA_LOOKBACK='USA' \
    --from-literal=COUNTRY_USA_MODEL='usa' \
    --from-literal=API_KEY='your-api-key'
```

### 4. Deploy to Kubernetes

```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

# Verify deployment
kubectl get pods -l app=smmarketvalue
kubectl get svc smmarketvalue-service
```

### 5. Get Load Balancer URL

```bash
kubectl get svc smmarketvalue-service -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'
```
### Updating the Application

**All in One:**
```bash
echo "Switching to 'default' profile (Account: 398646198502)..."
export AWS_PROFILE=default
aws sts get-caller-identity
aws eks list-clusters --region ap-southeast-2
aws eks update-kubeconfig --name data-feed --region ap-southeast-2 --profile default
kubectl get nodes
kubectl get pods

cd /Users/fs/Projects/sm-market-value && \
docker build -t smmarketvalue . && \
docker tag smmarketvalue:latest 398646198502.dkr.ecr.ap-southeast-2.amazonaws.com/smmarketvalue:latest && \
aws ecr get-login-password --region ap-southeast-2 | docker login --username AWS --password-stdin 398646198502.dkr.ecr.ap-southeast-2.amazonaws.com && \
docker push 398646198502.dkr.ecr.ap-southeast-2.amazonaws.com/smmarketvalue:latest && \
kubectl set image deployment/smmarketvalue smmarketvalue=398646198502.dkr.ecr.ap-southeast-2.amazonaws.com/smmarketvalue:latest && \
kubectl rollout restart deployment smmarketvalue && \
kubectl get pods
```


### Update Deployment

```bash
# Build and push new image
docker build -t smmarketvalue:latest .
docker tag smmarketvalue:latest 398646198502.dkr.ecr.ap-southeast-2.amazonaws.com/smmarketvalue:latest
docker push 398646198502.dkr.ecr.ap-southeast-2.amazonaws.com/smmarketvalue:latest

# Restart deployment to pull new image
kubectl rollout restart deployment/smmarketvalue
kubectl rollout status deployment/smmarketvalue
```

### Update Secrets

```bash
kubectl delete secret aws-secret-smmarketvalue
./create-k8s-secret.sh
kubectl rollout restart deployment/smmarketvalue
```

### Custom Domain

To use `sm-market-value.stallionmatch.horse`:

1. Get Load Balancer hostname:
   ```bash
   kubectl get svc smmarketvalue-service -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'
   ```

2. In Route53, create a CNAME record pointing `sm-market-value.stallionmatch.horse` to the Load Balancer hostname.

### EKS Troubleshooting

```bash
# Pod status
kubectl describe pod -l app=smmarketvalue

# View logs
kubectl logs -l app=smmarketvalue -f

# Test DB connection
kubectl exec -it <pod-name> -- python -c "from src.score_sale import get_db_connection; get_db_connection()"
```
