# Deployment Guide

This guide covers deploying the Market Value Scoring system.

**Production URL:** https://smmarketvalue.stallionmatch.horse

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

### CLI Reference

```
usage: python3 src/score_sale.py [-h] --sale-id SALE_ID [--dry-run] [--session-median SESSION_MEDIAN]

Score yearling lots for market value prediction

Arguments:
  --sale-id SALE_ID     Sale ID to score (required)
  --dry-run             Write to CSV instead of database
  --session-median      Override session median (required for pre-sale scoring)
```

### Score a Sale (Post-Sale)

```bash
source .venv/bin/activate

# Preview to CSV
python3 src/score_sale.py --sale-id 2096 --dry-run

# Write to database
python3 src/score_sale.py --sale-id 2096
```

### Score a Sale (Pre-Sale)

```bash
# Provide expected session median
python3 src/score_sale.py --sale-id 2098 --session-median 360000 --dry-run
```

### Typical Session Medians

| Sale | Median |
|------|--------|
| **AUS** | |
| Inglis Easter | $300,000 - $360,000 AUD |
| Gold Coast Yearling Sale Book 1 | $200,000 AUD |
| Gold Coast Yearling Sale Book 2 | $35,000 AUD |
| Inglis Premier | $80,000 AUD |
| Inglis Classic | $70,000 AUD |
| **NZL** | |
| Karaka Book 1 | $110,000 NZD |
| Karaka Book 2 | $27,500 NZD |
| Karaka Summer Sale | $10,000 NZD |

Query historical medians:
```sql
SELECT YEAR(S.startDate) as year, S.salesName,
       (SELECT TOP 1 PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY LT.price) OVER ()
        FROM tblSalesLot LT
        WHERE LT.salesId = S.Id AND LT.price > 0
          AND ISNULL(LT.isPassedIn, 0) = 0 AND ISNULL(LT.isWithdrawn, 0) = 0) as median
FROM tblSales S
WHERE S.salesName LIKE '%Easter%'
ORDER BY S.startDate DESC;
```

---

## Running the API

### Development

```bash
source .venv/bin/activate
uvicorn src.api:app --reload --port 8000
```

### Production

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 4
```

### With Process Manager (PM2)

```bash
# Install PM2
npm install -g pm2

# Start API
pm2 start "uvicorn src.api:app --host 0.0.0.0 --port 8000" --name mv-api

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
ExecStart=/path/to/sm-market-value/.venv/bin/uvicorn src.api:app --host 0.0.0.0 --port 8000
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
- `local_url`: `http://localhost:8000`
- `remote_url`: `https://smmarketvalue.stallionmatch.horse`
- `base_url`: `{{remote_url}}` (change to `{{local_url}}` for local dev)
- `api_key`: Your API key

---

## Health Check

### CLI

```bash
python3 src/score_sale.py --sale-id 2096 --dry-run
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
