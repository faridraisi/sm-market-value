# Market Value API Documentation

API documentation for the Market Value scoring system. This document covers all endpoints, authentication, request/response formats, and integration examples.

**Base URL:** `http://localhost:8000` (development) or your production server
**API Version:** 2.7.0

---

## Table of Contents

1. [Authentication](#authentication)
2. [Endpoints Overview](#endpoints-overview)
3. [Scoring Endpoints](#scoring-endpoints)
4. [Model Management](#model-management)
5. [Configuration Endpoints](#configuration-endpoints)
6. [Response Types](#response-types)
7. [Error Handling](#error-handling)
8. [Integration Examples](#integration-examples)

---

## Authentication

All endpoints (except `/health`) require API key authentication via the `X-API-Key` header.

### Request Header

```
X-API-Key: <your-api-key>
```

### Environment Setup

Create a `.env` file with the following credentials:

```env
# Database credentials (required)
DB_SERVER=127.0.0.1,1433
DB_NAME=G1StallionMatchProductionV5
DB_USER=your_user
DB_PASSWORD=your_password

# API authentication (required)
API_KEY=your_api_key
```

### Authentication Errors

| Status | Description |
|--------|-------------|
| `403` | Invalid or missing API key |

---

## Endpoints Overview

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/health` | No | Health check |
| `POST` | `/api/score/{sale_id}` | Yes | Score all lots for a sale |
| `POST` | `/api/score/{sale_id}/compare` | Yes | Score and compare with existing DB values |
| `POST` | `/api/score/{sale_id}/commit` | Yes | Commit selected lots to database |
| `POST` | `/api/train/{country}` | Yes | Train new model (background) |
| `GET` | `/api/models/{country}` | Yes | List all models for country |
| `GET` | `/api/config/models` | Yes | Get active models |
| `PUT` | `/api/config/models/{country}` | Yes | Set active model |
| `GET` | `/api/config/years` | Yes | Get year range |
| `PUT` | `/api/config/years` | Yes | Set year range |
| `GET` | `/api/config/test-years` | Yes | Get test years config |
| `PUT` | `/api/config/test-years` | Yes | Set test years config |
| `GET` | `/api/config/hist-countries` | Yes | Get all historical mappings |
| `GET` | `/api/config/hist-countries/{country}` | Yes | Get historical countries |
| `PUT` | `/api/config/hist-countries/{country}` | Yes | Set historical countries |
| `DELETE` | `/api/config/hist-countries/{country}` | Yes | Remove historical override |

---

## Scoring Endpoints

### Health Check

Check API and database connectivity.

```
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "database_connected": true
}
```

| Status Value | Description |
|--------------|-------------|
| `healthy` | All systems operational |
| `degraded` | Database connection failed |

---

### Score Sale

Score all lots for a given sale. Returns market value predictions with confidence bands.

```
POST /api/score/{sale_id}
```

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `sale_id` | integer | The sale ID to score |

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output` | string | `"none"` | Output destination: `"none"`, `"csv"`, or `"db"` |

**Response:**
```json
{
  "sale_id": 2094,
  "country_code": "AUS",
  "model_dir": "models/aus_v3",
  "total_lots": 806,
  "summary": {
    "gross": {
      "low": 43419300,
      "expected": 73225800,
      "high": 122159300
    },
    "median_prices": {
      "low": 53850,
      "expected": 90800,
      "high": 151550
    },
    "confidence_tiers": {
      "high": 15,
      "medium": 480,
      "low": 311
    },
    "elite_scaling_count": 12,
    "elite_scaling_percent": 1.49
  },
  "lots": [
    {
      "lot_id": 123456,
      "horse_id": 789012,
      "sales_id": 2094,
      "lot_number": 1,
      "horse_name": "EXAMPLE HORSE",
      "sire_name": "EXAMPLE SIRE",
      "session_median_price": 85000,
      "mv_expected_price": 120000,
      "mv_low_price": 75000,
      "mv_high_price": 195000,
      "mv_expected_index": 1.41,
      "mv_confidence_tier": "medium"
    }
  ],
  "output_written": null
}
```

**Summary Fields:**

| Field | Description |
|-------|-------------|
| `gross.low` | Sum of all low price predictions |
| `gross.expected` | Sum of all expected (P50) price predictions |
| `gross.high` | Sum of all high price predictions |
| `median_prices.low` | Median of low prices across lots |
| `median_prices.expected` | Median of expected prices across lots |
| `median_prices.high` | Median of high prices across lots |
| `confidence_tiers.high` | Count of lots with high confidence |
| `confidence_tiers.medium` | Count of lots with medium confidence |
| `confidence_tiers.low` | Count of lots with low confidence |
| `elite_scaling_count` | Lots with expected price >= elite threshold ($300k) |
| `elite_scaling_percent` | Percentage of lots above elite threshold |

**Lot Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `lot_id` | integer | Unique lot identifier |
| `horse_id` | integer\|null | Horse ID (if available) |
| `sales_id` | integer | Parent sale ID |
| `lot_number` | integer\|null | Catalogue lot number |
| `horse_name` | string\|null | Horse name |
| `sire_name` | string\|null | Sire name |
| `session_median_price` | float | Session median price used for calibration |
| `mv_expected_price` | float | P50 predicted price (expected value) |
| `mv_low_price` | float | P25 predicted price (low estimate) |
| `mv_high_price` | float | P75 predicted price (high estimate) |
| `mv_expected_index` | float | Price multiplier vs session median |
| `mv_confidence_tier` | string | Confidence level: `"high"`, `"medium"`, `"low"` |

**Output Options:**

| Value | Description |
|-------|-------------|
| `none` | Return JSON only (default) |
| `csv` | Save to `csv/sale_{sale_id}_scored.csv` |
| `db` | Write predictions to `tblHorseAnalytics` |

---

### Compare with Existing Values

Score a sale and compare new predictions with existing values in `tblHorseAnalytics`. This is a read-only operation that enables previewing changes before committing.

```
POST /api/score/{sale_id}/compare
```

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `sale_id` | integer | The sale ID to score and compare |

**Response:**
```json
{
  "sale_id": 2101,
  "country_code": "AUS",
  "model_dir": "models/aus_v3",
  "total_lots": 490,
  "new_lots": 12,
  "existing_lots": 478,
  "changed_lots": 156,
  "unchanged_lots": 322,
  "avg_price_delta": 3500.00,
  "avg_price_delta_pct": 2.80,
  "lots": [
    {
      "lot_id": 548629,
      "horse_id": 1792852,
      "sales_id": 2101,
      "lot_number": 1,
      "horse_name": "EXAMPLE HORSE",
      "sire_name": "EXAMPLE SIRE",
      "session_median_price": 85000,
      "new": {
        "lot_id": 548629,
        "horse_id": 1792852,
        "sales_id": 2101,
        "lot_number": 1,
        "horse_name": "EXAMPLE HORSE",
        "sire_name": "EXAMPLE SIRE",
        "session_median_price": 85000,
        "mv_expected_price": 125000,
        "mv_low_price": 80000,
        "mv_high_price": 200000,
        "mv_expected_index": 1.47,
        "mv_confidence_tier": "medium"
      },
      "existing": {
        "mv_expected_price": 120000,
        "mv_low_price": 75000,
        "mv_high_price": 195000,
        "mv_expected_index": 1.41,
        "mv_confidence_tier": "medium",
        "session_median_price": 85000
      },
      "delta": {
        "mv_expected_price": 5000,
        "mv_expected_price_pct": 4.17,
        "mv_low_price": 5000,
        "mv_high_price": 5000
      },
      "is_new": false
    }
  ]
}
```

**Summary Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `total_lots` | integer | Total lots scored |
| `new_lots` | integer | Lots not in database (no existing prediction) |
| `existing_lots` | integer | Lots with existing predictions |
| `changed_lots` | integer | Existing lots where new != existing |
| `unchanged_lots` | integer | Existing lots where new == existing |
| `avg_price_delta` | float | Average absolute price change (existing lots only) |
| `avg_price_delta_pct` | float | Average percentage price change (existing lots only) |

**Lot Comparison Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `new` | LotScore | New prediction from current model |
| `existing` | ExistingValues\|null | Current DB values (null if new lot) |
| `delta` | DeltaValues\|null | Calculated differences (null if new lot) |
| `is_new` | boolean | `true` if lot has no existing prediction |

**ExistingValues Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `mv_expected_price` | float\|null | Current marketValue in DB |
| `mv_low_price` | float\|null | Current marketValueLow in DB |
| `mv_high_price` | float\|null | Current marketValueHigh in DB |
| `mv_expected_index` | float\|null | Current marketValueMultiplier in DB |
| `mv_confidence_tier` | string\|null | Current marketValueConfidence in DB |
| `session_median_price` | float\|null | Current sessionMedianPrice in DB |

**DeltaValues Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `mv_expected_price` | float | Absolute change (new - existing) |
| `mv_expected_price_pct` | float | Percentage change |
| `mv_low_price` | float\|null | Absolute change for low price |
| `mv_high_price` | float\|null | Absolute change for high price |

**Use Cases:**

- Preview changes before committing to database
- Identify which lots are new vs. updated
- Review price deltas (positive = increase, negative = decrease)
- Filter lots by change magnitude before selective commit

---

### Commit Selected Lots

Commit selected lots to the database after scoring. This enables a two-step workflow:
1. Score sale → view all predictions
2. Select specific lots → commit only those to DB

```
POST /api/score/{sale_id}/commit
```

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `sale_id` | integer | The sale ID (must match `sales_id` in each lot) |

**Request Body:**

```json
{
  "lots": [
    {
      "horse_id": 789012,
      "sales_id": 2094,
      "mv_expected_price": 120000,
      "mv_low_price": 75000,
      "mv_high_price": 195000,
      "mv_expected_index": 1.41,
      "mv_confidence_tier": "medium",
      "session_median_price": 85000
    }
  ]
}
```

**Request Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `horse_id` | integer | Horse identifier |
| `sales_id` | integer | Sale identifier (must match path `sale_id`) |
| `mv_expected_price` | float | P50 predicted price |
| `mv_low_price` | float | P25 predicted price |
| `mv_high_price` | float | P75 predicted price |
| `mv_expected_index` | float | Price multiplier vs session median |
| `mv_confidence_tier` | string | `"high"`, `"medium"`, or `"low"` |
| `session_median_price` | float | Session median price |

**Response:**

```json
{
  "sale_id": 2094,
  "country_code": "AUS",
  "inserted": 5,
  "updated": 3,
  "total": 8
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `sale_id` | integer | The sale ID |
| `country_code` | string | Country code derived from sale |
| `inserted` | integer | Number of new records created |
| `updated` | integer | Number of existing records updated |
| `total` | integer | Total records processed (inserted + updated) |

**Errors:**

| Status | Cause |
|--------|-------|
| `400` | Empty lots array |
| `400` | `sales_id` in lot doesn't match path `sale_id` |
| `404` | Sale not found |

---

## Model Management

### Train Model

Start training a new model version for a country. Training runs in the background.

```
POST /api/train/{country}
```

**Path Parameters:**

| Parameter | Type | Values | Description |
|-----------|------|--------|-------------|
| `country` | string | `aus`, `nzl`, `usa` | Country code |

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `version` | string | auto | Force specific version (e.g., `v5`) |

**Response:**
```json
{
  "message": "Training started for AUS. Check GET /api/models/aus for completion.",
  "country": "AUS",
  "version": "v5",
  "output_dir": "models/aus_v5"
}
```

---

### List Models

List all model versions for a country with training metrics.

```
GET /api/models/{country}
```

**Path Parameters:**

| Parameter | Type | Values | Description |
|-----------|------|--------|-------------|
| `country` | string | `aus`, `nzl`, `usa` | Country code |

**Response:**
```json
{
  "country": "AUS",
  "active_model": "aus_v3",
  "models": [
    {
      "version": "v3",
      "directory": "aus_v3",
      "is_active": true,
      "training_info": {
        "generated_at": "2026-01-29 00:25:43",
        "model_version": "v3",
        "country": "AUS"
      },
      "data_summary": {
        "total_samples": 45000,
        "train_rows": 38000,
        "validation_rows": 4000,
        "test_rows": 3000,
        "features_count": 42
      },
      "baseline_model": {
        "name": "Elastic Net",
        "train_mae": 25000.5,
        "train_r2": 0.72,
        "test_mae": 28000.3,
        "test_r2": 0.68,
        "naive_mae": 45000.0,
        "passes": true
      },
      "quantile_models": {
        "q25_trees": 150,
        "q50_trees": 150,
        "q75_trees": 150
      },
      "evaluation": {
        "p50_mae": 26500.0,
        "p50_rmse": 42000.0,
        "p50_r2": 0.70,
        "raw_coverage_p25": 23.5,
        "raw_coverage_p75": 76.2,
        "mape": 28.5
      },
      "calibration": {
        "offset_p25": -0.075,
        "offset_p75": 0.048,
        "calibrated_coverage_p25": 25.3,
        "calibrated_coverage_p75": 75.3
      },
      "feature_importance": {
        "average": {
          "session_median_price": 0.35,
          "sire_avg_price": 0.18,
          "dam_produce_avg": 0.12
        },
        "q25": {...},
        "q50": {...},
        "q75": {...}
      }
    }
  ]
}
```

---

## Configuration Endpoints

### Get Active Models

Get active model for all countries.

```
GET /api/config/models
```

**Response:**
```json
{
  "models": {
    "aus": "aus_v3",
    "nzl": "nzl",
    "usa": "usa"
  }
}
```

---

### Set Active Model

Set the active model for a country.

```
PUT /api/config/models/{country}?model={model_name}
```

**Path Parameters:**

| Parameter | Type | Values | Description |
|-----------|------|--------|-------------|
| `country` | string | `aus`, `nzl`, `usa` | Country code |

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | string | Model directory name (e.g., `aus_v5`) |

**Response:**
```json
{
  "country": "AUS",
  "model": "aus_v5"
}
```

---

### Get Year Range

Get the year range used for training/scoring.

```
GET /api/config/years
```

**Response:**
```json
{
  "year_start": 2020,
  "year_end": 2026
}
```

---

### Set Year Range

Set the year range for training/scoring.

```
PUT /api/config/years?year_start={start}&year_end={end}
```

**Query Parameters:**

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `year_start` | integer | 2000-2100 | Start year (required) |
| `year_end` | integer\|null | 2000-2100 | End year (null = current year) |

**Response:**
```json
{
  "year_start": 2020,
  "year_end": 2026
}
```

---

### Get Test Years

Get number of years held out for model testing.

```
GET /api/config/test-years
```

**Response:**
```json
{
  "model_test_last_years": 2
}
```

---

### Set Test Years

Set number of years held out for model testing.

```
PUT /api/config/test-years?model_test_last_years={years}
```

**Query Parameters:**

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `model_test_last_years` | integer | 1-10 | Years to hold out |

**Response:**
```json
{
  "model_test_last_years": 2
}
```

---

### Get Historical Countries

Get historical country mappings (for cross-country data lookback).

```
GET /api/config/hist-countries
```

**Response:**
```json
{
  "hist_countries": {
    "NZL": ["NZL", "AUS"]
  }
}
```

---

### Get Historical Countries (Single)

Get historical countries for a specific country.

```
GET /api/config/hist-countries/{country}
```

**Response:**
```json
{
  "country": "NZL",
  "hist_countries": ["NZL", "AUS"]
}
```

---

### Set Historical Countries

Set historical countries for lookback.

```
PUT /api/config/hist-countries/{country}?hist_countries=NZL&hist_countries=AUS
```

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `hist_countries` | array | List of country codes (repeatable) |

**Response:**
```json
{
  "country": "NZL",
  "hist_countries": ["NZL", "AUS"]
}
```

---

### Delete Historical Countries

Remove historical country override.

```
DELETE /api/config/hist-countries/{country}
```

**Response:**
```json
{
  "message": "Removed hist_countries override for NZL"
}
```

---

## Response Types

### Price Range

Used for gross totals and median prices.

```typescript
interface PriceRange {
  low: number;      // P25 estimate
  expected: number; // P50 estimate
  high: number;     // P75 estimate
}
```

### Confidence Tier Counts

```typescript
interface ConfidenceTierCounts {
  high: number;   // High confidence predictions
  medium: number; // Medium confidence predictions
  low: number;    // Low confidence predictions
}
```

### Score Summary

```typescript
interface ScoreSummary {
  gross: PriceRange;
  median_prices: PriceRange;
  confidence_tiers: ConfidenceTierCounts;
  elite_scaling_count: number;   // Lots >= $300k threshold
  elite_scaling_percent: number; // Percentage of elite lots
}
```

### Lot Score

```typescript
interface LotScore {
  lot_id: number;
  horse_id: number | null;
  sales_id: number;
  lot_number: number | null;
  horse_name: string | null;
  sire_name: string | null;
  session_median_price: number;
  mv_expected_price: number;
  mv_low_price: number;
  mv_high_price: number;
  mv_expected_index: number;
  mv_confidence_tier: "high" | "medium" | "low";
}
```

### Lot Commit (Request)

```typescript
interface LotCommit {
  horse_id: number;
  sales_id: number;
  mv_expected_price: number;
  mv_low_price: number;
  mv_high_price: number;
  mv_expected_index: number;
  mv_confidence_tier: string;
  session_median_price: number;
}

interface CommitLotsRequest {
  lots: LotCommit[];
}
```

### Commit Response

```typescript
interface CommitLotsResponse {
  sale_id: number;
  country_code: string;
  inserted: number;
  updated: number;
  total: number;
}
```

### Existing Values

```typescript
interface ExistingValues {
  mv_expected_price: number | null;
  mv_low_price: number | null;
  mv_high_price: number | null;
  mv_expected_index: number | null;
  mv_confidence_tier: string | null;
  session_median_price: number | null;
}
```

### Delta Values

```typescript
interface DeltaValues {
  mv_expected_price: number | null;
  mv_expected_price_pct: number | null;
  mv_low_price: number | null;
  mv_high_price: number | null;
}
```

### Lot Score Comparison

```typescript
interface LotScoreComparison {
  lot_id: number;
  horse_id: number | null;
  sales_id: number;
  lot_number: number | null;
  horse_name: string | null;
  sire_name: string | null;
  session_median_price: number;
  new: LotScore;
  existing: ExistingValues | null;
  delta: DeltaValues | null;
  is_new: boolean;
}
```

### Compare Response

```typescript
interface CompareResponse {
  sale_id: number;
  country_code: string;
  model_dir: string;
  total_lots: number;
  new_lots: number;
  existing_lots: number;
  changed_lots: number;
  unchanged_lots: number;
  avg_price_delta: number;
  avg_price_delta_pct: number;
  lots: LotScoreComparison[];
}
```

---

## Error Handling

### Error Response Format

```json
{
  "detail": "Error message describing what went wrong"
}
```

### HTTP Status Codes

| Code | Description |
|------|-------------|
| `200` | Success |
| `400` | Bad request (invalid parameters) |
| `403` | Authentication failed (invalid API key) |
| `404` | Resource not found (sale, model, etc.) |
| `500` | Internal server error |

### Common Errors

| Error | Status | Cause |
|-------|--------|-------|
| Invalid API key | 403 | Missing or incorrect `X-API-Key` header |
| Sale not found | 404 | `sale_id` doesn't exist in database |
| No lots found | 404 | Sale exists but has no lots |
| Invalid country | 400 | Country not in `aus`, `nzl`, `usa` |
| Model not found | 404 | Specified model directory doesn't exist |

---

## Integration Examples

### JavaScript/TypeScript (fetch)

```typescript
const API_KEY = 'your-api-key';
const BASE_URL = 'http://localhost:8000';

async function scoreSale(saleId: number): Promise<ScoreResponse> {
  const response = await fetch(`${BASE_URL}/api/score/${saleId}`, {
    method: 'POST',
    headers: {
      'X-API-Key': API_KEY,
      'Content-Type': 'application/json'
    }
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail);
  }

  return response.json();
}

// Usage
const result = await scoreSale(2094);
console.log(`Total lots: ${result.total_lots}`);
console.log(`Expected gross: $${result.summary.gross.expected.toLocaleString()}`);
```

### JavaScript/TypeScript (axios)

```typescript
import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:8000',
  headers: {
    'X-API-Key': 'your-api-key'
  }
});

// Score a sale
const { data } = await api.post('/api/score/2094');
console.log(data.summary);

// Two-step workflow: score then commit selected lots
const scoreResult = await api.post('/api/score/2094');
const selectedLots = scoreResult.data.lots.filter(lot => lot.mv_confidence_tier !== 'low');
const commitResult = await api.post('/api/score/2094/commit', {
  lots: selectedLots.map(lot => ({
    horse_id: lot.horse_id,
    sales_id: lot.sales_id,
    mv_expected_price: lot.mv_expected_price,
    mv_low_price: lot.mv_low_price,
    mv_high_price: lot.mv_high_price,
    mv_expected_index: lot.mv_expected_index,
    mv_confidence_tier: lot.mv_confidence_tier,
    session_median_price: lot.session_median_price
  }))
});
console.log(`Committed: ${commitResult.data.inserted} new, ${commitResult.data.updated} updated`);

// Get active models
const { data: models } = await api.get('/api/config/models');
console.log(models);

// Set active model
await api.put('/api/config/models/aus', null, {
  params: { model: 'aus_v5' }
});
```

### React Hook Example

```typescript
import { useState, useEffect } from 'react';

interface UseScoreSaleResult {
  data: ScoreResponse | null;
  loading: boolean;
  error: string | null;
  refetch: () => void;
}

function useScoreSale(saleId: number): UseScoreSaleResult {
  const [data, setData] = useState<ScoreResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`/api/score/${saleId}`, {
        method: 'POST',
        headers: { 'X-API-Key': process.env.REACT_APP_API_KEY! }
      });
      if (!response.ok) throw new Error((await response.json()).detail);
      setData(await response.json());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchData(); }, [saleId]);

  return { data, loading, error, refetch: fetchData };
}

// Usage in component
function SaleScores({ saleId }: { saleId: number }) {
  const { data, loading, error } = useScoreSale(saleId);

  if (loading) return <Spinner />;
  if (error) return <Error message={error} />;

  return (
    <div>
      <h2>Sale {data.sale_id} - {data.total_lots} lots</h2>
      <p>Expected Gross: ${data.summary.gross.expected.toLocaleString()}</p>
      <LotTable lots={data.lots} />
    </div>
  );
}
```

### cURL Examples

```bash
# Set your API key
export API_KEY="your-api-key"

# Health check (no auth required)
curl http://localhost:8000/health

# Score a sale
curl -X POST "http://localhost:8000/api/score/2094" \
  -H "X-API-Key: $API_KEY"

# Score and save to CSV
curl -X POST "http://localhost:8000/api/score/2094?output=csv" \
  -H "X-API-Key: $API_KEY"

# Score and write to database
curl -X POST "http://localhost:8000/api/score/2094?output=db" \
  -H "X-API-Key: $API_KEY"

# Compare with existing DB values
curl -X POST "http://localhost:8000/api/score/2101/compare" \
  -H "X-API-Key: $API_KEY"

# Compare - get summary only
curl -X POST "http://localhost:8000/api/score/2101/compare" \
  -H "X-API-Key: $API_KEY" | jq '{
    total: .total_lots,
    new: .new_lots,
    existing: .existing_lots,
    changed: .changed_lots,
    avg_delta_pct: .avg_price_delta_pct
  }'

# Compare - get changed lots only
curl -X POST "http://localhost:8000/api/score/2101/compare" \
  -H "X-API-Key: $API_KEY" | jq '[.lots[] | select(.delta.mv_expected_price != 0)]'

# Commit selected lots to database
curl -X POST "http://localhost:8000/api/score/2094/commit" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "lots": [
      {
        "horse_id": 789012,
        "sales_id": 2094,
        "mv_expected_price": 120000,
        "mv_low_price": 75000,
        "mv_high_price": 195000,
        "mv_expected_index": 1.41,
        "mv_confidence_tier": "medium",
        "session_median_price": 85000
      }
    ]
  }'

# Get summary only (using jq)
curl -X POST "http://localhost:8000/api/score/2094" \
  -H "X-API-Key: $API_KEY" | jq '.summary'

# List models for Australia
curl "http://localhost:8000/api/models/aus" \
  -H "X-API-Key: $API_KEY"

# Get active models
curl "http://localhost:8000/api/config/models" \
  -H "X-API-Key: $API_KEY"

# Set active model
curl -X PUT "http://localhost:8000/api/config/models/aus?model=aus_v5" \
  -H "X-API-Key: $API_KEY"

# Train new model
curl -X POST "http://localhost:8000/api/train/aus" \
  -H "X-API-Key: $API_KEY"

# Get year config
curl "http://localhost:8000/api/config/years" \
  -H "X-API-Key: $API_KEY"

# Set year range
curl -X PUT "http://localhost:8000/api/config/years?year_start=2020&year_end=2026" \
  -H "X-API-Key: $API_KEY"
```

---

## OpenAPI / Swagger

The API provides auto-generated documentation at:

- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`
- **OpenAPI JSON:** `http://localhost:8000/openapi.json`

---

## Country Codes

| Code | Country |
|------|---------|
| `AUS` | Australia |
| `NZL` | New Zealand |
| `USA` | United States |

---

## Confidence Tiers

Price predictions are assigned confidence tiers based on data quality:

| Tier | Description |
|------|-------------|
| `high` | Strong historical data, reliable prediction |
| `medium` | Moderate data, reasonable prediction |
| `low` | Limited data, use with caution |

---

## Elite Scaling

Lots with expected prices >= $300,000 receive special "elite scaling" adjustments to account for the non-linear nature of high-value yearling sales.

The summary includes:
- `elite_scaling_count` - Number of lots above threshold
- `elite_scaling_percent` - Percentage of catalogue above threshold

---

## Rate Limits

Currently no rate limits are enforced. This may change in production deployments.

---

## Support

For issues or questions:
- Check the health endpoint first: `GET /health`
- Verify your API key is correct
- Check the API logs for detailed error messages
