# Market Value API Documentation

API documentation for the Market Value scoring system. This document covers all endpoints, authentication, request/response formats, and integration examples.

**Base URL:** `http://localhost:8000` (development) or `https://smmarketvalue.stallionmatch.horse` (production)
**API Version:** 2.13.0

---

## Table of Contents

1. [Authentication](#authentication)
2. [Endpoints Overview](#endpoints-overview)
3. [Auth Endpoints](#auth-endpoints)
4. [Scoring Endpoints](#scoring-endpoints)
5. [Model Management](#model-management)
6. [Configuration Endpoints](#configuration-endpoints)
7. [Response Types](#response-types)
8. [Error Handling](#error-handling)
9. [Integration Examples](#integration-examples)

---

## Authentication

The API supports two authentication methods:

### 1. API Key Authentication

Use the `X-API-Key` header for server-to-server integration:

```
X-API-Key: <your-api-key>
```

### 2. JWT Token Authentication (OTP)

For user-based authentication, use the email OTP flow:

1. Request OTP: `POST /auth/request-otp` with your whitelisted email
2. Verify OTP: `POST /auth/verify-otp` with email and code
3. Use the returned JWT token in the `Authorization` header:

```
Authorization: Bearer <jwt-token>
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

# Email-based OTP authentication (optional)
EMAIL_SERVICE_URL=https://your-email-service.com/send
EMAIL_SERVICE_API_KEY=your_email_service_api_key
AUTH_EMAIL_WHITELIST=admin@company.com,analyst@company.com
JWT_SECRET=your_random_32_char_secret_key_here
JWT_EXPIRY_HOURS=24

# Dev mode (skip email, log OTP to console)
AUTH_DEV_MODE=false
```

### Authentication Errors

| Status | Description |
|--------|-------------|
| `403` | Invalid or missing API key / JWT token |
| `401` | OTP expired, not found, or invalid |

---

## Endpoints Overview

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/health` | No | Health check |
| `POST` | `/auth/request-otp` | No | Request OTP for email login |
| `POST` | `/auth/verify-otp` | No | Verify OTP and get JWT token |
| `GET` | `/api/sales/search` | Yes | Search sales by name or company |
| `GET` | `/api/sales/{sale_id}` | Yes | Get detailed sale information |
| `POST` | `/api/score/{sale_id}` | Yes | Score all lots for a sale |
| `POST` | `/api/score/{sale_id}/compare` | Yes | Score and compare with existing DB values |
| `POST` | `/api/score/{sale_id}/commit` | Yes | Commit selected lots to database |
| `POST` | `/api/train/{country}` | Yes | Train new model (background) |
| `GET` | `/api/train/status` | Yes | Get training job status |
| `GET` | `/api/models/{country}` | Yes | List all models for country |
| `GET` | `/api/models/{model_name}/download` | Yes | Download model as ZIP |
| `POST` | `/api/models/{model_name}` | Yes | Upload new model from ZIP |
| `DELETE` | `/api/models/{model_name}` | Yes | Delete a model |
| `GET` | `/api/config` | Yes | Get full configuration |
| `GET` | `/api/config/years` | Yes | Get year range |
| `PUT` | `/api/config/years` | Yes | Set year range |
| `GET` | `/api/config/test-years` | Yes | Get test years config |
| `PUT` | `/api/config/test-years` | Yes | Set test years config |
| `GET` | `/api/config/sale-history-years` | Yes | Get sale history years config |
| `PUT` | `/api/config/sale-history-years` | Yes | Set sale history years config |
| `GET` | `/api/config/{country}` | Yes | Get region config |
| `POST` | `/api/config/{country}` | Yes | Partial update region config |
| `PUT` | `/api/config/{country}` | Yes | Create/replace region config |
| `DELETE` | `/api/config/{country}` | Yes | Remove region |

---

## Auth Endpoints

### Request OTP

Request a one-time password for email-based authentication. Only whitelisted emails can request OTPs.

```
POST /auth/request-otp
Content-Type: application/json
```

**Request Body:**
```json
{
  "email": "user@company.com"
}
```

**Response:**
```json
{
  "message": "OTP sent to user@company.com"
}
```

**Errors:**

| Status | Description |
|--------|-------------|
| `403` | Email not in whitelist |
| `500` | Email service not configured or failed |

---

### Verify OTP

Verify the OTP code and receive a JWT token for API access.

```
POST /auth/verify-otp
Content-Type: application/json
```

**Request Body:**
```json
{
  "email": "user@company.com",
  "code": "123456"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 86400
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `access_token` | string | JWT token for API authentication |
| `token_type` | string | Always `"bearer"` |
| `expires_in` | integer | Token validity in seconds (default: 86400 = 24 hours) |

**Errors:**

| Status | Description |
|--------|-------------|
| `401` | OTP expired, not found, or invalid code |

**OTP Notes:**
- OTPs are valid for 10 minutes
- Each OTP can only be used once
- OTPs are 6 digits (e.g., `"123456"`)

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

### Search Sales

Search for sales by name or company name. Useful for typeahead/autocomplete in the frontend.

```
GET /api/sales/search
```

**Query Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `q` | string | Yes | - | Search query (min 3 characters) |
| `limit` | integer | No | 20 | Max results to return (1-100) |

**Response:**
```json
{
  "query": "Magic",
  "results": [
    {
      "sale_id": 2094,
      "sale_name": "Gold Coast Yearling Sale",
      "sale_date": "2026-01-13",
      "country_code": "AUS",
      "currency_code": "AUD",
      "lot_count": 1221,
      "sale_company": "Magic Millions",
      "status": "past"
    },
    {
      "sale_id": 2101,
      "sale_name": "Gold Coast March Yearling Sale",
      "sale_date": "2026-03-12",
      "country_code": "AUS",
      "currency_code": "AUD",
      "lot_count": 490,
      "sale_company": "Magic Millions",
      "status": "upcoming"
    }
  ]
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `query` | string | The search query |
| `results` | array | Array of matching sales |
| `results[].sale_id` | integer | Sale identifier |
| `results[].sale_name` | string | Full sale name |
| `results[].sale_date` | string\|null | Sale start date (YYYY-MM-DD) |
| `results[].country_code` | string | Country code (e.g., `AUS`, `NZL`) |
| `results[].currency_code` | string\|null | Currency code (e.g., `AUD`, `NZD`, `USD`) |
| `results[].lot_count` | integer | Number of lots in the sale |
| `results[].sale_company` | string | Sale company name |
| `results[].status` | string | `"upcoming"` if sale_date >= today, otherwise `"past"` |

**Notes:**
- Searches both sale name and company name (OR logic)
- Results are ordered by sale date descending (most recent first)
- Minimum query length is 3 characters

---

### Get Sale Detail

Get detailed information about a sale including lot statistics, enhanced price statistics, prior year comparison, and book breakdown.

```
GET /api/sales/{sale_id}
```

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `sale_id` | integer | The sale ID |

**Response:**
```json
{
  "sale_id": 789,
  "sale_code": "INPY25",
  "sale_name": "Premier Yearling Sale",
  "start_date": "2025-03-02",
  "end_date": "2025-03-04",
  "sale_type": "Yearling",
  "sale_status": "Verified",
  "is_online": false,
  "is_public": false,
  "sale_company": "Inglis",
  "company_website": "https://inglis.com.au",
  "country_code": "AUS",
  "country_name": "Australia",
  "currency_code": "AUD",
  "currency_symbol": "$",
  "status": "past",
  "lot_stats": {
    "total_lots": 803,
    "sold_count": 551,
    "passed_in_count": 139,
    "withdrawn_count": 113,
    "clearance_rate": 79.9
  },
  "price_stats": {
    "gross": 60584137.0,
    "avg_price": 109953.06,
    "median_price": 80000.0,
    "q1_price": 40000.0,
    "q3_price": 150000.0,
    "min_price": 3500.0,
    "max_price": 1000000.0,
    "top10_avg": 563000.0,
    "std_dev": 109436.27,
    "price_bands": {
      "under_50k": 164,
      "band_50k_100k": 155,
      "band_100k_200k": 136,
      "band_200k_500k": 92,
      "band_500k_1m": 2,
      "over_1m": 2
    }
  },
  "prior_year": {
    "sale_ids": [636],
    "sale_names": ["Premier Yearling Sale"],
    "start_date": "2024-03-03",
    "end_date": "2024-03-05",
    "lot_stats": {
      "total_lots": 800,
      "sold_count": 559,
      "passed_in_count": 0,
      "withdrawn_count": 0,
      "clearance_rate": 69.9
    },
    "price_stats": {
      "gross": 58373500.0,
      "avg_price": 104424.87,
      "median_price": 75000.0,
      "q1_price": 40000.0,
      "q3_price": 140000.0,
      "min_price": 5000.0,
      "max_price": 925000.0,
      "top10_avg": 571500.0,
      "std_dev": 105949.28,
      "price_bands": {
        "under_50k": 186,
        "band_50k_100k": 159,
        "band_100k_200k": 134,
        "band_200k_500k": 72,
        "band_500k_1m": 8,
        "over_1m": 0
      }
    },
    "yoy_change": {
      "gross_pct": 3.8,
      "avg_price_pct": 5.3,
      "median_price_pct": 6.7,
      "sold_count_change": -8,
      "clearance_rate_change": 10.0
    }
  },
  "history": [
    {
      "year": 2024,
      "sale_ids": [636],
      "start_date": "2024-01-07",
      "end_date": "2024-01-10",
      "total_lots": 800,
      "sold_count": 559,
      "passed_in_count": 185,
      "withdrawn_count": 56,
      "clearance_rate": 69.9,
      "gross": 58373500.0,
      "avg_price": 104424.87,
      "median_price": 75000.0,
      "q1_price": 40000.0,
      "q3_price": 135000.0,
      "top10_avg": 650000.0
    },
    {
      "year": 2023,
      "sale_ids": [510],
      "start_date": "2023-01-08",
      "end_date": "2023-01-11",
      "total_lots": 780,
      "sold_count": 540,
      "passed_in_count": 170,
      "withdrawn_count": 70,
      "clearance_rate": 72.5,
      "gross": 52000000.0,
      "avg_price": 96296.30,
      "median_price": 70000.0,
      "q1_price": 38000.0,
      "q3_price": 125000.0,
      "top10_avg": 580000.0
    }
  ],
  "queue_stats": {
    "completed": 723,
    "in_queue": 0,
    "failed": 0,
    "postponed": 0,
    "last_completed": "2025-12-02 03:29:12.743333"
  },
  "books": [
    {"book_number": 1, "day_number": 0, "lot_count": 87},
    {"book_number": 1, "day_number": 1, "lot_count": 196},
    {"book_number": 1, "day_number": 2, "lot_count": 282},
    {"book_number": 2, "day_number": 3, "lot_count": 238}
  ]
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `sale_id` | integer | Sale identifier |
| `sale_code` | string\|null | Sale code (e.g., "INPY25") |
| `sale_name` | string | Full sale name |
| `start_date` | string\|null | Sale start date (YYYY-MM-DD) |
| `end_date` | string\|null | Sale end date (YYYY-MM-DD) |
| `sale_type` | string\|null | Sale type (e.g., "Yearling", "Broodmare") |
| `sale_status` | string\|null | Sale status (e.g., "Verified") |
| `is_online` | boolean | Whether this is an online sale |
| `is_public` | boolean | Whether the sale is public |
| `sale_company` | string | Sale company name |
| `company_website` | string\|null | Sale company website URL |
| `country_code` | string | Country code (e.g., "AUS") |
| `country_name` | string | Full country name |
| `currency_code` | string\|null | Currency code (e.g., "AUD") |
| `currency_symbol` | string\|null | Currency symbol (e.g., "$") |
| `status` | string | `"upcoming"` if start_date >= today, otherwise `"past"` |

**Lot Stats Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `total_lots` | integer | Total catalogue lots |
| `sold_count` | integer | Number of lots sold |
| `passed_in_count` | integer | Number of lots passed in |
| `withdrawn_count` | integer | Number of lots withdrawn |
| `clearance_rate` | float\|null | Clearance rate: sold / (total - withdrawn) × 100 |

**Price Stats Fields (null for upcoming sales):**

| Field | Type | Description |
|-------|------|-------------|
| `gross` | float\|null | Total gross revenue |
| `avg_price` | float\|null | Average sale price |
| `median_price` | float\|null | Median sale price (P50) |
| `q1_price` | float\|null | 25th percentile price |
| `q3_price` | float\|null | 75th percentile price |
| `min_price` | float\|null | Lowest sale price |
| `max_price` | float\|null | Highest sale price (top lot) |
| `top10_avg` | float\|null | Average of top 10 prices |
| `std_dev` | float\|null | Standard deviation of prices |
| `price_bands` | object\|null | Distribution by price range |

**Price Bands Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `under_50k` | integer | Lots sold under $50,000 |
| `band_50k_100k` | integer | Lots sold $50,000-$99,999 |
| `band_100k_200k` | integer | Lots sold $100,000-$199,999 |
| `band_200k_500k` | integer | Lots sold $200,000-$499,999 |
| `band_500k_1m` | integer | Lots sold $500,000-$999,999 |
| `over_1m` | integer | Lots sold $1,000,000+ |

**Prior Year Fields (null if no matching prior year sale):**

Prior year is matched by: same company + same sale type + same month of the year.

| Field | Type | Description |
|-------|------|-------------|
| `sale_ids` | array | IDs of matching prior year sales |
| `sale_names` | array | Names of matching prior year sales |
| `start_date` | string\|null | Earliest start date (if multiple sales) |
| `end_date` | string\|null | Latest end date (if multiple sales) |
| `lot_stats` | object | Aggregated lot statistics |
| `price_stats` | object\|null | Aggregated price statistics |
| `yoy_change` | object\|null | Year-over-year changes |

**YoY Change Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `gross_pct` | float\|null | Gross revenue change (%) |
| `avg_price_pct` | float\|null | Average price change (%) |
| `median_price_pct` | float\|null | Median price change (%) |
| `sold_count_change` | integer\|null | Sold count difference (absolute) |
| `clearance_rate_change` | float\|null | Clearance rate difference (percentage points) |

**History Fields (configurable via `sale_history_years` in config):**

| Field | Type | Description |
|-------|------|-------------|
| `year` | integer | Sale year |
| `sale_ids` | array | IDs of matching sales in that year |
| `start_date` | string\|null | First sale start date (YYYY-MM-DD) |
| `end_date` | string\|null | Last sale end date (YYYY-MM-DD) |
| `total_lots` | integer | Total lots in sale |
| `sold_count` | integer | Number of lots sold |
| `passed_in_count` | integer | Number of lots passed in (unsold) |
| `withdrawn_count` | integer | Number of lots withdrawn |
| `clearance_rate` | float\|null | Clearance rate percentage |
| `gross` | float\|null | Total gross revenue |
| `avg_price` | float\|null | Average sale price |
| `median_price` | float\|null | Median sale price (P50) |
| `q1_price` | float\|null | 25th percentile price |
| `q3_price` | float\|null | 75th percentile price |
| `top10_avg` | float\|null | Average price of top 10 lots |

**Queue Stats Fields (report generation status):**

| Field | Type | Description |
|-------|------|-------------|
| `completed` | integer | Lot reports generated |
| `in_queue` | integer | Lots waiting to process |
| `failed` | integer | Failed to generate |
| `postponed` | integer | Deferred processing |
| `last_completed` | string\|null | Timestamp of last report generated |

**Book Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `book_number` | integer | Book/catalogue number |
| `day_number` | integer\|null | Day of sale |
| `lot_count` | integer | Number of lots in this session |

**Notes:**
- `price_stats` is `null` for upcoming sales with no sold lots
- `prior_year` is `null` if no matching prior year sale found
- `yoy_change` is `null` if either current or prior year has no price stats
- `history` is empty if `sale_history_years` is 0 or no matching historical sales found
- `clearance_rate` is calculated as: sold ÷ (total - withdrawn) × 100
- Positive `yoy_change` values indicate increase vs prior year

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
| `session_median` | float | `null` | Manual session median price override. For future sales, defaults to prior year's median if not specified. |

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

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `session_median` | float | `null` | Manual session median price override. For future sales, defaults to prior year's median if not specified. |

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
  "message": "Training started for AUS. Check GET /api/train/status for progress.",
  "country": "AUS",
  "version": "v5",
  "output_dir": "models/aus_v5"
}
```

**Errors:**

| Status | Cause |
|--------|-------|
| `400` | Invalid country code |
| `409` | Training already in progress |

---

### Get Training Status

Get the current or most recent training job status.

```
GET /api/train/status
```

**Response (idle):**
```json
{
  "active": false,
  "country": null,
  "version": null,
  "phase": null,
  "started_at": null,
  "completed_at": null,
  "status": "idle",
  "error": null
}
```

**Response (in progress):**
```json
{
  "active": true,
  "country": "AUS",
  "version": "v5",
  "phase": "training_models",
  "started_at": "2026-02-10T03:15:00+00:00",
  "completed_at": null,
  "status": "training",
  "error": null
}
```

**Response (completed):**
```json
{
  "active": false,
  "country": "AUS",
  "version": "v5",
  "phase": "done",
  "started_at": "2026-02-10T03:15:00+00:00",
  "completed_at": "2026-02-10T03:25:00+00:00",
  "status": "completed",
  "error": null
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `active` | boolean | Whether training is currently running |
| `country` | string\|null | Country being trained |
| `version` | string\|null | Model version being trained |
| `phase` | string\|null | Current phase: `starting`, `exporting_data`, `preparing_features`, `splitting_data`, `training_models`, `evaluating`, `saving_artifacts`, `done` |
| `started_at` | string\|null | ISO 8601 timestamp when training started |
| `completed_at` | string\|null | ISO 8601 timestamp when training finished |
| `status` | string | Status: `idle`, `training`, `completed`, `failed` |
| `error` | string\|null | Error message if status is `failed` |

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

### Download Model

Download a model as a ZIP file containing all model files.

```
GET /api/models/{model_name}/download
```

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_name` | string | Model directory name (e.g., `aus_v8`) |

**Response:**
- Content-Type: `application/zip`
- Content-Disposition: `attachment; filename={model_name}.zip`

**ZIP Contents:**
- `calibration_offsets.json`
- `feature_cols.json`
- `mv_v1_q25.txt`, `mv_v1_q50.txt`, `mv_v1_q75.txt`
- `feature_importance_*.json` (if present)
- `training_report.txt` (if present)

---

### Upload Model

Upload a new model from a ZIP file.

```
POST /api/models/{model_name}
Content-Type: multipart/form-data
```

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_name` | string | New model name (lowercase alphanumeric + underscores) |

**Form Data:**

| Field | Type | Description |
|-------|------|-------------|
| `file` | file | ZIP file containing model files |

**Required Files in ZIP:**
- `calibration_offsets.json`
- `feature_cols.json`
- `mv_v1_q25.txt`, `mv_v1_q50.txt`, `mv_v1_q75.txt`

**Response:**
```json
{
  "model_name": "aus_v9",
  "files": [
    "calibration_offsets.json",
    "feature_cols.json",
    "mv_v1_q25.txt",
    "mv_v1_q50.txt",
    "mv_v1_q75.txt",
    "training_report.txt"
  ],
  "size_bytes": 3082226,
  "message": "Model 'aus_v9' uploaded successfully"
}
```

**Errors:**
- `400`: Invalid model name, invalid ZIP, or missing required files
- `409`: Model already exists

---

### Delete Model

Delete a model from the server.

```
DELETE /api/models/{model_name}
```

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_name` | string | Model directory name |

**Response:**
```json
{
  "model_name": "aus_v9",
  "message": "Model 'aus_v9' deleted successfully"
}
```

**Errors:**
- `404`: Model not found
- `409`: Cannot delete - model is active for one or more regions

---

## Configuration Endpoints

### Get Full Configuration

Get full configuration including all regions.

```
GET /api/config
```

**Response:**
```json
{
  "year_start": 2020,
  "year_end": null,
  "model_test_last_years": 2,
  "sale_history_years": 5,
  "audit_user_id": 2,
  "regions": {
    "AUS": {
      "model": "aus",
      "currency_id": 1,
      "hist_countries": ["AUS"],
      "elite_scaling": {
        "threshold": 500000,
        "base_offset": 0.25,
        "scaling_factor": 0.5
      },
      "confidence_tiers": {
        "close_threshold": 0.7,
        "extreme_threshold": 1.0
      },
      "sire_sample_min_count": 10
    },
    "NZL": {...},
    "USA": {...},
    "GBR": {...},
    "IRE": {...},
    "FRA": {...},
    "GER": {...},
    "ZAF": {...},
    "JPN": {...},
    "CAN": {...},
    "HKG": {...}
  }
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

### Get Sale History Years

Get number of years of history to include in sale detail.

```
GET /api/config/sale-history-years
```

**Response:**
```json
{
  "sale_history_years": 5
}
```

---

### Set Sale History Years

Set number of years of history to include in sale detail.

```
PUT /api/config/sale-history-years?sale_history_years={years}
```

**Query Parameters:**

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `sale_history_years` | integer | 0-20 | Years of history (0 to disable) |

**Response:**
```json
{
  "sale_history_years": 5
}
```

---

### Get Region Config

Get configuration for a specific region.

```
GET /api/config/{country}
```

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `country` | string | Country code (e.g., `AUS`, `NZL`, `USA`) |

**Response:**
```json
{
  "model": "aus",
  "currency_id": 1,
  "hist_countries": ["AUS"],
  "elite_scaling": {
    "threshold": 500000,
    "base_offset": 0.25,
    "scaling_factor": 0.5
  },
  "confidence_tiers": {
    "close_threshold": 0.7,
    "extreme_threshold": 1.0
  },
  "sire_sample_min_count": 10
}
```

---

### Update Region Config (Partial)

Partial or full update for an existing region. Supports nested partial updates.

```
POST /api/config/{country}
Content-Type: application/json
```

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `country` | string | Country code (e.g., `AUS`, `NZL`, `USA`) |

**Request Body Examples:**

```json
// Update just the model
{"model": "aus_v5"}

// Update just elite scaling threshold
{"elite_scaling": {"threshold": 600000}}

// Update multiple fields
{
  "model": "aus_v5",
  "confidence_tiers": {
    "close_threshold": 0.6,
    "extreme_threshold": 0.9
  }
}
```

**Response:** Returns the updated region config.

---

### Create Region Config

Add a new region (full config required). PUT creates if not exists.

```
PUT /api/config/{country}
Content-Type: application/json
```

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `country` | string | Country code (e.g., `GBR`) |

**Request Body:**
```json
{
  "model": "gbr",
  "currency_id": 2,
  "hist_countries": ["GBR"],
  "elite_scaling": {
    "threshold": 400000,
    "base_offset": 0.25,
    "scaling_factor": 0.5
  },
  "confidence_tiers": {
    "close_threshold": 0.7,
    "extreme_threshold": 1.0
  },
  "sire_sample_min_count": 10
}
```

**Response:** Returns the created region config.

---

### Delete Region Config

Remove a region from configuration.

```
DELETE /api/config/{country}
```

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `country` | string | Country code to remove |

**Response:**
```json
{
  "message": "Region GBR removed"
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

### Elite Scaling Config

```typescript
interface EliteScalingConfig {
  threshold: number;
  base_offset: number;
  scaling_factor: number;
}
```

### Confidence Tiers Config

```typescript
interface ConfidenceTiersConfig {
  close_threshold: number;
  extreme_threshold: number;
}
```

### Region Config

```typescript
interface RegionConfig {
  model: string;
  currency_id: number;
  hist_countries: string[];
  elite_scaling: EliteScalingConfig;
  confidence_tiers: ConfidenceTiersConfig;
  sire_sample_min_count: number;
}
```

### Full Config Response

```typescript
interface FullConfigResponse {
  year_start: number;
  year_end: number | null;
  model_test_last_years: number;
  audit_user_id: number;
  regions: Record<string, RegionConfig>;
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
| `409` | Conflict (training already in progress, model already exists) |
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

// Get full config
const { data: config } = await api.get('/api/config');
console.log(config);

// Update region config (partial)
await api.post('/api/config/AUS', { model: 'aus_v5' });
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

# ==========================================
# OTP Authentication Flow
# ==========================================

# 1. Request OTP (dev mode - check console for code)
curl -X POST "http://localhost:8000/auth/request-otp" \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com"}'

# 2. Verify OTP and get JWT token
curl -X POST "http://localhost:8000/auth/verify-otp" \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com", "code": "123456"}'

# 3. Use JWT token for API requests
export JWT_TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
curl "http://localhost:8000/api/config" \
  -H "Authorization: Bearer $JWT_TOKEN"

# ==========================================
# API Key Authentication
# ==========================================

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

# Train new model
curl -X POST "http://localhost:8000/api/train/aus" \
  -H "X-API-Key: $API_KEY"

# Get full config
curl "http://localhost:8000/api/config" \
  -H "X-API-Key: $API_KEY"

# Get year config
curl "http://localhost:8000/api/config/years" \
  -H "X-API-Key: $API_KEY"

# Set year range
curl -X PUT "http://localhost:8000/api/config/years?year_start=2020&year_end=2026" \
  -H "X-API-Key: $API_KEY"

# Get region config for AUS
curl "http://localhost:8000/api/config/AUS" \
  -H "X-API-Key: $API_KEY"

# Partial update - just change model
curl -X POST "http://localhost:8000/api/config/AUS" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "aus_v5"}'

# Partial update - nested field
curl -X POST "http://localhost:8000/api/config/AUS" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"elite_scaling": {"threshold": 600000}}'

# Add new region (full config required)
curl -X PUT "http://localhost:8000/api/config/GBR" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gbr",
    "currency_id": 2,
    "hist_countries": ["GBR"],
    "elite_scaling": {"threshold": 400000, "base_offset": 0.25, "scaling_factor": 0.5},
    "confidence_tiers": {"close_threshold": 0.7, "extreme_threshold": 1.0},
    "sire_sample_min_count": 10
  }'

# Remove region
curl -X DELETE "http://localhost:8000/api/config/GBR" \
  -H "X-API-Key: $API_KEY"
```

---

## OpenAPI / Swagger

The API provides auto-generated documentation at:

**Development:**
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`
- **OpenAPI JSON:** `http://localhost:8000/openapi.json`

**Production:**
- **Swagger UI:** `https://smmarketvalue.stallionmatch.horse/docs`
- **ReDoc:** `https://smmarketvalue.stallionmatch.horse/redoc`
- **OpenAPI JSON:** `https://smmarketvalue.stallionmatch.horse/openapi.json`

---

## Country Codes

| Code | Country | Currency |
|------|---------|----------|
| `AUS` | Australia | AUD |
| `NZL` | New Zealand | NZD |
| `USA` | United States | USD |
| `GBR` | Great Britain | GBP |
| `IRE` | Ireland | EUR |
| `FRA` | France | EUR |
| `GER` | Germany | EUR |
| `ZAF` | South Africa | ZAR |
| `JPN` | Japan | JPY |
| `CAN` | Canada | CAD |
| `HKG` | Hong Kong | HKD |

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

Lots with expected prices above the region's elite threshold receive special "elite scaling" adjustments to account for the non-linear nature of high-value yearling sales.

**Thresholds by Region:**
| Region | Currency | Threshold |
|--------|----------|-----------|
| AUS | AUD | 500,000 |
| NZL | NZD | 100,000 |
| USA | USD | 500,000 |
| GBR | GBP | 300,000 |
| IRE | EUR | 300,000 |
| FRA | EUR | 200,000 |
| GER | EUR | 150,000 |
| ZAF | ZAR | 2,000,000 |
| JPN | JPY | 100,000,000 |
| CAN | CAD | 300,000 |
| HKG | HKD | 4,000,000 |

Elite scaling parameters are configurable per region via `POST /api/config/{country}`.

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
