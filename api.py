#!/usr/bin/env python3
"""
Market Value API - Score yearling lots and manage configuration.
Usage: uvicorn api:app --host 0.0.0.0 --port 8000
"""

import io
import json
import os
import random
import re
import shutil
import time
import zipfile
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, Optional

import httpx
import jwt
import pandas as pd
from fastapi import FastAPI, HTTPException, Security, Query, BackgroundTasks, UploadFile, File, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, EmailStr
from dotenv import load_dotenv

from src import rebuild_sale_features, get_connection, fetch_sale_country, MODEL_VERSION
from src.score_sale import load_models, get_model_dir, score_lots, upsert_to_database, fetch_existing_predictions
from src.train_model import train_model
from src.config import config, CONFIG_PATH

load_dotenv()

app = FastAPI(title="Market Value API", version=MODEL_VERSION)

# CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Auth
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


# OTP storage (in-memory)
@dataclass
class OTPRecord:
    code: str
    expires_at: float


otp_store: dict[str, OTPRecord] = {}  # email -> OTPRecord
OTP_EXPIRY_SECONDS = 600  # 10 minutes


def get_email_whitelist() -> list[str]:
    """Get list of whitelisted emails from environment."""
    whitelist = os.getenv("AUTH_EMAIL_WHITELIST", "")
    return [e.strip().lower() for e in whitelist.split(",") if e.strip()]


def send_otp_email(email: str, code: str) -> bool:
    """Send OTP code via email service."""
    if os.getenv("AUTH_DEV_MODE", "").lower() == "true":
        print(f"[DEV] OTP for {email}: {code}")
        return True

    url = os.getenv("EMAIL_SERVICE_URL")
    api_key = os.getenv("EMAIL_SERVICE_API_KEY")

    if not url or not api_key:
        print("[ERROR] EMAIL_SERVICE_URL or EMAIL_SERVICE_API_KEY not configured")
        return False

    try:
        response = httpx.post(
            url,
            headers={
                "x-api-key": api_key,
                "Content-Type": "application/json"
            },
            json={
                "to": email,
                "subject": f"Market Value API - Verification Code: {code}",
                "body_text": (
                    f"Hi,\n\n"
                    f"Your one-time verification code for the Market Value API is:\n\n"
                    f"    {code}\n\n"
                    f"This code is valid for 10 minutes. If you did not request this code, "
                    f"please ignore this email.\n\n"
                    f"- StallionMatch Market Value"
                ),
            },
            timeout=10.0
        )
        if response.status_code == 200:
            return True
        print(f"[ERROR] Email service returned {response.status_code}: {response.text}")
        return False
    except Exception as e:
        print(f"[ERROR] Failed to send OTP email: {e}")
        return False


def create_jwt_token(email: str) -> str:
    """Create a JWT token for the authenticated user."""
    expiry_hours = int(os.getenv("JWT_EXPIRY_HOURS", 24))
    payload = {
        "sub": email,
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(hours=expiry_hours)
    }
    return jwt.encode(payload, os.getenv("JWT_SECRET", ""), algorithm="HS256")


def verify_jwt_token(token: str) -> str | None:
    """Verify JWT token and return the email (subject) if valid."""
    secret = os.getenv("JWT_SECRET", "")
    if not secret:
        return None
    try:
        payload = jwt.decode(token, secret, algorithms=["HS256"])
        return payload.get("sub")
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


async def verify_auth(
    api_key: str = Security(api_key_header),
    authorization: str = Header(None)
) -> str:
    """Verify authentication via API key or JWT token. Returns user identity."""
    # Check API key first
    if api_key and api_key == os.getenv("API_KEY"):
        return "api_key"

    # Check JWT token
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:]
        email = verify_jwt_token(token)
        if email:
            return email

    raise HTTPException(status_code=403, detail="Invalid authentication")


# ============================================================================
# ACTIVITY LOG
# ============================================================================

ACTIVITY_LOG_PATH = Path("logs/activity.jsonl")
ACTIVITY_LOG_MAX_BYTES = int(os.getenv("ACTIVITY_LOG_MAX_MB", "20")) * 1024 * 1024
ACTIVITY_LOG_MAX_DAYS = int(os.getenv("ACTIVITY_LOG_MAX_DAYS", "26"))


def _rotate_if_needed():
    """Rotate activity log if size or age limit exceeded."""
    if not ACTIVITY_LOG_PATH.exists():
        return
    try:
        stat = ACTIVITY_LOG_PATH.stat()
        age_days = (time.time() - stat.st_mtime) / 86400
        if stat.st_size >= ACTIVITY_LOG_MAX_BYTES or age_days >= ACTIVITY_LOG_MAX_DAYS:
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            ACTIVITY_LOG_PATH.rename(ACTIVITY_LOG_PATH.with_name(f"activity.{ts}.jsonl"))
    except OSError:
        pass


def log_activity(user: str, method: str, path: str, detail: str,
                 category: str = "general", status: str = "success"):
    """Append a JSON line to the activity log."""
    ACTIVITY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _rotate_if_needed()
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user": user,
        "action": f"{method} {path}",
        "detail": detail,
        "category": category,
        "status": status,
    }
    try:
        with open(ACTIVITY_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass


# ============================================================================
# CONFIG HELPERS
# ============================================================================


def load_config_json() -> dict:
    """Load config.json as dict."""
    with open(CONFIG_PATH) as f:
        return json.load(f)


def save_config_json(data: dict):
    """Save dict to config.json and reload config singleton."""
    with open(CONFIG_PATH, "w") as f:
        json.dump(data, f, indent=2)
    config.reload()


def deep_merge(base: dict, updates: dict) -> dict:
    """Deep merge updates into base dict. Returns new dict."""
    result = base.copy()
    for key, value in updates.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


# ============================================================================
# RESPONSE MODELS
# ============================================================================


class LotScore(BaseModel):
    lot_id: int
    horse_id: Optional[int]
    sales_id: int
    lot_number: Optional[int]
    horse_name: Optional[str]
    sire_name: Optional[str]
    session_median_price: float
    mv_expected_price: float
    mv_low_price: float
    mv_high_price: float
    mv_expected_index: float
    mv_confidence_tier: str


class PriceRange(BaseModel):
    low: float
    expected: float
    high: float


class ConfidenceTierCounts(BaseModel):
    high: int
    medium: int
    low: int


class ScoreSummary(BaseModel):
    gross: PriceRange
    median_prices: PriceRange
    confidence_tiers: ConfidenceTierCounts
    elite_scaling_count: int
    elite_scaling_percent: float


class ScoreResponse(BaseModel):
    sale_id: int
    country_code: str
    model_dir: str
    total_lots: int
    summary: ScoreSummary
    lots: list[LotScore]
    output_written: Optional[str] = None


class LotCommit(BaseModel):
    horse_id: int
    sales_id: int
    mv_expected_price: float
    mv_low_price: float
    mv_high_price: float
    mv_expected_index: float
    mv_confidence_tier: str
    session_median_price: float


class CommitLotsRequest(BaseModel):
    lots: list[LotCommit]


class CommitLotsResponse(BaseModel):
    sale_id: int
    country_code: str
    inserted: int
    updated: int
    total: int


class ExistingValues(BaseModel):
    mv_expected_price: Optional[float] = None
    mv_low_price: Optional[float] = None
    mv_high_price: Optional[float] = None
    mv_expected_index: Optional[float] = None
    mv_confidence_tier: Optional[str] = None
    session_median_price: Optional[float] = None


class DeltaValues(BaseModel):
    mv_expected_price: Optional[float] = None
    mv_expected_price_pct: Optional[float] = None
    mv_low_price: Optional[float] = None
    mv_high_price: Optional[float] = None


class LotScoreComparison(BaseModel):
    lot_id: int
    horse_id: Optional[int]
    sales_id: int
    lot_number: Optional[int]
    horse_name: Optional[str]
    sire_name: Optional[str]
    session_median_price: float

    # New predictions
    new: LotScore

    # Existing DB values (None if new record)
    existing: Optional[ExistingValues] = None

    # Delta values
    delta: Optional[DeltaValues] = None

    is_new: bool


class CompareResponse(BaseModel):
    sale_id: int
    country_code: str
    model_dir: str
    total_lots: int
    new_lots: int
    existing_lots: int
    changed_lots: int
    unchanged_lots: int
    avg_price_delta: float
    avg_price_delta_pct: float
    lots: list[LotScoreComparison]


class HealthResponse(BaseModel):
    status: str
    database_connected: bool


class SaleSearchResult(BaseModel):
    sale_id: int
    sale_name: str
    sale_date: Optional[str] = None
    country_code: str
    currency_code: Optional[str] = None
    lot_count: int
    sale_company: str
    status: str  # "past" or "upcoming"


class SaleSearchResponse(BaseModel):
    query: str
    results: list[SaleSearchResult]


class SaleBook(BaseModel):
    book_number: int
    day_number: Optional[int] = None
    lot_count: int


class SaleLotStats(BaseModel):
    total_lots: int
    sold_count: int
    passed_in_count: int
    withdrawn_count: int
    clearance_rate: Optional[float] = None  # sold / (total - withdrawn) * 100


class SalePriceBands(BaseModel):
    under_50k: int = 0
    band_50k_100k: int = 0
    band_100k_200k: int = 0
    band_200k_500k: int = 0
    band_500k_1m: int = 0
    over_1m: int = 0


class SalePriceStats(BaseModel):
    gross: Optional[float] = None
    avg_price: Optional[float] = None
    median_price: Optional[float] = None
    q1_price: Optional[float] = None  # 25th percentile
    q3_price: Optional[float] = None  # 75th percentile
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    top10_avg: Optional[float] = None  # Average of top 10 prices
    std_dev: Optional[float] = None  # Standard deviation
    price_bands: Optional[SalePriceBands] = None


class YoyChange(BaseModel):
    gross_pct: Optional[float] = None
    avg_price_pct: Optional[float] = None
    median_price_pct: Optional[float] = None
    sold_count_change: Optional[int] = None
    clearance_rate_change: Optional[float] = None


class PriorYearStats(BaseModel):
    sale_ids: list[int]  # Could be multiple matching sales
    sale_names: list[str]
    start_date: Optional[str] = None  # Earliest start date
    end_date: Optional[str] = None  # Latest end date
    lot_stats: SaleLotStats
    price_stats: Optional[SalePriceStats] = None
    yoy_change: Optional[YoyChange] = None


class SaleHistoryYear(BaseModel):
    year: int
    sale_ids: list[int]
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    total_lots: int
    sold_count: int
    passed_in_count: int
    withdrawn_count: int
    clearance_rate: Optional[float] = None
    gross: Optional[float] = None
    avg_price: Optional[float] = None
    median_price: Optional[float] = None
    q1_price: Optional[float] = None
    q3_price: Optional[float] = None
    top10_avg: Optional[float] = None


class SaleQueueStats(BaseModel):
    completed: int
    in_queue: int
    failed: int
    postponed: int
    last_completed: Optional[str] = None


class SaleDetailResponse(BaseModel):
    sale_id: int
    sale_code: Optional[str] = None
    sale_name: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    sale_type: Optional[str] = None
    sale_status: Optional[str] = None
    is_online: bool
    is_public: bool
    sale_company: str
    company_website: Optional[str] = None
    country_code: str
    country_name: str
    currency_code: Optional[str] = None
    currency_symbol: Optional[str] = None
    status: str  # "past" or "upcoming"
    lot_stats: SaleLotStats
    price_stats: Optional[SalePriceStats] = None
    prior_year: Optional[PriorYearStats] = None
    history: list[SaleHistoryYear] = []
    queue_stats: Optional[SaleQueueStats] = None
    books: list[SaleBook]


class TrainResponse(BaseModel):
    message: str
    country: str
    version: str
    output_dir: str


class DataSummary(BaseModel):
    total_samples: Optional[int] = None
    train_rows: Optional[int] = None
    validation_rows: Optional[int] = None
    test_rows: Optional[int] = None
    features_count: Optional[int] = None


class BaselineModel(BaseModel):
    name: str = "Elastic Net"
    train_mae: Optional[float] = None
    train_r2: Optional[float] = None
    test_mae: Optional[float] = None
    test_r2: Optional[float] = None
    naive_mae: Optional[float] = None
    passes: Optional[bool] = None


class QuantileModels(BaseModel):
    q25_trees: Optional[int] = None
    q50_trees: Optional[int] = None
    q75_trees: Optional[int] = None


class Evaluation(BaseModel):
    p50_mae: Optional[float] = None
    p50_rmse: Optional[float] = None
    p50_r2: Optional[float] = None
    raw_coverage_p25: Optional[float] = None
    raw_coverage_p75: Optional[float] = None
    mape: Optional[float] = None


class Calibration(BaseModel):
    offset_p25: Optional[float] = None
    offset_p75: Optional[float] = None
    calibrated_coverage_p25: Optional[float] = None
    calibrated_coverage_p75: Optional[float] = None


class FeatureImportance(BaseModel):
    average: Optional[dict[str, float]] = None
    q25: Optional[dict[str, float]] = None
    q50: Optional[dict[str, float]] = None
    q75: Optional[dict[str, float]] = None


class ModelInfo(BaseModel):
    version: str
    directory: str
    is_active: bool
    training_info: Optional[dict] = None
    data_summary: Optional[DataSummary] = None
    baseline_model: Optional[BaselineModel] = None
    quantile_models: Optional[QuantileModels] = None
    evaluation: Optional[Evaluation] = None
    calibration: Optional[Calibration] = None
    feature_importance: Optional[FeatureImportance] = None


class ModelsListResponse(BaseModel):
    country: str
    active_model: str
    models: list[ModelInfo]


class ModelUploadResponse(BaseModel):
    model_name: str
    files: list[str]
    size_bytes: int
    message: str


class ModelDeleteResponse(BaseModel):
    model_name: str
    message: str


class YearsConfigResponse(BaseModel):
    year_start: int
    year_end: int


class TestYearsConfigResponse(BaseModel):
    model_test_last_years: int


class SaleHistoryYearsConfigResponse(BaseModel):
    sale_history_years: int


# ============================================================================
# AUTH REQUEST/RESPONSE MODELS
# ============================================================================


class OTPRequest(BaseModel):
    email: EmailStr


class OTPRequestResponse(BaseModel):
    message: str


class OTPVerifyRequest(BaseModel):
    email: EmailStr
    code: str


class OTPVerifyResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


# ============================================================================
# CONFIG REQUEST/RESPONSE MODELS
# ============================================================================


class EliteScalingConfig(BaseModel):
    threshold: int = 500000
    base_offset: float = 0.25
    scaling_factor: float = 0.5


class ConfidenceTiersConfig(BaseModel):
    close_threshold: float = 0.7
    extreme_threshold: float = 1.0


class RegionConfig(BaseModel):
    model: str
    currency_id: int
    hist_countries: list[str]
    elite_scaling: EliteScalingConfig
    confidence_tiers: ConfidenceTiersConfig
    sire_sample_min_count: int = 10


class FullConfigResponse(BaseModel):
    year_start: int
    year_end: int | None
    model_test_last_years: int
    sale_history_years: int
    audit_user_id: int
    regions: dict[str, RegionConfig]


# ============================================================================
# ACTIVITY LOG MODELS
# ============================================================================


class ActivityEntry(BaseModel):
    timestamp: str
    user: str
    action: str
    detail: str
    category: str = "general"
    status: str = "success"


class ActivityResponse(BaseModel):
    entries: list[ActivityEntry]
    total_in_file: int


# ============================================================================
# AUTH ENDPOINTS
# ============================================================================


@app.post("/auth/request-otp", response_model=OTPRequestResponse)
async def request_otp(request: OTPRequest):
    """Request OTP for whitelisted email."""
    email = request.email.lower()
    whitelist = get_email_whitelist()

    if not whitelist:
        raise HTTPException(status_code=500, detail="Email whitelist not configured")

    if email not in whitelist:
        log_activity(email, "POST", "/auth/request-otp", f"OTP rejected - email not authorized", category="auth", status="error")
        raise HTTPException(status_code=403, detail="Email not authorized")

    # Generate 6-digit OTP
    code = f"{random.randint(0, 999999):06d}"
    otp_store[email] = OTPRecord(code=code, expires_at=time.time() + OTP_EXPIRY_SECONDS)

    if not send_otp_email(email, code):
        raise HTTPException(status_code=500, detail="Failed to send email")

    log_activity(email, "POST", "/auth/request-otp", f"OTP requested for {email}", category="auth")
    return OTPRequestResponse(message=f"OTP sent to {email}")


@app.post("/auth/verify-otp", response_model=OTPVerifyResponse)
async def verify_otp(request: OTPVerifyRequest):
    """Verify OTP and return JWT token."""
    email = request.email.lower()
    record = otp_store.get(email)

    if not record or time.time() > record.expires_at:
        log_activity(email, "POST", "/auth/verify-otp", f"Login failed for {email}", category="auth", status="error")
        raise HTTPException(status_code=401, detail="OTP expired or not found")

    if record.code != request.code:
        log_activity(email, "POST", "/auth/verify-otp", f"Login failed for {email}", category="auth", status="error")
        raise HTTPException(status_code=401, detail="Invalid OTP")

    # Remove OTP after successful verification (one-time use)
    del otp_store[email]

    # Create JWT token
    token = create_jwt_token(email)
    expiry_hours = int(os.getenv("JWT_EXPIRY_HOURS", 24))

    log_activity(email, "POST", "/auth/verify-otp", f"Login successful for {email}", category="auth")
    return OTPVerifyResponse(
        access_token=token,
        expires_in=expiry_hours * 3600
    )


# ============================================================================
# SCORING ENDPOINTS
# ============================================================================


@app.get("/health", response_model=HealthResponse)
async def health_check():
    db_ok = False
    try:
        conn = get_connection()
        conn.close()
        db_ok = True
    except Exception:
        pass
    return HealthResponse(status="healthy" if db_ok else "degraded", database_connected=db_ok)


@app.get("/api/sales/search", response_model=SaleSearchResponse)
async def search_sales(
    q: str = Query(..., min_length=3, description="Search query"),
    limit: int = Query(default=20, ge=1, le=100),
    user: str = Security(verify_auth),
):
    """Search sales by name. Returns matching sales with lot counts."""
    conn = get_connection()
    cursor = conn.cursor()

    query = """
        SELECT TOP (?)
            SL.Id AS sale_id,
            SL.salesName AS sale_name,
            CAST(SL.startDate AS DATE) AS sale_date,
            CN.countryCode AS country_code,
            CUR.currencyCode AS currency_code,
            SC.salescompanyName AS sale_company,
            (SELECT COUNT(*) FROM tblSalesLot LT
             WHERE LT.salesId = SL.Id) AS lot_count
        FROM tblSales SL
        JOIN tblCountry CN ON SL.countryId = CN.id
        JOIN tblSalesCompany SC ON SL.salesCompanyId = SC.Id
        LEFT JOIN tblCurrency CUR ON CN.preferredCurrencyId = CUR.id
        WHERE SL.salesName LIKE ? OR SC.salescompanyName LIKE ?
        ORDER BY SL.startDate DESC
    """

    cursor.execute(query, (limit, f"%{q}%", f"%{q}%"))
    rows = cursor.fetchall()
    conn.close()

    today = datetime.now().date()
    results = [
        SaleSearchResult(
            sale_id=row.sale_id,
            sale_name=row.sale_name,
            sale_date=str(row.sale_date) if row.sale_date else None,
            country_code=row.country_code,
            currency_code=row.currency_code,
            lot_count=row.lot_count,
            sale_company=row.sale_company,
            status="upcoming" if row.sale_date and row.sale_date >= today else "past"
        )
        for row in rows
    ]

    return SaleSearchResponse(query=q, results=results)


def compute_price_stats(cursor, sale_ids: list[int]) -> tuple[SalePriceStats | None, int, int, int, int, float | None]:
    """
    Compute enhanced price stats for one or more sales.
    Returns (price_stats, total_lots, sold_count, passed_in_count, withdrawn_count, clearance_rate)
    """
    placeholders = ",".join("?" * len(sale_ids))

    # Get lot statistics
    stats_query = f"""
        SELECT
            COUNT(*) AS total_lots,
            SUM(CASE WHEN isWithdrawn = 1 THEN 1 ELSE 0 END) AS withdrawn_count,
            SUM(CASE WHEN ISNULL(isWithdrawn, 0) = 0 AND isPassedIn = 1 THEN 1 ELSE 0 END) AS passed_in_count,
            SUM(CASE WHEN price > 0 AND ISNULL(isWithdrawn, 0) = 0 AND ISNULL(isPassedIn, 0) = 0 THEN 1 ELSE 0 END) AS sold_count,
            SUM(CASE WHEN price > 0 AND ISNULL(isWithdrawn, 0) = 0 AND ISNULL(isPassedIn, 0) = 0 THEN price ELSE 0 END) AS gross,
            AVG(CASE WHEN price > 0 AND ISNULL(isWithdrawn, 0) = 0 AND ISNULL(isPassedIn, 0) = 0 THEN CAST(price AS FLOAT) END) AS avg_price,
            MIN(CASE WHEN price > 0 AND ISNULL(isWithdrawn, 0) = 0 AND ISNULL(isPassedIn, 0) = 0 THEN price END) AS min_price,
            MAX(CASE WHEN price > 0 AND ISNULL(isWithdrawn, 0) = 0 AND ISNULL(isPassedIn, 0) = 0 THEN price END) AS max_price,
            STDEV(CASE WHEN price > 0 AND ISNULL(isWithdrawn, 0) = 0 AND ISNULL(isPassedIn, 0) = 0 THEN CAST(price AS FLOAT) END) AS std_dev,
            -- Price bands
            SUM(CASE WHEN price > 0 AND price < 50000 AND ISNULL(isWithdrawn, 0) = 0 AND ISNULL(isPassedIn, 0) = 0 THEN 1 ELSE 0 END) AS under_50k,
            SUM(CASE WHEN price >= 50000 AND price < 100000 AND ISNULL(isWithdrawn, 0) = 0 AND ISNULL(isPassedIn, 0) = 0 THEN 1 ELSE 0 END) AS band_50k_100k,
            SUM(CASE WHEN price >= 100000 AND price < 200000 AND ISNULL(isWithdrawn, 0) = 0 AND ISNULL(isPassedIn, 0) = 0 THEN 1 ELSE 0 END) AS band_100k_200k,
            SUM(CASE WHEN price >= 200000 AND price < 500000 AND ISNULL(isWithdrawn, 0) = 0 AND ISNULL(isPassedIn, 0) = 0 THEN 1 ELSE 0 END) AS band_200k_500k,
            SUM(CASE WHEN price >= 500000 AND price < 1000000 AND ISNULL(isWithdrawn, 0) = 0 AND ISNULL(isPassedIn, 0) = 0 THEN 1 ELSE 0 END) AS band_500k_1m,
            SUM(CASE WHEN price >= 1000000 AND ISNULL(isWithdrawn, 0) = 0 AND ISNULL(isPassedIn, 0) = 0 THEN 1 ELSE 0 END) AS over_1m
        FROM tblSalesLot
        WHERE salesId IN ({placeholders})
    """
    cursor.execute(stats_query, sale_ids)
    stats_row = cursor.fetchone()

    total_lots = stats_row.total_lots or 0
    withdrawn = stats_row.withdrawn_count or 0
    passed_in = stats_row.passed_in_count or 0
    sold = stats_row.sold_count or 0

    # Clearance rate: sold / (total - withdrawn)
    eligible_lots = total_lots - withdrawn
    clearance_rate = round(100.0 * sold / eligible_lots, 1) if eligible_lots > 0 else None

    if sold == 0:
        return None, total_lots, sold, passed_in, withdrawn, clearance_rate

    # Get quartiles (Q1, median, Q3) - separate query for PERCENTILE_CONT
    quartiles_query = f"""
        SELECT DISTINCT
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY price) OVER () AS q1_price,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY price) OVER () AS median_price,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY price) OVER () AS q3_price
        FROM tblSalesLot
        WHERE salesId IN ({placeholders})
          AND price > 0
          AND ISNULL(isWithdrawn, 0) = 0
          AND ISNULL(isPassedIn, 0) = 0
    """
    cursor.execute(quartiles_query, sale_ids)
    quartiles_row = cursor.fetchone()

    # Get top 10 average
    top10_query = f"""
        SELECT AVG(CAST(price AS FLOAT)) AS top10_avg
        FROM (
            SELECT TOP 10 price
            FROM tblSalesLot
            WHERE salesId IN ({placeholders})
              AND price > 0
              AND ISNULL(isWithdrawn, 0) = 0
              AND ISNULL(isPassedIn, 0) = 0
            ORDER BY price DESC
        ) t
    """
    cursor.execute(top10_query, sale_ids)
    top10_row = cursor.fetchone()

    price_bands = SalePriceBands(
        under_50k=stats_row.under_50k or 0,
        band_50k_100k=stats_row.band_50k_100k or 0,
        band_100k_200k=stats_row.band_100k_200k or 0,
        band_200k_500k=stats_row.band_200k_500k or 0,
        band_500k_1m=stats_row.band_500k_1m or 0,
        over_1m=stats_row.over_1m or 0,
    )

    price_stats = SalePriceStats(
        gross=round(stats_row.gross, 2) if stats_row.gross else None,
        avg_price=round(stats_row.avg_price, 2) if stats_row.avg_price else None,
        median_price=round(quartiles_row.median_price, 2) if quartiles_row and quartiles_row.median_price else None,
        q1_price=round(quartiles_row.q1_price, 2) if quartiles_row and quartiles_row.q1_price else None,
        q3_price=round(quartiles_row.q3_price, 2) if quartiles_row and quartiles_row.q3_price else None,
        min_price=round(stats_row.min_price, 2) if stats_row.min_price else None,
        max_price=round(stats_row.max_price, 2) if stats_row.max_price else None,
        top10_avg=round(top10_row.top10_avg, 2) if top10_row and top10_row.top10_avg else None,
        std_dev=round(stats_row.std_dev, 2) if stats_row.std_dev else None,
        price_bands=price_bands,
    )

    return price_stats, total_lots, sold, passed_in, withdrawn, clearance_rate


@app.get("/api/sales/{sale_id}", response_model=SaleDetailResponse)
async def get_sale_detail(
    sale_id: int,
    user: str = Security(verify_auth),
):
    """Get detailed information about a sale including enhanced price stats and prior year comparison."""
    conn = get_connection()
    cursor = conn.cursor()

    # Get basic sale info
    query = """
        SELECT
            SL.Id AS sale_id,
            SL.salesCode AS sale_code,
            SL.salesName AS sale_name,
            CAST(SL.startDate AS DATE) AS start_date,
            CAST(SL.endDate AS DATE) AS end_date,
            SL.isOnlineSales AS is_online,
            SL.isPublic AS is_public,
            SL.salesCompanyId AS sales_company_id,
            SL.salesTypeId AS sales_type_id,
            SC.salescompanyName AS sale_company,
            SC.salescompanyWebsite AS company_website,
            CN.countryCode AS country_code,
            CN.countryName AS country_name,
            CUR.currencyCode AS currency_code,
            CUR.currencySymbol AS currency_symbol,
            ST.status AS sale_status,
            STP.salesTypeName AS sale_type
        FROM tblSales SL
        JOIN tblCountry CN ON SL.countryId = CN.id
        JOIN tblSalesCompany SC ON SL.salesCompanyId = SC.Id
        LEFT JOIN tblCurrency CUR ON CN.preferredCurrencyId = CUR.id
        LEFT JOIN tblSalesStatus ST ON SL.statusId = ST.id
        LEFT JOIN tblSalesType STP ON SL.salesTypeId = STP.Id
        WHERE SL.Id = ?
    """
    cursor.execute(query, (sale_id,))
    sale_row = cursor.fetchone()

    if not sale_row:
        conn.close()
        raise HTTPException(status_code=404, detail=f"Sale {sale_id} not found")

    # Compute price stats for current sale
    price_stats, total_lots, sold, passed_in, withdrawn, clearance_rate = compute_price_stats(cursor, [sale_id])

    lot_stats = SaleLotStats(
        total_lots=total_lots,
        sold_count=sold,
        passed_in_count=passed_in,
        withdrawn_count=withdrawn,
        clearance_rate=clearance_rate,
    )

    # Get book breakdown
    books_query = """
        SELECT bookNumber, dayNumber, COUNT(*) AS lot_count
        FROM tblSalesLot
        WHERE salesId = ?
        GROUP BY bookNumber, dayNumber
        ORDER BY bookNumber, dayNumber
    """
    cursor.execute(books_query, (sale_id,))
    books_rows = cursor.fetchall()

    # Get queue stats (report generation status)
    queue_query = """
        SELECT Completed, InQueue, Failed, Postponed, LastCompleted
        FROM vw_fr_SalesQueueOverview
        WHERE SaleId = ?
    """
    cursor.execute(queue_query, (sale_id,))
    queue_row = cursor.fetchone()

    # Find prior year sales (same company, type, and month from previous year)
    prior_year = None
    if sale_row.start_date and sale_row.sales_company_id and sale_row.sales_type_id:
        prior_year_query = """
            SELECT Id, salesName, CAST(startDate AS DATE) AS start_date, CAST(endDate AS DATE) AS end_date
            FROM tblSales
            WHERE salesCompanyId = ?
              AND salesTypeId = ?
              AND MONTH(startDate) = MONTH(?)
              AND YEAR(startDate) = YEAR(?) - 1
            ORDER BY startDate
        """
        cursor.execute(prior_year_query, (
            sale_row.sales_company_id,
            sale_row.sales_type_id,
            sale_row.start_date,
            sale_row.start_date,
        ))
        prior_sales = cursor.fetchall()

        if prior_sales:
            prior_sale_ids = [row.Id for row in prior_sales]
            prior_sale_names = [row.salesName for row in prior_sales]
            # Get earliest start and latest end across all matching sales
            prior_start_date = min((row.start_date for row in prior_sales if row.start_date), default=None)
            prior_end_date = max((row.end_date for row in prior_sales if row.end_date), default=None)

            # Compute stats for prior year sales
            prior_price_stats, prior_total, prior_sold, prior_passed_in, prior_withdrawn, prior_clearance = compute_price_stats(
                cursor, prior_sale_ids
            )

            prior_lot_stats = SaleLotStats(
                total_lots=prior_total,
                sold_count=prior_sold,
                passed_in_count=prior_passed_in,
                withdrawn_count=prior_withdrawn,
                clearance_rate=prior_clearance,
            )

            # Calculate YoY changes
            yoy_change = None
            if price_stats and prior_price_stats:
                gross_pct = None
                if price_stats.gross and prior_price_stats.gross and prior_price_stats.gross > 0:
                    gross_pct = round(100.0 * (price_stats.gross - prior_price_stats.gross) / prior_price_stats.gross, 1)

                avg_pct = None
                if price_stats.avg_price and prior_price_stats.avg_price and prior_price_stats.avg_price > 0:
                    avg_pct = round(100.0 * (price_stats.avg_price - prior_price_stats.avg_price) / prior_price_stats.avg_price, 1)

                median_pct = None
                if price_stats.median_price and prior_price_stats.median_price and prior_price_stats.median_price > 0:
                    median_pct = round(100.0 * (price_stats.median_price - prior_price_stats.median_price) / prior_price_stats.median_price, 1)

                sold_change = sold - prior_sold if sold and prior_sold else None

                clearance_change = None
                if clearance_rate is not None and prior_clearance is not None:
                    clearance_change = round(clearance_rate - prior_clearance, 1)

                yoy_change = YoyChange(
                    gross_pct=gross_pct,
                    avg_price_pct=avg_pct,
                    median_price_pct=median_pct,
                    sold_count_change=sold_change,
                    clearance_rate_change=clearance_change,
                )

            prior_year = PriorYearStats(
                sale_ids=prior_sale_ids,
                sale_names=prior_sale_names,
                start_date=str(prior_start_date) if prior_start_date else None,
                end_date=str(prior_end_date) if prior_end_date else None,
                lot_stats=prior_lot_stats,
                price_stats=prior_price_stats,
                yoy_change=yoy_change,
            )

    # Build history (configurable number of years)
    history = []
    history_years = config.app.sale_history_years
    if history_years > 0 and sale_row.start_date and sale_row.sales_company_id and sale_row.sales_type_id:
        current_year = sale_row.start_date.year
        history_query = """
            SELECT Id, YEAR(startDate) AS sale_year,
                   CAST(startDate AS DATE) AS start_date,
                   CAST(endDate AS DATE) AS end_date
            FROM tblSales
            WHERE salesCompanyId = ?
              AND salesTypeId = ?
              AND MONTH(startDate) = MONTH(?)
              AND YEAR(startDate) >= ?
              AND YEAR(startDate) < ?
            ORDER BY startDate
        """
        cursor.execute(history_query, (
            sale_row.sales_company_id,
            sale_row.sales_type_id,
            sale_row.start_date,
            current_year - history_years,
            current_year,
        ))
        history_sales = cursor.fetchall()

        if history_sales:
            # Group by year with dates
            from collections import defaultdict
            sales_by_year = defaultdict(lambda: {"ids": [], "start_dates": [], "end_dates": []})
            for row in history_sales:
                sales_by_year[row.sale_year]["ids"].append(row.Id)
                if row.start_date:
                    sales_by_year[row.sale_year]["start_dates"].append(row.start_date)
                if row.end_date:
                    sales_by_year[row.sale_year]["end_dates"].append(row.end_date)

            # Compute stats for each year
            for year in sorted(sales_by_year.keys(), reverse=True):
                year_data = sales_by_year[year]
                year_sale_ids = year_data["ids"]
                year_start = min(year_data["start_dates"]) if year_data["start_dates"] else None
                year_end = max(year_data["end_dates"]) if year_data["end_dates"] else None

                year_price_stats, year_total, year_sold, year_passed_in, year_withdrawn, year_clearance = compute_price_stats(
                    cursor, year_sale_ids
                )
                history.append(SaleHistoryYear(
                    year=year,
                    sale_ids=year_sale_ids,
                    start_date=str(year_start) if year_start else None,
                    end_date=str(year_end) if year_end else None,
                    total_lots=year_total,
                    sold_count=year_sold,
                    passed_in_count=year_passed_in,
                    withdrawn_count=year_withdrawn,
                    clearance_rate=year_clearance,
                    gross=year_price_stats.gross if year_price_stats else None,
                    avg_price=year_price_stats.avg_price if year_price_stats else None,
                    median_price=year_price_stats.median_price if year_price_stats else None,
                    q1_price=year_price_stats.q1_price if year_price_stats else None,
                    q3_price=year_price_stats.q3_price if year_price_stats else None,
                    top10_avg=year_price_stats.top10_avg if year_price_stats else None,
                ))

    conn.close()

    # Build response
    today = datetime.now().date()
    start_date = sale_row.start_date

    # Books
    books = [
        SaleBook(
            book_number=row.bookNumber or 1,
            day_number=row.dayNumber,
            lot_count=row.lot_count,
        )
        for row in books_rows
    ]

    # Queue stats
    queue_stats = None
    if queue_row:
        queue_stats = SaleQueueStats(
            completed=queue_row.Completed or 0,
            in_queue=queue_row.InQueue or 0,
            failed=queue_row.Failed or 0,
            postponed=queue_row.Postponed or 0,
            last_completed=str(queue_row.LastCompleted) if queue_row.LastCompleted else None,
        )

    return SaleDetailResponse(
        sale_id=sale_row.sale_id,
        sale_code=sale_row.sale_code,
        sale_name=sale_row.sale_name,
        start_date=str(sale_row.start_date) if sale_row.start_date else None,
        end_date=str(sale_row.end_date) if sale_row.end_date else None,
        sale_type=sale_row.sale_type,
        sale_status=sale_row.sale_status,
        is_online=bool(sale_row.is_online),
        is_public=bool(sale_row.is_public),
        sale_company=sale_row.sale_company,
        company_website=sale_row.company_website,
        country_code=sale_row.country_code,
        country_name=sale_row.country_name,
        currency_code=sale_row.currency_code,
        currency_symbol=sale_row.currency_symbol,
        status="upcoming" if start_date and start_date >= today else "past",
        lot_stats=lot_stats,
        price_stats=price_stats,
        prior_year=prior_year,
        history=history,
        queue_stats=queue_stats,
        books=books,
    )


@app.post("/api/score/{sale_id}", response_model=ScoreResponse)
async def score_sale(
    sale_id: int,
    output: Literal["none", "csv", "db"] = Query(default="none"),
    session_median: Optional[float] = Query(default=None, description="Manual session median price override"),
    user: str = Security(verify_auth),
):
    """Score all lots for a sale.

    For future sales without sold lots, the session median defaults to the prior year's
    median for the same sale. Use `session_median` to override this with a custom value.
    """
    try:
        conn = get_connection()
        country_code = fetch_sale_country(conn, sale_id)
        conn.close()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Rebuild features
    features_df = rebuild_sale_features(
        sale_id,
        export_csv=(output == "csv"),
        session_median_override=session_median
    )
    if features_df.empty:
        raise HTTPException(status_code=404, detail=f"No lots found for sale {sale_id}")

    # Load models and score
    model_dir = get_model_dir(country_code)
    models, offsets, feature_cols = load_models(model_dir)
    results_df = score_lots(features_df, models, offsets, feature_cols, country_code)

    # Handle output
    if output == "csv":
        os.makedirs("csv", exist_ok=True)
        results_df.to_csv(f"csv/sale_{sale_id}_scored.csv", index=False)
    elif output == "db":
        upsert_to_database(results_df, country_code)

    # Compute summary statistics
    gross = PriceRange(
        low=float(results_df["mv_low_price"].sum()),
        expected=float(results_df["mv_expected_price"].sum()),
        high=float(results_df["mv_high_price"].sum()),
    )
    median_prices = PriceRange(
        low=float(results_df["mv_low_price"].median()),
        expected=float(results_df["mv_expected_price"].median()),
        high=float(results_df["mv_high_price"].median()),
    )
    tier_counts = results_df["mv_confidence_tier"].value_counts()
    confidence_tiers = ConfidenceTierCounts(
        high=int(tier_counts.get("high", 0)),
        medium=int(tier_counts.get("medium", 0)),
        low=int(tier_counts.get("low", 0)),
    )
    # Elite scaling: count lots where mv_expected_price >= threshold
    elite_threshold = offsets.get("elite_scaling", {}).get("threshold", 300000)
    elite_count = int((results_df["mv_expected_price"] >= elite_threshold).sum())
    elite_percent = round(100.0 * elite_count / len(results_df), 2) if len(results_df) > 0 else 0.0

    summary = ScoreSummary(
        gross=gross,
        median_prices=median_prices,
        confidence_tiers=confidence_tiers,
        elite_scaling_count=elite_count,
        elite_scaling_percent=elite_percent,
    )

    lots = [
        LotScore(
            lot_id=int(row["lot_id"]),
            horse_id=int(row["horseId"]) if row.get("horseId") else None,
            sales_id=int(row["salesId"]),
            lot_number=int(row["lot_number"]) if row.get("lot_number") else None,
            horse_name=row.get("horse_name"),
            sire_name=row.get("sire_name"),
            session_median_price=float(row["session_median_price"]),
            mv_expected_price=float(row["mv_expected_price"]),
            mv_low_price=float(row["mv_low_price"]),
            mv_high_price=float(row["mv_high_price"]),
            mv_expected_index=float(row["mv_expected_index"]),
            mv_confidence_tier=row["mv_confidence_tier"],
        )
        for _, row in results_df.iterrows()
    ]

    log_activity(user, "POST", f"/api/score/{sale_id}", f"Scored {len(lots)} lots for sale {sale_id} (output={output})", category="score")

    return ScoreResponse(
        sale_id=sale_id,
        country_code=country_code,
        model_dir=model_dir,
        total_lots=len(lots),
        summary=summary,
        lots=lots,
        output_written=output if output != "none" else None,
    )


@app.post("/api/score/{sale_id}/commit", response_model=CommitLotsResponse)
async def commit_lots(
    sale_id: int,
    request: CommitLotsRequest,
    user: str = Security(verify_auth),
):
    """
    Commit selected lots to database.

    Use this endpoint after scoring to selectively persist predictions.
    The lots must have been scored for the specified sale_id.
    """
    if not request.lots:
        raise HTTPException(status_code=400, detail="No lots provided")

    # Validate all lots have sales_id matching path param
    for lot in request.lots:
        if lot.sales_id != sale_id:
            raise HTTPException(
                status_code=400,
                detail=f"Lot sales_id {lot.sales_id} does not match path sale_id {sale_id}",
            )

    # Get country_code from sale_id
    try:
        conn = get_connection()
        country_code = fetch_sale_country(conn, sale_id)
        conn.close()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Convert lots to DataFrame with expected column names
    results_df = pd.DataFrame([
        {
            "horseId": lot.horse_id,
            "salesId": lot.sales_id,
            "mv_expected_price": lot.mv_expected_price,
            "mv_low_price": lot.mv_low_price,
            "mv_high_price": lot.mv_high_price,
            "mv_expected_index": lot.mv_expected_index,
            "mv_confidence_tier": lot.mv_confidence_tier,
            "session_median_price": lot.session_median_price,
        }
        for lot in request.lots
    ])

    # Upsert to database
    inserted, updated = upsert_to_database(results_df, country_code)

    log_activity(user, "POST", f"/api/score/{sale_id}/commit", f"Committed {len(request.lots)} lots (inserted={inserted}, updated={updated})", category="score")

    return CommitLotsResponse(
        sale_id=sale_id,
        country_code=country_code,
        inserted=inserted,
        updated=updated,
        total=inserted + updated,
    )


@app.post("/api/score/{sale_id}/compare", response_model=CompareResponse)
async def score_and_compare(
    sale_id: int,
    session_median: Optional[float] = Query(default=None, description="Manual session median price override"),
    user: str = Security(verify_auth),
):
    """
    Score sale and compare with existing database values.

    Returns new predictions alongside existing values from tblHorseAnalytics,
    with calculated deltas. Read-only operation - does not modify database.

    For future sales without sold lots, the session median defaults to the prior year's
    median for the same sale. Use `session_median` to override this with a custom value.
    """
    # Get country and validate sale exists
    try:
        conn = get_connection()
        country_code = fetch_sale_country(conn, sale_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Rebuild features
    features_df = rebuild_sale_features(
        sale_id,
        export_csv=False,
        session_median_override=session_median
    )
    if features_df.empty:
        conn.close()
        raise HTTPException(status_code=404, detail=f"No lots found for sale {sale_id}")

    # Load models and score
    model_dir = get_model_dir(country_code)
    models, offsets, feature_cols = load_models(model_dir)
    results_df = score_lots(features_df, models, offsets, feature_cols, country_code)

    # Fetch existing predictions from database
    horse_ids = [int(h) for h in results_df["horseId"].dropna().tolist()]
    existing_predictions = fetch_existing_predictions(conn, horse_ids, sale_id)
    conn.close()

    # Build comparison results
    comparison_lots = []
    deltas = []  # For computing averages

    for _, row in results_df.iterrows():
        horse_id = int(row["horseId"]) if pd.notna(row.get("horseId")) else None

        # Build new prediction (LotScore)
        new_score = LotScore(
            lot_id=int(row["lot_id"]),
            horse_id=horse_id,
            sales_id=int(row["salesId"]),
            lot_number=int(row["lot_number"]) if pd.notna(row.get("lot_number")) else None,
            horse_name=row.get("horse_name"),
            sire_name=row.get("sire_name"),
            session_median_price=float(row["session_median_price"]),
            mv_expected_price=float(row["mv_expected_price"]),
            mv_low_price=float(row["mv_low_price"]),
            mv_high_price=float(row["mv_high_price"]),
            mv_expected_index=float(row["mv_expected_index"]),
            mv_confidence_tier=row["mv_confidence_tier"],
        )

        # Check for existing prediction
        existing = existing_predictions.get(horse_id) if horse_id else None
        is_new = existing is None

        existing_values = None
        delta_values = None

        if existing:
            existing_values = ExistingValues(
                mv_expected_price=existing.get("mv_expected_price"),
                mv_low_price=existing.get("mv_low_price"),
                mv_high_price=existing.get("mv_high_price"),
                mv_expected_index=existing.get("mv_expected_index"),
                mv_confidence_tier=existing.get("mv_confidence_tier"),
                session_median_price=existing.get("session_median_price"),
            )

            # Calculate deltas
            if existing.get("mv_expected_price") is not None:
                delta_expected = float(row["mv_expected_price"]) - existing["mv_expected_price"]
                delta_pct = (delta_expected / existing["mv_expected_price"] * 100) if existing["mv_expected_price"] != 0 else 0.0

                delta_low = None
                if existing.get("mv_low_price") is not None:
                    delta_low = float(row["mv_low_price"]) - existing["mv_low_price"]

                delta_high = None
                if existing.get("mv_high_price") is not None:
                    delta_high = float(row["mv_high_price"]) - existing["mv_high_price"]

                delta_values = DeltaValues(
                    mv_expected_price=round(delta_expected, 2),
                    mv_expected_price_pct=round(delta_pct, 2),
                    mv_low_price=round(delta_low, 2) if delta_low is not None else None,
                    mv_high_price=round(delta_high, 2) if delta_high is not None else None,
                )

                deltas.append({
                    "absolute": delta_expected,
                    "pct": delta_pct,
                })

        comparison_lots.append(LotScoreComparison(
            lot_id=int(row["lot_id"]),
            horse_id=horse_id,
            sales_id=int(row["salesId"]),
            lot_number=int(row["lot_number"]) if pd.notna(row.get("lot_number")) else None,
            horse_name=row.get("horse_name"),
            sire_name=row.get("sire_name"),
            session_median_price=float(row["session_median_price"]),
            new=new_score,
            existing=existing_values,
            delta=delta_values,
            is_new=is_new,
        ))

    # Compute summary stats
    total_lots = len(comparison_lots)
    new_lots = sum(1 for lot in comparison_lots if lot.is_new)
    existing_lots = total_lots - new_lots

    # Changed = any difference in expected price (new != existing)
    changed_lots = sum(
        1 for lot in comparison_lots
        if not lot.is_new and lot.delta and lot.delta.mv_expected_price != 0
    )
    unchanged_lots = existing_lots - changed_lots

    avg_price_delta = sum(d["absolute"] for d in deltas) / len(deltas) if deltas else 0.0
    avg_price_delta_pct = sum(d["pct"] for d in deltas) / len(deltas) if deltas else 0.0

    log_activity(user, "POST", f"/api/score/{sale_id}/compare", f"Compared {total_lots} lots for sale {sale_id}", category="score")

    return CompareResponse(
        sale_id=sale_id,
        country_code=country_code,
        model_dir=model_dir,
        total_lots=total_lots,
        new_lots=new_lots,
        existing_lots=existing_lots,
        changed_lots=changed_lots,
        unchanged_lots=unchanged_lots,
        avg_price_delta=round(avg_price_delta, 2),
        avg_price_delta_pct=round(avg_price_delta_pct, 2),
        lots=comparison_lots,
    )


# ============================================================================
# TRAINING ENDPOINTS
# ============================================================================


training_state: dict = {
    "active": False,
    "country": None,
    "version": None,
    "phase": None,
    "started_at": None,
    "completed_at": None,
    "status": "idle",  # idle | training | completed | failed
    "error": None,
}


class TrainingStatusResponse(BaseModel):
    active: bool
    country: Optional[str] = None
    version: Optional[str] = None
    phase: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    status: str
    error: Optional[str] = None


def run_training(country: str, version: str, user: str):
    """Background task to run model training."""
    training_state["active"] = True
    training_state["country"] = country.upper()
    training_state["version"] = version
    training_state["phase"] = "starting"
    training_state["started_at"] = datetime.now(timezone.utc).isoformat()
    training_state["completed_at"] = None
    training_state["status"] = "training"
    training_state["error"] = None

    def on_progress(phase: str):
        training_state["phase"] = phase

    try:
        train_model(country=country, version=version, on_progress=on_progress)
        training_state["status"] = "completed"
        training_state["phase"] = "done"
        log_activity(user, "POST", f"/api/train/{country}", f"Completed training {country.upper()} {version}", category="train")
    except Exception as e:
        training_state["status"] = "failed"
        training_state["error"] = str(e)
        log_activity(user, "POST", f"/api/train/{country}", f"Training failed for {country.upper()} {version}: {e}", category="train", status="error")
    finally:
        training_state["active"] = False
        training_state["completed_at"] = datetime.now(timezone.utc).isoformat()


@app.get("/api/train/status", response_model=TrainingStatusResponse)
async def get_training_status(
    user: str = Security(verify_auth),
):
    """Get current training job status."""
    return TrainingStatusResponse(**training_state)


@app.post("/api/train/{country}", response_model=TrainResponse)
async def train_model_endpoint(
    country: str,
    background_tasks: BackgroundTasks,
    version: Optional[str] = Query(default=None, description="Force specific version (e.g., v5)"),
    user: str = Security(verify_auth),
):
    """
    Train a new model for a specific country.

    Training runs in the background. Check GET /api/train/status for progress.
    """
    country = country.lower()
    if country not in ["aus", "nzl", "usa"]:
        raise HTTPException(status_code=400, detail=f"Invalid country: {country}. Must be aus, nzl, or usa.")

    if training_state["active"]:
        raise HTTPException(
            status_code=409,
            detail=f"Training already in progress for {training_state['country']} ({training_state['phase']}). Check GET /api/train/status.",
        )

    # Determine version
    if version is None:
        from src.train_model import get_next_version
        version = get_next_version(country)

    output_dir = f"models/{country}_{version}"

    # Start training in background
    background_tasks.add_task(run_training, country, version, user)

    log_activity(user, "POST", f"/api/train/{country}", f"Started training {country.upper()} {version}", category="train")

    return TrainResponse(
        message=f"Training started for {country.upper()}",
        country=country.upper(),
        version=version,
        output_dir=output_dir,
    )


# ============================================================================
# MODEL LISTING ENDPOINTS
# ============================================================================


def parse_training_report(report_path: Path) -> dict:
    """Parse complete training_report.txt for all metrics."""
    if not report_path.exists():
        return {}

    content = report_path.read_text()
    result = {
        "training_info": {},
        "data_summary": {},
        "baseline_model": {},
        "quantile_models": {},
        "evaluation": {},
        "calibration": {},
    }

    # Training info
    date_match = re.search(r"Generated:\s*(\d{4}-\d{2}-\d{2}\s*\d{2}:\d{2}:\d{2})", content)
    if date_match:
        result["training_info"]["generated_at"] = date_match.group(1)

    version_match = re.search(r"Model Version:\s*(\w+)", content)
    if version_match:
        result["training_info"]["model_version"] = version_match.group(1)

    country_match = re.search(r"Country:\s*(\w+)", content)
    if country_match:
        result["training_info"]["country"] = country_match.group(1)

    # Data summary
    total_match = re.search(r"Total samples:\s*([\d,]+)", content)
    if total_match:
        result["data_summary"]["total_samples"] = int(total_match.group(1).replace(",", ""))

    train_match = re.search(r"Train:\s*([\d,]+)\s*rows", content)
    if train_match:
        result["data_summary"]["train_rows"] = int(train_match.group(1).replace(",", ""))

    val_match = re.search(r"Validation:\s*([\d,]+)\s*rows", content)
    if val_match:
        result["data_summary"]["validation_rows"] = int(val_match.group(1).replace(",", ""))

    test_rows_match = re.search(r"Test:\s*([\d,]+)\s*rows", content)
    if test_rows_match:
        result["data_summary"]["test_rows"] = int(test_rows_match.group(1).replace(",", ""))

    features_match = re.search(r"Features:\s*(\d+)", content)
    if features_match:
        result["data_summary"]["features_count"] = int(features_match.group(1))

    # Baseline model
    result["baseline_model"]["name"] = "Elastic Net"

    train_mae_match = re.search(r"Train MAE:\s*([\d.]+)", content)
    if train_mae_match:
        result["baseline_model"]["train_mae"] = float(train_mae_match.group(1))

    train_r2_match = re.search(r"Train R²:\s*([\d.]+)", content)
    if train_r2_match:
        result["baseline_model"]["train_r2"] = float(train_r2_match.group(1))

    test_mae_match = re.search(r"Test MAE:\s*([\d.]+)", content)
    if test_mae_match:
        result["baseline_model"]["test_mae"] = float(test_mae_match.group(1))

    test_r2_match = re.search(r"Test R²:\s*([\d.]+)", content)
    if test_r2_match:
        result["baseline_model"]["test_r2"] = float(test_r2_match.group(1))

    naive_mae_match = re.search(r"Naive MAE:\s*([\d.]+)", content)
    if naive_mae_match:
        result["baseline_model"]["naive_mae"] = float(naive_mae_match.group(1))

    result["baseline_model"]["passes"] = "✓ Baseline passes" in content

    # Quantile models
    q25_match = re.search(r"Q25:\s*(\d+)\s*trees", content)
    if q25_match:
        result["quantile_models"]["q25_trees"] = int(q25_match.group(1))

    q50_match = re.search(r"Q50:\s*(\d+)\s*trees", content)
    if q50_match:
        result["quantile_models"]["q50_trees"] = int(q50_match.group(1))

    q75_match = re.search(r"Q75:\s*(\d+)\s*trees", content)
    if q75_match:
        result["quantile_models"]["q75_trees"] = int(q75_match.group(1))

    # Evaluation - P50 metrics
    p50_mae_match = re.search(r"P50.*?MAE:\s*([\d.]+)", content, re.DOTALL)
    if p50_mae_match:
        result["evaluation"]["p50_mae"] = float(p50_mae_match.group(1))

    p50_rmse_match = re.search(r"RMSE:\s*([\d.]+)", content)
    if p50_rmse_match:
        result["evaluation"]["p50_rmse"] = float(p50_rmse_match.group(1))

    p50_r2_match = re.search(r"P50.*?R²:\s*([\d.]+)", content, re.DOTALL)
    if p50_r2_match:
        result["evaluation"]["p50_r2"] = float(p50_r2_match.group(1))

    raw_p25_match = re.search(r"% below P25:\s*([\d.]+)%", content)
    if raw_p25_match:
        result["evaluation"]["raw_coverage_p25"] = float(raw_p25_match.group(1))

    raw_p75_match = re.search(r"% below P75:\s*([\d.]+)%", content)
    if raw_p75_match:
        result["evaluation"]["raw_coverage_p75"] = float(raw_p75_match.group(1))

    mape_match = re.search(r"MAPE:\s*([\d.]+)%", content)
    if mape_match:
        result["evaluation"]["mape"] = float(mape_match.group(1))

    # Calibration
    offset_p25_match = re.search(r"P25:\s*(-?[\d.]+)\s*\n", content)
    if offset_p25_match:
        result["calibration"]["offset_p25"] = float(offset_p25_match.group(1))

    offset_p75_match = re.search(r"P75:\s*(-?[\d.]+)\s*\n", content)
    if offset_p75_match:
        result["calibration"]["offset_p75"] = float(offset_p75_match.group(1))

    cal_p25_match = re.search(r"Calibrated.*?% below P25:\s*([\d.]+)%", content, re.DOTALL)
    if cal_p25_match:
        result["calibration"]["calibrated_coverage_p25"] = float(cal_p25_match.group(1))

    cal_p75_match = re.search(r"Calibrated.*?% below P75:\s*([\d.]+)%", content, re.DOTALL)
    if cal_p75_match:
        result["calibration"]["calibrated_coverage_p75"] = float(cal_p75_match.group(1))

    return result


def parse_feature_importance(importance_path: Path) -> dict:
    """Get complete feature importance from JSON."""
    if not importance_path.exists():
        return {}

    with open(importance_path) as f:
        data = json.load(f)

    # Round values for cleaner output
    return {
        "average": {k: round(v, 2) for k, v in data.get("average", {}).items()},
        "q25": {k: round(v, 2) for k, v in data.get("q25", {}).items()},
        "q50": {k: round(v, 2) for k, v in data.get("q50", {}).items()},
        "q75": {k: round(v, 2) for k, v in data.get("q75", {}).items()},
    }


@app.get("/api/models/{country}", response_model=ModelsListResponse)
async def list_models(
    country: str,
    user: str = Security(verify_auth),
):
    """
    List all model versions for a country with training metrics.

    Returns feature importance and training report info for each model.
    """
    country = country.lower()
    if country not in ["aus", "nzl", "usa"]:
        raise HTTPException(status_code=400, detail=f"Invalid country: {country}. Must be aus, nzl, or usa.")

    models_dir = Path("models")
    if not models_dir.exists():
        raise HTTPException(status_code=404, detail="No models directory found")

    # Get active model from config
    active_model = config.app.models.get(country, country)

    # Find all model directories for this country
    model_infos = []
    pattern = re.compile(rf"^{country}(_v\d+)?$", re.IGNORECASE)

    for entry in sorted(models_dir.iterdir()):
        if entry.is_dir() and pattern.match(entry.name):
            # Extract version
            if "_v" in entry.name:
                version = entry.name.split("_")[-1]
            else:
                version = "v1"

            # Check if this is the active model
            is_active = entry.name == active_model

            # Parse training report
            report_path = entry / "training_report.txt"
            report_data = parse_training_report(report_path)

            # Parse feature importance
            importance_files = list(entry.glob("feature_importance_*.json"))
            feature_importance = None
            if importance_files:
                fi_data = parse_feature_importance(importance_files[0])
                if fi_data:
                    feature_importance = FeatureImportance(**fi_data)

            # Build structured response
            data_summary = None
            if report_data.get("data_summary"):
                data_summary = DataSummary(**report_data["data_summary"])

            baseline_model = None
            if report_data.get("baseline_model"):
                baseline_model = BaselineModel(**report_data["baseline_model"])

            quantile_models = None
            if report_data.get("quantile_models"):
                quantile_models = QuantileModels(**report_data["quantile_models"])

            evaluation = None
            if report_data.get("evaluation"):
                evaluation = Evaluation(**report_data["evaluation"])

            calibration = None
            if report_data.get("calibration"):
                calibration = Calibration(**report_data["calibration"])

            model_infos.append(ModelInfo(
                version=version,
                directory=entry.name,
                is_active=is_active,
                training_info=report_data.get("training_info") or None,
                data_summary=data_summary,
                baseline_model=baseline_model,
                quantile_models=quantile_models,
                evaluation=evaluation,
                calibration=calibration,
                feature_importance=feature_importance,
            ))

    if not model_infos:
        raise HTTPException(status_code=404, detail=f"No models found for country: {country}")

    return ModelsListResponse(
        country=country.upper(),
        active_model=active_model,
        models=model_infos,
    )


# Required files for a valid model
MODEL_REQUIRED_FILES = [
    "calibration_offsets.json",
    "feature_cols.json",
    "mv_v1_q25.txt",
    "mv_v1_q50.txt",
    "mv_v1_q75.txt",
]


@app.get("/api/models/{model_name}/download")
async def download_model(
    model_name: str,
    user: str = Security(verify_auth),
):
    """
    Download a model as a ZIP file.

    Returns all files in the model directory as a ZIP archive.
    """
    # Validate model name format
    if not re.match(r"^[a-z0-9_]+$", model_name):
        raise HTTPException(status_code=400, detail="Invalid model name. Use lowercase alphanumeric and underscores only.")

    model_dir = Path("models") / model_name
    if not model_dir.exists():
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    # Create ZIP in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in model_dir.iterdir():
            if file_path.is_file():
                zf.write(file_path, file_path.name)

    zip_buffer.seek(0)

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={model_name}.zip"}
    )


@app.post("/api/models/{model_name}", response_model=ModelUploadResponse)
async def upload_model(
    model_name: str,
    file: UploadFile = File(...),
    user: str = Security(verify_auth),
):
    """
    Upload a new model from a ZIP file.

    The ZIP must contain required model files:
    - calibration_offsets.json
    - feature_cols.json
    - mv_v1_q25.txt, mv_v1_q50.txt, mv_v1_q75.txt

    Optional files (feature_importance_*.json, training_report.txt) are also accepted.
    """
    # Validate model name format
    if not re.match(r"^[a-z0-9_]+$", model_name):
        raise HTTPException(status_code=400, detail="Invalid model name. Use lowercase alphanumeric and underscores only.")

    # Check if model already exists
    model_dir = Path("models") / model_name
    if model_dir.exists():
        raise HTTPException(status_code=409, detail=f"Model '{model_name}' already exists. Delete it first to replace.")

    # Validate file is a ZIP
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="File must be a ZIP archive")

    # Read and validate ZIP contents
    try:
        content = await file.read()
        zip_buffer = io.BytesIO(content)

        with zipfile.ZipFile(zip_buffer, "r") as zf:
            file_names = zf.namelist()

            # Check for required files
            missing = [f for f in MODEL_REQUIRED_FILES if f not in file_names]
            if missing:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required files: {', '.join(missing)}"
                )

            # Extract to model directory
            model_dir.mkdir(parents=True, exist_ok=True)
            zf.extractall(model_dir)

    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid ZIP file")

    # Calculate total size
    total_size = sum(f.stat().st_size for f in model_dir.iterdir() if f.is_file())
    extracted_files = [f.name for f in model_dir.iterdir() if f.is_file()]

    log_activity(user, "POST", f"/api/models/{model_name}", f"Uploaded model {model_name}", category="model")

    return ModelUploadResponse(
        model_name=model_name,
        files=extracted_files,
        size_bytes=total_size,
        message=f"Model '{model_name}' uploaded successfully"
    )


@app.delete("/api/models/{model_name}", response_model=ModelDeleteResponse)
async def delete_model(
    model_name: str,
    user: str = Security(verify_auth),
):
    """
    Delete a model.

    Cannot delete models that are currently active (referenced in config).
    """
    # Validate model name format
    if not re.match(r"^[a-z0-9_]+$", model_name):
        raise HTTPException(status_code=400, detail="Invalid model name. Use lowercase alphanumeric and underscores only.")

    model_dir = Path("models") / model_name
    if not model_dir.exists():
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    # Check if model is active for any region
    active_regions = []
    for country, region in config.app.regions.items():
        if region.model == model_name:
            active_regions.append(country)

    if active_regions:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot delete model '{model_name}' - it is active for: {', '.join(active_regions)}"
        )

    # Delete the model directory
    shutil.rmtree(model_dir)

    log_activity(user, "DELETE", f"/api/models/{model_name}", f"Deleted model {model_name}", category="model")

    return ModelDeleteResponse(
        model_name=model_name,
        message=f"Model '{model_name}' deleted successfully"
    )


# ============================================================================
# CONFIG ENDPOINTS
# ============================================================================


@app.get("/api/config", response_model=FullConfigResponse)
async def get_full_config(
    user: str = Security(verify_auth),
):
    """Get full configuration including all regions."""
    regions = {}
    for country, region in config.app.regions.items():
        regions[country] = RegionConfig(
            model=region.model,
            currency_id=region.currency_id,
            hist_countries=region.hist_countries,
            elite_scaling=EliteScalingConfig(
                threshold=region.elite_scaling.threshold,
                base_offset=region.elite_scaling.base_offset,
                scaling_factor=region.elite_scaling.scaling_factor,
            ),
            confidence_tiers=ConfidenceTiersConfig(
                close_threshold=region.confidence_tiers.close_threshold,
                extreme_threshold=region.confidence_tiers.extreme_threshold,
            ),
            sire_sample_min_count=region.sire_sample_min_count,
        )

    return FullConfigResponse(
        year_start=config.app.year_start,
        year_end=config.app.year_end,
        model_test_last_years=config.app.model_test_last_years,
        sale_history_years=config.app.sale_history_years,
        audit_user_id=config.app.audit_user_id,
        regions=regions,
    )


# Years endpoints - must be defined before {country} to avoid route conflict
@app.get("/api/config/years", response_model=YearsConfigResponse)
async def get_years_config(
    user: str = Security(verify_auth),
):
    """Get the year range for training/scoring. year_end uses current year if not set."""
    return YearsConfigResponse(
        year_start=config.app.year_start,
        year_end=config.app.get_year_end(),
    )


@app.put("/api/config/years", response_model=YearsConfigResponse)
async def set_years_config(
    year_start: int = Query(..., ge=2000, le=2100, description="Start year"),
    year_end: Optional[int] = Query(default=None, ge=2000, le=2100, description="End year (null = current year)"),
    user: str = Security(verify_auth),
):
    """Set the year range for training/scoring. Set year_end to null to use current year."""
    effective_year_end = year_end if year_end is not None else config.app.get_year_end()
    if year_start > effective_year_end:
        raise HTTPException(status_code=400, detail="year_start must be <= year_end")

    data = load_config_json()
    data["year_start"] = year_start
    data["year_end"] = year_end  # Can be null
    save_config_json(data)

    log_activity(user, "PUT", "/api/config/years", f"Set years {year_start}-{effective_year_end}", category="config")

    return YearsConfigResponse(year_start=year_start, year_end=effective_year_end)


@app.get("/api/config/test-years", response_model=TestYearsConfigResponse)
async def get_test_years_config(
    user: str = Security(verify_auth),
):
    """Get model_test_last_years - number of years to hold out for testing."""
    return TestYearsConfigResponse(model_test_last_years=config.app.model_test_last_years)


@app.put("/api/config/test-years", response_model=TestYearsConfigResponse)
async def set_test_years_config(
    model_test_last_years: int = Query(..., ge=1, le=10, description="Number of years to hold out for testing"),
    user: str = Security(verify_auth),
):
    """Set model_test_last_years - number of years to hold out for testing."""
    data = load_config_json()
    data["model_test_last_years"] = model_test_last_years
    save_config_json(data)

    log_activity(user, "PUT", "/api/config/test-years", f"Set test years to {model_test_last_years}", category="config")

    return TestYearsConfigResponse(model_test_last_years=model_test_last_years)


@app.get("/api/config/sale-history-years", response_model=SaleHistoryYearsConfigResponse)
async def get_sale_history_years_config(
    user: str = Security(verify_auth),
):
    """Get sale_history_years - number of years of history to include in sale detail."""
    return SaleHistoryYearsConfigResponse(sale_history_years=config.app.sale_history_years)


@app.put("/api/config/sale-history-years", response_model=SaleHistoryYearsConfigResponse)
async def set_sale_history_years_config(
    sale_history_years: int = Query(..., ge=0, le=20, description="Number of years of history (0 to disable)"),
    user: str = Security(verify_auth),
):
    """Set sale_history_years - number of years of history to include in sale detail."""
    data = load_config_json()
    data["sale_history_years"] = sale_history_years
    save_config_json(data)

    log_activity(user, "PUT", "/api/config/sale-history-years", f"Set sale history years to {sale_history_years}", category="config")

    return SaleHistoryYearsConfigResponse(sale_history_years=sale_history_years)


# Region endpoints - {country} parameter routes must come after specific routes
@app.get("/api/config/{country}", response_model=RegionConfig)
async def get_region_config(
    country: str,
    user: str = Security(verify_auth),
):
    """Get configuration for a specific region."""
    country = country.upper()
    try:
        region = config.app.get_region(country)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Region {country} not found")

    return RegionConfig(
        model=region.model,
        currency_id=region.currency_id,
        hist_countries=region.hist_countries,
        elite_scaling=EliteScalingConfig(
            threshold=region.elite_scaling.threshold,
            base_offset=region.elite_scaling.base_offset,
            scaling_factor=region.elite_scaling.scaling_factor,
        ),
        confidence_tiers=ConfidenceTiersConfig(
            close_threshold=region.confidence_tiers.close_threshold,
            extreme_threshold=region.confidence_tiers.extreme_threshold,
        ),
        sire_sample_min_count=region.sire_sample_min_count,
    )


@app.post("/api/config/{country}", response_model=RegionConfig)
async def update_region_config(
    country: str,
    updates: dict,
    user: str = Security(verify_auth),
):
    """
    Partial or full update for an existing region.

    Supports nested partial updates, e.g.:
    - {"model": "aus_v5"} - update just the model
    - {"elite_scaling": {"threshold": 600000}} - update just the threshold
    """
    country = country.upper()

    data = load_config_json()
    if country not in data.get("regions", {}):
        raise HTTPException(status_code=404, detail=f"Region {country} not found")

    # Deep merge updates into existing config
    existing = data["regions"][country]
    data["regions"][country] = deep_merge(existing, updates)
    save_config_json(data)

    log_activity(user, "POST", f"/api/config/{country}", f"Updated config for {country}", category="config")

    # Return updated config
    region = config.app.get_region(country)
    return RegionConfig(
        model=region.model,
        currency_id=region.currency_id,
        hist_countries=region.hist_countries,
        elite_scaling=EliteScalingConfig(
            threshold=region.elite_scaling.threshold,
            base_offset=region.elite_scaling.base_offset,
            scaling_factor=region.elite_scaling.scaling_factor,
        ),
        confidence_tiers=ConfidenceTiersConfig(
            close_threshold=region.confidence_tiers.close_threshold,
            extreme_threshold=region.confidence_tiers.extreme_threshold,
        ),
        sire_sample_min_count=region.sire_sample_min_count,
    )


@app.put("/api/config/{country}", response_model=RegionConfig)
async def create_region_config(
    country: str,
    region_config: RegionConfig,
    user: str = Security(verify_auth),
):
    """
    Add a new region (full config required).

    PUT creates if not exists, requires complete configuration.
    """
    country = country.upper()

    data = load_config_json()
    data.setdefault("regions", {})[country] = {
        "model": region_config.model,
        "currency_id": region_config.currency_id,
        "hist_countries": region_config.hist_countries,
        "elite_scaling": {
            "threshold": region_config.elite_scaling.threshold,
            "base_offset": region_config.elite_scaling.base_offset,
            "scaling_factor": region_config.elite_scaling.scaling_factor,
        },
        "confidence_tiers": {
            "close_threshold": region_config.confidence_tiers.close_threshold,
            "extreme_threshold": region_config.confidence_tiers.extreme_threshold,
        },
        "sire_sample_min_count": region_config.sire_sample_min_count,
    }
    save_config_json(data)

    log_activity(user, "PUT", f"/api/config/{country}", f"Created region {country}", category="config")

    return region_config


@app.delete("/api/config/{country}")
async def delete_region_config(
    country: str,
    user: str = Security(verify_auth),
):
    """Remove a region from configuration."""
    country = country.upper()

    data = load_config_json()
    if country not in data.get("regions", {}):
        raise HTTPException(status_code=404, detail=f"Region {country} not found")

    del data["regions"][country]
    save_config_json(data)

    log_activity(user, "DELETE", f"/api/config/{country}", f"Deleted region {country}", category="config")

    return {"message": f"Region {country} removed"}


# ============================================================================
# ACTIVITY LOG ENDPOINT
# ============================================================================


@app.get("/api/activity", response_model=ActivityResponse)
async def get_activity_log(
    limit: int = Query(default=50, ge=1, le=500, description="Number of recent entries to return"),
    category: Optional[str] = Query(default=None, description="Filter: score, train, model, config, auth"),
    status: Optional[str] = Query(default=None, description="Filter: success, error"),
    user: str = Security(verify_auth),
):
    """Get recent activity log entries, optionally filtered by category and/or status."""
    if not ACTIVITY_LOG_PATH.exists():
        return ActivityResponse(entries=[], total_in_file=0)

    with open(ACTIVITY_LOG_PATH) as f:
        all_lines = f.readlines()

    total = len(all_lines)

    # Parse all entries
    parsed = []
    for line in all_lines:
        line = line.strip()
        if line:
            try:
                parsed.append(ActivityEntry(**json.loads(line)))
            except (json.JSONDecodeError, ValueError):
                pass

    # Apply filters
    if category:
        parsed = [e for e in parsed if e.category == category]
    if status:
        parsed = [e for e in parsed if e.status == status]

    # Take last N entries
    entries = list(deque(parsed, maxlen=limit))

    return ActivityResponse(entries=entries, total_in_file=total)


# ============================================================================
# UI
# ============================================================================


@app.get("/", response_class=FileResponse)
async def serve_ui():
    """Serve the local UI."""
    return FileResponse("ui.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
