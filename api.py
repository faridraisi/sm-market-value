#!/usr/bin/env python3
"""
Market Value API - Score yearling lots and manage configuration.
Usage: uvicorn api:app --host 0.0.0.0 --port 8000
"""

import json
import os
import re
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Security, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
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
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)


async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=403, detail="Invalid API key")


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


class ModelConfigResponse(BaseModel):
    country: str
    model: str


class AllModelsConfigResponse(BaseModel):
    models: dict[str, str]


class YearsConfigResponse(BaseModel):
    year_start: int
    year_end: int


class TestYearsConfigResponse(BaseModel):
    model_test_last_years: int


class HistCountriesResponse(BaseModel):
    country: str
    hist_countries: list[str]


class AllHistCountriesResponse(BaseModel):
    hist_countries: dict[str, list[str]]


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


@app.post("/api/score/{sale_id}", response_model=ScoreResponse)
async def score_sale(
    sale_id: int,
    output: Literal["none", "csv", "db"] = Query(default="none"),
    api_key: str = Security(verify_api_key),
):
    """Score all lots for a sale."""
    try:
        conn = get_connection()
        country_code = fetch_sale_country(conn, sale_id)
        conn.close()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Rebuild features
    features_df = rebuild_sale_features(sale_id, export_csv=(output == "csv"))
    if features_df.empty:
        raise HTTPException(status_code=404, detail=f"No lots found for sale {sale_id}")

    # Load models and score
    model_dir = get_model_dir(country_code)
    models, offsets, feature_cols = load_models(model_dir)
    results_df = score_lots(features_df, models, offsets, feature_cols)

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
    api_key: str = Security(verify_api_key),
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
    api_key: str = Security(verify_api_key),
):
    """
    Score sale and compare with existing database values.

    Returns new predictions alongside existing values from tblHorseAnalytics,
    with calculated deltas. Read-only operation - does not modify database.
    """
    # Get country and validate sale exists
    try:
        conn = get_connection()
        country_code = fetch_sale_country(conn, sale_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Rebuild features
    features_df = rebuild_sale_features(sale_id, export_csv=False)
    if features_df.empty:
        conn.close()
        raise HTTPException(status_code=404, detail=f"No lots found for sale {sale_id}")

    # Load models and score
    model_dir = get_model_dir(country_code)
    models, offsets, feature_cols = load_models(model_dir)
    results_df = score_lots(features_df, models, offsets, feature_cols)

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


def run_training(country: str, version: str):
    """Background task to run model training."""
    train_model(country=country, version=version)


@app.post("/api/train/{country}", response_model=TrainResponse)
async def train_model_endpoint(
    country: str,
    background_tasks: BackgroundTasks,
    version: Optional[str] = Query(default=None, description="Force specific version (e.g., v5)"),
    api_key: str = Security(verify_api_key),
):
    """
    Train a new model for a specific country.

    Training runs in the background. Check models list endpoint for completion.
    """
    country = country.lower()
    if country not in ["aus", "nzl", "usa"]:
        raise HTTPException(status_code=400, detail=f"Invalid country: {country}. Must be aus, nzl, or usa.")

    # Determine version
    if version is None:
        from src.train_model import get_next_version
        version = get_next_version(country)

    output_dir = f"models/{country}_{version}"

    # Start training in background
    background_tasks.add_task(run_training, country, version)

    return TrainResponse(
        message=f"Training started for {country.upper()}. Check GET /api/models/{country} for completion.",
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
    api_key: str = Security(verify_api_key),
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


# ============================================================================
# CONFIG ENDPOINTS - MODELS
# ============================================================================


@app.get("/api/config/models", response_model=AllModelsConfigResponse)
async def get_all_model_configs(
    api_key: str = Security(verify_api_key),
):
    """Get active models for all countries."""
    return AllModelsConfigResponse(models=config.app.models)


@app.put("/api/config/models/{country}", response_model=ModelConfigResponse)
async def set_model_config(
    country: str,
    model: str = Query(..., description="Model directory name (e.g., 'aus_v5')"),
    api_key: str = Security(verify_api_key),
):
    """
    Set the active model for a country.

    The model directory must exist in the models/ folder.
    """
    country = country.lower()
    if country not in ["aus", "nzl", "usa"]:
        raise HTTPException(status_code=400, detail=f"Invalid country: {country}. Must be aus, nzl, or usa.")

    # Verify model directory exists
    model_path = Path("models") / model
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model directory not found: {model}")

    # Update config
    data = load_config_json()
    data["models"][country] = model
    save_config_json(data)

    return ModelConfigResponse(country=country.upper(), model=model)


# ============================================================================
# CONFIG ENDPOINTS - YEARS
# ============================================================================


@app.get("/api/config/years", response_model=YearsConfigResponse)
async def get_years_config(
    api_key: str = Security(verify_api_key),
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
    api_key: str = Security(verify_api_key),
):
    """Set the year range for training/scoring. Set year_end to null to use current year."""
    effective_year_end = year_end if year_end is not None else config.app.get_year_end()
    if year_start > effective_year_end:
        raise HTTPException(status_code=400, detail="year_start must be <= year_end")

    data = load_config_json()
    data["year_start"] = year_start
    data["year_end"] = year_end  # Can be null
    save_config_json(data)

    return YearsConfigResponse(year_start=year_start, year_end=effective_year_end)


@app.get("/api/config/test-years", response_model=TestYearsConfigResponse)
async def get_test_years_config(
    api_key: str = Security(verify_api_key),
):
    """Get model_test_last_years - number of years to hold out for testing."""
    return TestYearsConfigResponse(model_test_last_years=config.app.model_test_last_years)


@app.put("/api/config/test-years", response_model=TestYearsConfigResponse)
async def set_test_years_config(
    model_test_last_years: int = Query(..., ge=1, le=10, description="Number of years to hold out for testing"),
    api_key: str = Security(verify_api_key),
):
    """Set model_test_last_years - number of years to hold out for testing."""
    data = load_config_json()
    data["model_test_last_years"] = model_test_last_years
    save_config_json(data)

    return TestYearsConfigResponse(model_test_last_years=model_test_last_years)


# ============================================================================
# CONFIG ENDPOINTS - HIST COUNTRIES
# ============================================================================


@app.get("/api/config/hist-countries", response_model=AllHistCountriesResponse)
async def get_all_hist_countries(
    api_key: str = Security(verify_api_key),
):
    """Get all historical country mappings."""
    return AllHistCountriesResponse(hist_countries=config.app.hist_countries)


@app.get("/api/config/hist-countries/{country}", response_model=HistCountriesResponse)
async def get_hist_countries_config(
    country: str,
    api_key: str = Security(verify_api_key),
):
    """Get historical countries for lookback for a specific country."""
    country = country.upper()
    hist = config.app.get_hist_countries(country)
    return HistCountriesResponse(country=country, hist_countries=hist)


@app.put("/api/config/hist-countries/{country}", response_model=HistCountriesResponse)
async def set_hist_countries_config(
    country: str,
    hist_countries: list[str] = Query(..., description="List of country codes for historical lookback"),
    api_key: str = Security(verify_api_key),
):
    """
    Set historical countries for lookback.

    Example: For NZL sales, include both NZL and AUS historical data.
    """
    country = country.upper()
    hist_countries = [c.upper() for c in hist_countries]

    if not hist_countries:
        raise HTTPException(status_code=400, detail="hist_countries cannot be empty")

    data = load_config_json()
    data["hist_countries"][country] = hist_countries
    save_config_json(data)

    return HistCountriesResponse(country=country, hist_countries=hist_countries)


@app.delete("/api/config/hist-countries/{country}")
async def delete_hist_countries_config(
    country: str,
    api_key: str = Security(verify_api_key),
):
    """
    Remove historical country override.

    After deletion, the country will use only its own historical data.
    """
    country = country.upper()

    data = load_config_json()
    if country in data["hist_countries"]:
        del data["hist_countries"][country]
        save_config_json(data)
        return {"message": f"Removed hist_countries override for {country}"}

    return {"message": f"No override existed for {country}"}


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
