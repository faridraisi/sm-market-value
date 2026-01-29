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

from fastapi import FastAPI, HTTPException, Security, Query, BackgroundTasks
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from dotenv import load_dotenv

from src import rebuild_sale_features, get_connection, fetch_sale_country, MODEL_VERSION
from src.score_sale import load_models, get_model_dir, score_lots, upsert_to_database
from src.train_model import train_model
from src.config import config, CONFIG_PATH

load_dotenv()

app = FastAPI(title="Market Value API", version=MODEL_VERSION)

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
    mv_confidence_tier: str


class ScoreResponse(BaseModel):
    sale_id: int
    country_code: str
    total_lots: int
    lots: list[LotScore]
    output_written: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    database_connected: bool


class TrainResponse(BaseModel):
    message: str
    country: str
    version: str
    output_dir: str


class ModelInfo(BaseModel):
    version: str
    directory: str
    is_active: bool
    training_date: Optional[str] = None
    test_mae: Optional[float] = None
    test_r2: Optional[float] = None
    mape: Optional[float] = None
    top_features: Optional[list[str]] = None


class ModelsListResponse(BaseModel):
    country: str
    active_model: str
    models: list[ModelInfo]


class ModelConfigResponse(BaseModel):
    country: str
    model: str


class YearsConfigResponse(BaseModel):
    year_start: int
    year_end: int


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
            mv_confidence_tier=row["mv_confidence_tier"],
        )
        for _, row in results_df.iterrows()
    ]

    return ScoreResponse(
        sale_id=sale_id,
        country_code=country_code,
        total_lots=len(lots),
        lots=lots,
        output_written=output if output != "none" else None,
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
    """Parse training_report.txt for key metrics."""
    if not report_path.exists():
        return {}

    content = report_path.read_text()
    result = {}

    # Extract date
    date_match = re.search(r"Generated:\s*(\d{4}-\d{2}-\d{2})", content)
    if date_match:
        result["training_date"] = date_match.group(1)

    # Extract test metrics
    mae_match = re.search(r"Test MAE:\s*([\d.]+)", content)
    if mae_match:
        result["test_mae"] = float(mae_match.group(1))

    r2_match = re.search(r"Test R²:\s*([\d.]+)", content)
    if r2_match:
        result["test_r2"] = float(r2_match.group(1))

    mape_match = re.search(r"Dollar MAPE:\s*([\d.]+)%", content)
    if mape_match:
        result["mape"] = float(mape_match.group(1))

    return result


def parse_feature_importance(importance_path: Path, top_n: int = 5) -> list[str]:
    """Get top N features from feature_importance JSON."""
    if not importance_path.exists():
        return []

    with open(importance_path) as f:
        data = json.load(f)

    avg_importance = data.get("average", {})
    sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
    return [f[0] for f in sorted_features[:top_n]]


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
            top_features = []
            if importance_files:
                top_features = parse_feature_importance(importance_files[0])

            model_infos.append(ModelInfo(
                version=version,
                directory=entry.name,
                is_active=is_active,
                training_date=report_data.get("training_date"),
                test_mae=report_data.get("test_mae"),
                test_r2=report_data.get("test_r2"),
                mape=report_data.get("mape"),
                top_features=top_features if top_features else None,
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


@app.get("/api/config/models/{country}", response_model=ModelConfigResponse)
async def get_model_config(
    country: str,
    api_key: str = Security(verify_api_key),
):
    """Get the active model for a country."""
    country = country.lower()
    if country not in ["aus", "nzl", "usa"]:
        raise HTTPException(status_code=400, detail=f"Invalid country: {country}. Must be aus, nzl, or usa.")

    model = config.app.models.get(country, country)
    return ModelConfigResponse(country=country.upper(), model=model)


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
    """Get the year range for training/scoring."""
    return YearsConfigResponse(
        year_start=config.app.year_start,
        year_end=config.app.year_end,
    )


@app.put("/api/config/years", response_model=YearsConfigResponse)
async def set_years_config(
    year_start: int = Query(..., ge=2000, le=2100, description="Start year"),
    year_end: int = Query(..., ge=2000, le=2100, description="End year"),
    api_key: str = Security(verify_api_key),
):
    """Set the year range for training/scoring."""
    if year_start > year_end:
        raise HTTPException(status_code=400, detail="year_start must be <= year_end")

    data = load_config_json()
    data["year_start"] = year_start
    data["year_end"] = year_end
    save_config_json(data)

    return YearsConfigResponse(year_start=year_start, year_end=year_end)


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
