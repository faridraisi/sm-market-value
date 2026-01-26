"""
Market Value Model API
======================
FastAPI endpoint for scoring yearling lots.

Usage:
    uvicorn src.api:app --reload --port 8000

Endpoints:
    POST /api/score/{sale_id}  - Score a sale
    GET  /docs                 - Swagger documentation
"""

import os
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Security, Query, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from dotenv import load_dotenv

# Import scoring functions from score_sale.py
from src.score_sale import (
    get_db_connection,
    get_sale_info,
    get_country_config,
    load_features,
    load_models,
    score_lots,
    save_to_db,
    MODEL_VERSION,
)

# Load environment variables
load_dotenv()

# ============================================================================
# API Configuration
# ============================================================================

app = FastAPI(
    title="Market Value Scoring API",
    description="Score yearling lots for market value prediction",
    version=MODEL_VERSION,
)

API_KEY = os.getenv("API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)):
    """Validate API key from header."""
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API_KEY not configured on server")
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


# ============================================================================
# Response Models
# ============================================================================

class LotPrediction(BaseModel):
    lot_id: int
    lot_number: Optional[int] = None
    horse_name: Optional[str] = None
    sire_name: Optional[str] = None
    sex: Optional[str] = None
    mv_expected_price: float
    mv_low_price: float
    mv_high_price: float
    mv_confidence_tier: str


class ScoreResponse(BaseModel):
    sale_id: int
    sale_name: str
    country: str
    lookback_countries: List[str]
    model: str
    model_version: str
    session_median_price: float
    total_lots: int
    written_to_db: bool
    predictions: List[LotPrediction]


class ErrorResponse(BaseModel):
    detail: str


# ============================================================================
# Endpoints
# ============================================================================

@app.post(
    "/api/score/{sale_id}",
    response_model=ScoreResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid API key"},
        404: {"model": ErrorResponse, "description": "Sale not found"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
    summary="Score a sale",
    description="Score all lots in a sale and optionally write to database.",
)
async def score_sale(
    sale_id: int,
    write_to_db: bool = Query(default=True, description="Write results to tblHorseAnalytics"),
    session_median: Optional[float] = Query(default=None, description="Override session median (for pre-sale scoring)"),
    api_key: str = Depends(verify_api_key),
):
    """
    Score all yearling lots in a sale.

    - **sale_id**: Sale ID to score
    - **write_to_db**: If true, write predictions to tblHorseAnalytics
    - **session_median**: Optional override for session median price (required for pre-sale)
    """
    try:
        # Connect to database
        conn = get_db_connection()

        # Get sale info
        try:
            sale_info = get_sale_info(conn, sale_id)
        except SystemExit:
            raise HTTPException(status_code=404, detail=f"Sale {sale_id} not found")

        # Get country configuration
        country_config = get_country_config(sale_info['country_code'])

        # Load models
        try:
            models, offsets, feature_cols = load_models(country_config['model_dir'])
        except FileNotFoundError as e:
            raise HTTPException(status_code=500, detail=f"Model not found: {e}")

        # Load features
        try:
            df = load_features(conn, sale_id, country_config['lookback_countries'], session_median)
        except SystemExit:
            if session_median is None:
                raise HTTPException(
                    status_code=400,
                    detail="No sold lots found. Use session_median parameter for pre-sale scoring."
                )
            raise HTTPException(status_code=500, detail="Failed to load features")

        # Score lots
        results = score_lots(df, models, offsets, feature_cols)

        # Optionally write to database
        written = False
        if write_to_db:
            save_to_db(conn, results, sale_id, sale_info['currency_id'])
            written = True

        conn.close()

        # Build response
        predictions = []
        for _, row in results.iterrows():
            predictions.append(LotPrediction(
                lot_id=int(row['lot_id']),
                lot_number=int(row['lot_number']) if row['lot_number'] is not None else None,
                horse_name=row['horse_name'] if row['horse_name'] is not None else None,
                sire_name=row['sire_name'] if row['sire_name'] is not None else None,
                sex=row['sex'] if row['sex'] is not None else None,
                mv_expected_price=float(row['mv_expected_price']),
                mv_low_price=float(row['mv_low_price']),
                mv_high_price=float(row['mv_high_price']),
                mv_confidence_tier=str(row['mv_confidence_tier']),
            ))

        return ScoreResponse(
            sale_id=sale_id,
            sale_name=sale_info['sale_name'],
            country=sale_info['country_code'],
            lookback_countries=country_config['lookback_countries'],
            model=country_config['model_dir'],
            model_version=MODEL_VERSION,
            session_median_price=float(results['session_median_price'].iloc[0]),
            total_lots=len(results),
            written_to_db=written,
            predictions=predictions,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": MODEL_VERSION}
