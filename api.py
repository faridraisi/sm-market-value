#!/usr/bin/env python3
"""
Market Value API - Score yearling lots.
Usage: uvicorn api:app --host 0.0.0.0 --port 8000
"""

import os
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Security, Query
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from dotenv import load_dotenv

from src import rebuild_sale_features, get_connection, fetch_sale_country, score_sale_lots, MODEL_VERSION

load_dotenv()

app = FastAPI(title="Market Value API", version=MODEL_VERSION)

# Auth
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)


async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=403, detail="Invalid API key")


# Response models
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


# Endpoints
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

    features_df = rebuild_sale_features(sale_id, export_csv=(output == "csv"))
    if features_df.empty:
        raise HTTPException(status_code=404, detail=f"No lots found for sale {sale_id}")

    results_df = score_sale_lots(sale_id, features_df, output)

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
