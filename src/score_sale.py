#!/usr/bin/env python3
"""
Market Value Model — Score Sale Pipeline
=========================================
Complete pipeline: rebuild features + score lots.

All database access is read-only (except optional --output db).

Usage:
    python src/score_sale.py --sale-id 2094                # Output to CSV (default)
    python src/score_sale.py --sale-id 2094 --output csv   # Output to CSV
    python src/score_sale.py --sale-id 2094 --output db    # Output to database
"""

import argparse
import json
import os
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyodbc
from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder

# Import feature rebuild functions
try:
    from src.run_rebuild import (
        get_connection,
        fetch_sale_country,
        get_hist_countries,
        fetch_base_lots,
        fetch_hist_lots,
        build_features,
    )
except ModuleNotFoundError:
    from run_rebuild import (
        get_connection,
        fetch_sale_country,
        get_hist_countries,
        fetch_base_lots,
        fetch_hist_lots,
        build_features,
    )

MODEL_VERSION = "v2.1"

# Currency mapping by country code
CURRENCY_MAP = {"AUS": 1, "NZL": 6, "USA": 7}


# ============================================================================
# MODEL LOADING
# ============================================================================


def get_model_dir(country_code: str) -> str:
    """
    Get model directory from .env config.

    Looks for {COUNTRY_CODE}_MODEL in .env (e.g., NZL_MODEL=aus).
    Falls back to the country code in lowercase if not configured.
    """
    load_dotenv()
    env_key = f"{country_code}_MODEL"
    model_name = os.getenv(env_key, country_code.lower())
    return f"models/{model_name}"


def load_models(model_dir: str):
    """Load trained models and metadata from specified directory."""

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    models = {
        "q25": lgb.Booster(model_file=f"{model_dir}/mv_v1_q25.txt"),
        "q50": lgb.Booster(model_file=f"{model_dir}/mv_v1_q50.txt"),
        "q75": lgb.Booster(model_file=f"{model_dir}/mv_v1_q75.txt"),
    }

    with open(f"{model_dir}/calibration_offsets.json", "r") as f:
        offsets = json.load(f)

    with open(f"{model_dir}/feature_cols.json", "r") as f:
        feature_cols = json.load(f)

    return models, offsets, feature_cols


# ============================================================================
# ELITE SCALING LOGIC (v2.1)
# ============================================================================


def apply_elite_scaling(raw_prices: np.ndarray, offsets: dict) -> np.ndarray:
    """
    Apply elite scaling for predictions above threshold.

    Formula: offset = base_offset + (ln(raw_p50) - ln(threshold)) * scaling_factor
    Only applied when raw_p50 >= threshold.
    """
    elite_config = offsets.get("elite_scaling", {})
    threshold = elite_config.get("threshold", 300000)
    base_offset = elite_config.get("base_offset", 0.5)
    scaling_factor = elite_config.get("scaling_factor", 1.2)

    offsets_array = np.zeros_like(raw_prices)
    mask = raw_prices >= threshold

    if mask.any():
        offsets_array[mask] = (
            base_offset
            + (np.log(raw_prices[mask]) - np.log(threshold)) * scaling_factor
        )

    return offsets_array


def apply_adjustments(
    session_median: np.ndarray,
    pred_q25: np.ndarray,
    pred_q50: np.ndarray,
    pred_q75: np.ndarray,
    offsets: dict,
) -> tuple:
    """Apply all adjustments: elite scaling + base calibration."""
    raw_prices = session_median * np.exp(pred_q50)
    elite_offsets = apply_elite_scaling(raw_prices, offsets)

    adj_q50 = pred_q50 + elite_offsets
    adj_q25 = pred_q25 + offsets["offset_p25"] + elite_offsets
    adj_q75 = pred_q75 + offsets["offset_p75"] + elite_offsets

    return adj_q25, adj_q50, adj_q75, raw_prices


# ============================================================================
# CONFIDENCE TIER (v2.1)
# ============================================================================


def calculate_confidence_tier(df: pd.DataFrame, raw_prices: np.ndarray) -> pd.Series:
    """
    Calculate confidence tier based on data flags AND predicted price tier.

    Logic:
    - Predicted >= $300k: LOW
    - Predicted $200k-$300k + any flag: LOW
    - Predicted $200k-$300k + no flags: MEDIUM
    - Predicted < $200k + 2+ flags: LOW
    - Predicted < $200k + 1 flag: MEDIUM
    - Predicted < $200k + no flags: HIGH
    """
    flags = pd.DataFrame(index=df.index)
    flags["sire_flag"] = (
        df.get("sire_sample_flag_36m", pd.Series(0, index=df.index)).fillna(0).astype(int)
    )
    flags["dam_flag"] = (
        df.get("dam_first_foal_flag", pd.Series(0, index=df.index)).fillna(0).astype(int)
    )
    flags["vendor_flag"] = (
        df.get("vendor_first_seen_flag", pd.Series(0, index=df.index)).fillna(0).astype(int)
    )
    flag_count = flags.sum(axis=1)

    tier = pd.Series("high", index=df.index)

    tier[raw_prices >= 300000] = "low"

    mid_high_mask = (raw_prices >= 200000) & (raw_prices < 300000)
    tier[mid_high_mask & (flag_count >= 1)] = "low"
    tier[mid_high_mask & (flag_count == 0)] = "medium"

    lower_mask = raw_prices < 200000
    tier[lower_mask & (flag_count >= 2)] = "low"
    tier[lower_mask & (flag_count == 1)] = "medium"
    tier[lower_mask & (flag_count == 0)] = "high"

    return tier


# ============================================================================
# SCORING
# ============================================================================


def score_lots(
    df: pd.DataFrame, models: dict, offsets: dict, feature_cols: list
) -> pd.DataFrame:
    """Score lots and return predictions."""

    df = df.copy()

    # Log transform price features
    price_cols = [
        "session_median_price",
        "sire_median_price_36m",
        "sire_median_price_12m",
        "dam_progeny_median_price",
        "vendor_median_price_36m",
    ]
    for col in price_cols:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col].fillna(0))

    # Encode categoricals
    for col in ["sex", "vendor_volume_bucket", "sale_company"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")
            le = LabelEncoder()
            df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))

    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    X = df[feature_cols].fillna(0)

    # Ensure numeric types for LightGBM
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    # Predict (raw log-space)
    pred_q25 = models["q25"].predict(X)
    pred_q50 = models["q50"].predict(X)
    pred_q75 = models["q75"].predict(X)

    session_median = df["session_median_price"].values

    # Apply adjustments
    adj_q25, adj_q50, adj_q75, raw_prices = apply_adjustments(
        session_median, pred_q25, pred_q50, pred_q75, offsets
    )

    # Build results DataFrame
    results = pd.DataFrame(
        {
            "lot_id": df["lot_id"],
            "horseId": df["horseId"] if "horseId" in df.columns else None,
            "salesId": df["salesId"],
            "lot_number": df["lot_number"] if "lot_number" in df.columns else None,
            "book_number": df["book_number"] if "book_number" in df.columns else None,
            "sex": df["sex"] if "sex" in df.columns else None,
            "horse_name": df["horse_name"] if "horse_name" in df.columns else None,
            "sire_name": df["sire_name"] if "sire_name" in df.columns else None,
            "session_median_price": session_median,
            "mv_expected_index": np.exp(adj_q50),
            "mv_low_index": np.exp(adj_q25),
            "mv_high_index": np.exp(adj_q75),
            "mv_expected_price": np.round(session_median * np.exp(adj_q50), -2),
            "mv_low_price": np.round(session_median * np.exp(adj_q25), -2),
            "mv_high_price": np.round(session_median * np.exp(adj_q75), -2),
            "mv_raw_price": np.round(raw_prices, -2),
            "mv_confidence_tier": calculate_confidence_tier(df, raw_prices),
            "mv_model_version": MODEL_VERSION,
            "mv_generated_at": datetime.now(timezone.utc),
        }
    )

    # Ensure low <= expected <= high
    results["mv_low_price"] = np.minimum(
        results["mv_low_price"], results["mv_expected_price"]
    )
    results["mv_high_price"] = np.maximum(
        results["mv_high_price"], results["mv_expected_price"]
    )
    results["mv_low_price"] = np.maximum(results["mv_low_price"], 0)

    return results


# ============================================================================
# DATABASE OUTPUT
# ============================================================================


def upsert_to_database(results: pd.DataFrame, country_code: str):
    """Insert/update market value predictions to tblHorseAnalytics."""
    conn = get_connection()
    cursor = conn.cursor()

    currency_id = CURRENCY_MAP.get(country_code, 1)
    modified_by = int(os.getenv("AUDIT_USER_ID", 2))

    inserted = 0
    updated = 0

    for _, row in results.iterrows():
        horse_id = int(row["horseId"])
        sales_id = int(row["salesId"])

        cursor.execute(
            "SELECT id FROM tblHorseAnalytics WHERE horseId = ? AND salesId = ?",
            (horse_id, sales_id),
        )
        existing = cursor.fetchone()

        if existing:
            cursor.execute(
                """
                UPDATE tblHorseAnalytics SET
                    marketValue = ?, marketValueLow = ?, marketValueHigh = ?,
                    marketValueMultiplier = ?, marketValueConfidence = ?,
                    sessionMedianPrice = ?, currencyId = ?,
                    modifiedBy = ?, modifiedOn = GETDATE()
                WHERE horseId = ? AND salesId = ?
                """,
                (
                    float(row["mv_expected_price"]),
                    float(row["mv_low_price"]),
                    float(row["mv_high_price"]),
                    float(row["mv_expected_index"]),
                    row["mv_confidence_tier"],
                    float(row["session_median_price"]),
                    currency_id,
                    modified_by,
                    horse_id,
                    sales_id,
                ),
            )
            updated += 1
        else:
            cursor.execute(
                """
                INSERT INTO tblHorseAnalytics
                    (horseId, salesId, marketValue, marketValueLow, marketValueHigh,
                     marketValueMultiplier, marketValueConfidence, sessionMedianPrice,
                     currencyId, createdBy, createdOn, modifiedBy, modifiedOn)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, GETDATE(), ?, GETDATE())
                """,
                (
                    horse_id,
                    sales_id,
                    float(row["mv_expected_price"]),
                    float(row["mv_low_price"]),
                    float(row["mv_high_price"]),
                    float(row["mv_expected_index"]),
                    row["mv_confidence_tier"],
                    float(row["session_median_price"]),
                    currency_id,
                    modified_by,
                    modified_by,
                ),
            )
            inserted += 1

    conn.commit()
    cursor.close()
    conn.close()

    print(f"\nDatabase update complete:")
    print(f"  Inserted: {inserted}")
    print(f"  Updated:  {updated}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================


def score_sale(sale_id: int, output: str = "csv") -> pd.DataFrame:
    """
    Complete pipeline: rebuild features + score lots.

    Args:
        sale_id: Sale ID to process
        output: "csv", "db", or "none"

    Returns:
        DataFrame with scored results
    """
    print("=" * 60)
    print(f"MARKET VALUE MODEL — SCORE SALE {sale_id}")
    print("=" * 60)

    # Step 1: Connect and get sale info
    print("\nConnecting to database...")
    conn = get_connection()
    country_code = fetch_sale_country(conn, sale_id)
    hist_countries = get_hist_countries(country_code)
    print(f"  Sale country: {country_code}")
    print(f"  Historical countries: {hist_countries}")

    # Step 2: Fetch data
    print(f"\nFetching base lots for sale {sale_id}...")
    base_lots = fetch_base_lots(conn, sale_id)
    print(f"  Found {len(base_lots)} lots")

    if base_lots.empty:
        print("No lots found for this sale.")
        conn.close()
        return pd.DataFrame()

    print(f"Fetching historical lots for {hist_countries}...")
    hist_lots = fetch_hist_lots(conn, hist_countries)
    print(f"  Found {len(hist_lots)} historical lots")

    # Step 3: Build features
    print("\nBuilding features...")
    features = build_features(base_lots, hist_lots, conn)
    conn.close()
    print(f"  Built features for {len(features)} lots")

    # Step 4: Load models
    model_dir = get_model_dir(country_code)
    print(f"\nLoading models from: {model_dir}/")
    models, offsets, feature_cols = load_models(model_dir)
    print(f"  Model version: {MODEL_VERSION}")

    # Show elite scaling config
    if "elite_scaling" in offsets:
        cfg = offsets["elite_scaling"]
        print(f"\nElite scaling (predictions >= ${cfg['threshold']:,}):")
        print(f"  Base offset: {cfg['base_offset']}")
        print(f"  Scaling factor: {cfg['scaling_factor']}")

    # Step 5: Score lots
    print("\nScoring lots...")
    results = score_lots(features, models, offsets, feature_cols)

    # Step 6: Output
    if output == "csv":
        os.makedirs("csv", exist_ok=True)
        # Save inference data
        inference_path = f"csv/sale_{sale_id}_inference.csv"
        inference_cols = [
            "lot_id", "horseId", "salesId", "lot_number", "book_number", "sex",
            "horse_name", "sire_name", "sale_company", "sale_year", "day_number",
            "session_median_price",
            "sire_sold_count_36m", "sire_total_offered_36m", "sire_clearance_rate_36m", "sire_median_price_36m",
            "sire_sold_count_12m", "sire_total_offered_12m", "sire_clearance_rate_12m", "sire_median_price_12m",
            "sire_momentum", "sire_sample_flag_36m",
            "dam_progeny_sold_count", "dam_progeny_total_offered_count", "dam_progeny_median_price", "dam_first_foal_flag",
            "vendor_sold_count_36m", "vendor_total_offered_36m", "vendor_clearance_rate_36m", "vendor_median_price_36m",
            "vendor_volume_bucket", "vendor_first_seen_flag",
        ]
        available_cols = [c for c in inference_cols if c in features.columns]
        features[available_cols].to_csv(inference_path, index=False)
        print(f"\nSaved inference data: {inference_path}")

        # Save scored results
        output_path = f"csv/sale_{sale_id}_scored.csv"
        results.to_csv(output_path, index=False)
        print(f"Saved scored results: {output_path}")

    elif output == "db":
        print(f"\nWriting to database (tblHorseAnalytics)...")
        upsert_to_database(results, country_code)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total lots: {len(results)}")
    print(f"\nMedian prices:")
    print(f"  Low:      ${results['mv_low_price'].median():,.0f}")
    print(f"  Expected: ${results['mv_expected_price'].median():,.0f}")
    print(f"  High:     ${results['mv_high_price'].median():,.0f}")
    print(f"\nConfidence tiers:")
    print(results["mv_confidence_tier"].value_counts().to_string())

    adjusted_count = (results["mv_raw_price"] >= 300000).sum()
    print(f"\nElite scaling applied to: {adjusted_count} lots ({adjusted_count/len(results)*100:.1f}%)")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Score a sale: rebuild features + run model inference."
    )
    parser.add_argument(
        "--sale-id",
        type=int,
        required=True,
        help="Sale ID to process",
    )
    parser.add_argument(
        "--output",
        choices=["csv", "db"],
        default="csv",
        help="Output to CSV file or database (default: csv)",
    )
    args = parser.parse_args()

    score_sale(args.sale_id, args.output)


if __name__ == "__main__":
    main()
