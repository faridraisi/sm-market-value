#!/usr/bin/env python3
"""
Market Value Model — Training Script
=====================================
Automates retraining for a specific country with auto-versioning.

Features are computed in Python/pandas using read-only SQL queries (no database writes).

Usage:
    python src/train_model.py --country aus                          # Retrain with database export
    python src/train_model.py --country aus --csv training_data.csv  # Use existing CSV
    python src/train_model.py --country nzl --version v3             # Force specific version
"""

import argparse
import json
import os
import re
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyodbc
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Import feature computation functions from run_rebuild
try:
    from src.run_rebuild import (
        compute_sire_metrics,
        compute_dam_stats,
        compute_vendor_metrics,
    )
    from src.config import config
except ModuleNotFoundError:
    from run_rebuild import (
        compute_sire_metrics,
        compute_dam_stats,
        compute_vendor_metrics,
    )
    from config import config


# ============================================================================
# DATABASE CONNECTION
# ============================================================================


def get_connection():
    """Create and return a database connection."""
    db = config.db
    if not all([db.server, db.name, db.user, db.password]):
        raise ValueError("Missing required database credentials in .env file")

    conn_str = (
        f"DRIVER={{ODBC Driver 18 for SQL Server}};"
        f"SERVER={db.server};"
        f"DATABASE={db.name};"
        f"UID={db.user};"
        f"PWD={db.password};"
        f"TrustServerCertificate=yes"
    )
    return pyodbc.connect(conn_str)


# ============================================================================
# VERSION DETECTION
# ============================================================================


def get_next_version(country: str) -> str:
    """
    Scan models/ for existing versions, return next (v2, v3, etc.).

    Looks for directories matching pattern: {country} or {country}_v{N}
    Also supports legacy hyphen format: {country}-v{N}
    """
    models_dir = "models"
    if not os.path.exists(models_dir):
        return "v1"

    # Match both underscore (new) and hyphen (legacy) formats
    pattern = re.compile(rf"^{country}[_-]?v?(\d+)?$", re.IGNORECASE)
    max_version = 0

    for entry in os.listdir(models_dir):
        if os.path.isdir(os.path.join(models_dir, entry)):
            # Check for patterns like: aus, aus_v2, aus-v2, aus_v3
            if entry.lower() == country.lower():
                # Base directory (e.g., "aus" without version) counts as v1
                max_version = max(max_version, 1)
            elif entry.lower().startswith(country.lower()):
                # Extract version number from aus_v2, aus-v2, etc.
                suffix = entry[len(country):]
                version_match = re.match(r"[_-]v(\d+)$", suffix, re.IGNORECASE)
                if version_match:
                    version_num = int(version_match.group(1))
                    max_version = max(max_version, version_num)

    return f"v{max_version + 1}"


# ============================================================================
# DATA EXPORT (Read-only queries, features computed in Python)
# ============================================================================


def fetch_training_lots(conn, country: str) -> pd.DataFrame:
    """
    Fetch base training lots for a country (sold yearlings only).

    Uses read-only SQL queries - no writes to database.
    """
    year_start = config.app.year_start
    year_end = config.app.get_year_end()

    query = f"""
    SELECT
        LT.Id AS lot_id,
        LT.salesId,
        CAST(SL.startDate AS DATE) AS asOfDate,
        SC.salescompanyName AS sale_company,
        SL.salesName AS sale_name,
        YEAR(SL.startDate) AS sale_year,
        LT.bookNumber AS book_number,
        LT.dayNumber AS day_number,
        LT.lotNumber AS lot_number,
        LT.horseGender AS sex,
        H.sireId,
        H.damId,
        LT.vendorId,
        LT.horseId,
        CAST(LT.price AS DECIMAL(12,2)) AS hammer_price,
        CASE WHEN LT.price > 0 AND ISNULL(LT.isPassedIn,0) = 0 AND ISNULL(LT.isWithdrawn,0) = 0 THEN 1 ELSE 0 END AS isSold_int,
        CASE WHEN ISNULL(LT.isPassedIn,0) = 1 THEN 1 ELSE 0 END AS isPassedIn_int
    FROM tblSalesLot LT
    JOIN tblSales SL ON LT.salesId = SL.Id
    JOIN tblSalesCompany SC ON SL.salesCompanyId = SC.Id
    JOIN tblSalesLotType LTP ON LT.lotType = LTP.Id
    JOIN tblCountry CN ON SL.countryId = CN.id
    JOIN tblHorse H ON LT.horseId = H.id
    WHERE CN.countryCode = ?
        AND YEAR(SL.startDate) BETWEEN {year_start} AND {year_end}
        AND LTP.salesLotTypeName = 'Yearling'
        AND ISNULL(LT.isWithdrawn, 0) = 0
        AND ISNULL(LT.isPassedIn, 0) = 0
        AND LT.price > 0
    """
    df = pd.read_sql(query, conn, params=[country.upper()])
    df["asOfDate"] = pd.to_datetime(df["asOfDate"])
    return df


def fetch_historical_lots(conn, country: str) -> pd.DataFrame:
    """Fetch all historical yearling lots for feature computation."""
    query = """
    SELECT
        LT.Id AS lot_id,
        LT.salesId,
        CAST(SL.startDate AS DATE) AS saleDate,
        H.sireId,
        H.damId,
        LT.vendorId,
        CAST(LT.price AS DECIMAL(12,2)) AS hammer_price,
        CASE WHEN LT.price > 0 AND ISNULL(LT.isPassedIn,0) = 0 AND ISNULL(LT.isWithdrawn,0) = 0 THEN 1 ELSE 0 END AS isSold_int,
        CASE WHEN ISNULL(LT.isPassedIn,0) = 1 THEN 1 ELSE 0 END AS isPassedIn_int
    FROM tblSalesLot LT
    JOIN tblSales SL ON LT.salesId = SL.Id
    JOIN tblSalesLotType LTP ON LT.lotType = LTP.Id
    JOIN tblCountry CN ON SL.countryId = CN.id
    JOIN tblHorse H ON LT.horseId = H.id
    WHERE CN.countryCode = ?
        AND LTP.salesLotTypeName = 'Yearling'
        AND ISNULL(LT.isWithdrawn, 0) = 0
    """
    df = pd.read_sql(query, conn, params=[country.upper()])
    df["saleDate"] = pd.to_datetime(df["saleDate"])
    return df


def compute_session_medians(base_lots: pd.DataFrame) -> pd.DataFrame:
    """Compute session median price per sale."""
    sold = base_lots[base_lots["isSold_int"] == 1]
    medians = sold.groupby("salesId")["hammer_price"].median().reset_index()
    medians.columns = ["salesId", "session_median_price"]
    return medians


def build_training_features(base_lots: pd.DataFrame, hist_lots: pd.DataFrame, country_code: str) -> pd.DataFrame:
    """
    Build all features for training lots.

    Computes features in Python/pandas (no database writes).
    """
    if base_lots.empty:
        return base_lots

    print("  Computing session medians...")
    session_medians = compute_session_medians(base_lots)
    result = base_lots.merge(session_medians, on="salesId", how="left")

    # Process each sale date to compute point-in-time features
    print("  Computing sire/dam/vendor metrics (this may take a while)...")
    unique_dates = result["asOfDate"].unique()
    all_features = []

    for i, as_of_date in enumerate(sorted(unique_dates)):
        if (i + 1) % 10 == 0:
            print(f"    Processing sale {i + 1}/{len(unique_dates)}...")

        date_lots = result[result["asOfDate"] == as_of_date].copy()

        # Get unique IDs for this sale date
        sire_ids = date_lots["sireId"].dropna().unique()
        dam_ids = date_lots["damId"].dropna().unique()
        vendor_ids = date_lots["vendorId"].dropna().unique()

        # Compute metrics as of this date
        sire_36m = compute_sire_metrics(hist_lots, sire_ids, as_of_date, 36)
        sire_12m = compute_sire_metrics(hist_lots, sire_ids, as_of_date, 12)
        sire_metrics = sire_36m.merge(sire_12m, on="sireId", how="outer")
        sire_metrics["sire_momentum"] = sire_metrics["sire_median_price_12m"] - sire_metrics["sire_median_price_36m"]
        min_count = config.app.get_sire_sample_min_count(country_code)
        sire_metrics["sire_sample_flag_36m"] = (sire_metrics["sire_sold_count_36m"] < min_count).astype(int)

        dam_stats = compute_dam_stats(hist_lots, dam_ids, as_of_date)
        vendor_metrics = compute_vendor_metrics(hist_lots, vendor_ids, as_of_date)

        # Merge features
        date_lots = date_lots.merge(sire_metrics, on="sireId", how="left")
        date_lots = date_lots.merge(dam_stats, on="damId", how="left")
        date_lots = date_lots.merge(vendor_metrics, on="vendorId", how="left")

        all_features.append(date_lots)

    result = pd.concat(all_features, ignore_index=True)

    # Compute log_price_index (target variable)
    result["log_price_index"] = np.log(result["hammer_price"] / result["session_median_price"])

    # Filter out invalid rows
    result = result[
        (result["session_median_price"] > 0) &
        (result["hammer_price"] > 0) &
        (result["log_price_index"].notna())
    ]

    return result


def export_training_data(country: str) -> pd.DataFrame:
    """
    Export training data by computing features in Python.

    Uses read-only SQL queries - no writes to database.
    Features are computed point-in-time for each sale date.

    Args:
        country: Country code (aus, nzl, usa)

    Returns:
        DataFrame with training data and computed features
    """
    print(f"Building training data for {country.upper()} (read-only queries)...")

    conn = get_connection()

    print("  Fetching training lots...")
    base_lots = fetch_training_lots(conn, country)
    print(f"    Found {len(base_lots)} sold yearling lots")

    print("  Fetching historical lots...")
    hist_lots = fetch_historical_lots(conn, country)
    print(f"    Found {len(hist_lots)} historical lots")

    conn.close()

    print("  Building features...")
    df = build_training_features(base_lots, hist_lots, country)
    print(f"  Built {len(df)} training rows with features")

    return df


def load_training_data(csv_path: str) -> pd.DataFrame:
    """Load training data from CSV file."""
    print(f"Loading training data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} rows")
    return df


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================


def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Prepare features for training.

    Returns:
        Tuple of (X, y, feature_cols, df_processed)
    """
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
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            df[f"log_{col}"] = np.log1p(df[col])

    # Encode categoricals
    for col in ["sex", "vendor_volume_bucket", "sale_company"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")
            le = LabelEncoder()
            df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))

    # Define feature columns
    feature_cols = [
        "session_median_price",
        "sire_sold_count_36m",
        "sire_total_offered_36m",
        "sire_clearance_rate_36m",
        "sire_median_price_36m",
        "sire_sold_count_12m",
        "sire_total_offered_12m",
        "sire_clearance_rate_12m",
        "sire_median_price_12m",
        "sire_momentum",
        "dam_progeny_sold_count",
        "dam_progeny_total_offered_count",
        "dam_progeny_median_price",
        "vendor_sold_count_36m",
        "vendor_total_offered_36m",
        "vendor_clearance_rate_36m",
        "vendor_median_price_36m",
        "log_session_median_price",
        "log_sire_median_price_36m",
        "log_sire_median_price_12m",
        "log_dam_progeny_median_price",
        "log_vendor_median_price_36m",
        "sex_encoded",
        "vendor_volume_bucket_encoded",
        "sale_company_encoded",
        "sire_sample_flag_36m",
        "dam_first_foal_flag",
        "vendor_first_seen_flag",
    ]

    # Filter to columns that exist
    feature_cols = [col for col in feature_cols if col in df.columns]

    X = df[feature_cols].fillna(0)

    # Ensure numeric types for LightGBM (handles object dtype from database)
    for col in X.columns:
        if X[col].dtype == "object" or X[col].dtype == "bool":
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0).astype(float)

    y = df["log_price_index"].astype(float)

    return X, y, feature_cols, df


# ============================================================================
# BASELINE MODEL
# ============================================================================


def train_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Train Elastic Net baseline model for sanity check.

    Returns:
        Dictionary with baseline metrics and comparison results
    """
    print("\nTraining baseline model (Elastic Net)...")

    baseline = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=1000)
    baseline.fit(X_train, y_train)

    train_pred = baseline.predict(X_train)
    test_pred = baseline.predict(X_test)

    train_mae = mean_absolute_error(y_train, train_pred)
    train_r2 = r2_score(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_r2 = r2_score(y_test, test_pred)

    # Naive predictor: always predict mean of training set
    naive_pred = np.full_like(y_test, y_train.mean())
    naive_mae = mean_absolute_error(y_test, naive_pred)

    # Baseline passes if it beats naive predictor
    passes = test_mae < naive_mae

    print(f"  Train MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
    print(f"  Test MAE:  {test_mae:.4f}, R²: {test_r2:.4f}")
    print(f"  Naive MAE: {naive_mae:.4f}")
    print(f"  {'✓' if passes else '✗'} Baseline {'passes' if passes else 'FAILS'}: {'Better' if passes else 'Worse'} than naive predictor")

    return {
        "train_mae": train_mae,
        "train_r2": train_r2,
        "test_mae": test_mae,
        "test_r2": test_r2,
        "naive_mae": naive_mae,
        "passes": passes,
    }


# ============================================================================
# MODEL TRAINING
# ============================================================================


def train_quantile_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_cols: list,
) -> dict:
    """
    Train Q25, Q50, Q75 LightGBM models.

    Returns:
        Dictionary with trained models
    """
    print("\nTraining quantile models...")

    # LightGBM parameters
    base_params = {
        "objective": "quantile",
        "metric": "quantile",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "n_estimators": 500,
        "early_stopping_rounds": 50,
    }

    models = {}

    for quantile, name in [(0.25, "q25"), (0.50, "q50"), (0.75, "q75")]:
        print(f"  Training {name} (alpha={quantile})...")

        params = base_params.copy()
        params["alpha"] = quantile

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            callbacks=[lgb.log_evaluation(period=0)],  # Suppress verbose output
        )

        models[name] = model
        print(f"    Best iteration: {model.best_iteration}")

    return models


def get_feature_importance(models: dict, feature_cols: list) -> dict:
    """Extract feature importance from trained models."""
    importance = {}

    for name, model in models.items():
        imp = model.feature_importance(importance_type="gain")
        importance[name] = {
            feature_cols[i]: float(imp[i]) for i in range(len(feature_cols))
        }

    # Also compute average importance across all quantiles
    avg_importance = {}
    for col in feature_cols:
        avg_importance[col] = np.mean([importance[q][col] for q in models.keys()])

    importance["average"] = avg_importance

    return importance


# ============================================================================
# EVALUATION
# ============================================================================


def evaluate_models(
    y_test: pd.Series,
    pred_q25: np.ndarray,
    pred_q50: np.ndarray,
    pred_q75: np.ndarray,
    df_test: pd.DataFrame,
) -> dict:
    """
    Calculate evaluation metrics for quantile models.

    Args:
        y_test: True log_price_index values
        pred_q25, pred_q50, pred_q75: Predictions from quantile models
        df_test: Test dataframe with hammer_price and session_median_price

    Returns:
        Dictionary with MAE, RMSE, R², coverage, and MAPE
    """
    print("\nEvaluating models...")

    # P50 metrics (log space)
    mae = mean_absolute_error(y_test, pred_q50)
    rmse = np.sqrt(mean_squared_error(y_test, pred_q50))
    r2 = r2_score(y_test, pred_q50)

    print(f"  P50 MAE:  {mae:.4f}")
    print(f"  P50 RMSE: {rmse:.4f}")
    print(f"  P50 R²:   {r2:.4f}")

    # Raw coverage (before calibration)
    coverage_p25 = (y_test < pred_q25).mean() * 100
    coverage_p75 = (y_test < pred_q75).mean() * 100

    print(f"  Coverage P25: {coverage_p25:.1f}% (target: 25%)")
    print(f"  Coverage P75: {coverage_p75:.1f}% (target: 75%)")

    # Dollar-space MAPE (if we have the required columns)
    mape = None
    if "hammer_price" in df_test.columns and "session_median_price" in df_test.columns:
        # Convert predictions back to dollar space
        # log_price_index = log(hammer_price / session_median_price)
        # So: predicted_price = session_median_price * exp(pred_q50)
        session_median = df_test["session_median_price"].values
        actual_price = df_test["hammer_price"].values
        predicted_price = session_median * np.exp(pred_q50)

        # MAPE: mean of |actual - predicted| / actual
        valid_mask = actual_price > 0
        if valid_mask.sum() > 0:
            mape = (
                np.abs(actual_price[valid_mask] - predicted_price[valid_mask])
                / actual_price[valid_mask]
            ).mean() * 100
            print(f"  Dollar MAPE: {mape:.1f}%")

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "raw_coverage_p25": coverage_p25,
        "raw_coverage_p75": coverage_p75,
        "mape": mape,
    }


# ============================================================================
# CALIBRATION
# ============================================================================


def calibrate_models(
    y_test: pd.Series, pred_q25: np.ndarray, pred_q50: np.ndarray, pred_q75: np.ndarray
) -> dict:
    """
    Calculate calibration offsets for P25/P75 to achieve target coverage.

    Target: P25 should cover 25% of actuals below, P75 should cover 75% below.
    """
    print("\nCalibrating models...")

    # Calculate raw coverage
    raw_coverage_p25 = (y_test < pred_q25).mean() * 100
    raw_coverage_p75 = (y_test < pred_q75).mean() * 100

    print(f"  Raw coverage P25: {raw_coverage_p25:.1f}% (target: 25%)")
    print(f"  Raw coverage P75: {raw_coverage_p75:.1f}% (target: 75%)")

    # Calculate required offsets
    # For P25: if raw coverage is too high, we need to shift down (negative offset)
    # For P75: if raw coverage is too low, we need to shift up (positive offset)
    residuals = y_test.values - pred_q50

    offset_p25 = np.percentile(residuals, 25) - np.percentile(
        pred_q25 - pred_q50, 50
    )
    offset_p75 = np.percentile(residuals, 75) - np.percentile(
        pred_q75 - pred_q50, 50
    )

    # Verify calibrated coverage
    calibrated_p25 = pred_q25 + offset_p25
    calibrated_p75 = pred_q75 + offset_p75

    calibrated_coverage_p25 = (y_test < calibrated_p25).mean() * 100
    calibrated_coverage_p75 = (y_test < calibrated_p75).mean() * 100

    print(f"  Calibrated coverage P25: {calibrated_coverage_p25:.1f}%")
    print(f"  Calibrated coverage P75: {calibrated_coverage_p75:.1f}%")

    offsets = {
        "offset_p25": float(offset_p25),
        "offset_p75": float(offset_p75),
        "calibrated_coverage_p25": calibrated_coverage_p25,
        "calibrated_coverage_p75": calibrated_coverage_p75,
    }

    return offsets


# ============================================================================
# SAVE MODELS
# ============================================================================


def save_models(
    models: dict,
    offsets: dict,
    feature_cols: list,
    feature_importance: dict,
    output_dir: str,
    version: str,
    country: str,
):
    """Save all artifacts to versioned directory."""
    print(f"\nSaving models to {output_dir}/...")

    os.makedirs(output_dir, exist_ok=True)

    # Save LightGBM models
    for name, model in models.items():
        model_path = os.path.join(output_dir, f"mv_v1_{name}.txt")
        model.save_model(model_path)
        print(f"  Saved {model_path}")

    # Save calibration offsets with metadata
    offsets["generated_at"] = datetime.now(timezone.utc).isoformat()
    offsets["model_version"] = version
    offsets["notes"] = (
        f"Trained on {datetime.now().strftime('%Y-%m-%d')}. "
        "Elite scaling: offset = base_offset + (ln(raw_p50) - ln(threshold)) * scaling_factor"
    )

    offsets_path = os.path.join(output_dir, "calibration_offsets.json")
    with open(offsets_path, "w") as f:
        json.dump(offsets, f, indent=2)
    print(f"  Saved {offsets_path}")

    # Save feature columns
    feature_cols_path = os.path.join(output_dir, "feature_cols.json")
    with open(feature_cols_path, "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"  Saved {feature_cols_path}")

    # Save feature importance
    importance_path = os.path.join(output_dir, f"feature_importance_{country}_{version}.json")
    with open(importance_path, "w") as f:
        json.dump(feature_importance, f, indent=2)
    print(f"  Saved {importance_path}")


# ============================================================================
# TRAINING REPORT
# ============================================================================


def generate_training_report(
    country: str,
    version: str,
    n_total: int,
    n_train: int,
    n_val: int,
    n_test: int,
    n_features: int,
    baseline_metrics: dict,
    models: dict,
    eval_metrics: dict,
    offsets: dict,
    feature_importance: dict,
) -> str:
    """
    Generate formatted training report text.

    Returns:
        Formatted report string
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "=" * 60,
        f"MARKET VALUE MODEL — TRAINING REPORT ({country.upper()})",
        "=" * 60,
        f"Generated: {now}",
        f"Model Version: {version}",
        f"Country: {country.upper()}",
        "",
        "DATA",
        "-" * 60,
        f"Total samples: {n_total:,}",
        f"Train: {n_train:,} rows",
        f"Validation: {n_val:,} rows",
        f"Test: {n_test:,} rows",
        f"Features: {n_features}",
        "",
        "BASELINE MODEL: Elastic Net",
        "-" * 60,
        f"Train MAE: {baseline_metrics['train_mae']:.4f}",
        f"Train R²:  {baseline_metrics['train_r2']:.4f}",
        f"Test MAE:  {baseline_metrics['test_mae']:.4f}",
        f"Test R²:   {baseline_metrics['test_r2']:.4f}",
        f"Naive MAE: {baseline_metrics['naive_mae']:.4f}",
        f"{'✓' if baseline_metrics['passes'] else '✗'} Baseline {'passes' if baseline_metrics['passes'] else 'FAILS'}: {'Better' if baseline_metrics['passes'] else 'Worse'} than naive predictor",
        "",
        "QUANTILE MODELS",
        "-" * 60,
        f"Q25: {models['q25'].best_iteration} trees (early stopped)",
        f"Q50: {models['q50'].best_iteration} trees (early stopped)",
        f"Q75: {models['q75'].best_iteration} trees (early stopped)",
        "",
        "MODEL EVALUATION",
        "-" * 60,
        "P50 (Expected) Performance:",
        f"  MAE:  {eval_metrics['mae']:.4f}",
        f"  RMSE: {eval_metrics['rmse']:.4f}",
        f"  R²:   {eval_metrics['r2']:.4f}",
        "",
        "Coverage (before calibration):",
        f"  % below P25: {eval_metrics['raw_coverage_p25']:.1f}% (target: 25%)",
        f"  % below P75: {eval_metrics['raw_coverage_p75']:.1f}% (target: 75%)",
    ]

    if eval_metrics["mape"] is not None:
        lines.extend(
            [
                "",
                "Dollar-space (P50):",
                f"  MAPE: {eval_metrics['mape']:.1f}%",
            ]
        )

    lines.extend(
        [
            "",
            "CALIBRATION",
            "-" * 60,
            "Offsets (log space):",
            f"  P25: {offsets['offset_p25']:.4f}",
            f"  P75: {offsets['offset_p75']:.4f}",
            "",
            "Calibrated coverage:",
            f"  % below P25: {offsets['calibrated_coverage_p25']:.1f}% (target: 25%)",
            f"  % below P75: {offsets['calibrated_coverage_p75']:.1f}% (target: 75%)",
            "",
            "FEATURE IMPORTANCE (Top 15)",
            "-" * 60,
        ]
    )

    # Sort features by average importance
    avg_importance = feature_importance.get("average", {})
    sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)

    for feature, importance in sorted_features[:15]:
        lines.append(f"  {feature:40} {importance:,.1f}")

    lines.append("")

    return "\n".join(lines)


def save_training_report(report: str, output_dir: str):
    """Save training report to file."""
    report_path = os.path.join(output_dir, "training_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Saved {report_path}")


# ============================================================================
# DATA SPLITTING
# ============================================================================


def split_data_time_based(
    X: pd.DataFrame, y: pd.Series, df: pd.DataFrame
) -> tuple:
    """
    Split data using time-based approach based on config settings.

    Uses model_test_last_years from config to determine test set.
    Falls back to random split if sale_year column doesn't exist.

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, df_test, split_type)
    """
    if "sale_year" in df.columns:
        # Time-based split using config
        year_end = config.app.get_year_end()
        cutoff_year = year_end - config.app.model_test_last_years + 1

        train_mask = df["sale_year"] < cutoff_year
        test_mask = df["sale_year"] >= cutoff_year

        X_train_full = X[train_mask]
        y_train_full = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]
        df_test = df[test_mask]

        # Split train into train/validation (90/10)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.1, random_state=42
        )

        split_type = f"time-based ({config.app.year_start}-{cutoff_year-1} train, {cutoff_year}-{year_end} test)"
        print(f"  Using time-based split: train years < {cutoff_year}, test years >= {cutoff_year}")
    else:
        # Fall back to random split
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, X.index, test_size=0.2, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=42
        )
        df_test = df.loc[idx_test]

        split_type = "random 80/20"
        print(f"  Using random split (sale_year column not found)")

    return X_train, X_val, X_test, y_train, y_val, y_test, df_test, split_type


# ============================================================================
# MAIN
# ============================================================================


def train_model(country: str, version: str = None, csv_path: str = None, on_progress=None):
    """
    Main training function.

    Args:
        country: Country code (aus, nzl, usa)
        version: Optional version override (default: auto-increment)
        csv_path: Optional CSV path (default: export from database)
        on_progress: Optional callback function(phase: str) called at each major step
    """
    def _progress(phase):
        if on_progress:
            on_progress(phase)

    print("=" * 60)
    print(f"MARKET VALUE MODEL — TRAINING ({country.upper()})")
    print("=" * 60)

    # Determine version
    if version is None:
        version = get_next_version(country)
    print(f"\nModel version: {version}")

    # Load or export training data
    _progress("exporting_data")
    if csv_path:
        df = load_training_data(csv_path)
    else:
        df = export_training_data(country)

    # Prepare features
    _progress("preparing_features")
    print("\nPreparing features...")
    X, y, feature_cols, df_processed = prepare_features(df)
    print(f"  Features: {len(feature_cols)}")
    print(f"  Samples: {len(X)}")

    # Split data (time-based if sale_year exists, otherwise random)
    _progress("splitting_data")
    print("\nSplitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test, df_test, split_type = split_data_time_based(
        X, y, df_processed
    )

    print(f"\nData split ({split_type}):")
    print(f"  Train: {len(X_train)}")
    print(f"  Validation: {len(X_val)}")
    print(f"  Test: {len(X_test)}")

    # Train baseline model for sanity check
    _progress("training_models")
    baseline_metrics = train_baseline(X_train, y_train, X_test, y_test)

    # Train quantile models
    models = train_quantile_models(X_train, y_train, X_val, y_val, feature_cols)

    # Get predictions on test set
    pred_q25 = models["q25"].predict(X_test)
    pred_q50 = models["q50"].predict(X_test)
    pred_q75 = models["q75"].predict(X_test)

    # Evaluate models
    _progress("evaluating")
    eval_metrics = evaluate_models(y_test, pred_q25, pred_q50, pred_q75, df_test)

    # Calibrate
    offsets = calibrate_models(y_test, pred_q25, pred_q50, pred_q75)

    # Get feature importance
    feature_importance = get_feature_importance(models, feature_cols)

    # Determine output directory (using underscore: aus_v2)
    output_dir = f"models/{country}_{version}"

    # Save all artifacts
    _progress("saving_artifacts")
    save_models(
        models, offsets, feature_cols, feature_importance, output_dir, version, country
    )

    # Generate and save training report
    report = generate_training_report(
        country=country,
        version=version,
        n_total=len(X),
        n_train=len(X_train),
        n_val=len(X_val),
        n_test=len(X_test),
        n_features=len(feature_cols),
        baseline_metrics=baseline_metrics,
        models=models,
        eval_metrics=eval_metrics,
        offsets=offsets,
        feature_importance=feature_importance,
    )
    save_training_report(report, output_dir)

    # Print summary and next steps
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nModel saved to: {output_dir}/")
    print(f"\nFiles created:")
    print(f"  - mv_v1_q25.txt")
    print(f"  - mv_v1_q50.txt")
    print(f"  - mv_v1_q75.txt")
    print(f"  - calibration_offsets.json")
    print(f"  - feature_cols.json")
    print(f"  - feature_importance_{country}_{version}.json")
    print(f"  - training_report.txt")

    print(f"\n" + "-" * 60)
    print("NEXT STEPS")
    print("-" * 60)
    print(f"\nTo activate the new model, update .env:")
    print(f"  {country.upper()}_MODEL={country}_{version}")
    print(f"\nTo test before deploying:")
    print(f"  python src/score_sale.py --sale-id <SALE_ID>")


def main():
    parser = argparse.ArgumentParser(
        description="Train market value model for a specific country."
    )
    parser.add_argument(
        "--country",
        type=str,
        required=True,
        choices=["aus", "nzl", "usa"],
        help="Country code (aus, nzl, usa)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Override version (default: auto-increment, e.g., v2, v3)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Use CSV file instead of database query",
    )
    args = parser.parse_args()

    train_model(
        country=args.country.lower(),
        version=args.version,
        csv_path=args.csv,
    )


if __name__ == "__main__":
    main()
