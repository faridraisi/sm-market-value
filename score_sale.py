"""
Market Value Model v2.1 - Database-Driven Scoring Script
=========================================================
Scores yearling lots directly from database tables.

Features:
- Auto-detects country from sale_id
- Queries base tables directly (no staging tables needed)
- Outputs to tblHorseAnalytics or CSV (--dry-run)

Usage:
    python3 score_sale.py --sale-id 2096            # Score and write to DB
    python3 score_sale.py --sale-id 2096 --dry-run  # Score and write to CSV
"""

import argparse
import os
import sys
import json
import pyodbc
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime, timezone
from dotenv import load_dotenv

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_VERSION = "v2.1.1"

COUNTRY_CURRENCY_MAP = {
    'AUS': 1,  # AUD
    'NZL': 6,  # NZD
    'USA': 7,  # USD
}

# ============================================================================
# CLI ARGUMENTS
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Score yearling lots for market value prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 score_sale.py --sale-id 2096 --dry-run
    python3 score_sale.py --sale-id 2096
    python3 score_sale.py --sale-id 2098 --session-median 200000 --dry-run  # Pre-sale
        """
    )
    parser.add_argument('--sale-id', type=int, required=True,
                        help='Sale ID to score')
    parser.add_argument('--dry-run', action='store_true',
                        help='Write to CSV instead of database')
    parser.add_argument('--session-median', type=float, default=None,
                        help='Override session median price (required for pre-sale scoring)')
    return parser.parse_args()


# ============================================================================
# DATABASE CONNECTION
# ============================================================================

def get_db_connection():
    """Create database connection from .env credentials."""
    server = os.getenv('DB_SERVER', '127.0.0.1,1433')
    database = os.getenv('DB_NAME', 'G1StallionMatchProductionV5')
    username = os.getenv('DB_USER')
    password = os.getenv('DB_PASSWORD')

    if not username or not password:
        print("Error: DB_USER and DB_PASSWORD must be set in .env file")
        sys.exit(1)

    connection_string = (
        f"DRIVER={{ODBC Driver 18 for SQL Server}};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={username};"
        f"PWD={password};"
        f"TrustServerCertificate=yes;"
    )

    try:
        return pyodbc.connect(connection_string)
    except pyodbc.Error as e:
        print(f"Error connecting to database: {e}")
        sys.exit(1)


# ============================================================================
# COUNTRY CONFIGURATION
# ============================================================================

def get_country_config(country_code: str) -> dict:
    """
    Get country-specific configuration from .env

    Returns dict with:
        - lookback_countries: List of country codes for historical data
        - model_dir: Model directory to use for scoring
    """
    # Get lookback countries (default to same country)
    lookback_env = os.getenv(f'COUNTRY_{country_code}_LOOKBACK', country_code)
    lookback_countries = [c.strip() for c in lookback_env.split(',')]

    # Get model directory (default to lowercase country code)
    model_dir = os.getenv(f'COUNTRY_{country_code}_MODEL', country_code.lower())

    return {
        'lookback_countries': lookback_countries,
        'model_dir': model_dir,
    }


# ============================================================================
# SALE INFO LOOKUP
# ============================================================================

def get_sale_info(conn, sale_id: int) -> dict:
    """
    Lookup sale info and auto-detect country.

    Returns dict with:
        - sale_name: Sale name
        - country_code: 'AUS' or 'NZL'
        - currency_id: Currency ID for the country
        - start_date: Sale start date
    """
    query = """
    SELECT
        S.salesName,
        C.countryCode,
        C.preferredCurrencyId,
        S.startDate
    FROM tblSales S
    JOIN tblCountry C ON S.countryId = C.id
    WHERE S.Id = ?
    """

    cursor = conn.cursor()
    cursor.execute(query, (sale_id,))
    row = cursor.fetchone()

    if not row:
        print(f"Error: Sale ID {sale_id} not found")
        sys.exit(1)

    country_code = row.countryCode
    if country_code not in COUNTRY_CURRENCY_MAP:
        print(f"Error: Unsupported country code '{country_code}' for sale {sale_id}")
        print(f"Supported countries: {list(COUNTRY_CURRENCY_MAP.keys())}")
        sys.exit(1)

    return {
        'sale_name': row.salesName,
        'country_code': country_code,
        'currency_id': COUNTRY_CURRENCY_MAP[country_code],
        'start_date': row.startDate,
    }


# ============================================================================
# FEATURE LOADING
# ============================================================================

def load_features(conn, sale_id: int, lookback_countries: list, session_median_override: float = None) -> pd.DataFrame:
    """
    Load all features for a sale directly from base tables.

    Executes the parameterized query that calculates:
    - Session median price (or uses override for pre-sale scoring)
    - Sire metrics (36m and 12m lookback)
    - Dam stats
    - Vendor metrics

    Args:
        lookback_countries: List of country codes for historical lookback (e.g., ['AUS', 'NZL'])
        session_median_override: Optional override for session median (for pre-sale scoring)
    """

    # Handle session median - use override if provided, otherwise calculate from sold lots
    if session_median_override is not None:
        session_median_sql = f"SELECT {session_median_override} AS session_median_price"
    else:
        session_median_sql = f"""
        SELECT TOP 1
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY LT.price) OVER () AS session_median_price
        FROM tblSalesLot LT
        WHERE LT.salesId = {sale_id}
            AND LT.price > 0
            AND ISNULL(LT.isPassedIn, 0) = 0
            AND ISNULL(LT.isWithdrawn, 0) = 0
        """

    # Build SQL IN clause from country list
    countries_sql = ", ".join([f"'{c}'" for c in lookback_countries])

    # TODO: Currency conversion for multi-country lookback
    # When pooling historical data from multiple countries (e.g., AUS+NZL),
    # prices are currently mixed without currency conversion.
    # AUD/NZD rates are close (~0.90-0.95) so impact is minimal.
    # For more accurate metrics, convert using tblCurrencyrate:
    #   converted_price = price * (target_rate / source_rate)
    # where target_rate is the sale's currency rate.

    # Build the query with parameters embedded (pyodbc doesn't support DECLARE with params)
    query = f"""
    DECLARE @sale_id INT = {sale_id};
    DECLARE @as_of_date DATE = (SELECT startDate FROM tblSales WHERE Id = @sale_id);

    -- Session median for this sale
    ;WITH SessionMedian AS (
        {session_median_sql}
    ),
    -- Historical lots for lookback calculations
    HistLots AS (
        SELECT
            LT.Id AS lot_id,
            LT.salesId,
            CAST(SL.startDate AS DATE) AS saleDate,
            H.sireId,
            H.damId,
            LT.vendorId,
            CAST(LT.price AS DECIMAL(12,2)) AS hammer_price,
            CASE WHEN LT.price > 0 AND ISNULL(LT.isPassedIn,0) = 0
                 AND ISNULL(LT.isWithdrawn,0) = 0 THEN 1 ELSE 0 END AS isSold_int,
            CASE WHEN ISNULL(LT.isPassedIn,0) = 1 THEN 1 ELSE 0 END AS isPassedIn_int
        FROM tblSalesLot LT
        JOIN tblSales SL ON LT.salesId = SL.Id
        JOIN tblSalesLotType LTP ON LT.lotType = LTP.Id
        JOIN tblCountry CN ON SL.countryId = CN.id
        JOIN tblHorse H ON LT.horseId = H.id
        WHERE CN.countryCode IN ({countries_sql})
            AND LTP.salesLotTypeName = 'Yearling'
            AND ISNULL(LT.isWithdrawn, 0) = 0
            AND SL.startDate < @as_of_date
    )
    SELECT
        LT.Id AS lot_id,
        LT.salesId,
        LT.horseId,
        LT.lotNumber AS lot_number,
        LT.bookNumber AS book_number,
        LT.dayNumber AS day_number,
        LT.horseGender AS sex,
        LT.price AS hammer_price,
        H.horseName AS horse_name,
        SIRE.horseName AS sire_name,
        H.sireId,
        H.damId,
        LT.vendorId,
        SM.session_median_price,
        SC.salescompanyName AS sale_company,

        -- Sire 36m metrics
        COALESCE(SR36.sire_sold_count_36m, 0) AS sire_sold_count_36m,
        COALESCE(SR36.sire_total_offered_36m, 0) AS sire_total_offered_36m,
        SR36.sire_clearance_rate_36m,
        SR36.sire_median_price_36m,
        CASE WHEN COALESCE(SR36.sire_sold_count_36m, 0) >= 5 THEN 1 ELSE 0 END AS sire_sample_flag_36m,

        -- Sire 12m metrics
        COALESCE(SR12.sire_sold_count_12m, 0) AS sire_sold_count_12m,
        COALESCE(SR12.sire_total_offered_12m, 0) AS sire_total_offered_12m,
        SR12.sire_clearance_rate_12m,
        SR12.sire_median_price_12m,

        -- Sire momentum
        CASE
            WHEN SR12.sire_median_price_12m IS NOT NULL AND SR36.sire_median_price_36m IS NOT NULL
            THEN SR12.sire_median_price_12m - SR36.sire_median_price_36m
        END AS sire_momentum,

        -- Dam stats
        COALESCE(DS.dam_progeny_sold_count, 0) AS dam_progeny_sold_count,
        COALESCE(DS.dam_progeny_total_offered_count, 0) AS dam_progeny_total_offered_count,
        DS.dam_progeny_median_price,
        CASE WHEN COALESCE(DS.dam_progeny_sold_count, 0) = 0 THEN 1 ELSE 0 END AS dam_first_foal_flag,

        -- Vendor metrics
        COALESCE(VD.vendor_sold_count_36m, 0) AS vendor_sold_count_36m,
        COALESCE(VD.vendor_total_offered_36m, 0) AS vendor_total_offered_36m,
        VD.vendor_clearance_rate_36m,
        VD.vendor_median_price_36m,
        CASE
            WHEN COALESCE(VD.vendor_sold_count_36m, 0) = 0 THEN 'New'
            WHEN VD.vendor_sold_count_36m BETWEEN 1 AND 5 THEN 'Small'
            WHEN VD.vendor_sold_count_36m BETWEEN 6 AND 20 THEN 'Medium'
            ELSE 'Large'
        END AS vendor_volume_bucket,
        CASE WHEN COALESCE(VD.vendor_sold_count_36m, 0) = 0 THEN 1 ELSE 0 END AS vendor_first_seen_flag

    FROM tblSalesLot LT
    JOIN tblSales SL ON LT.salesId = SL.Id
    JOIN tblSalesCompany SC ON SL.salesCompanyId = SC.Id
    JOIN tblHorse H ON LT.horseId = H.id
    LEFT JOIN tblHorse SIRE ON H.sireId = SIRE.id
    CROSS JOIN SessionMedian SM

    -- Sire 36m lookback
    OUTER APPLY (
        SELECT
            SUM(HL.isSold_int) AS sire_sold_count_36m,
            SUM(HL.isSold_int + HL.isPassedIn_int) AS sire_total_offered_36m,
            CAST(100.0 * SUM(HL.isSold_int) / NULLIF(SUM(HL.isSold_int + HL.isPassedIn_int), 0) AS DECIMAL(5,1)) AS sire_clearance_rate_36m,
            (
                SELECT TOP 1 PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY HL2.hammer_price) OVER ()
                FROM HistLots HL2
                WHERE HL2.sireId = H.sireId
                    AND HL2.saleDate < @as_of_date
                    AND HL2.saleDate >= DATEADD(MONTH, -36, @as_of_date)
                    AND HL2.isSold_int = 1
                    AND HL2.hammer_price > 0
            ) AS sire_median_price_36m
        FROM HistLots HL
        WHERE HL.sireId = H.sireId
            AND HL.saleDate < @as_of_date
            AND HL.saleDate >= DATEADD(MONTH, -36, @as_of_date)
    ) SR36

    -- Sire 12m lookback
    OUTER APPLY (
        SELECT
            SUM(HL.isSold_int) AS sire_sold_count_12m,
            SUM(HL.isSold_int + HL.isPassedIn_int) AS sire_total_offered_12m,
            CAST(100.0 * SUM(HL.isSold_int) / NULLIF(SUM(HL.isSold_int + HL.isPassedIn_int), 0) AS DECIMAL(5,1)) AS sire_clearance_rate_12m,
            (
                SELECT TOP 1 PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY HL2.hammer_price) OVER ()
                FROM HistLots HL2
                WHERE HL2.sireId = H.sireId
                    AND HL2.saleDate < @as_of_date
                    AND HL2.saleDate >= DATEADD(MONTH, -12, @as_of_date)
                    AND HL2.isSold_int = 1
                    AND HL2.hammer_price > 0
            ) AS sire_median_price_12m
        FROM HistLots HL
        WHERE HL.sireId = H.sireId
            AND HL.saleDate < @as_of_date
            AND HL.saleDate >= DATEADD(MONTH, -12, @as_of_date)
    ) SR12

    -- Dam stats (all-time prior)
    OUTER APPLY (
        SELECT
            SUM(HL.isSold_int) AS dam_progeny_sold_count,
            SUM(HL.isSold_int + HL.isPassedIn_int) AS dam_progeny_total_offered_count,
            (
                SELECT TOP 1 PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY HL2.hammer_price) OVER ()
                FROM HistLots HL2
                WHERE HL2.damId = H.damId
                    AND HL2.saleDate < @as_of_date
                    AND HL2.isSold_int = 1
                    AND HL2.hammer_price > 0
            ) AS dam_progeny_median_price
        FROM HistLots HL
        WHERE HL.damId = H.damId
            AND HL.saleDate < @as_of_date
    ) DS

    -- Vendor 36m lookback
    OUTER APPLY (
        SELECT
            SUM(HL.isSold_int) AS vendor_sold_count_36m,
            SUM(HL.isSold_int + HL.isPassedIn_int) AS vendor_total_offered_36m,
            CAST(100.0 * SUM(HL.isSold_int) / NULLIF(SUM(HL.isSold_int + HL.isPassedIn_int), 0) AS DECIMAL(5,1)) AS vendor_clearance_rate_36m,
            (
                SELECT TOP 1 PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY HL2.hammer_price) OVER ()
                FROM HistLots HL2
                WHERE HL2.vendorId = LT.vendorId
                    AND HL2.saleDate < @as_of_date
                    AND HL2.saleDate >= DATEADD(MONTH, -36, @as_of_date)
                    AND HL2.isSold_int = 1
                    AND HL2.hammer_price > 0
            ) AS vendor_median_price_36m
        FROM HistLots HL
        WHERE HL.vendorId = LT.vendorId
            AND HL.saleDate < @as_of_date
            AND HL.saleDate >= DATEADD(MONTH, -36, @as_of_date)
    ) VD

    WHERE LT.salesId = @sale_id
        AND ISNULL(LT.isWithdrawn, 0) = 0
    ORDER BY LT.lotNumber;
    """

    df = pd.read_sql(query, conn)

    if len(df) == 0:
        print(f"Error: No lots found for sale {sale_id}")
        print("Possible reasons:")
        print("  - Sale has no yearling lots")
        print("  - All lots are withdrawn")
        print("  - No sold lots for session median calculation (use --session-median for pre-sale)")
        sys.exit(1)

    # Check for null session_median (pre-sale without override)
    if df['session_median_price'].isna().all():
        print(f"Error: No session median available for sale {sale_id}")
        print("For pre-sale scoring, use --session-median to provide expected median price")
        sys.exit(1)

    return df


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_models(model_dir: str):
    """
    Load trained models and metadata from specified directory.

    Args:
        model_dir: Model directory name (e.g., 'nzl', 'aus')
    """
    model_path = f"models/{model_dir}"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    models = {
        'q25': lgb.Booster(model_file=f'{model_path}/mv_v1_q25.txt'),
        'q50': lgb.Booster(model_file=f'{model_path}/mv_v1_q50.txt'),
        'q75': lgb.Booster(model_file=f'{model_path}/mv_v1_q75.txt'),
    }

    with open(f'{model_path}/calibration_offsets.json', 'r') as f:
        offsets = json.load(f)

    with open(f'{model_path}/feature_cols.json', 'r') as f:
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
    elite_config = offsets.get('elite_scaling', {})
    threshold = elite_config.get('threshold', 300000)
    base_offset = elite_config.get('base_offset', 0.5)
    scaling_factor = elite_config.get('scaling_factor', 1.2)

    offsets_array = np.zeros_like(raw_prices)
    mask = raw_prices >= threshold

    if mask.any():
        offsets_array[mask] = base_offset + (np.log(raw_prices[mask]) - np.log(threshold)) * scaling_factor

    return offsets_array


def apply_adjustments(session_median: np.ndarray, pred_q25: np.ndarray,
                      pred_q50: np.ndarray, pred_q75: np.ndarray,
                      offsets: dict) -> tuple:
    """Apply all adjustments: elite scaling + base calibration."""
    raw_prices = session_median * np.exp(pred_q50)
    elite_offsets = apply_elite_scaling(raw_prices, offsets)

    adj_q50 = pred_q50 + elite_offsets
    adj_q25 = pred_q25 + offsets['offset_p25'] + elite_offsets
    adj_q75 = pred_q75 + offsets['offset_p75'] + elite_offsets

    return adj_q25, adj_q50, adj_q75, raw_prices


# ============================================================================
# CONFIDENCE TIER (v2.1)
# ============================================================================

def calculate_confidence_tier(df: pd.DataFrame, raw_prices: np.ndarray) -> pd.Series:
    """Calculate confidence tier based on data flags and predicted price tier."""

    flags = pd.DataFrame(index=df.index)
    flags['sire_flag'] = df.get('sire_sample_flag_36m', pd.Series(0, index=df.index)).fillna(0).astype(int)
    flags['dam_flag'] = df.get('dam_first_foal_flag', pd.Series(0, index=df.index)).fillna(0).astype(int)
    flags['vendor_flag'] = df.get('vendor_first_seen_flag', pd.Series(0, index=df.index)).fillna(0).astype(int)
    flag_count = flags.sum(axis=1)

    tier = pd.Series('high', index=df.index)

    tier[raw_prices >= 300000] = 'low'

    mid_high_mask = (raw_prices >= 200000) & (raw_prices < 300000)
    tier[mid_high_mask & (flag_count >= 1)] = 'low'
    tier[mid_high_mask & (flag_count == 0)] = 'medium'

    lower_mask = raw_prices < 200000
    tier[lower_mask & (flag_count >= 2)] = 'low'
    tier[lower_mask & (flag_count == 1)] = 'medium'
    tier[lower_mask & (flag_count == 0)] = 'high'

    return tier


# ============================================================================
# SCORING
# ============================================================================

def score_lots(df: pd.DataFrame, models: dict, offsets: dict, feature_cols: list) -> pd.DataFrame:
    """Score lots and return predictions."""

    df = df.copy()

    # Log transform price features
    price_cols = ['session_median_price', 'sire_median_price_36m', 'sire_median_price_12m',
                  'dam_progeny_median_price', 'vendor_median_price_36m']
    for col in price_cols:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(df[col].fillna(0))

    # Encode categoricals
    from sklearn.preprocessing import LabelEncoder
    for col in ['sex', 'vendor_volume_bucket', 'sale_company']:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))

    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    X = df[feature_cols].fillna(0)

    # Predict (raw log-space)
    pred_q25 = models['q25'].predict(X)
    pred_q50 = models['q50'].predict(X)
    pred_q75 = models['q75'].predict(X)

    session_median = df['session_median_price'].values

    # Apply adjustments (v2.1)
    adj_q25, adj_q50, adj_q75, raw_prices = apply_adjustments(
        session_median, pred_q25, pred_q50, pred_q75, offsets
    )

    # Convert to dollar prices
    results = pd.DataFrame({
        'lot_id': df['lot_id'],
        'horseId': df['horseId'] if 'horseId' in df.columns else None,
        'salesId': df['salesId'],
        'lot_number': df['lot_number'] if 'lot_number' in df.columns else None,
        'book_number': df['book_number'] if 'book_number' in df.columns else None,
        'sex': df['sex'] if 'sex' in df.columns else None,
        'horse_name': df['horse_name'] if 'horse_name' in df.columns else None,
        'sire_name': df['sire_name'] if 'sire_name' in df.columns else None,
        'session_median_price': session_median,

        # Log-space predictions (adjusted)
        'mv_expected_index': np.exp(adj_q50),
        'mv_low_index': np.exp(adj_q25),
        'mv_high_index': np.exp(adj_q75),

        # Dollar predictions
        'mv_expected_price': np.round(session_median * np.exp(adj_q50), -2),
        'mv_low_price': np.round(session_median * np.exp(adj_q25), -2),
        'mv_high_price': np.round(session_median * np.exp(adj_q75), -2),

        # Raw prediction (before adjustment)
        'mv_raw_price': np.round(raw_prices, -2),

        # Confidence tier (v2.1)
        'mv_confidence_tier': calculate_confidence_tier(df, raw_prices),

        # Metadata
        'mv_model_version': MODEL_VERSION,
        'mv_generated_at': datetime.now(timezone.utc),
    })

    # Ensure low <= expected <= high
    results['mv_low_price'] = np.minimum(results['mv_low_price'], results['mv_expected_price'])
    results['mv_high_price'] = np.maximum(results['mv_high_price'], results['mv_expected_price'])
    results['mv_low_price'] = np.maximum(results['mv_low_price'], 0)

    return results


# ============================================================================
# OUTPUT
# ============================================================================

def save_to_csv(results: pd.DataFrame, sale_id: int) -> str:
    """Save results to CSV file."""
    os.makedirs("reports", exist_ok=True)
    output_path = f"reports/sale_{sale_id}_predictions.csv"
    results.to_csv(output_path, index=False)
    return output_path


def save_to_db(conn, results: pd.DataFrame, sale_id: int, currency_id: int):
    """
    Save results to tblHorseAnalytics using MERGE (upsert).

    Upserts on (salesId, horseId) - updates if exists, inserts if new.
    """
    audit_user_id = int(os.getenv('AUDIT_USER_ID', 2))

    cursor = conn.cursor()

    # Use MERGE for upsert
    merge_sql = """
    MERGE INTO tblHorseAnalytics AS target
    USING (SELECT ? AS salesId, ? AS horseId) AS source
    ON target.salesId = source.salesId AND target.horseId = source.horseId
    WHEN MATCHED THEN
        UPDATE SET
            marketValue = ?,
            marketValueLow = ?,
            marketValueHigh = ?,
            marketValueConfidence = ?,
            marketValueMultiplier = ?,
            sessionMedianPrice = ?,
            currencyId = ?,
            modifiedBy = ?,
            modifiedOn = GETUTCDATE()
    WHEN NOT MATCHED THEN
        INSERT (salesId, horseId, marketValue, marketValueLow, marketValueHigh,
                marketValueConfidence, marketValueMultiplier, sessionMedianPrice,
                currencyId, createdBy, createdOn, modifiedBy, modifiedOn)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, GETUTCDATE(), ?, GETUTCDATE());
    """

    for _, row in results.iterrows():
        params = (
            # Source (for matching)
            int(row['salesId']),
            int(row['horseId']),
            # Update values
            float(row['mv_expected_price']),
            float(row['mv_low_price']),
            float(row['mv_high_price']),
            str(row['mv_confidence_tier']),
            float(row['mv_expected_index']),
            float(row['session_median_price']),
            currency_id,
            audit_user_id,
            # Insert values (same as update + keys)
            int(row['salesId']),
            int(row['horseId']),
            float(row['mv_expected_price']),
            float(row['mv_low_price']),
            float(row['mv_high_price']),
            str(row['mv_confidence_tier']),
            float(row['mv_expected_index']),
            float(row['session_median_price']),
            currency_id,
            audit_user_id,
            audit_user_id,
        )

        cursor.execute(merge_sql, params)

    conn.commit()
    return len(results)


def print_summary(results: pd.DataFrame):
    """Print scoring summary statistics."""
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total lots: {len(results)}")
    print(f"\nMedian prices:")
    print(f"  Low:      ${results['mv_low_price'].median():,.0f}")
    print(f"  Expected: ${results['mv_expected_price'].median():,.0f}")
    print(f"  High:     ${results['mv_high_price'].median():,.0f}")
    print(f"\nConfidence tiers:")
    print(results['mv_confidence_tier'].value_counts().to_string())

    adjusted_count = (results['mv_raw_price'] >= 300000).sum()
    print(f"\nElite scaling applied to: {adjusted_count} lots ({adjusted_count/len(results)*100:.1f}%)")

    if adjusted_count > 0:
        adjusted = results[results['mv_raw_price'] >= 300000]
        avg_boost = (adjusted['mv_expected_price'] / adjusted['mv_raw_price']).mean()
        print(f"Average boost for adjusted lots: {avg_boost:.2f}x")


# ============================================================================
# MAIN
# ============================================================================

def main():
    args = parse_args()
    load_dotenv()

    print("="*60)
    print("MARKET VALUE MODEL - DATABASE-DRIVEN SCORING")
    print("="*60)

    # Connect to database
    print("\nConnecting to database...")
    conn = get_db_connection()

    # Get sale info and auto-detect country
    print(f"Looking up sale {args.sale_id}...")
    sale_info = get_sale_info(conn, args.sale_id)

    # Get country-specific configuration
    country_config = get_country_config(sale_info['country_code'])

    print(f"\nSale: {sale_info['sale_name']}")
    print(f"Country: {sale_info['country_code']}")
    print(f"Lookback: {', '.join(country_config['lookback_countries'])}")
    print(f"Model: models/{country_config['model_dir']}/")
    print(f"Start Date: {sale_info['start_date']}")

    # Load models from configured directory
    print(f"\nLoading models from: models/{country_config['model_dir']}/")
    models, offsets, feature_cols = load_models(country_config['model_dir'])
    print(f"Model Version: {MODEL_VERSION}")

    # Show elite scaling config
    if 'elite_scaling' in offsets:
        cfg = offsets['elite_scaling']
        print(f"\nElite scaling (predictions >= ${cfg['threshold']:,}):")
        print(f"  Base offset: {cfg['base_offset']}")
        print(f"  Scaling factor: {cfg['scaling_factor']}")

    # Load features using configured lookback countries
    print("\nLoading features from base tables...")
    print(f"  Lookback countries: {', '.join(country_config['lookback_countries'])}")
    if args.session_median:
        print(f"  Using session median override: ${args.session_median:,.0f}")
    df = load_features(conn, args.sale_id, country_config['lookback_countries'], args.session_median)
    print(f"  Loaded {len(df)} lots")
    if not args.session_median:
        print(f"  Calculated session median: ${df['session_median_price'].iloc[0]:,.0f}")

    # Score
    print("\nScoring...")
    results = score_lots(df, models, offsets, feature_cols)

    # Output
    if args.dry_run:
        output_path = save_to_csv(results, args.sale_id)
        print(f"\n[DRY-RUN] Saved to: {output_path}")
    else:
        count = save_to_db(conn, results, args.sale_id, sale_info['currency_id'])
        print(f"\nSaved {count} rows to tblHorseAnalytics")

    print_summary(results)

    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
