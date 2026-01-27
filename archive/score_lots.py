"""
Market Value Model v2.1 — Scoring Script
=========================================
Loads trained models and scores new lots.

v2.1 Changes:
- Elite/Premium boost for predictions >= $300k to fix undervaluation
- Revised confidence tier logic (factors in predicted price tier)

Usage:
    python3 score_lots.py
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import json
import os
from datetime import datetime, timezone

# ============================================================================
# CONFIGURATION
# ============================================================================

# Set country: 'aus' or 'nzl'
COUNTRY = "aus"

# Set to True to load from CSV instead of database
USE_CSV = True
CSV_PATH = "sale_2094_inference.csv"

MODEL_VERSION = "v2.1"

# ============================================================================
# PATHS
# ============================================================================

def get_model_path(country: str) -> str:
    """Get model directory for country."""
    return f"models/{country}"

# ============================================================================
# LOAD MODELS
# ============================================================================

def load_models(country: str):
    """Load trained models and metadata for specified country."""
    
    model_dir = get_model_path(country)
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    models = {
        'q25': lgb.Booster(model_file=f'{model_dir}/mv_v1_q25.txt'),
        'q50': lgb.Booster(model_file=f'{model_dir}/mv_v1_q50.txt'),
        'q75': lgb.Booster(model_file=f'{model_dir}/mv_v1_q75.txt'),
    }
    
    with open(f'{model_dir}/calibration_offsets.json', 'r') as f:
        offsets = json.load(f)
    
    with open(f'{model_dir}/feature_cols.json', 'r') as f:
        feature_cols = json.load(f)
    
    return models, offsets, feature_cols


# ============================================================================
# LOAD DATA
# ============================================================================

def load_inference_data():
    """Load inference data from CSV or database."""
    
    if USE_CSV:
        print(f"Loading from CSV: {CSV_PATH}")
        return pd.read_csv(CSV_PATH)
    
    # Database fallback
    import pyodbc
    CONNECTION_STRING = (
        "DRIVER={ODBC Driver 18 for SQL Server};"
        "SERVER=127.0.0.1,1433;"
        "DATABASE=G1StallionMatchProductionV5;"
        "UID=dev_matthew;"
        "PWD=YOUR_PASSWORD_HERE;"
        "TrustServerCertificate=yes;"
    )
    
    conn = pyodbc.connect(CONNECTION_STRING)
    df = pd.read_sql("SELECT * FROM dbo.mv_yearling_lot_features_v1 WHERE isWithdrawn = 0", conn)
    conn.close()
    return df


# ============================================================================
# ELITE SCALING LOGIC (v2.1)
# ============================================================================

def apply_elite_scaling(raw_prices: np.ndarray, offsets: dict) -> np.ndarray:
    """
    Apply elite scaling for predictions above threshold.
    
    Formula: offset = base_offset + (ln(raw_p50) - ln(threshold)) * scaling_factor
    Only applied when raw_p50 >= threshold.
    
    This corrects model compression at elite/premium tiers while leaving
    mid-tier predictions unchanged.
    """
    elite_config = offsets.get('elite_scaling', {})
    threshold = elite_config.get('threshold', 300000)
    base_offset = elite_config.get('base_offset', 0.5)
    scaling_factor = elite_config.get('scaling_factor', 1.2)
    
    # Calculate offset for each lot
    offsets_array = np.zeros_like(raw_prices)
    mask = raw_prices >= threshold
    
    if mask.any():
        offsets_array[mask] = base_offset + (np.log(raw_prices[mask]) - np.log(threshold)) * scaling_factor
    
    return offsets_array


def apply_adjustments(session_median: np.ndarray, pred_q25: np.ndarray, 
                      pred_q50: np.ndarray, pred_q75: np.ndarray, 
                      offsets: dict) -> tuple:
    """
    Apply all adjustments: elite scaling + base calibration.
    
    Returns adjusted log predictions for P25, P50, P75 and raw prices.
    """
    # Calculate raw prices to determine scaling
    raw_prices = session_median * np.exp(pred_q50)
    
    # Get elite scaling offset for each lot
    elite_offsets = apply_elite_scaling(raw_prices, offsets)
    
    # Apply elite offset to P50
    adj_q50 = pred_q50 + elite_offsets
    
    # Apply base calibration offsets + elite offset to P25/P75
    adj_q25 = pred_q25 + offsets['offset_p25'] + elite_offsets
    adj_q75 = pred_q75 + offsets['offset_p75'] + elite_offsets
    
    return adj_q25, adj_q50, adj_q75, raw_prices


# ============================================================================
# CONFIDENCE TIER (v2.1)
# ============================================================================

def calculate_confidence_tier(df: pd.DataFrame, raw_prices: np.ndarray) -> pd.Series:
    """
    Calculate confidence tier based on data flags AND predicted price tier.
    
    Logic:
    - Predicted >= $300k: LOW (model less reliable at elite tier, even after adjustment)
    - Predicted $200k-$300k + any flag: LOW
    - Predicted $200k-$300k + no flags: MEDIUM
    - Predicted < $200k + 2+ flags: LOW
    - Predicted < $200k + 1 flag: MEDIUM
    - Predicted < $200k + no flags: HIGH
    """
    
    # Count data quality flags
    flags = pd.DataFrame(index=df.index)
    flags['sire_flag'] = df.get('sire_sample_flag_36m', pd.Series(0, index=df.index)).fillna(0).astype(int)
    flags['dam_flag'] = df.get('dam_first_foal_flag', pd.Series(0, index=df.index)).fillna(0).astype(int)
    flags['vendor_flag'] = df.get('vendor_first_seen_flag', pd.Series(0, index=df.index)).fillna(0).astype(int)
    flag_count = flags.sum(axis=1)
    
    # Initialize all as HIGH
    tier = pd.Series('high', index=df.index)
    
    # Predicted >= $300k: always LOW (even after adjustment, high uncertainty)
    tier[raw_prices >= 300000] = 'low'
    
    # Predicted $200k-$300k
    mid_high_mask = (raw_prices >= 200000) & (raw_prices < 300000)
    tier[mid_high_mask & (flag_count >= 1)] = 'low'
    tier[mid_high_mask & (flag_count == 0)] = 'medium'
    
    # Predicted < $200k (standard flag-based logic)
    lower_mask = raw_prices < 200000
    tier[lower_mask & (flag_count >= 2)] = 'low'
    tier[lower_mask & (flag_count == 1)] = 'medium'
    tier[lower_mask & (flag_count == 0)] = 'high'
    
    return tier


# ============================================================================
# SCORE
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
        
        # Raw prediction (before adjustment, for debugging/comparison)
        'mv_raw_price': np.round(raw_prices, -2),
        
        # Confidence tier (v2.1 - price-aware)
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
# MAIN
# ============================================================================

def main():
    print("="*60)
    print(f"MARKET VALUE MODEL — SCORING ({COUNTRY.upper()})")
    print("="*60)
    
    print(f"\nCountry: {COUNTRY.upper()}")
    print(f"Model Version: {MODEL_VERSION}")
    print(f"Loading models from: models/{COUNTRY}/")
    models, offsets, feature_cols = load_models(COUNTRY)
    
    # Show elite scaling config
    if 'elite_scaling' in offsets:
        cfg = offsets['elite_scaling']
        print(f"\nElite scaling (predictions >= ${cfg['threshold']:,}):")
        print(f"  Base offset: {cfg['base_offset']}")
        print(f"  Scaling factor: {cfg['scaling_factor']}")
        print(f"  At $300k: {cfg['base_offset']:.2f} ({np.exp(cfg['base_offset']):.2f}x)")
        example_400k = cfg['base_offset'] + (np.log(400000) - np.log(cfg['threshold'])) * cfg['scaling_factor']
        print(f"  At $400k: {example_400k:.2f} ({np.exp(example_400k):.2f}x)")
        example_500k = cfg['base_offset'] + (np.log(500000) - np.log(cfg['threshold'])) * cfg['scaling_factor']
        print(f"  At $500k: {example_500k:.2f} ({np.exp(example_500k):.2f}x)")
    
    print("\nLoading inference data...")
    df = load_inference_data()
    print(f"  Loaded {len(df)} lots")
    
    print("\nScoring...")
    results = score_lots(df, models, offsets, feature_cols)
    
    # Create reports directory if needed
    os.makedirs("reports", exist_ok=True)
    
    results.to_csv("reports/scored_lots.csv", index=False)
    print(f"\nSaved: reports/scored_lots.csv")
    
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
    
    # Show adjustment impact
    adjusted_count = (results['mv_raw_price'] >= 300000).sum()
    print(f"\nElite scaling applied to: {adjusted_count} lots ({adjusted_count/len(results)*100:.1f}%)")
    
    if adjusted_count > 0:
        adjusted = results[results['mv_raw_price'] >= 300000]
        avg_boost = (adjusted['mv_expected_price'] / adjusted['mv_raw_price']).mean()
        print(f"Average boost for adjusted lots: {avg_boost:.2f}x")


if __name__ == "__main__":
    main()
