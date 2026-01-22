"""
Market Value Model v2.0 — Scoring Script
=========================================
Loads trained models and scores new lots.

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
COUNTRY = "nzl"

# Set to True to load from CSV instead of database
USE_CSV = True
CSV_PATH = "sale_2096_inference.csv"

MODEL_VERSION = "v2.0"

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
    
    # Predict
    pred_q25 = models['q25'].predict(X)
    pred_q50 = models['q50'].predict(X)
    pred_q75 = models['q75'].predict(X)
    
    # Apply calibration
    cal_q25 = pred_q25 + offsets['offset_p25']
    cal_q75 = pred_q75 + offsets['offset_p75']
    
    # Convert to dollar prices
    session_median = df['session_median_price'].values
    
    results = pd.DataFrame({
        'lot_id': df['lot_id'],
        'horseId': df['horseId'] if 'horseId' in df.columns else None,
        'salesId': df['salesId'],
        'lot_number': df['lot_number'] if 'lot_number' in df.columns else None,
        'book_number': df['book_number'] if 'book_number' in df.columns else None,
        'sex': df['sex'] if 'sex' in df.columns else None,
        'horse_name': df['horse_name'] if 'horse_name' in df.columns else None,
        'sire_name': df['sire_name'] if 'sire_name' in df.columns else None,
        'session_median_price': df['session_median_price'],
        
        # Log-space predictions
        'mv_expected_index': np.exp(pred_q50),
        'mv_low_index': np.exp(cal_q25),
        'mv_high_index': np.exp(cal_q75),
        
        # Dollar predictions
        'mv_expected_price': np.round(session_median * np.exp(pred_q50), -2),
        'mv_low_price': np.round(session_median * np.exp(cal_q25), -2),
        'mv_high_price': np.round(session_median * np.exp(cal_q75), -2),
        
        # Confidence tier
        'mv_confidence_tier': calculate_confidence_tier(df),
        
        # Metadata
        'mv_model_version': MODEL_VERSION,
        'mv_generated_at': datetime.now(timezone.utc),
    })
    
    # Ensure low <= expected <= high
    results['mv_low_price'] = np.minimum(results['mv_low_price'], results['mv_expected_price'])
    results['mv_high_price'] = np.maximum(results['mv_high_price'], results['mv_expected_price'])
    results['mv_low_price'] = np.maximum(results['mv_low_price'], 0)
    
    return results


def calculate_confidence_tier(df: pd.DataFrame) -> pd.Series:
    """Calculate confidence tier based on data flags."""
    
    flags = pd.DataFrame(index=df.index)
    flags['sire_flag'] = df.get('sire_sample_flag_36m', pd.Series(0, index=df.index)).fillna(0).astype(int)
    flags['dam_flag'] = df.get('dam_first_foal_flag', pd.Series(0, index=df.index)).fillna(0).astype(int)
    flags['vendor_flag'] = df.get('vendor_first_seen_flag', pd.Series(0, index=df.index)).fillna(0).astype(int)
    
    flag_count = flags.sum(axis=1)
    
    tier = pd.Series('high', index=df.index)
    tier[flag_count == 1] = 'medium'
    tier[flag_count >= 2] = 'low'
    
    return tier


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*60)
    print(f"MARKET VALUE MODEL — SCORING ({COUNTRY.upper()})")
    print("="*60)
    
    print(f"\nCountry: {COUNTRY.upper()}")
    print(f"Loading models from: models/{COUNTRY}/")
    models, offsets, feature_cols = load_models(COUNTRY)
    
    print("\nLoading inference data...")
    df = load_inference_data()
    print(f"  Loaded {len(df)} lots")
    
    print("\nScoring...")
    results = score_lots(df, models, offsets, feature_cols)
    
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


if __name__ == "__main__":
    main()
