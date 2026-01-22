"""
Market Value Model v2.0 — Training Script
==========================================
Trains quantile GBDT models for yearling price prediction.

Usage:
    python3 train_market_value_model.py
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import json
import os
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

# Set country: 'aus' or 'nzl'
COUNTRY = "nzl"

# Set to True to load from CSV instead of database
USE_CSV = True
CSV_PATH = "training_data_nzl.csv"

MODEL_VERSION = "v2.0"

# ============================================================================
# PATHS
# ============================================================================

def get_model_path(country: str) -> str:
    """Get model directory for country."""
    path = f"models/{country}"
    os.makedirs(path, exist_ok=True)
    return path

def get_report_path() -> str:
    """Get reports directory."""
    path = "reports"
    os.makedirs(path, exist_ok=True)
    return path

# ============================================================================
# LOAD DATA
# ============================================================================

def get_training_data():
    """Load training data from CSV or database."""
    
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
    
    print("Connecting to database...")
    conn = pyodbc.connect(CONNECTION_STRING)
    
    # Adjust table name based on country
    if COUNTRY == "nzl":
        table = "mv_yearling_lot_features_nz_v1"
    else:
        table = "mv_yearling_lot_features_v1"
    
    query = f"""
    SELECT *
    FROM dbo.{table}
    WHERE isWithdrawn = 0
      AND isPassedIn = 0
      AND hammer_price > 0
      AND session_median_price > 0
      AND log_price_index IS NOT NULL
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    return df


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def prepare_features(df: pd.DataFrame) -> tuple:
    """Prepare features for training."""
    
    df = df.copy()
    
    # Target
    y = df['log_price_index']
    
    # Log transform price features
    price_cols = ['session_median_price', 'sire_median_price_36m', 'sire_median_price_12m', 
                  'dam_progeny_median_price', 'vendor_median_price_36m']
    for col in price_cols:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(df[col].fillna(0))
    
    # Encode categoricals
    for col in ['sex', 'vendor_volume_bucket', 'sale_company']:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
    
    # Select feature columns
    feature_cols = [
        # Sale context
        'book_number', 'day_number',
        'log_session_median_price',
        'sex_encoded', 'sale_company_encoded',
        
        # Sire features
        'sire_sold_count_36m', 'sire_total_offered_36m', 'sire_clearance_rate_36m',
        'log_sire_median_price_36m',
        'sire_sold_count_12m', 'sire_total_offered_12m', 'sire_clearance_rate_12m',
        'log_sire_median_price_12m',
        'sire_momentum', 'sire_sample_flag_36m',
        
        # Dam features
        'dam_progeny_sold_count', 'dam_progeny_total_offered_count',
        'log_dam_progeny_median_price', 'dam_first_foal_flag',
        
        # Vendor features
        'vendor_sold_count_36m', 'vendor_total_offered_36m',
        'vendor_clearance_rate_36m', 'log_vendor_median_price_36m',
        'vendor_volume_bucket_encoded', 'vendor_first_seen_flag',
    ]
    
    # Keep only columns that exist
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    X = df[feature_cols].fillna(0)
    
    return X, y, feature_cols, df


def time_split(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series) -> dict:
    """Split data by time: train on older, test on recent."""
    
    # For AUS: Train 2020-2023, Test 2024-2025
    # For NZL: Train 2020-2023, Test 2024-2025
    train_mask = df['sale_year'] <= 2023
    test_mask = df['sale_year'] >= 2024
    
    return {
        'X_train': X[train_mask],
        'X_test': X[test_mask],
        'y_train': y[train_mask],
        'y_test': y[test_mask],
        'df_test': df[test_mask],
    }


# ============================================================================
# BASELINE MODEL
# ============================================================================

def train_baseline(splits: dict) -> dict:
    """Train Elastic Net baseline for sanity check."""
    
    print("\n" + "="*60)
    print("BASELINE MODEL: Elastic Net")
    print("="*60)
    
    model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)
    model.fit(splits['X_train'], splits['y_train'])
    
    train_pred = model.predict(splits['X_train'])
    test_pred = model.predict(splits['X_test'])
    
    train_mae = mean_absolute_error(splits['y_train'], train_pred)
    train_r2 = r2_score(splits['y_train'], train_pred)
    test_mae = mean_absolute_error(splits['y_test'], test_pred)
    test_r2 = r2_score(splits['y_test'], test_pred)
    
    # Naive baseline: predict 0 (session median)
    naive_mae = mean_absolute_error(splits['y_test'], np.zeros(len(splits['y_test'])))
    
    print(f"Train MAE: {train_mae:.4f}")
    print(f"Train R²:  {train_r2:.4f}")
    print(f"Test MAE:  {test_mae:.4f}")
    print(f"Test R²:   {test_r2:.4f}")
    print(f"Naive MAE (predict 0): {naive_mae:.4f}")
    
    if test_mae < naive_mae:
        print("✓ Baseline passes: Better than naive predictor")
    else:
        print("⚠ Warning: Baseline worse than naive predictor")
    
    return {'model': model, 'test_mae': test_mae, 'test_r2': test_r2}


# ============================================================================
# QUANTILE MODELS
# ============================================================================

def train_quantile_models(splits: dict, feature_cols: list) -> dict:
    """Train Q25, Q50, Q75 LightGBM models."""
    
    print("\n" + "="*60)
    print("PRODUCTION MODELS: Quantile GBDT")
    print("="*60)
    
    # Identify categorical feature indices
    cat_features = [i for i, c in enumerate(feature_cols) if '_encoded' in c]
    
    # LightGBM parameters
    base_params = {
        'objective': 'quantile',
        'metric': 'quantile',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'min_data_in_leaf': 100,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42,
    }
    
    # Create datasets
    train_data = lgb.Dataset(
        splits['X_train'], 
        label=splits['y_train'],
        categorical_feature=cat_features
    )
    
    models = {}
    predictions = {}
    
    for quantile, name in [(0.25, 'q25'), (0.50, 'q50'), (0.75, 'q75')]:
        print(f"\nTraining {name.upper()} (alpha={quantile})...")
        
        params = base_params.copy()
        params['alpha'] = quantile
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
        )
        
        models[name] = model
        predictions[f'pred_{name}'] = model.predict(splits['X_test'])
        
        print(f"  Trained {model.num_trees()} trees")
    
    return models, predictions


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_models(splits: dict, predictions: dict) -> dict:
    """Evaluate model performance."""
    
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    y_test = splits['y_test']
    pred_q50 = predictions['pred_q50']
    pred_q25 = predictions['pred_q25']
    pred_q75 = predictions['pred_q75']
    
    mae = mean_absolute_error(y_test, pred_q50)
    rmse = np.sqrt(mean_squared_error(y_test, pred_q50))
    r2 = r2_score(y_test, pred_q50)
    
    print(f"P50 (Expected) Performance:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}")
    
    # Coverage before calibration
    below_q25 = (y_test < pred_q25).mean()
    below_q75 = (y_test < pred_q75).mean()
    in_range = ((y_test >= pred_q25) & (y_test <= pred_q75)).mean()
    
    print(f"\nCoverage (before calibration):")
    print(f"  % below P25: {below_q25*100:.1f}% (target: 25%)")
    print(f"  % below P75: {below_q75*100:.1f}% (target: 75%)")
    print(f"  % in P25-P75 range: {in_range*100:.1f}% (target: 50%)")
    
    # Dollar-space metrics
    df_test = splits['df_test']
    session_median = df_test['session_median_price'].values
    actual_price = df_test['hammer_price'].values
    pred_price = session_median * np.exp(pred_q50)
    
    mape = np.mean(np.abs(actual_price - pred_price) / actual_price) * 100
    print(f"\nDollar-space (P50):")
    print(f"  MAPE: {mape:.1f}%")
    
    return {'mae': mae, 'r2': r2, 'mape': mape}


# ============================================================================
# CALIBRATION
# ============================================================================

def calibrate_models(splits: dict, predictions: dict) -> dict:
    """Calculate calibration offsets for P25/P75."""
    
    print("\n" + "="*60)
    print("CALIBRATION")
    print("="*60)
    
    y_test = splits['y_test'].values
    pred_q25 = predictions['pred_q25']
    pred_q75 = predictions['pred_q75']
    
    # Calculate residuals
    residuals_q25 = y_test - pred_q25
    residuals_q75 = y_test - pred_q75
    
    # Find offsets to achieve target coverage
    offset_p25 = np.percentile(residuals_q25, 25)
    offset_p75 = np.percentile(residuals_q75, 75)
    
    print(f"Calibration offsets (log space):")
    print(f"  P25 offset: {offset_p25:.4f}")
    print(f"  P75 offset: {offset_p75:.4f}")
    
    # Verify calibrated coverage
    cal_q25 = pred_q25 + offset_p25
    cal_q75 = pred_q75 + offset_p75
    
    below_q25_cal = (y_test < cal_q25).mean()
    below_q75_cal = (y_test < cal_q75).mean()
    
    print(f"\nCalibrated coverage:")
    print(f"  % below P25: {below_q25_cal*100:.1f}% (target: 25%)")
    print(f"  % below P75: {below_q75_cal*100:.1f}% (target: 75%)")
    
    return {'offset_p25': offset_p25, 'offset_p75': offset_p75}


# ============================================================================
# SAVE MODELS
# ============================================================================

def save_models(models: dict, offsets: dict, feature_cols: list, country: str):
    """Save trained models and metadata."""
    
    print("\n" + "="*60)
    print("SAVING MODELS")
    print("="*60)
    
    model_dir = get_model_path(country)
    
    # Save LightGBM models
    for name, model in models.items():
        path = f"{model_dir}/mv_v1_{name}.txt"
        model.save_model(path)
        print(f"Saved: {path}")
    
    # Save calibration offsets
    path = f"{model_dir}/calibration_offsets.json"
    with open(path, 'w') as f:
        json.dump(offsets, f, indent=2)
    print(f"Saved: {path}")
    
    # Save feature columns
    path = f"{model_dir}/feature_cols.json"
    with open(path, 'w') as f:
        json.dump(feature_cols, f, indent=2)
    print(f"Saved: {path}")


def save_feature_importance(models: dict, feature_cols: list):
    """Save feature importance report."""
    
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE (P50 Model)")
    print("="*60)
    
    importance = models['q50'].feature_importance(importance_type='gain')
    
    fi_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print("\nTop 15 features:")
    for _, row in fi_df.head(15).iterrows():
        print(f"  {row['feature']:40s} {row['importance']:,.0f}")
    
    report_dir = get_report_path()
    path = f"{report_dir}/feature_importance_{COUNTRY}.csv"
    fi_df.to_csv(path, index=False)
    print(f"\nSaved: {path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*60)
    print(f"MARKET VALUE MODEL {MODEL_VERSION} — TRAINING ({COUNTRY.upper()})")
    print("="*60)
    print(f"Started: {datetime.now()}")
    print(f"Country: {COUNTRY.upper()}")
    
    # Load data
    df = get_training_data()
    print(f"\nLoaded {len(df)} rows")
    
    # Prepare features
    X, y, feature_cols, df = prepare_features(df)
    print(f"Feature count: {len(feature_cols)}")
    
    # Time split
    splits = time_split(df, X, y)
    print(f"Train: {len(splits['X_train']):,} rows (2020-2023)")
    print(f"Test:  {len(splits['X_test']):,} rows (2024-2025)")
    
    # Train baseline
    baseline = train_baseline(splits)
    
    # Train quantile models
    models, predictions = train_quantile_models(splits, feature_cols)
    
    # Evaluate
    metrics = evaluate_models(splits, predictions)
    
    # Calibrate
    offsets = calibrate_models(splits, predictions)
    
    # Save
    save_models(models, offsets, feature_cols, COUNTRY)
    save_feature_importance(models, feature_cols)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Finished: {datetime.now()}")
    print(f"Model artifacts saved to ./models/{COUNTRY}/")
    print(f"Reports saved to ./reports/")


if __name__ == "__main__":
    main()
