#!/usr/bin/env python3
"""
Run the golden table rebuild and export sale inference data to CSV.
All computation done in Python/pandas using read-only SQL queries.

Usage:
    python run_rebuild.py --sale-id 2094
"""

import argparse
import os
import sys
from datetime import timedelta

import numpy as np
import pandas as pd
import pyodbc

try:
    from src.config import config
except ModuleNotFoundError:
    from config import config


def get_connection():
    """Create and return a database connection."""
    db = config.db
    if not all([db.server, db.name, db.user, db.password]):
        raise ValueError("Missing required database credentials in .env file")

    conn_str = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={db.server};"
        f"DATABASE={db.name};"
        f"UID={db.user};"
        f"PWD={db.password}"
    )
    return pyodbc.connect(conn_str)


def fetch_sale_country(conn, sale_id):
    """Get the country code for a sale."""
    query = """
    SELECT CN.countryCode
    FROM tblSales SL
    JOIN tblCountry CN ON SL.countryId = CN.id
    WHERE SL.Id = ?
    """
    cursor = conn.cursor()
    cursor.execute(query, (sale_id,))
    row = cursor.fetchone()
    cursor.close()
    if row:
        return row[0]
    raise ValueError(f"Sale {sale_id} not found or has no country")


def get_hist_countries(country_code):
    """Get list of country codes to include in historical lookback.

    Checks config.json for hist_countries mapping.
    Falls back to just the sale's country if not configured.
    """
    return config.app.get_hist_countries(country_code)


def fetch_base_lots(conn, sale_id):
    """Fetch base lots for the target sale."""
    year_start = config.app.year_start
    year_end = config.app.get_year_end()

    query = f"""
    SELECT
        LT.Id AS lot_id,
        LT.salesId,
        CAST(SL.startDate AS DATE) AS asOfDate,
        SC.Id AS salesCompanyId,
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
        H.horseName AS horse_name,
        SIRE.horseName AS sire_name,
        CAST(LT.price AS DECIMAL(12,2)) AS hammer_price,
        CAST(ISNULL(LT.isPassedIn, 0) AS BIT) AS isPassedIn,
        CAST(ISNULL(LT.isWithdrawn, 0) AS BIT) AS isWithdrawn,
        CASE WHEN LT.price > 0 AND ISNULL(LT.isPassedIn,0) = 0 AND ISNULL(LT.isWithdrawn,0) = 0 THEN 1 ELSE 0 END AS isSold_int,
        CASE WHEN ISNULL(LT.isPassedIn,0) = 1 THEN 1 ELSE 0 END AS isPassedIn_int
    FROM tblSalesLot LT
    JOIN tblSales SL ON LT.salesId = SL.Id
    JOIN tblSalesCompany SC ON SL.salesCompanyId = SC.Id
    JOIN tblSalesLotType LTP ON LT.lotType = LTP.Id
    JOIN tblCountry CN ON SL.countryId = CN.id
    JOIN tblHorse H ON LT.horseId = H.id
    LEFT JOIN tblHorse SIRE ON H.sireId = SIRE.id
    WHERE LT.salesId = ?
        AND YEAR(SL.startDate) BETWEEN {year_start} AND {year_end}
        AND LTP.salesLotTypeName = 'Yearling'
        AND ISNULL(LT.isWithdrawn, 0) = 0
    """
    df = pd.read_sql(query, conn, params=[sale_id])
    df["asOfDate"] = pd.to_datetime(df["asOfDate"])
    return df


def fetch_hist_lots(conn, country_codes):
    """Fetch historical yearling lots for the given countries."""
    placeholders = ','.join(['?' for _ in country_codes])
    query = f"""
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
    WHERE CN.countryCode IN ({placeholders})
        AND LTP.salesLotTypeName = 'Yearling'
        AND ISNULL(LT.isWithdrawn, 0) = 0
    """
    df = pd.read_sql(query, conn, params=country_codes)
    df["saleDate"] = pd.to_datetime(df["saleDate"])
    return df


def compute_sale_median(base_lots):
    """Compute session median price for the sale."""
    sold = base_lots[(base_lots["isSold_int"] == 1) & (base_lots["hammer_price"] > 0)]
    if len(sold) > 0:
        return sold["hammer_price"].median()
    return None


def fetch_prior_year_median(conn, sale_name, sale_year):
    """Fetch prior year's median for the same sale name."""
    query = """
    SELECT TOP 1
        (
            SELECT TOP 1 PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY LT.price) OVER ()
            FROM tblSalesLot LT
            JOIN tblSales SL ON LT.salesId = SL.Id
            JOIN tblSalesLotType LTP ON LT.lotType = LTP.Id
            WHERE SL.salesName = ?
                AND YEAR(SL.startDate) = ?
                AND LTP.salesLotTypeName = 'Yearling'
                AND LT.price > 0
                AND ISNULL(LT.isPassedIn, 0) = 0
                AND ISNULL(LT.isWithdrawn, 0) = 0
        ) AS median_price
    """
    prior_year = int(sale_year) - 1
    cursor = conn.cursor()
    cursor.execute(query, (str(sale_name), prior_year))
    row = cursor.fetchone()
    cursor.close()
    if row and row[0]:
        return float(row[0])
    return None


def compute_sire_metrics(hist_lots, sire_ids, as_of_date, months):
    """Compute sire metrics for given lookback period."""
    cutoff = as_of_date - pd.DateOffset(months=months)

    # Filter to lookback window, before as_of_date
    window = hist_lots[
        (hist_lots["saleDate"] < as_of_date) &
        (hist_lots["saleDate"] >= cutoff) &
        (hist_lots["sireId"].isin(sire_ids))
    ].copy()

    if window.empty:
        return pd.DataFrame({
            "sireId": sire_ids,
            f"sire_sold_count_{months}m": 0,
            f"sire_passedin_count_{months}m": 0,
            f"sire_total_offered_{months}m": 0,
            f"sire_clearance_rate_{months}m": None,
            f"sire_median_price_{months}m": None,
        })

    # Aggregate counts
    agg = window.groupby("sireId").agg(
        sold_count=("isSold_int", "sum"),
        passedin_count=("isPassedIn_int", "sum"),
    ).reset_index()

    agg["total_offered"] = agg["sold_count"] + agg["passedin_count"]
    agg["clearance_rate"] = np.where(
        agg["total_offered"] > 0,
        np.round(100.0 * agg["sold_count"] / agg["total_offered"], 1),
        None
    )

    # Compute median for sold lots only
    sold_window = window[(window["isSold_int"] == 1) & (window["hammer_price"] > 0)]
    medians = sold_window.groupby("sireId")["hammer_price"].median().reset_index()
    medians.columns = ["sireId", "median_price"]

    agg = agg.merge(medians, on="sireId", how="left")

    # Rename columns
    agg = agg.rename(columns={
        "sold_count": f"sire_sold_count_{months}m",
        "passedin_count": f"sire_passedin_count_{months}m",
        "total_offered": f"sire_total_offered_{months}m",
        "clearance_rate": f"sire_clearance_rate_{months}m",
        "median_price": f"sire_median_price_{months}m",
    })

    # Fill in missing sires
    all_sires = pd.DataFrame({"sireId": sire_ids})
    result = all_sires.merge(agg, on="sireId", how="left")
    result[f"sire_sold_count_{months}m"] = result[f"sire_sold_count_{months}m"].fillna(0).astype(int)
    result[f"sire_passedin_count_{months}m"] = result[f"sire_passedin_count_{months}m"].fillna(0).astype(int)
    result[f"sire_total_offered_{months}m"] = result[f"sire_total_offered_{months}m"].fillna(0).astype(int)

    return result


def compute_dam_stats(hist_lots, dam_ids, as_of_date):
    """Compute dam progeny stats (all history before as_of_date)."""
    window = hist_lots[
        (hist_lots["saleDate"] < as_of_date) &
        (hist_lots["damId"].isin(dam_ids))
    ].copy()

    if window.empty:
        return pd.DataFrame({
            "damId": dam_ids,
            "dam_progeny_sold_count": 0,
            "dam_progeny_passedin_count": 0,
            "dam_progeny_total_offered_count": 0,
            "dam_progeny_median_price": None,
            "dam_first_foal_flag": 1,
        })

    # Aggregate counts
    agg = window.groupby("damId").agg(
        dam_progeny_sold_count=("isSold_int", "sum"),
        dam_progeny_passedin_count=("isPassedIn_int", "sum"),
    ).reset_index()

    agg["dam_progeny_total_offered_count"] = agg["dam_progeny_sold_count"] + agg["dam_progeny_passedin_count"]

    # Compute median for sold lots only
    sold_window = window[(window["isSold_int"] == 1) & (window["hammer_price"] > 0)]
    medians = sold_window.groupby("damId")["hammer_price"].median().reset_index()
    medians.columns = ["damId", "dam_progeny_median_price"]

    agg = agg.merge(medians, on="damId", how="left")
    agg["dam_first_foal_flag"] = (agg["dam_progeny_sold_count"] == 0).astype(int)

    # Fill in missing dams
    all_dams = pd.DataFrame({"damId": dam_ids})
    result = all_dams.merge(agg, on="damId", how="left")
    result["dam_progeny_sold_count"] = result["dam_progeny_sold_count"].fillna(0).astype(int)
    result["dam_progeny_passedin_count"] = result["dam_progeny_passedin_count"].fillna(0).astype(int)
    result["dam_progeny_total_offered_count"] = result["dam_progeny_total_offered_count"].fillna(0).astype(int)
    result["dam_first_foal_flag"] = result["dam_first_foal_flag"].fillna(1).astype(int)

    return result


def compute_vendor_metrics(hist_lots, vendor_ids, as_of_date):
    """Compute vendor metrics for 36-month lookback."""
    cutoff = as_of_date - pd.DateOffset(months=36)

    window = hist_lots[
        (hist_lots["saleDate"] < as_of_date) &
        (hist_lots["saleDate"] >= cutoff) &
        (hist_lots["vendorId"].isin(vendor_ids))
    ].copy()

    if window.empty:
        return pd.DataFrame({
            "vendorId": vendor_ids,
            "vendor_sold_count_36m": 0,
            "vendor_passedin_count_36m": 0,
            "vendor_total_offered_36m": 0,
            "vendor_clearance_rate_36m": None,
            "vendor_median_price_36m": None,
            "vendor_volume_bucket": "New",
            "vendor_first_seen_flag": 1,
        })

    # Aggregate counts
    agg = window.groupby("vendorId").agg(
        vendor_sold_count_36m=("isSold_int", "sum"),
        vendor_passedin_count_36m=("isPassedIn_int", "sum"),
    ).reset_index()

    agg["vendor_total_offered_36m"] = agg["vendor_sold_count_36m"] + agg["vendor_passedin_count_36m"]
    agg["vendor_clearance_rate_36m"] = np.where(
        agg["vendor_total_offered_36m"] > 0,
        np.round(100.0 * agg["vendor_sold_count_36m"] / agg["vendor_total_offered_36m"], 1),
        None
    )

    # Compute median for sold lots only
    sold_window = window[(window["isSold_int"] == 1) & (window["hammer_price"] > 0)]
    medians = sold_window.groupby("vendorId")["hammer_price"].median().reset_index()
    medians.columns = ["vendorId", "vendor_median_price_36m"]

    agg = agg.merge(medians, on="vendorId", how="left")

    # Volume bucket
    def volume_bucket(count):
        if count == 0:
            return "New"
        elif count <= 5:
            return "Small"
        elif count <= 20:
            return "Medium"
        else:
            return "Large"

    agg["vendor_volume_bucket"] = agg["vendor_sold_count_36m"].apply(volume_bucket)
    agg["vendor_first_seen_flag"] = (agg["vendor_sold_count_36m"] == 0).astype(int)

    # Fill in missing vendors
    all_vendors = pd.DataFrame({"vendorId": vendor_ids})
    result = all_vendors.merge(agg, on="vendorId", how="left")
    result["vendor_sold_count_36m"] = result["vendor_sold_count_36m"].fillna(0).astype(int)
    result["vendor_passedin_count_36m"] = result["vendor_passedin_count_36m"].fillna(0).astype(int)
    result["vendor_total_offered_36m"] = result["vendor_total_offered_36m"].fillna(0).astype(int)
    result["vendor_volume_bucket"] = result["vendor_volume_bucket"].fillna("New")
    result["vendor_first_seen_flag"] = result["vendor_first_seen_flag"].fillna(1).astype(int)

    return result


def build_features(base_lots, hist_lots, country_code: str, conn=None):
    """Build all features for the base lots."""
    if base_lots.empty:
        return base_lots

    as_of_date = base_lots["asOfDate"].iloc[0]
    sale_name = base_lots["sale_name"].iloc[0]
    sale_year = base_lots["sale_year"].iloc[0]

    # Compute sale median with prior year fallback
    session_median_price = compute_sale_median(base_lots)
    if session_median_price is None and conn is not None:
        print("  No sold lots - fetching prior year median...")
        session_median_price = fetch_prior_year_median(conn, sale_name, sale_year)
        if session_median_price:
            print(f"  Using prior year median: {session_median_price}")
    base_lots["session_median_price"] = session_median_price

    # Get unique IDs
    sire_ids = base_lots["sireId"].dropna().unique()
    dam_ids = base_lots["damId"].dropna().unique()
    vendor_ids = base_lots["vendorId"].dropna().unique()

    # Compute sire metrics (36m and 12m)
    print("  Computing sire metrics...")
    sire_36m = compute_sire_metrics(hist_lots, sire_ids, as_of_date, 36)
    sire_12m = compute_sire_metrics(hist_lots, sire_ids, as_of_date, 12)
    sire_metrics = sire_36m.merge(sire_12m, on="sireId", how="outer")

    # Sire momentum and sample flag
    sire_metrics["sire_momentum"] = sire_metrics["sire_median_price_12m"] - sire_metrics["sire_median_price_36m"]
    min_count = config.app.get_sire_sample_min_count(country_code)
    sire_metrics["sire_sample_flag_36m"] = (sire_metrics["sire_sold_count_36m"] < min_count).astype(int)

    # Compute dam stats
    print("  Computing dam stats...")
    dam_stats = compute_dam_stats(hist_lots, dam_ids, as_of_date)

    # Compute vendor metrics
    print("  Computing vendor metrics...")
    vendor_metrics = compute_vendor_metrics(hist_lots, vendor_ids, as_of_date)

    # Merge all features back to base lots
    print("  Assembling features...")
    result = base_lots.merge(sire_metrics, on="sireId", how="left")
    result = result.merge(dam_stats, on="damId", how="left")
    result = result.merge(vendor_metrics, on="vendorId", how="left")

    # Compute delta features
    result["sire_vs_sale_median_delta"] = result["sire_median_price_36m"] - result["session_median_price"]
    result["dam_vs_sale_median_delta"] = result["dam_progeny_median_price"] - result["session_median_price"]
    result["vendor_vs_sale_median_delta"] = result["vendor_median_price_36m"] - result["session_median_price"]

    return result


def export_inference(df, output_path):
    """Export the inference data to CSV."""
    columns = [
        "lot_id",
        "horseId",
        "salesId",
        "lot_number",
        "book_number",
        "sex",
        "horse_name",
        "sire_name",
        "sale_company",
        "sale_year",
        "day_number",
        "session_median_price",
        "sire_sold_count_36m", "sire_total_offered_36m", "sire_clearance_rate_36m", "sire_median_price_36m",
        "sire_sold_count_12m", "sire_total_offered_12m", "sire_clearance_rate_12m", "sire_median_price_12m",
        "sire_momentum", "sire_sample_flag_36m",
        "dam_progeny_sold_count", "dam_progeny_total_offered_count", "dam_progeny_median_price", "dam_first_foal_flag",
        "vendor_sold_count_36m", "vendor_total_offered_36m", "vendor_clearance_rate_36m", "vendor_median_price_36m",
        "vendor_volume_bucket", "vendor_first_seen_flag",
    ]

    # Only include columns that exist
    available = [c for c in columns if c in df.columns]
    output_df = df[available].sort_values("lot_number")

    output_df.to_csv(output_path, index=False)
    print(f"Exported {len(output_df)} rows to {output_path}")
    return output_df


def rebuild_sale_features(sale_id: int, export_csv: bool = True) -> pd.DataFrame:
    """Rebuild features for a sale. Returns DataFrame."""
    conn = get_connection()
    country_code = fetch_sale_country(conn, sale_id)
    hist_countries = get_hist_countries(country_code)

    base_lots = fetch_base_lots(conn, sale_id)
    if base_lots.empty:
        conn.close()
        return base_lots

    hist_lots = fetch_hist_lots(conn, hist_countries)
    features = build_features(base_lots, hist_lots, country_code, conn)
    conn.close()

    if export_csv:
        os.makedirs("csv", exist_ok=True)
        export_inference(features, f"csv/sale_{sale_id}_inference.csv")

    return features


def main():
    parser = argparse.ArgumentParser(
        description="Run golden table rebuild and export sale inference data."
    )
    parser.add_argument(
        "--sale-id",
        type=int,
        required=True,
        help="Sale ID to export inference data for",
    )
    args = parser.parse_args()

    os.makedirs("csv", exist_ok=True)
    output_file = f"csv/sale_{args.sale_id}_inference.csv"

    try:
        conn = get_connection()
        print("Connected to database.")

        # Fetch country for the sale and determine historical countries
        country_code = fetch_sale_country(conn, args.sale_id)
        hist_countries = get_hist_countries(country_code)
        print(f"Sale country: {country_code}")
        print(f"Historical countries: {hist_countries}")

        # Fetch data with read-only queries
        print(f"Fetching base lots for sale {args.sale_id}...")
        base_lots = fetch_base_lots(conn, args.sale_id)
        print(f"  Found {len(base_lots)} lots")

        if base_lots.empty:
            print("No lots found for this sale.")
            conn.close()
            sys.exit(1)

        print(f"Fetching historical lots for {hist_countries}...")
        hist_lots = fetch_hist_lots(conn, hist_countries)
        print(f"  Found {len(hist_lots)} historical lots")

        # Build features in pandas (pass conn for prior year median fallback)
        print("Building features...")
        features = build_features(base_lots, hist_lots, country_code, conn)

        conn.close()
        print("Database connection closed.")

        # Export to CSV
        print(f"Exporting inference data...")
        export_inference(features, output_file)

        print("Done.")

    except pyodbc.Error as e:
        print(f"Database error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
