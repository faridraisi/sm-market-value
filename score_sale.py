#!/usr/bin/env python3
"""
Score a sale: rebuild features + run model inference.

This script orchestrates the full pipeline:
1. Runs run_rebuild.py to generate sale_{sale_id}_inference.csv
2. Runs score_lots.py to score lots and output to CSV or database

Model selection is configurable via .env:
- AUS_MODEL=aus (default)
- NZL_MODEL=nzl (or NZL_MODEL=aus to use AUS model for NZL sales)
- USA_MODEL=usa

Usage:
    python score_sale.py --sale-id 2094                # Output to CSV (default)
    python score_sale.py --sale-id 2094 --output csv   # Output to CSV
    python score_sale.py --sale-id 2094 --output db    # Output to database
"""

import argparse
import subprocess
import sys


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

    # Step 1: Rebuild features
    print("=" * 60)
    print(f"Step 1: Rebuilding features for sale {args.sale_id}")
    print("=" * 60)
    subprocess.run(
        [sys.executable, "run_rebuild.py", "--sale-id", str(args.sale_id)],
        check=True,
    )

    # Step 2: Score lots
    print("\n" + "=" * 60)
    print(f"Step 2: Scoring lots for sale {args.sale_id}")
    print("=" * 60)
    subprocess.run(
        [
            sys.executable,
            "score_lots.py",
            "--sale-id",
            str(args.sale_id),
            "--output",
            args.output,
        ],
        check=True,
    )

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)
    print(f"  Inference data: csv/sale_{args.sale_id}_inference.csv")
    if args.output == "csv":
        print(f"  Scored lots:    csv/sale_{args.sale_id}_scored.csv")
    else:
        print("  Scored lots:    tblHorseAnalytics (database)")


if __name__ == "__main__":
    main()
