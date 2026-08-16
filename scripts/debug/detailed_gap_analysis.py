"""
Detailed gap analysis - find ALL missing dates in existing data
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import os

DATA_DIR = Path("data/historical")


def find_all_gaps():
    """Find ALL missing business days in each stock's data"""
    files = sorted(DATA_DIR.glob("*.csv"))
    files = [f for f in files if not f.name.startswith("_")]

    print("=" * 80)
    print("🔍 DETAILED GAP ANALYSIS - FINDING ALL MISSING DATES")
    print("=" * 80)

    # Get all business days from 2010-01-01 to today
    today = datetime.now().strftime("%Y-%m-%d")
    all_business_days = pd.bdate_range(start="2010-01-01", end=today)
    print(f"Expected Business Days (2010-01-01 to {today}): {len(all_business_days)}")
    print("=" * 80)

    total_summary = []

    for file in files:
        try:
            df = pd.read_csv(file, parse_dates=["date"])
            stock_code = df["code"].iloc[0] if "code" in df.columns else file.stem

            # Get all dates in the file
            actual_dates = set(pd.to_datetime(df["date"]).dt.date)
            expected_dates = set(all_business_days.date)

            # Find ALL missing dates (internal + future gaps)
            missing_dates = sorted(expected_dates - actual_dates)

            # Categorize gaps
            last_date_in_file = df["date"].max().date()
            future_gaps = [d for d in missing_dates if d > last_date_in_file]
            internal_gaps = [d for d in missing_dates if d <= last_date_in_file]

            if len(missing_dates) > 0:
                # Show first 10 missing dates for this stock
                print(f"\n📊 {stock_code} ({file.name})")
                print(f"   Records: {len(df)}, Range: {df['date'].min().date()} to {df['date'].max().date()}")
                print(f"   Total Missing: {len(missing_dates)} days")
                print(f"   Internal Gaps: {len(internal_gaps)} days")
                print(f"   Future Gaps: {len(future_gaps)} days")

                if internal_gaps:
                    # Show sample gaps
                    print(f"   Sample Internal Gaps (first 10):")
                    for gap_date in internal_gaps[:10]:
                        print(f"      ❌ {gap_date}")
                    if len(internal_gaps) > 10:
                        print(f"      ... and {len(internal_gaps) - 10} more")

                total_summary.append({
                    "code": stock_code,
                    "records": len(df),
                    "total_missing": len(missing_dates),
                    "internal_gaps": len(internal_gaps),
                    "future_gaps": len(future_gaps),
                })

        except Exception as e:
            print(f"❌ Error reading {file.name}: {e}")

    # Summary
    if total_summary:
        summary_df = pd.DataFrame(total_summary)
        print("\n" + "=" * 80)
        print("📋 SUMMARY TABLE:")
        print("=" * 80)
        print(summary_df.to_string(index=False))

        print("\n" + "=" * 80)
        print("📊 TOTALS:")
        print("=" * 80)
        print(f"Total Missing Records: {summary_df['total_missing'].sum():,}")
        print(f"Total Internal Gaps:   {summary_df['internal_gaps'].sum():,}")
        print(f"Total Future Gaps:     {summary_df['future_gaps'].sum():,}")

        os.makedirs("logs", exist_ok=True)
        summary_df.to_csv("logs/detailed_gaps_summary.csv", index=False)


if __name__ == "__main__":
    find_all_gaps()
