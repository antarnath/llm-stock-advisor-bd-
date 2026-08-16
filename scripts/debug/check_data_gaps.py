"""
Analyze gaps in existing data
Shows what dates are missing for each stock
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import os

DATA_DIR = Path("data/historical")


def analyze_gaps():
    """Check what dates are missing in each stock's data"""
    files = sorted(DATA_DIR.glob("*.csv"))
    files = [f for f in files if not f.name.startswith("_")]

    print("=" * 80)
    print("📊 DATA GAP ANALYSIS")
    print("=" * 80)
    print(f"Current Date: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"Total Stock Files: {len(files)}")
    print("=" * 80)

    # Expected date range
    expected_start = pd.Timestamp("2010-01-01")
    expected_end = pd.Timestamp(datetime.now().strftime("%Y-%m-%d"))

    # Business days in range
    all_business_days = pd.bdate_range(start=expected_start, end=expected_end)
    print(f"Expected Business Days: {len(all_business_days)}")
    print(f"Expected Range: {expected_start.date()} to {expected_end.date()}")
    print("=" * 80)

    summary = []

    for file in files:
        try:
            df = pd.read_csv(file, parse_dates=["date"])
            stock_code = df["code"].iloc[0] if "code" in df.columns else file.stem

            # Get actual date range
            actual_start = df["date"].min()
            actual_end = df["date"].max()
            actual_days = len(df)

            # Check for missing dates in range
            actual_dates = set(df["date"].dt.date)
            all_dates = set(all_business_days.date)
            missing_dates = sorted(all_dates - actual_dates)

            # Find gap (missing after the data ends)
            future_gap_days = 0
            if actual_end < expected_end:
                future_gap_days = len(
                    pd.bdate_range(start=actual_end + pd.Timedelta(days=1),
                                  end=expected_end)
                )

            # Find internal gaps
            internal_gaps = 0
            if len(missing_dates) > 0 and missing_dates[0] > actual_start.date():
                # Count missing dates within the range
                internal_gaps = len(missing_dates) - future_gap_days

            summary.append({
                "code": stock_code,
                "file": file.name,
                "records": actual_days,
                "date_from": actual_start.date(),
                "date_to": actual_end.date(),
                "days_to_update": future_gap_days,
                "internal_gaps": internal_gaps,
                "total_missing": len(missing_dates),
            })

        except Exception as e:
            print(f"❌ Error reading {file.name}: {e}")

    summary_df = pd.DataFrame(summary)

    # Show summary
    print("\n📋 SUMMARY:")
    print("-" * 80)
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("📊 STATISTICS:")
    print("=" * 80)
    total_missing = summary_df["total_missing"].sum()
    total_to_update = summary_df["days_to_update"].sum()
    stocks_with_gaps = (summary_df["days_to_update"] > 0).sum()

    print(f"Total Missing Records: {total_missing:,}")
    print(f"Days to Update (2026): {total_to_update:,}")
    print(f"Stocks with Future Gaps: {stocks_with_gaps}/{len(summary_df)}")
    print(f"Stocks with Internal Gaps: {(summary_df['internal_gaps'] > 0).sum()}/{len(summary_df)}")

    # Show specific date range to update
    print("\n📅 DATE RANGE TO UPDATE:")
    print("-" * 80)
    print(f"Start: 2026-01-01")
    print(f"End:   {datetime.now().strftime('%Y-%m-%d')}")
    print(f"Business Days: {len(pd.bdate_range(start='2026-01-01', end=datetime.now().strftime('%Y-%m-%d')))}")

    # Save summary
    summary_df.to_csv("logs/data_gaps_analysis.csv", index=False)
    print(f"\n💾 Saved analysis to: logs/data_gaps_analysis.csv")
    print("=" * 80)


if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    analyze_gaps()
