"""
Verify data integrity - check if all business days are present
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

DATA_DIR = Path("data/historical")


def verify_integrity():
    """Verify if all business days are present in each file"""
    files = sorted(DATA_DIR.glob("*.csv"))
    files = [f for f in files if not f.name.startswith("_")]

    print("=" * 80)
    print("✅ DATA INTEGRITY VERIFICATION")
    print("=" * 80)
    print(f"Checking if all business days (Mon-Fri, excluding holidays) are present")
    print("=" * 80)

    issues_found = 0

    for file in files[:5]:  # Check first 5 stocks as samples
        df = pd.read_csv(file, parse_dates=["date"])
        stock_code = df["code"].iloc[0] if "code" in df.columns else file.stem

        # Get all business days in the file's date range
        date_min = df["date"].min()
        date_max = df["date"].max()
        expected_business_days = pd.bdate_range(start=date_min, end=date_max)

        # Get actual dates in file
        actual_dates = set(df["date"].dt.date)
        expected_dates = set(expected_business_days.date)

        # Find missing dates
        missing = sorted(expected_dates - actual_dates)
        extra = sorted(actual_dates - expected_dates)

        print(f"\n📊 {stock_code} ({file.name})")
        print(f"   Range: {date_min.date()} to {date_max.date()}")
        print(f"   Records: {len(df)}")
        print(f"   Expected Business Days: {len(expected_business_days)}")
        print(f"   Missing Business Days: {len(missing)}")
        print(f"   Extra Dates (non-business): {len(extra)}")

        if missing:
            print(f"   ⚠️  Missing dates (first 10):")
            for d in missing[:10]:
                dt = pd.to_datetime(d)
                print(f"      - {d} ({dt.day_name()})")
            issues_found += 1
        else:
            print(f"   ✅ All business days present!")

        if extra:
            print(f"   ⚠️  Extra dates (first 5):")
            for d in extra[:5]:
                dt = pd.to_datetime(d)
                print(f"      + {d} ({dt.day_name()})")

    print("\n" + "=" * 80)
    print(f"Issues found: {issues_found}/5")
    print("=" * 80)


if __name__ == "__main__":
    verify_integrity()
