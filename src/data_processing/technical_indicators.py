"""
Data Processing Pipeline - Phase 2
Adds technical indicators to stock data for ML model training

Technical Indicators Added:
- SMA (Simple Moving Average): 5, 10, 20, 50, 100, 200 days
- EMA (Exponential Moving Average): 12, 26, 50 days
- RSI (Relative Strength Index): 14 days
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands: 20 days, 2 std
- ATR (Average True Range): 14 days
- Returns: 1d, 5d, 20d
- Volatility: 30 days, 60 days

Input:  data/historical/{STOCK}.csv (30 files)
Output: data/processed/{STOCK}_processed.csv (30 files with features)

Usage:
    python scripts/data_processing.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
from datetime import datetime


# Project paths (centralized)
from src.utils.config import (
    PROJECT_ROOT,
    RAW_STOCKS_DIR as INPUT_DIR,
    PROCESSED_DATA_DIR as OUTPUT_DIR,
    LOGS_DIR,
)


class DataProcessor:
    """Process stock data and add technical indicators"""

    def __init__(self):
        self.input_dir = INPUT_DIR
        self.output_dir = OUTPUT_DIR
        self.logs_dir = LOGS_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

    def calculate_sma(self, df, windows=[5, 10, 20, 50, 100, 200]):
        """Simple Moving Average"""
        for window in windows:
            df[f'SMA_{window}'] = df['close'].rolling(window=window).mean()
        return df

    def calculate_ema(self, prices, span):
        """Exponential Moving Average"""
        return prices.ewm(span=span, adjust=False).mean()

    def calculate_rsi(self, prices, period=14):
        """Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, df, fast=12, slow=26, signal=9):
        """Moving Average Convergence Divergence"""
        ema_fast = self.calculate_ema(df['close'], fast)
        ema_slow = self.calculate_ema(df['close'], slow)
        df['MACD'] = ema_fast - ema_slow
        df['MACD_Signal'] = self.calculate_ema(df['MACD'], signal)
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        return df

    def calculate_bollinger_bands(self, df, window=20, num_std=2):
        """Bollinger Bands"""
        df['BB_Middle'] = df['close'].rolling(window=window).mean()
        bb_std = df['close'].rolling(window=window).std()
        df['BB_Upper'] = df['BB_Middle'] + (num_std * bb_std)
        df['BB_Lower'] = df['BB_Middle'] - (num_std * bb_std)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['close'] - df['BB_Lower']) / df['BB_Width']
        return df

    def calculate_atr(self, df, period=14):
        """Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    def calculate_returns(self, df):
        """Calculate returns"""
        df['Returns_1d'] = df['close'].pct_change(1)
        df['Returns_5d'] = df['close'].pct_change(5)
        df['Returns_20d'] = df['close'].pct_change(20)
        df['Log_Returns'] = np.log(df['close'] / df['close'].shift(1))
        return df

    def calculate_volatility(self, df, windows=[30, 60]):
        """Calculate volatility"""
        returns = df['close'].pct_change()
        for window in windows:
            df[f'Volatility_{window}d'] = returns.rolling(window=window).std() * np.sqrt(252)
        return df

    def calculate_volume_indicators(self, df):
        """Volume-based indicators"""
        df['Volume_SMA_20'] = df['volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['volume'] / df['Volume_SMA_20']
        return df

    def process_stock(self, filepath):
        """Process a single stock file"""
        try:
            # Read data
            df = pd.read_csv(filepath, parse_dates=['date'])
            stock_code = df['code'].iloc[0] if 'code' in df.columns else filepath.stem

            print(f"   📊 Processing {stock_code}...")
            print(f"      Records: {len(df)}")

            # Calculate all indicators
            df = self.calculate_sma(df)
            print(f"      ✓ SMA calculated")

            # EMAs
            for span in [12, 26, 50]:
                df[f'EMA_{span}'] = self.calculate_ema(df['close'], span)
            print(f"      ✓ EMA calculated")

            # RSI
            df['RSI_14'] = self.calculate_rsi(df['close'], 14)
            print(f"      ✓ RSI calculated")

            # MACD
            df = self.calculate_macd(df)
            print(f"      ✓ MACD calculated")

            # Bollinger Bands
            df = self.calculate_bollinger_bands(df)
            print(f"      ✓ Bollinger Bands calculated")

            # ATR
            df['ATR_14'] = self.calculate_atr(df, 14)
            print(f"      ✓ ATR calculated")

            # Returns
            df = self.calculate_returns(df)
            print(f"      ✓ Returns calculated")

            # Volatility
            df = self.calculate_volatility(df)
            print(f"      ✓ Volatility calculated")

            # Volume indicators
            df = self.calculate_volume_indicators(df)
            print(f"      ✓ Volume indicators calculated")

            # Target variable: next day return
            df['Target_Return_1d'] = df['close'].pct_change().shift(-1)
            df['Target_Price_1d'] = df['close'].shift(-1)

            # Drop rows with NaN (from rolling calculations)
            initial_rows = len(df)
            df = df.dropna()
            final_rows = len(df)

            print(f"      ✓ Cleaned: {initial_rows} → {final_rows} rows (dropped {initial_rows - final_rows})")

            # Save processed file
            output_path = self.output_dir / f"{stock_code}_processed.csv"
            df.to_csv(output_path, index=False)
            print(f"      💾 Saved: {output_path.name}")

            return {
                'code': stock_code,
                'input_rows': initial_rows,
                'output_rows': final_rows,
                'features_added': len(df.columns) - 11,  # Original columns
                'status': 'success'
            }

        except Exception as e:
            print(f"   ❌ Error processing {filepath.name}: {e}")
            return {'code': filepath.stem, 'status': 'failed', 'error': str(e)}

    def process_all_stocks(self):
        """Process all stock files"""
        print("""
╔══════════════════════════════════════════════════════════════╗
║   Data Processing Pipeline - Phase 2                        ║
║   Adding Technical Indicators to Stock Data                 ║
╚══════════════════════════════════════════════════════════════╝
        """)

        # Get all stock files
        files = sorted([f for f in self.input_dir.glob("*.csv") if not f.name.startswith("_")])

        print(f"📁 Input Directory: {self.input_dir}")
        print(f"📁 Output Directory: {self.output_dir}")
        print(f"📊 Total Files: {len(files)}")
        print("=" * 70)

        results = []
        successful = 0
        failed = []

        for i, filepath in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}] {filepath.name}")
            result = self.process_stock(filepath)

            if result['status'] == 'success':
                successful += 1
                results.append(result)
            else:
                failed.append(filepath.name)

        # Create summary report
        if results:
            results_df = pd.DataFrame(results)
            summary_path = self.output_dir / "_processing_summary.csv"
            results_df.to_csv(summary_path, index=False)

            print("\n" + "=" * 70)
            print("📊 PROCESSING SUMMARY")
            print("=" * 70)
            print(f"Total Files: {len(files)}")
            print(f"✅ Successful: {successful}")
            print(f"❌ Failed: {len(failed)}")

            if failed:
                print(f"Failed files: {', '.join(failed)}")

            # Show statistics
            total_features = results_df['features_added'].iloc[0] if len(results_df) > 0 else 0
            avg_rows = results_df['output_rows'].mean()

            print(f"\n📈 Features Added per Stock: {total_features}")
            print(f"📊 Average Clean Rows: {avg_rows:.0f}")
            print(f"💾 Summary saved: {summary_path}")

        print("=" * 70)
        print("✨ Data processing complete!")
        print(f"📁 Processed files in: {self.output_dir}")
        print("=" * 70)


def main():
    processor = DataProcessor()
    processor.process_all_stocks()


if __name__ == "__main__":
    main()
