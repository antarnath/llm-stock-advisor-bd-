"""
Data Processing Pipeline - Phase 2 (v2, LEAK-FREE)
Adds technical indicators to stock data for ML model training, with strict
forward-shift to eliminate same-day target leakage.

What changed vs v1:
- Every engineered indicator is computed first, then shifted by 1 day so that
  row t contains the indicator computed from data up to day t-1.
- Raw OHLCV columns are kept in the file (for reference) but excluded from
  features by the trainer (see baseline_trainer_v2.py).
- Targets remain: Target_Return_1d = close[t+1]/close[t] - 1, Target_Price_1d = close[t+1].
- Output filename suffix: `_processed_v2.csv` (preserves v1 files).

Technical Indicators Added (all lag-1, no same-day info):
- SMA (Simple Moving Average): 5, 10, 20, 50, 100, 200 days
- EMA (Exponential Moving Average): 12, 26, 50 days
- RSI (Relative Strength Index): 14 days
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands: 20 days, 2 std
- ATR (Average True Range): 14 days
- Returns: 1d, 5d, 20d, Log
- Volatility: 30 days, 60 days
- Volume: SMA_20, Volume_Ratio

Input:  data/raw/stocks/{STOCK}.csv (30 files)
Output: data/processed/{STOCK}_processed_v2.csv (30 files)

Usage:
    python src/data_processing/technical_indicators_v2.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

# Project paths (centralized)
from src.utils.config import (
    PROJECT_ROOT,
    RAW_STOCKS_DIR as INPUT_DIR,
    PROCESSED_DATA_DIR as OUTPUT_DIR,
    LOGS_DIR,
)
from src.utils.logger import get_logger

logger = get_logger("technical_indicators_v2")


class DataProcessorV2:
    """Process stock data and add leak-free technical indicators."""

    # Final file suffix
    SUFFIX = "_v2"

    def __init__(self):
        self.input_dir = INPUT_DIR
        self.output_dir = OUTPUT_DIR
        self.logs_dir = LOGS_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

    # ----- Indicator calculation (copied from v1, identical math) -----

    def calculate_sma(self, df, windows=(5, 10, 20, 50, 100, 200)):
        """Simple Moving Average."""
        for window in windows:
            df[f'SMA_{window}'] = df['close'].rolling(window=window).mean()
        return df

    def calculate_ema(self, prices, span):
        """Exponential Moving Average."""
        return prices.ewm(span=span, adjust=False).mean()

    def calculate_rsi(self, prices, period=14):
        """Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, df, fast=12, slow=26, signal=9):
        """Moving Average Convergence Divergence."""
        ema_fast = self.calculate_ema(df['close'], fast)
        ema_slow = self.calculate_ema(df['close'], slow)
        df['MACD'] = ema_fast - ema_slow
        df['MACD_Signal'] = self.calculate_ema(df['MACD'], signal)
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        return df

    def calculate_bollinger_bands(self, df, window=20, num_std=2):
        """Bollinger Bands."""
        df['BB_Middle'] = df['close'].rolling(window=window).mean()
        bb_std = df['close'].rolling(window=window).std()
        df['BB_Upper'] = df['BB_Middle'] + (num_std * bb_std)
        df['BB_Lower'] = df['BB_Middle'] - (num_std * bb_std)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['close'] - df['BB_Lower']) / df['BB_Width']
        return df

    def calculate_atr(self, df, period=14):
        """Average True Range."""
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
        """Calculate returns."""
        df['Returns_1d'] = df['close'].pct_change(1)
        df['Returns_5d'] = df['close'].pct_change(5)
        df['Returns_20d'] = df['close'].pct_change(20)
        df['Log_Returns'] = np.log(df['close'] / df['close'].shift(1))
        return df

    def calculate_volatility(self, df, windows=(30, 60)):
        """Calculate volatility."""
        returns = df['close'].pct_change()
        for window in windows:
            df[f'Volatility_{window}d'] = returns.rolling(window=window).std() * np.sqrt(252)
        return df

    def calculate_volume_indicators(self, df):
        """Volume-based indicators."""
        df['Volume_SMA_20'] = df['volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['volume'] / df['Volume_SMA_20']
        return df

    # ----- Main per-stock pipeline -----

    def process_stock(self, filepath):
        """Process a single stock file with strict leak-free lag."""
        try:
            df = pd.read_csv(filepath, parse_dates=['date'])
            stock_code = df['code'].iloc[0] if 'code' in df.columns else filepath.stem

            logger.info(f"📊 Processing {stock_code}...")
            logger.info(f"   Records: {len(df)}")

            # Step 1: compute all indicators on a copy
            df_ind = df.copy()
            df_ind = self.calculate_sma(df_ind)
            for span in [12, 26, 50]:
                df_ind[f'EMA_{span}'] = self.calculate_ema(df_ind['close'], span)
            df_ind['RSI_14'] = self.calculate_rsi(df_ind['close'], 14)
            df_ind = self.calculate_macd(df_ind)
            df_ind = self.calculate_bollinger_bands(df_ind)
            df_ind['ATR_14'] = self.calculate_atr(df_ind, 14)
            df_ind = self.calculate_returns(df_ind)
            df_ind = self.calculate_volatility(df_ind)
            df_ind = self.calculate_volume_indicators(df_ind)

            # Step 2: bring raw OHLCV + meta into result, then attach LAGGED indicators
            result = df.copy()
            original_cols = set(df.columns)
            for col in df_ind.columns:
                if col not in original_cols:
                    # CRITICAL: shift by 1 so indicator at row[t] reflects info up to t-1
                    result[col] = df_ind[col].shift(1)

            # Step 3: targets (unchanged)
            result['Target_Return_1d'] = result['close'].pct_change().shift(-1)
            result['Target_Price_1d'] = result['close'].shift(-1)

            # Step 4: drop rows with NaN (lost ~200 rows × 1 lag + ~200 rolling)
            initial_rows = len(result)
            result = result.dropna().reset_index(drop=True)
            final_rows = len(result)

            logger.info(f"   Cleaned: {initial_rows} → {final_rows} rows (dropped {initial_rows - final_rows})")
            logger.info(f"   Total features: {len(result.columns)}")

            # Step 5: save
            output_path = self.output_dir / f"{stock_code}_processed{self.SUFFIX}.csv"
            result.to_csv(output_path, index=False)
            logger.info(f"   💾 Saved: {output_path.name}")

            return {
                'code': stock_code,
                'input_rows': initial_rows,
                'output_rows': final_rows,
                'features_added': len(result.columns) - len(original_cols),
                'status': 'success',
                'suffix': self.SUFFIX,
            }

        except Exception as e:
            logger.error(f"❌ Error processing {filepath.name}: {e}")
            return {'code': filepath.stem, 'status': 'failed', 'error': str(e)}

    def process_all_stocks(self):
        """Process all stock files with v2 pipeline."""
        logger.info("=" * 70)
        logger.info("📊 Data Processing Pipeline (v2 — LEAK-FREE)")
        logger.info("=" * 70)

        files = sorted([f for f in self.input_dir.glob("*.csv") if not f.name.startswith("_")])

        logger.info(f"📁 Input Directory: {self.input_dir}")
        logger.info(f"📁 Output Directory: {self.output_dir}")
        logger.info(f"📊 Total Files: {len(files)}")
        logger.info("=" * 70)

        results = []
        successful = 0
        failed = []

        for i, filepath in enumerate(files, 1):
            logger.info(f"\n[{i}/{len(files)}] {filepath.name}")
            result = self.process_stock(filepath)
            if result['status'] == 'success':
                successful += 1
                results.append(result)
            else:
                failed.append(filepath.name)

        # Summary report
        if results:
            results_df = pd.DataFrame(results)
            summary_path = self.output_dir / f"_processing_summary{self.SUFFIX}.csv"
            results_df.to_csv(summary_path, index=False)

            logger.info("\n" + "=" * 70)
            logger.info("📊 PROCESSING SUMMARY (v2)")
            logger.info("=" * 70)
            logger.info(f"Total Files: {len(files)}")
            logger.info(f"✅ Successful: {successful}")
            logger.info(f"❌ Failed: {len(failed)}")

            if failed:
                logger.warning(f"Failed files: {', '.join(failed)}")

            total_features = results_df['features_added'].iloc[0] if len(results_df) > 0 else 0
            avg_rows = results_df['output_rows'].mean()
            min_rows = results_df['output_rows'].min()
            max_rows = results_df['output_rows'].max()

            logger.info(f"\n📈 Features Added per Stock: {total_features}")
            logger.info(f"📊 Avg/Min/Max Clean Rows: {avg_rows:.0f} / {min_rows} / {max_rows}")
            logger.info(f"💾 Summary saved: {summary_path}")

        logger.info("=" * 70)
        logger.info("✨ Data processing complete (v2 — leak-free)!")
        logger.info(f"📁 Processed files in: {self.output_dir}")
        logger.info("=" * 70)


def main():
    processor = DataProcessorV2()
    processor.process_all_stocks()


if __name__ == "__main__":
    main()