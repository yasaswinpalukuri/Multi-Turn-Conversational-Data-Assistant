"""
download_data.py

Downloads and prepares the NYC Yellow Taxi dataset for the data assistant.

Usage:
    python download_data.py

What it does:
    1. Downloads yellow_tripdata_2023-01.parquet from NYC TLC (~50MB)
    2. Samples 100,000 rows with a fixed random seed for reproducibility
    3. Saves to data/nyc_taxi_sample.csv
    4. Prints column names and row count to confirm success

The CSV is gitignored — run this script once after cloning.
Requires: pandas, pyarrow (both in requirements.txt)
"""

import os
import sys
import urllib.request
from pathlib import Path


PARQUET_URL = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/"
    "yellow_tripdata_2023-01.parquet"
)
PARQUET_FILE = "yellow_tripdata_2023-01.parquet"
OUTPUT_CSV   = Path("data/nyc_taxi_sample.csv")
SAMPLE_ROWS  = 100_000
RANDOM_SEED  = 42


def show_progress(block_num: int, block_size: int, total_size: int) -> None:
    """Print a simple download progress indicator."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        mb  = downloaded / 1024 / 1024
        total_mb = total_size / 1024 / 1024
        print(f"\r  Downloading... {mb:.1f} / {total_mb:.1f} MB ({pct}%)", end="")
    else:
        mb = downloaded / 1024 / 1024
        print(f"\r  Downloading... {mb:.1f} MB", end="")


def main() -> None:
    try:
        import pandas as pd
    except ImportError:
        print("❌ pandas not installed. Run: pip install -r requirements.txt")
        sys.exit(1)

    # Create data directory
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Skip download if CSV already exists
    if OUTPUT_CSV.exists():
        print(f"✓ Dataset already exists at {OUTPUT_CSV}")
        df = pd.read_csv(OUTPUT_CSV, nrows=5)
        print(f"  Columns: {list(df.columns)}")
        print(f"  (Delete {OUTPUT_CSV} and re-run to refresh)")
        return

    # Download parquet
    print(f"Downloading NYC TLC Yellow Taxi data (Jan 2023)...")
    print(f"Source: {PARQUET_URL}")
    print()

    try:
        urllib.request.urlretrieve(PARQUET_URL, PARQUET_FILE, show_progress)
        print()  # newline after progress
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("Check your internet connection and try again.")
        sys.exit(1)

    # Convert to CSV sample
    print(f"\nConverting parquet → CSV sample ({SAMPLE_ROWS:,} rows)...")
    try:
        df_full = pd.read_parquet(PARQUET_FILE)
        print(f"  Full dataset: {len(df_full):,} rows × {len(df_full.columns)} columns")

        sample = df_full.sample(SAMPLE_ROWS, random_state=RANDOM_SEED)
        sample.to_csv(OUTPUT_CSV, index=False)
        print(f"  Sample saved: {OUTPUT_CSV}")

    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        sys.exit(1)

    # Clean up parquet
    os.remove(PARQUET_FILE)
    print(f"  Cleaned up: {PARQUET_FILE} removed")

    # Confirm
    print(f"\n✅ Dataset ready!")
    print(f"   Path:    {OUTPUT_CSV}")
    print(f"   Rows:    {SAMPLE_ROWS:,}")
    print(f"   Columns: {list(sample.columns)}")
    print(f"\nNext step: streamlit run app.py")


if __name__ == "__main__":
    main()
