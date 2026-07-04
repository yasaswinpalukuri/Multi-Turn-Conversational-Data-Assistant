"""
utils/data_loader.py

Loads the NYC TLC Yellow Taxi CSV, parses datetimes, engineers derived
columns, and exposes schema / profiling helpers used by the agent tools.

Design decisions:
- Loads once into a module-level singleton (get_dataframe()) so Streamlit
  reruns don't reload the 100k-row CSV on every interaction.
- Adds derived columns at load time so LLM-generated pandas code can
  reference them without extra transformation steps.
- payment_type integer codes are mapped to human-readable labels so the
  agent can answer "which payment type is most common?" sensibly.
- All datetime parsing uses explicit format strings for speed on 100k rows.
- Compatible with migration to n8n/LangGraph: no Streamlit imports here.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

# 2023 NYC TLC Yellow Taxi column layout (19 columns)
EXPECTED_COLUMNS: list[str] = [
    "VendorID",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "RatecodeID",
    "store_and_fwd_flag",
    "PULocationID",
    "DOLocationID",
    "payment_type",
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "total_amount",
    "congestion_surcharge",
    "airport_fee",
]

# Human-readable labels for payment_type integer codes
PAYMENT_TYPE_MAP: dict[int, str] = {
    1: "Credit card",
    2: "Cash",
    3: "No charge",
    4: "Dispute",
    5: "Unknown",
    6: "Voided trip",
}

# VendorID labels
VENDOR_MAP: dict[int, str] = {
    1: "Creative Mobile Technologies (CMT)",
    2: "VeriFone Inc. (VTS)",
}

# RatecodeID labels
RATECODE_MAP: dict[int, str] = {
    1: "Standard rate",
    2: "JFK",
    3: "Newark",
    4: "Nassau or Westchester",
    5: "Negotiated fare",
    6: "Group ride",
}

# Default config — overridden by .env values
DEFAULT_DATA_PATH = "data/nyc_taxi_sample.csv"
DEFAULT_MAX_ROWS = 100_000


# ── Loader ───────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_dataframe() -> pd.DataFrame:
    """
    Load and return the NYC Taxi DataFrame.

    Uses lru_cache so the CSV is read exactly once per process lifetime,
    even across Streamlit reruns. Call invalidate_cache() to force a reload.

    Returns
    -------
    pd.DataFrame
        Cleaned and feature-engineered taxi trip data.

    Raises
    ------
    FileNotFoundError
        If the CSV path from .env / default does not exist.
    ValueError
        If the CSV is missing required columns.
    """
    data_path = Path(os.getenv("DATA_PATH", DEFAULT_DATA_PATH))
    max_rows = int(os.getenv("DATA_MAX_ROWS", DEFAULT_MAX_ROWS))

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'.\n"
            f"Place your NYC Taxi CSV at that path, or update DATA_PATH in .env.\n"
            f"Download: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page"
        )

    logger.info(f"Loading dataset from {data_path} (max_rows={max_rows:,})")

    df = pd.read_csv(
        data_path,
        nrows=max_rows,
        # Parse datetimes inline for speed
        parse_dates=["tpep_pickup_datetime", "tpep_dropoff_datetime"],
    )

    logger.info(f"Raw load: {len(df):,} rows × {len(df.columns)} columns")

    # Validate schema
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing expected columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    df = _clean(df)
    df = _engineer_features(df)

    logger.info(f"Dataset ready: {len(df):,} rows × {len(df.columns)} columns")
    return df


def invalidate_cache() -> None:
    """
    Clear the lru_cache so get_dataframe() reloads from disk on next call.
    Use this if the user uploads a new CSV mid-session.
    """
    get_dataframe.cache_clear()
    logger.info("DataFrame cache invalidated")


# ── Cleaning ─────────────────────────────────────────────────────────────────

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply data quality fixes to the raw DataFrame.

    - Drops rows where trip_distance <= 0 or fare_amount <= 0
      (these are cancellations or data errors, not real trips)
    - Caps tip_amount and fare_amount at 99th percentile to remove outliers
      that would skew averages in LLM-generated queries
    - Fills nullable columns with sensible defaults
    - Maps integer codes to human-readable string columns

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame from pd.read_csv.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    original_len = len(df)

    # Drop obvious data errors
    df = df[df["trip_distance"] > 0].copy()
    df = df[df["fare_amount"] > 0].copy()
    df = df[df["total_amount"] > 0].copy()

    # Drop trips where pickup == dropoff time (invalid)
    df = df[df["tpep_pickup_datetime"] != df["tpep_dropoff_datetime"]].copy()

    dropped = original_len - len(df)
    if dropped > 0:
        logger.info(f"Dropped {dropped:,} invalid rows (zero distance/fare or bad timestamps)")

    # Fill nulls in columns with known reasonable defaults
    df["passenger_count"] = df["passenger_count"].fillna(1.0)
    df["congestion_surcharge"] = df["congestion_surcharge"].fillna(0.0)
    df["airport_fee"] = df["airport_fee"].fillna(0.0)
    df["RatecodeID"] = df["RatecodeID"].fillna(1.0)
    df["store_and_fwd_flag"] = df["store_and_fwd_flag"].fillna("N")

    # Cap fare and tip at 99th percentile to reduce outlier skew
    fare_cap = df["fare_amount"].quantile(0.99)
    tip_cap = df["tip_amount"].quantile(0.99)
    df["fare_amount"] = df["fare_amount"].clip(upper=fare_cap)
    df["tip_amount"] = df["tip_amount"].clip(upper=tip_cap)

    # Add human-readable label columns (keep originals for numeric operations)
    df["payment_type_label"] = df["payment_type"].map(PAYMENT_TYPE_MAP).fillna("Unknown")
    df["vendor_label"] = df["VendorID"].map(VENDOR_MAP).fillna("Unknown")
    df["ratecode_label"] = df["RatecodeID"].map(RATECODE_MAP).fillna("Unknown")

    return df


# ── Feature Engineering ───────────────────────────────────────────────────────

def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived columns that the agent tools and eval set questions need.

    New columns added:
    - trip_duration_minutes : float — trip length in minutes
    - pickup_hour           : int   — hour of day (0–23)
    - pickup_day_of_week    : str   — e.g. "Monday"
    - pickup_date           : date  — date portion of pickup timestamp
    - tip_percentage        : float — tip as % of fare (0 if cash)
    - speed_mph             : float — average speed (distance / duration)
    - has_surcharge         : bool  — True if any surcharge > 0
    - is_airport_trip       : bool  — True if RatecodeID is JFK or Newark

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with derived columns appended.
    """
    # Trip duration
    duration_seconds = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds()
    df["trip_duration_minutes"] = (duration_seconds / 60).round(2)

    # Drop negative durations (timestamp errors)
    df = df[df["trip_duration_minutes"] > 0].copy()

    # Time features
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["pickup_day_of_week"] = df["tpep_pickup_datetime"].dt.day_name()
    df["pickup_date"] = df["tpep_pickup_datetime"].dt.date

    # Tip percentage — only meaningful for credit card payments (payment_type == 1)
    # Cash tips are not recorded, so tip_percentage is 0 for cash trips
    df["tip_percentage"] = np.where(
        df["fare_amount"] > 0,
        (df["tip_amount"] / df["fare_amount"] * 100).round(2),
        0.0,
    )

    # Speed in mph — trip_distance (miles) / duration (hours)
    duration_hours = df["trip_duration_minutes"] / 60
    df["speed_mph"] = np.where(
        duration_hours > 0,
        (df["trip_distance"] / duration_hours).round(2),
        0.0,
    )
    # Cap unrealistic speeds (>100 mph = data error)
    df["speed_mph"] = df["speed_mph"].clip(upper=100.0)

    # Surcharge flag — any extra charge applied
    surcharge_cols = ["extra", "mta_tax", "tolls_amount",
                      "improvement_surcharge", "congestion_surcharge", "airport_fee"]
    df["has_surcharge"] = df[surcharge_cols].sum(axis=1) > 0

    # Airport trip flag (RatecodeID 2 = JFK, 3 = Newark)
    df["is_airport_trip"] = df["RatecodeID"].isin([2.0, 3.0])

    logger.info(
        f"Feature engineering done. Final shape: {df.shape}. "
        f"New columns: trip_duration_minutes, pickup_hour, pickup_day_of_week, "
        f"pickup_date, tip_percentage, speed_mph, has_surcharge, is_airport_trip"
    )

    return df


# ── Schema / Profiling Helpers ────────────────────────────────────────────────

def get_schema_summary() -> dict[str, Any]:
    """
    Return a compact schema summary dict for use in the agent's system prompt
    and schema_tool. Includes column names, dtypes, null counts, and ranges
    for numeric columns.

    Returns
    -------
    dict
        Schema summary with keys: columns, dtypes, null_counts,
        numeric_ranges, row_count, derived_columns.
    """
    df = get_dataframe()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_ranges = {}
    for col in numeric_cols:
        numeric_ranges[col] = {
            "min": round(float(df[col].min()), 4),
            "max": round(float(df[col].max()), 4),
            "mean": round(float(df[col].mean()), 4),
        }

    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "null_counts": df.isnull().sum().to_dict(),
        "numeric_ranges": numeric_ranges,
        "datetime_columns": ["tpep_pickup_datetime", "tpep_dropoff_datetime"],
        "derived_columns": [
            "trip_duration_minutes",
            "pickup_hour",
            "pickup_day_of_week",
            "pickup_date",
            "tip_percentage",
            "speed_mph",
            "has_surcharge",
            "is_airport_trip",
            "payment_type_label",
            "vendor_label",
            "ratecode_label",
        ],
        "payment_type_map": PAYMENT_TYPE_MAP,
        "notes": (
            "tip_percentage is 0 for cash trips (payment_type=2) because "
            "cash tips are not recorded by the meter. "
            "speed_mph is capped at 100 mph. "
            "fare_amount and tip_amount are capped at 99th percentile."
        ),
    }


def get_column_stats(column_name: str) -> dict[str, Any]:
    """
    Return descriptive statistics for a single column.

    Parameters
    ----------
    column_name : str
        Name of the column to profile.

    Returns
    -------
    dict
        Statistics including count, nulls, dtype, and type-specific stats.

    Raises
    ------
    ValueError
        If the column does not exist in the DataFrame.
    """
    df = get_dataframe()

    if column_name not in df.columns:
        available = ", ".join(df.columns.tolist())
        raise ValueError(
            f"Column '{column_name}' not found.\n"
            f"Available columns: {available}"
        )

    col = df[column_name]
    stats: dict[str, Any] = {
        "column": column_name,
        "dtype": str(col.dtype),
        "count": int(col.count()),
        "null_count": int(col.isnull().sum()),
        "null_pct": round(col.isnull().mean() * 100, 2),
    }

    if pd.api.types.is_numeric_dtype(col):
        desc = col.describe()
        stats.update({
            "min": round(float(desc["min"]), 4),
            "max": round(float(desc["max"]), 4),
            "mean": round(float(desc["mean"]), 4),
            "median": round(float(col.median()), 4),
            "std": round(float(desc["std"]), 4),
            "p25": round(float(desc["25%"]), 4),
            "p75": round(float(desc["75%"]), 4),
            "p99": round(float(col.quantile(0.99)), 4),
        })
    elif pd.api.types.is_datetime64_any_dtype(col):
        stats.update({
            "min": str(col.min()),
            "max": str(col.max()),
            "date_range_days": (col.max() - col.min()).days,
        })
    else:
        # Categorical / string
        value_counts = col.value_counts()
        stats.update({
            "unique_count": int(col.nunique()),
            "top_5_values": value_counts.head(5).to_dict(),
        })

    return stats


def get_memory_usage() -> dict[str, str]:
    """
    Return memory usage of the loaded DataFrame for the Streamlit sidebar.

    Returns
    -------
    dict
        Keys: total_mb, per_column (top 5 heaviest).
    """
    df = get_dataframe()
    mem = df.memory_usage(deep=True)
    total_mb = mem.sum() / 1024 / 1024

    per_col = (
        mem.drop("Index")
        .sort_values(ascending=False)
        .head(5)
        .apply(lambda x: f"{x / 1024:.1f} KB")
        .to_dict()
    )

    return {
        "total_mb": f"{total_mb:.1f} MB",
        "row_count": f"{len(df):,}",
        "column_count": str(len(df.columns)),
        "per_column_top5": per_col,
    }
