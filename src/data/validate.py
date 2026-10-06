"""
validate.py - Explicit checks on the CLEAN data.

clean.py removes bad rows; this module proves that it worked. If any check
fails, validate_candles() raises DataValidationError listing every problem,
so the pipeline stops before saving bad data.
"""

import pandas as pd

from src.config import CANDLE_COLUMNS, PRICE_COLUMNS

KEY_COLUMNS = ["symbol", "exchange", "timestamp"]


class DataValidationError(Exception):
    """Raised when the candle data breaks one or more rules."""


def validate_candles(df: pd.DataFrame) -> None:
    """Raise DataValidationError if df is not clean. Returns None if it passes."""
    if df.empty:
        raise DataValidationError("Dataset is empty: no rows left to save.")

    # Without the right columns the other checks cannot even run, so stop here
    missing_cols = [c for c in CANDLE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DataValidationError(f"Missing required columns: {missing_cols}")

    problems: list[str] = []

    # --- Types ---
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        problems.append(f"'timestamp' must be a datetime, got {df['timestamp'].dtype}")
    for col in PRICE_COLUMNS + ["volume"]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            problems.append(f"'{col}' must be numeric, got {df[col].dtype}")
    if problems:
        # Value checks below would fail confusingly on the wrong types
        raise DataValidationError("Data validation failed:\n  - " + "\n  - ".join(problems))

    # --- Missing values ---
    nulls = df[CANDLE_COLUMNS].isna().sum()
    for col, n in nulls[nulls > 0].items():
        problems.append(f"{n} missing value(s) in '{col}'")

    # --- Value rules (count the rows that break each one) ---
    rules = {
        "price <= 0": (df[PRICE_COLUMNS] <= 0).any(axis=1),
        "volume < 0": df["volume"] < 0,
        "high < low": df["high"] < df["low"],
        "high < open or close": df["high"] < df[["open", "close"]].max(axis=1),
        "low > open or close": df["low"] > df[["open", "close"]].min(axis=1),
    }
    for rule, broken in rules.items():
        if broken.any():
            problems.append(f"{int(broken.sum())} row(s) with {rule}")

    # --- Uniqueness and order ---
    dups = int(df.duplicated(KEY_COLUMNS).sum())
    if dups:
        problems.append(f"{dups} duplicate (symbol, exchange, timestamp) row(s)")

    # Sorted means: re-sorting would not change the order
    order = df[KEY_COLUMNS].sort_values(KEY_COLUMNS, kind="stable").index
    if not order.equals(df.index):
        problems.append("rows are not sorted by symbol, exchange, timestamp")

    if problems:
        raise DataValidationError("Data validation failed:\n  - " + "\n  - ".join(problems))
