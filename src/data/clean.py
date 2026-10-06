"""
clean.py - Turn raw OHLCV rows into clean candles.

Every function takes a DataFrame and returns a new one (the input is never
changed in place). clean_candles() also returns a dict that records how many
rows were removed and why. The pipeline logs that dict.

Cleaning order:
  1. fix types         (text -> numbers / dates; bad values become NaN)
  2. drop rows missing symbol or timestamp
  3. drop rows missing any price; fill missing volume with 0
  4. sort and drop duplicate (symbol, exchange, timestamp) rows
  5. drop invalid OHLC rows (non-positive price, negative volume,
     high/low not actually the high/low of the bar, and placeholder bars
     with a flat price and zero volume)
"""

import pandas as pd

from src.config import CANDLE_COLUMNS, DEFAULT_EXCHANGE, PRICE_COLUMNS

SORT_KEYS = ["symbol", "exchange", "timestamp"]


def fix_types(df: pd.DataFrame, default_exchange: str = DEFAULT_EXCHANGE) -> pd.DataFrame:
    """Convert every column to its proper type.

    errors="coerce" turns anything unparseable (e.g. "abc" in a price column)
    into NaN/NaT instead of crashing; step 3 then removes those rows.
    """
    df = df.copy()
    if "exchange" not in df.columns:
        df["exchange"] = default_exchange

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for col in PRICE_COLUMNS + ["volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # "string" dtype keeps real missing values as <NA> (plain astype(str)
    # would turn them into the text "nan"). Blank text also counts as missing.
    for col in ["symbol", "exchange"]:
        s = df[col].astype("string").str.strip().str.upper()
        df[col] = s.replace("", pd.NA)
    return df


def _drop(df: pd.DataFrame, bad: pd.Series, reason: str, removed: dict) -> pd.DataFrame:
    """Remove rows where `bad` is True and record the count under `reason`."""
    removed[reason] = int(bad.sum())
    return df[~bad]


def drop_missing_keys(df: pd.DataFrame, removed: dict) -> pd.DataFrame:
    """A candle without a symbol or a time cannot be placed anywhere, so drop it."""
    bad = df["symbol"].isna() | df["exchange"].isna() | df["timestamp"].isna()
    return _drop(df, bad, "missing_symbol_or_timestamp", removed)


def handle_missing_values(df: pd.DataFrame, removed: dict) -> pd.DataFrame:
    """Drop rows with any missing price; fill a missing volume with 0.

    We do not invent prices (e.g. forward-fill), because a made-up price could
    fill a simulated order at a price that never traded. A missing volume is
    harmless to treat as "no volume reported".
    """
    bad = df[PRICE_COLUMNS].isna().any(axis=1)
    df = _drop(df, bad, "missing_price", removed).copy()
    df["volume"] = df["volume"].fillna(0)
    return df


def drop_duplicates(df: pd.DataFrame, removed: dict) -> pd.DataFrame:
    """Sort by symbol and time, then keep the first row of each (symbol, exchange, timestamp).

    kind="stable" keeps duplicates in their original file order, so "first"
    always means the row that appeared first in the raw data.
    """
    df = df.sort_values(SORT_KEYS, kind="stable")
    bad = df.duplicated(SORT_KEYS, keep="first")
    return _drop(df, bad, "duplicate_symbol_timestamp", removed)


def is_placeholder_bar(df: pd.DataFrame) -> pd.Series:
    """True for fake "no trading" bars: open = high = low = close and volume 0.

    Data vendors sometimes insert these on days they have no data for. They
    look valid but would show a 0% return and leave no volume to trade against.
    """
    flat = (df["open"] == df["high"]) & (df["high"] == df["low"]) & (df["low"] == df["close"])
    return flat & (df["volume"] == 0)


def drop_invalid_ohlc(df: pd.DataFrame, removed: dict) -> pd.DataFrame:
    """Remove rows that cannot be a real candle. Each rule is counted separately."""
    df = _drop(df, (df[PRICE_COLUMNS] <= 0).any(axis=1), "non_positive_price", removed)
    df = _drop(df, df["volume"] < 0, "negative_volume", removed)

    # High must be the highest price of the bar and low the lowest
    bar_top = df[["open", "close"]].max(axis=1)
    bar_bottom = df[["open", "close"]].min(axis=1)
    bad = (df["high"] < df["low"]) | (df["high"] < bar_top) | (df["low"] > bar_bottom)
    df = _drop(df, bad, "high_low_inconsistent", removed)
    return _drop(df, is_placeholder_bar(df), "placeholder_bar", removed)


def clean_candles(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Run every cleaning step. Returns (clean DataFrame, {reason: rows removed})."""
    removed: dict[str, int] = {}

    df = fix_types(raw)
    df = drop_missing_keys(df, removed)
    df = handle_missing_values(df, removed)
    df = drop_duplicates(df, removed)
    df = drop_invalid_ohlc(df, removed)

    # Final shape: only the standard columns, fixed dtypes, fresh 0..n-1 index
    df = df[CANDLE_COLUMNS].copy()
    df["symbol"] = df["symbol"].astype(str)
    df["exchange"] = df["exchange"].astype(str)
    df["volume"] = df["volume"].round().astype("int64")
    for col in PRICE_COLUMNS:
        df[col] = df[col].astype("float64")
    return df.reset_index(drop=True), removed
