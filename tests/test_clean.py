"""Tests for src/data/clean.py."""

import numpy as np
import pandas as pd

from src.data.clean import clean_candles
from src.data.validate import validate_candles


def add_row(df: pd.DataFrame, **values) -> pd.DataFrame:
    """Copy the first row, override some fields, and append it."""
    row = df.iloc[[0]].copy()
    for col, val in values.items():
        row[col] = val
    return pd.concat([df, row], ignore_index=True)


def test_clean_data_fixes_types_and_defaults(raw_candles):
    clean, removed = clean_candles(raw_candles)

    assert len(clean) == 6
    assert sum(removed.values()) == 0
    assert pd.api.types.is_datetime64_any_dtype(clean["timestamp"])
    assert clean["close"].dtype == "float64"
    assert clean["volume"].dtype == "int64"
    assert set(clean["symbol"]) == {"TCS", "INFY"}  # upper-cased
    assert (clean["exchange"] == "NSE").all()       # default exchange added


def test_output_is_sorted_and_passes_validation(raw_candles):
    clean, _ = clean_candles(raw_candles)

    assert list(clean["symbol"]) == ["INFY"] * 3 + ["TCS"] * 3
    assert clean.loc[clean["symbol"] == "TCS", "timestamp"].is_monotonic_increasing
    validate_candles(clean)  # must not raise


def test_input_is_not_modified(raw_candles):
    before = raw_candles.copy()
    clean_candles(raw_candles)
    pd.testing.assert_frame_equal(raw_candles, before)


def test_drops_missing_or_unparseable_keys(raw_candles):
    df = add_row(raw_candles, symbol=None, timestamp="2024-02-01")
    df = add_row(df, symbol="   ", timestamp="2024-02-02")  # blank symbol
    df = add_row(df, timestamp="not-a-date")

    clean, removed = clean_candles(df)

    assert removed["missing_symbol_or_timestamp"] == 3
    assert len(clean) == 6


def test_drops_missing_price_but_fills_missing_volume(raw_candles):
    df = add_row(raw_candles, timestamp="2024-02-01", close=None)
    df = add_row(df, timestamp="2024-02-02", open="abc")         # unparseable -> NaN
    df = add_row(df, timestamp="2024-02-03", volume=np.nan)      # kept, volume -> 0

    clean, removed = clean_candles(df)

    assert removed["missing_price"] == 2
    filled = clean[clean["timestamp"] == "2024-02-03"]
    assert len(filled) == 1
    assert filled["volume"].iloc[0] == 0


def test_drops_duplicates_keeping_first(raw_candles):
    first = raw_candles.iloc[0]
    # Same symbol and timestamp as row 0, different close
    df = add_row(raw_candles, close="100.5")

    clean, removed = clean_candles(df)

    assert removed["duplicate_symbol_timestamp"] == 1
    kept = clean[(clean["symbol"] == "TCS") & (clean["timestamp"] == first["timestamp"])]
    assert kept["close"].iloc[0] == float(first["close"])  # the original row survived


def test_same_timestamp_on_different_exchanges_is_not_a_duplicate(raw_candles):
    df = raw_candles.assign(exchange="NSE")
    df = add_row(df, exchange="BSE")

    clean, removed = clean_candles(df)

    assert removed["duplicate_symbol_timestamp"] == 0
    assert len(clean) == 7


def test_drops_each_kind_of_invalid_ohlc(raw_candles):
    df = add_row(raw_candles, timestamp="2024-02-01", low="-1")                 # negative price
    df = add_row(df, timestamp="2024-02-02", open="0")                         # zero price
    df = add_row(df, timestamp="2024-02-03", volume="-5")                      # negative volume
    df = add_row(df, timestamp="2024-02-04", high="90", low="95")              # high < low
    df = add_row(df, timestamp="2024-02-05", high="100.5", close="101")        # high < close
    df = add_row(df, timestamp="2024-02-06", low="100.5", open="100")          # low > open

    clean, removed = clean_candles(df)

    assert removed["non_positive_price"] == 2
    assert removed["negative_volume"] == 1
    assert removed["high_low_inconsistent"] == 3
    assert len(clean) == 6


def test_every_reason_is_reported_even_when_zero(raw_candles):
    _, removed = clean_candles(raw_candles)
    assert set(removed) == {
        "missing_symbol_or_timestamp",
        "missing_price",
        "duplicate_symbol_timestamp",
        "non_positive_price",
        "negative_volume",
        "high_low_inconsistent",
    }
