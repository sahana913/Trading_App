"""Tests for src/data/validate.py."""

import pandas as pd
import pytest

from src.data.clean import clean_candles
from src.data.validate import DataValidationError, validate_candles


@pytest.fixture
def clean(raw_candles) -> pd.DataFrame:
    df, _ = clean_candles(raw_candles)
    return df


def test_clean_data_passes(clean):
    validate_candles(clean)  # no exception = pass


def test_empty_data_fails(clean):
    with pytest.raises(DataValidationError, match="empty"):
        validate_candles(clean.iloc[0:0])


def test_missing_column_fails(clean):
    with pytest.raises(DataValidationError, match="volume"):
        validate_candles(clean.drop(columns="volume"))


def test_wrong_dtype_fails(clean):
    df = clean.assign(close=clean["close"].astype(str))
    with pytest.raises(DataValidationError, match="'close' must be numeric"):
        validate_candles(df)


def test_missing_value_fails(clean):
    df = clean.copy()
    df.loc[0, "close"] = None
    with pytest.raises(DataValidationError, match="missing value"):
        validate_candles(df)


# Each case: (column to change, new value, words expected in the error)
@pytest.mark.parametrize(
    "column, value, message",
    [
        ("open", 0.0, "price <= 0"),
        ("volume", -1, "volume < 0"),
        ("low", 1000.0, "high < low"),
        ("close", 1000.0, "high < open or close"),
        ("low", 51.5, "low > open or close"),   # row 0 is INFY: open 50, close 51
    ],
)
def test_bad_values_fail(clean, column, value, message):
    df = clean.copy()
    df.loc[0, column] = value
    with pytest.raises(DataValidationError, match=message):
        validate_candles(df)


def test_placeholder_bar_fails(clean):
    df = clean.copy()
    df.loc[0, ["open", "high", "low", "close", "volume"]] = [50.0, 50.0, 50.0, 50.0, 0]
    with pytest.raises(DataValidationError, match="placeholder bar"):
        validate_candles(df)


def test_duplicates_fail(clean):
    df = pd.concat([clean.iloc[[0]], clean], ignore_index=True)
    with pytest.raises(DataValidationError, match="duplicate"):
        validate_candles(df)


def test_unsorted_fails(clean):
    df = clean.iloc[::-1].reset_index(drop=True)  # reverse the order
    with pytest.raises(DataValidationError, match="not sorted"):
        validate_candles(df)


def test_all_problems_are_reported_together(clean):
    df = clean.copy()
    df.loc[0, "volume"] = -1
    df.loc[1, "open"] = -5.0
    with pytest.raises(DataValidationError) as err:
        validate_candles(df)
    assert "volume < 0" in str(err.value)
    assert "price <= 0" in str(err.value)
