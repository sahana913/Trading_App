"""
conftest.py - Shared test data. pytest finds fixtures here automatically.
"""

import pandas as pd
import pytest


@pytest.fixture
def raw_candles() -> pd.DataFrame:
    """Small raw dataset: 3 good rows per symbol, all values still as text,
    like they arrive from a CSV."""
    return pd.DataFrame(
        {
            "symbol": ["tcs", "tcs", "tcs", "infy", "infy", "infy"],
            "timestamp": ["2024-01-03", "2024-01-01", "2024-01-02"] * 2,  # out of order
            "open": ["100", "98", "99", "50", "51", "52"],
            "high": ["102", "100", "101", "52", "53", "54"],
            "low": ["99", "97", "98", "49", "50", "51"],
            "close": ["101", "99", "100", "51", "52", "53"],
            "volume": ["1000", "1100", "1200", "500", "600", "700"],
        }
    )
