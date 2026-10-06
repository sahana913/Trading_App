"""
download.py - Fetch real daily NSE prices from Yahoo Finance (free, no API key)
into data/raw/, one CSV per stock.

Run from the project root (PowerShell):
    .venv\\Scripts\\python.exe -m src.data.download

The dates are fixed, so running it again gives the same dataset (apart from
rare corrections Yahoo makes to its own history). Prices are real; trading
on them in this project is simulated with virtual money.
"""

import pandas as pd
import yfinance as yf

from src.config import RAW_DIR

# 10 large, liquid NIFTY 50 stocks from different sectors.
# Yahoo marks NSE stocks with a ".NS" suffix.
SYMBOLS = [
    "RELIANCE",    # energy / conglomerate
    "TCS",         # IT
    "INFY",        # IT
    "HDFCBANK",    # banking
    "ICICIBANK",   # banking
    "SBIN",        # public-sector banking
    "ITC",         # consumer goods
    "HINDUNILVR",  # consumer goods
    "BHARTIARTL",  # telecom
    "LT",          # infrastructure
]
START = "2022-01-01"
END = "2025-12-31"  # yfinance treats END as exclusive


def download_symbol(symbol: str) -> pd.DataFrame:
    """Daily OHLCV for one NSE stock, in our raw-file column format."""
    df = yf.download(
        f"{symbol}.NS", start=START, end=END, interval="1d",
        # auto_adjust=False: keep the prices that actually traded. Yahoo still
        # adjusts for stock splits, so a split doesn't look like a 50% crash.
        auto_adjust=False, progress=False, multi_level_index=False,
    )
    if df.empty:
        raise RuntimeError(f"Yahoo returned no data for {symbol}")

    df = df.reset_index()  # the date is the index; make it a column
    df = df.rename(columns={"Date": "timestamp", "Open": "open", "High": "high",
                            "Low": "low", "Close": "close", "Volume": "volume"})
    # Yahoo stores prices as 32-bit floats (1539.800048828125); NSE prices have 2 decimals
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].round(2)
    df["symbol"] = symbol
    df["exchange"] = "NSE"
    return df[["symbol", "exchange", "timestamp", "open", "high", "low", "close", "volume"]]


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for symbol in SYMBOLS:
        df = download_symbol(symbol)
        out = RAW_DIR / f"{symbol}.csv"
        df.to_csv(out, index=False)
        print(f"{symbol:<11} {len(df):>5} rows  {df['timestamp'].min():%Y-%m-%d} -> "
              f"{df['timestamp'].max():%Y-%m-%d}  -> {out.name}")


if __name__ == "__main__":
    main()
