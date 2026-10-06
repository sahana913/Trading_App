"""
eda.py - Exploratory Data Analysis for the raw OHLCV dataset.

What it does (in order):
  1. Loads every .csv / .parquet file in data/raw/ (via src/data/load.py,
     which also standardises names like "Date" -> "timestamp").
  2. Fixes types without dropping anything, so problems stay visible.
  3. Prints: shape, columns + dtypes, symbols, date range, missing values,
     duplicates, bad rows, and return statistics per symbol.
  4. Saves interactive Plotly charts (HTML) to reports/eda/.

Run from the project root (PowerShell):
    .venv\\Scripts\\python.exe eda.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

from src.data.clean import fix_types, is_placeholder_bar
from src.data.load import load_raw_data

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
RAW_DIR = Path("data/raw")        # where the input files live
REPORT_DIR = Path("reports/eda")  # where the charts are written

REQUIRED_COLUMNS = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
PRICE_COLUMNS = ["open", "high", "low", "close"]

# Show every column when printing wide tables
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 50)


def section(title: str) -> None:
    """Print a visible header so the console output is easy to read."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# 1. Loading
# ---------------------------------------------------------------------------
def load_for_eda(raw_dir: Path) -> pd.DataFrame:
    """Load the raw files and fix types, but do NOT drop anything:
    EDA needs to see the bad rows so it can report them."""
    data = load_raw_data(raw_dir)  # shared with the pipeline (src/data/load.py)
    for f, n in data["source_file"].value_counts(sort=False).items():
        print(f"Loaded {f}: {n:,} rows")

    missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing:
        raise SystemExit(f"Missing required columns {missing}. Found: {list(data.columns)}")

    # Unparseable values become NaN/NaT, so they show up in the missing-value report
    data = fix_types(data)
    return data.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Quality checks
# ---------------------------------------------------------------------------
def find_bad_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return a table with one True/False column per rule; True = rule broken."""
    checks = pd.DataFrame(index=df.index)
    checks["high_lt_low"] = df["high"] < df["low"]
    # High must be the highest price of the bar, low the lowest
    checks["high_lt_open_or_close"] = df["high"] < df[["open", "close"]].max(axis=1)
    checks["low_gt_open_or_close"] = df["low"] > df[["open", "close"]].min(axis=1)
    checks["non_positive_price"] = (df[PRICE_COLUMNS] <= 0).any(axis=1)
    checks["negative_volume"] = df["volume"] < 0
    checks["placeholder_bar"] = is_placeholder_bar(df)  # flat price, zero volume
    return checks


def return_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Daily-return statistics for each symbol (based on close prices)."""
    df = df.copy()
    # pct_change inside each symbol, so we never compute a return
    # between the last row of one stock and the first row of the next
    df["ret"] = df.groupby("symbol")["close"].pct_change()

    stats = df.groupby("symbol")["ret"].agg(
        rows="count", mean="mean", std="std", min="min", max="max", skew="skew"
    )
    # Annualise with 252 trading days (only meaningful for daily bars)
    stats["ann_return"] = stats["mean"] * 252
    stats["ann_vol"] = stats["std"] * np.sqrt(252)
    stats["sharpe_rf0"] = stats["ann_return"] / stats["ann_vol"]  # risk-free rate = 0
    return stats.round(4)


# ---------------------------------------------------------------------------
# 3. Charts
# ---------------------------------------------------------------------------
def save_charts(df: pd.DataFrame, out_dir: Path) -> None:
    """Write a few Plotly charts as standalone HTML files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df = df.dropna(subset=["timestamp", "close"]).copy()
    df["ret"] = df.groupby("symbol")["close"].pct_change()

    # a) Close price per symbol, rebased to 100 so different price levels compare
    first_close = df.groupby("symbol")["close"].transform("first")
    df["rebased"] = df["close"] / first_close * 100
    fig = px.line(df, x="timestamp", y="rebased", color="symbol",
                  title="Close price rebased to 100")
    fig.write_html(out_dir / "close_rebased.html", include_plotlyjs="cdn")

    # b) Distribution of daily returns per symbol
    fig = px.box(df.dropna(subset=["ret"]), x="symbol", y="ret",
                 title="Daily return distribution by symbol")
    fig.write_html(out_dir / "returns_box.html", include_plotlyjs="cdn")

    # c) Average daily volume per symbol
    vol = df.groupby("symbol", as_index=False)["volume"].mean()
    fig = px.bar(vol, x="symbol", y="volume", title="Average volume by symbol")
    fig.write_html(out_dir / "avg_volume.html", include_plotlyjs="cdn")

    # d) Correlation of daily returns between symbols
    wide = df.pivot_table(index="timestamp", columns="symbol", values="ret")
    if wide.shape[1] > 1:
        fig = px.imshow(wide.corr().round(2), text_auto=True, zmin=-1, zmax=1,
                        color_continuous_scale="RdBu_r",
                        title="Correlation of daily returns")
        fig.write_html(out_dir / "return_correlation.html", include_plotlyjs="cdn")

    print(f"Charts saved to {out_dir.resolve()}")


# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------
def main() -> None:
    section("LOADING")
    df = load_for_eda(RAW_DIR)

    section("SHAPE")
    print(f"{df.shape[0]:,} rows x {df.shape[1]} columns")

    section("COLUMNS AND DTYPES")
    print(df.dtypes.to_string())

    section("SYMBOLS AND DATE RANGE")
    per_symbol = df.groupby("symbol")["timestamp"].agg(rows="count", start="min", end="max")
    print(per_symbol.to_string())
    print(f"\nSymbols: {df['symbol'].nunique()}")
    print(f"Overall range: {df['timestamp'].min()} -> {df['timestamp'].max()}")
    # Median gap between consecutive bars tells us the bar size (1 day, 5 min, ...)
    gap = df.groupby("symbol")["timestamp"].diff().median()
    print(f"Typical bar interval: {gap}")

    section("MISSING VALUES")
    missing = df.isna().sum()
    print(missing[missing > 0].to_string() if missing.any() else "None")

    section("DUPLICATES")
    print(f"Fully identical rows:          {df.duplicated().sum():,}")
    print(f"Duplicate (symbol, timestamp): {df.duplicated(['symbol', 'timestamp']).sum():,}")

    section("BAD ROWS")
    checks = find_bad_rows(df)
    print(checks.sum().to_string())
    bad = df[checks.any(axis=1)]
    print(f"\nRows breaking at least one rule: {len(bad):,}")
    if not bad.empty:
        print("\nFirst 10 bad rows:")
        print(bad.head(10).to_string())

    section("RETURN STATISTICS PER SYMBOL (daily close-to-close)")
    print(return_stats(df).to_string())

    section("CHARTS")
    save_charts(df, REPORT_DIR)


if __name__ == "__main__":
    main()
