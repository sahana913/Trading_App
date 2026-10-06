"""
load.py - Read the raw files exactly as they are, with only the column
names standardised. No cleaning happens here (that is clean.py's job),
so we can count exactly what cleaning removes.
"""

from pathlib import Path

import pandas as pd

# Common alternative column names -> the standard name we use everywhere.
# Keys are lower-case because we lower-case the headers before renaming.
COLUMN_ALIASES = {
    "date": "timestamp",
    "datetime": "timestamp",
    "time": "timestamp",
    "ticker": "symbol",
    "stock": "symbol",
    "tradingsymbol": "symbol",
    "o": "open",
    "h": "high",
    "l": "low",
    "c": "close",
    "v": "volume",
    "vol": "volume",
}


def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case and trim headers, then rename known aliases."""
    df = df.copy()
    # " Close " and "close" should be treated as the same column
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df.rename(columns=COLUMN_ALIASES)


def load_raw_data(raw_dir: Path) -> pd.DataFrame:
    """Read every CSV/Parquet file in raw_dir and stack them into one table."""
    files = sorted(list(raw_dir.glob("*.csv")) + list(raw_dir.glob("*.parquet")))
    if not files:
        raise FileNotFoundError(f"No .csv or .parquet files found in {raw_dir.resolve()}")

    frames = []
    for f in files:
        df = pd.read_csv(f) if f.suffix == ".csv" else pd.read_parquet(f)
        df = standardise_columns(df)
        # One file per symbol with no symbol column? Use the file name.
        if "symbol" not in df.columns:
            df["symbol"] = f.stem.upper()
        df["source_file"] = f.name  # remember where each row came from
        frames.append(df)

    return pd.concat(frames, ignore_index=True)
