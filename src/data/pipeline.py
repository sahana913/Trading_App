"""
pipeline.py - Raw files -> clean, validated Parquet.

Run from the project root (PowerShell):
    .venv\\Scripts\\python.exe -m src.data.pipeline

Same input always gives the same output: no randomness, a fixed sort order
and a fixed rule for which duplicate is kept.
"""

import logging
from pathlib import Path

import pandas as pd

from src.config import CANDLES_PARQUET, LOG_DIR, RAW_DIR
from src.data.clean import clean_candles
from src.data.load import load_raw_data
from src.data.validate import validate_candles

log = logging.getLogger("pipeline")


def setup_logging() -> None:
    """Print log lines to the console and also append them to logs/pipeline.log."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_DIR / "pipeline.log", encoding="utf-8"),
        ],
    )


def run_pipeline(raw_dir: Path = RAW_DIR, out_path: Path = CANDLES_PARQUET) -> pd.DataFrame:
    """Load, clean, validate and save. Returns the clean DataFrame."""
    raw = load_raw_data(raw_dir)
    log.info("Loaded %d raw rows from %s", len(raw), raw_dir)

    clean, removed = clean_candles(raw)
    for reason, count in removed.items():
        log.info("Removed %6d rows: %s", count, reason)
    log.info("Removed %d rows in total; %d clean rows remain", sum(removed.values()), len(clean))

    validate_candles(clean)  # raises DataValidationError if anything is wrong
    log.info("Validation passed")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(out_path, index=False)
    log.info("Saved %d rows for %d symbols to %s", len(clean), clean["symbol"].nunique(), out_path)
    return clean


if __name__ == "__main__":
    setup_logging()
    run_pipeline()
