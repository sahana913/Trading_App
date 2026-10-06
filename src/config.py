"""
config.py - One place for every path and default used across the project.

All paths are built from PROJECT_ROOT, so scripts work no matter which
folder you run them from.
"""

from datetime import time
from pathlib import Path

# src/config.py -> parent is src/, parent of that is the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Data folders ---------------------------------------------------------
RAW_DIR = PROJECT_ROOT / "data" / "raw"              # original files, never modified
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"  # cleaned output
CANDLES_PARQUET = PROCESSED_DIR / "candles.parquet"  # the main clean dataset
LOG_DIR = PROJECT_ROOT / "logs"

# --- Database -------------------------------------------------------------
DB_PATH = PROJECT_ROOT / "db" / "paper_trading.db"
# SQLAlchemy connection string for a local SQLite file
DB_URL = f"sqlite:///{DB_PATH.as_posix()}"

# --- Market defaults ------------------------------------------------------
# Used when the raw data has no "exchange" column. OpenAlgo needs an exchange
# on every order, and NSE is the usual one for Indian equities.
DEFAULT_EXCHANGE = "NSE"

# The columns every clean candle row must have, in this order
CANDLE_COLUMNS = ["symbol", "exchange", "timestamp", "open", "high", "low", "close", "volume"]
PRICE_COLUMNS = ["open", "high", "low", "close"]

# --- Paper-trading rules ----------------------------------------------------
STARTING_CASH = 1_000_000.0  # ₹10,00,000 virtual cash for every new user

# Margin = share of the order value that must be blocked as cash.
# CNC (delivery) needs the full amount; MIS (intraday) gets 5x leverage.
MARGIN_RATE = {"CNC": 1.0, "MIS": 0.20}

# One order can take at most this share of a bar's volume. A big order on a
# quiet bar is therefore filled over several bars (a partial fill).
MAX_VOLUME_PCT = 0.10

# MIS positions are closed automatically at/after this time, and no new MIS
# orders are accepted after it (Indian brokers use about 15:15-15:20).
SQUARE_OFF_TIME = time(15, 15)
