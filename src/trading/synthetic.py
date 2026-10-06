"""
synthetic.py - Invent the next day's candle for every stock, so the market
can keep going after the historical data ends ("synthetic" mode).

Model: a random walk in log prices (geometric Brownian motion, no drift).
For each stock:
  sigma      = std of its last 60 daily returns (how much it usually moves)
               x the admin's volatility multiplier (1.0 = as usual, 2.0 = twice as wild)
  close      = prev_close x exp(N(0, sigma))          the day's move
  open       = prev_close x exp(N(0, 0.3 x sigma))    a small overnight gap
  high / low = max/min(open, close) x exp(+/- |N(0, 0.5 x sigma)|)
               so high >= open, close and low <= open, close always hold
  volume     = median of the last 20 volumes x exp(N(0, 0.3))
Random numbers come from a generator seeded with (SEED, date), so the same
day always produces the same bars: reproducible.
"""

from datetime import datetime, timedelta

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.models import Candle, Instrument

SEED = 42
VOL_WINDOW = 60
DEFAULT_SIGMA = 0.015  # used when a stock has too little history to measure (1.5% a day)


def next_business_day(day: datetime) -> datetime:
    """The next Monday-Friday date (holidays are ignored)."""
    nxt = day + timedelta(days=1)
    while nxt.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        nxt += timedelta(days=1)
    return nxt


def recent_bars(session: Session, instrument_id: int, as_of: datetime, n: int) -> list[Candle]:
    """The last n candles at or before as_of, oldest first."""
    rows = session.scalars(select(Candle).where(Candle.instrument_id == instrument_id,
                                                Candle.timestamp <= as_of)
                           .order_by(Candle.timestamp.desc()).limit(n)).all()
    return rows[::-1]


def make_bar(prev_close: float, sigma: float, base_volume: float, rng: np.random.Generator) -> dict:
    """One synthetic OHLCV bar (see the formulas at the top)."""
    close = prev_close * np.exp(rng.normal(0, sigma))
    open_ = prev_close * np.exp(rng.normal(0, 0.3 * sigma))
    high = max(open_, close) * np.exp(abs(rng.normal(0, 0.5 * sigma)))
    low = min(open_, close) * np.exp(-abs(rng.normal(0, 0.5 * sigma)))
    volume = int(base_volume * np.exp(rng.normal(0, 0.3)))
    return {"open": round(open_, 2), "high": round(high, 2), "low": round(low, 2),
            "close": round(close, 2), "volume": max(volume, 1)}


def generate_next_day(session: Session, after: datetime, volatility: float = 1.0) -> datetime:
    """Add a synthetic candle for every active stock on the next business day
    after `after`. Returns that day. Does not commit."""
    day = next_business_day(after)
    rng = np.random.default_rng([SEED, day.toordinal()])  # same day -> same bars
    for inst in session.scalars(select(Instrument).where(Instrument.is_active).order_by(Instrument.id)):
        history = recent_bars(session, inst.id, after, VOL_WINDOW + 1)
        if not history:
            continue  # no price to start from
        closes = np.array([b.close for b in history])
        returns = np.diff(np.log(closes))
        sigma = float(returns.std()) if len(returns) >= 10 else DEFAULT_SIGMA
        base_volume = float(np.median([b.volume for b in history[-20:]]))
        bar = make_bar(closes[-1], sigma * volatility, base_volume, rng)
        session.add(Candle(instrument_id=inst.id, timestamp=day, is_synthetic=True, **bar))
    session.flush()
    return day
