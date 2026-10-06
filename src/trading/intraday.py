"""
intraday.py - Turn each daily candle into a trading day of 5-minute ticks.

The data only has daily bars (open, high, low, close). In intraday mode the
market clock moves in 5-minute steps from 09:15 to 15:30, so we need a price
for every step. For each stock and day we build ONE price path that:
  * starts at the real open (09:15) and ends at the real close (15:30),
  * touches the real high and the real low somewhere in between,
  * wiggles randomly in between (a "Brownian bridge": a random walk pinned
    at both ends).
The path is seeded by (stock, date), so the same day always replays the same
way: reproducible, and two users see the same prices.

No look-ahead for traders: at 11:00 they only see the path up to 11:00
(open, high-so-far, low-so-far, current price). The day's final high, low and
close appear only once the clock reaches 15:30.

Volume is spread over the day in a U-shape (busy at the open and the close,
quiet at lunch), like real markets.
"""

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from functools import lru_cache

import numpy as np

SESSION_OPEN = time(9, 15)
SESSION_CLOSE = time(15, 30)
STEP_MINUTES = 5
TICKS = 75  # (15:30 - 09:15) / 5 minutes; the path has TICKS + 1 points


def is_intraday(as_of: datetime | None) -> bool:
    """True if a clock time is inside a trading session (has a time of day).
    Midnight timestamps are the old daily mode: 'that day's close is known'."""
    return as_of is not None and as_of.time() != time(0)


def day_start(as_of: datetime) -> datetime:
    return as_of.replace(hour=0, minute=0, second=0, microsecond=0)


def session_open(day: datetime) -> datetime:
    return datetime.combine(day.date(), SESSION_OPEN)


def session_close(day: datetime) -> datetime:
    return datetime.combine(day.date(), SESSION_CLOSE)


def tick_index(as_of: datetime) -> int:
    """0 at 09:15, 1 at 09:20, ..., 75 at 15:30 (clipped to that range)."""
    minutes = (as_of - session_open(as_of)).total_seconds() / 60
    return int(min(max(minutes // STEP_MINUTES, 0), TICKS))


def next_tick(now: datetime) -> datetime | None:
    """The next 5-minute step, or None at/after the close."""
    if now.time() >= SESSION_CLOSE:
        return None
    nxt = session_open(now) + timedelta(minutes=STEP_MINUTES * (tick_index(now) + 1))
    return min(nxt, session_close(now))


@lru_cache(maxsize=8192)
def day_path(instrument_id: int, day_ordinal: int, o: float, h: float, l: float, c: float) -> tuple:
    """TICKS + 1 prices from open to close that hit the real high and low.

    1. random walk w, turned into a bridge: b_k = w_k - (k / N) * w_N  (0 at both ends)
    2. path = straight line from open to close + scaled bridge
    3. stretch only the parts above max(open, close) so the top equals the high,
       and only the parts below min(open, close) so the bottom equals the low.
       Start and end stay exactly at open and close.
    """
    hi_ref, lo_ref = max(o, c), min(o, c)
    if h - l < 1e-9:  # a flat day: nothing to simulate
        return tuple([round(float(o), 2)] * (TICKS + 1))
    for attempt in range(50):
        rng = np.random.default_rng([instrument_id, day_ordinal, attempt])
        walk = np.concatenate([[0.0], np.cumsum(rng.standard_normal(TICKS))])
        k = np.arange(TICKS + 1)
        bridge = walk - walk[-1] * k / TICKS
        path = o + (c - o) * k / TICKS + bridge * (h - l) / max(np.ptp(bridge), 1e-9)
        p_max, p_min = path.max(), path.min()
        need_up, need_down = h > hi_ref + 1e-9, l < lo_ref - 1e-9
        if (need_up and p_max <= hi_ref) or (need_down and p_min >= lo_ref):
            continue  # this walk never left the open-close range on that side: try another
        up = (h - hi_ref) / (p_max - hi_ref) if need_up else 0.0
        down = (lo_ref - l) / (lo_ref - p_min) if need_down else 0.0
        path = np.where(path > hi_ref, hi_ref + (path - hi_ref) * up, path)
        path = np.where(path < lo_ref, lo_ref - (lo_ref - path) * down, path)
        path[0], path[-1] = o, c
        return tuple(float(v) for v in np.round(path, 2))
    # Fallback (practically never used): straight line, still inside the range
    return tuple(float(v) for v in np.round(np.linspace(o, c, TICKS + 1), 2))


@lru_cache(maxsize=1)
def volume_weights() -> np.ndarray:
    """Share of the day's volume traded in each 5-minute step: a U-shape,
    3x busier at the open/close than at midday. Sums to 1."""
    x = np.linspace(-1, 1, TICKS + 1)
    w = 1 + 2 * x ** 2
    return w / w.sum()


@dataclass(frozen=True)
class PartialBar:
    """Today's candle as seen at `timestamp`: a stand-in for a Candle row
    (same attribute names), so the trading engine needs no changes.
    `volume` is this 5-minute step's volume (used for fill limits);
    `day_volume` is everything traded so far today (shown in quotes)."""
    instrument_id: int
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    day_volume: int


def partial_bar(candle, as_of: datetime) -> PartialBar:
    """The part of a daily candle that is visible at `as_of`."""
    path = day_path(candle.instrument_id, candle.timestamp.toordinal(),
                    float(candle.open), float(candle.high), float(candle.low), float(candle.close))
    k = tick_index(as_of)
    seen = path[: k + 1]
    weights = volume_weights()
    return PartialBar(
        instrument_id=candle.instrument_id,
        timestamp=as_of,
        open=path[0],
        high=max(seen),
        low=min(seen),
        close=seen[-1],
        volume=max(1, int(candle.volume * weights[k])),
        day_volume=int(candle.volume * weights[: k + 1].sum()),
    )
