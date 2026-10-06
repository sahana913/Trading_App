"""
market.py - Prices from the candles table, plus the OpenAlgo `quotes` and
`history` functions.

The simulated clock
-------------------
Many functions take `as_of` (a datetime). It means "pretend the time is now
as_of": only candles at or before it are visible. as_of=None means "use the
newest candle we have". The latest visible candle's close is the LTP (last
traded price), and its timestamp is the simulated "now". A market simulator
just has to move as_of forward bar by bar.

Intraday mode (as_of has a time of day, e.g. 11:05): a DAILY candle for
today (stamped at today's midnight) is NOT visible as a whole, because its
high, low and close are still in the future. It is replaced by a PartialBar
showing today up to as_of (see intraday.py). Any other candle is unaffected.
"""

from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.models import Candle, Instrument
from src.trading.intraday import day_start, is_intraday, partial_bar


def success(data=None, **extra) -> dict:
    """OpenAlgo-style success response."""
    response = {"status": "success"}
    if data is not None:
        response["data"] = data
    response.update(extra)
    return response


def error(message: str, **extra) -> dict:
    """OpenAlgo-style error response."""
    return {"status": "error", "message": message, **extra}


def get_instrument(session: Session, symbol: str, exchange: str) -> Instrument | None:
    """Find an instrument by symbol and exchange (case-insensitive)."""
    return session.scalar(
        select(Instrument).where(
            Instrument.symbol == str(symbol).upper(),
            Instrument.exchange == str(exchange).upper(),
        )
    )


def recent_candles(session: Session, instrument_id: int, as_of: datetime | None = None,
                   count: int = 1) -> list[Candle]:
    """The newest `count` candles at or before as_of, newest first.
    Mid-session, today's daily candle is swapped for its PartialBar."""
    query = select(Candle).where(Candle.instrument_id == instrument_id)
    if as_of is not None:
        query = query.where(Candle.timestamp <= as_of)  # never peek into the future
    bars = list(session.scalars(query.order_by(Candle.timestamp.desc()).limit(count)))
    if is_intraday(as_of) and bars and bars[0].timestamp == day_start(as_of):
        bars[0] = partial_bar(bars[0], as_of)  # today so far, not today's final candle
    return bars


def latest_candle(session: Session, instrument_id: int, as_of: datetime | None = None) -> Candle | None:
    """The current bar (or None if there is no data yet)."""
    bars = recent_candles(session, instrument_id, as_of, count=1)
    return bars[0] if bars else None


def get_ltp(session: Session, instrument_id: int, as_of: datetime | None = None) -> float | None:
    """Last traded price = close of the current bar."""
    bar = latest_candle(session, instrument_id, as_of)
    return bar.close if bar else None


# ---------------------------------------------------------------------------
# OpenAlgo API: quotes, history
# ---------------------------------------------------------------------------
def quotes(session: Session, symbol: str, exchange: str, as_of: datetime | None = None) -> dict:
    """Current quote for one symbol, in OpenAlgo's shape."""
    inst = get_instrument(session, symbol, exchange)
    if inst is None:
        return error(f"Unknown symbol {symbol} on {exchange}")
    bars = recent_candles(session, inst.id, as_of, count=2)
    if not bars:
        return error(f"No price data for {symbol} yet")

    bar = bars[0]
    # Previous close comes from the bar before; for the very first bar use its open
    prev_close = bars[1].close if len(bars) > 1 else bar.open
    return success({
        "ltp": bar.close,
        "open": bar.open,
        "high": bar.high,
        "low": bar.low,
        # intraday: everything traded so far today, not just the last 5 minutes
        "volume": getattr(bar, "day_volume", bar.volume),
        "prev_close": prev_close,
        # Simulated data has no order book, so bid and ask are both the LTP
        "bid": bar.close,
        "ask": bar.close,
        "timestamp": bar.timestamp,
    })


def history(session: Session, symbol: str, exchange: str,
            start_date: str | None = None, end_date: str | None = None,
            as_of: datetime | None = None) -> dict:
    """Candles between start_date and end_date (both "YYYY-MM-DD", both inclusive)."""
    inst = get_instrument(session, symbol, exchange)
    if inst is None:
        return error(f"Unknown symbol {symbol} on {exchange}")

    query = select(Candle).where(Candle.instrument_id == inst.id)
    if start_date:
        query = query.where(Candle.timestamp >= datetime.fromisoformat(start_date))
    if end_date:
        # "< next midnight" includes every bar during the end date itself
        query = query.where(Candle.timestamp < datetime.fromisoformat(end_date) + timedelta(days=1))
    if as_of is not None:
        query = query.where(Candle.timestamp <= as_of)

    rows = []
    for c in session.scalars(query.order_by(Candle.timestamp)):
        if is_intraday(as_of) and c.timestamp == day_start(as_of):
            c = partial_bar(c, as_of)  # today's candle as it looks right now (it grows during the day)
            rows.append({"timestamp": day_start(as_of), "open": c.open, "high": c.high,
                         "low": c.low, "close": c.close, "volume": c.day_volume})
        else:
            rows.append({"timestamp": c.timestamp, "open": c.open, "high": c.high,
                         "low": c.low, "close": c.close, "volume": c.volume})
    return success(rows)
