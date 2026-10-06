"""Tests for intraday mode (src/trading/intraday.py and its use in market.py,
the simulator and the ML signals): paths respect the real candle, nobody sees
the future mid-session, and MIS trading now has real intraday P&L."""

from datetime import datetime, time

import numpy as np
import pytest
from sqlalchemy import select

from src.db.models import Candle, DailyPnl, Instrument, Order, Position
from src.db.session import get_engine, get_session_factory, init_db
from src.ml.predict import candles_frame
from src.trading import create_user, history, placeorder, quotes
from src.trading.intraday import TICKS, day_path, next_tick, partial_bar, tick_index, volume_weights
from src.trading.simulator import get_clock, run, settings, start, step, update_settings

# Daily candles, Mon 1 - Fri 5 Jan 2024: (open, high, low, close)
BARS = {1: (100.0, 104.0, 97.0, 102.0), 2: (102.0, 103.5, 99.0, 99.5), 3: (99.5, 101.0, 96.0, 100.8),
        4: (100.8, 106.0, 100.2, 105.1), 5: (105.1, 105.9, 101.3, 102.2)}
VOLUME = 750_000


def day(d: int, hh: int = 0, mm: int = 0) -> datetime:
    return datetime(2024, 1, d, hh, mm)


@pytest.fixture
def session():
    engine = get_engine("sqlite:///:memory:")
    init_db(engine)
    with get_session_factory(engine)() as s:
        inst = Instrument(symbol="TCS", exchange="NSE", name="TCS", lot_size=1, tick_size=0.05, is_active=True)
        s.add(inst)
        s.flush()
        for d, (o, h, l, c) in BARS.items():
            s.add(Candle(instrument_id=inst.id, timestamp=day(d), open=o, high=h, low=l, close=c, volume=VOLUME))
        s.commit()
        yield s


@pytest.fixture
def user(session):
    return create_user(session, "alice", "password1")


def path_for(session, d: int) -> tuple:
    c = session.scalar(select(Candle).where(Candle.timestamp == day(d)))
    return day_path(c.instrument_id, c.timestamp.toordinal(), c.open, c.high, c.low, c.close)


# ---------------------------------------------------------------------------
# The price path
# ---------------------------------------------------------------------------
def test_path_is_pinned_to_the_real_candle():
    rng = np.random.default_rng(0)
    for i in range(300):  # many random but valid candles
        o, c = rng.uniform(90, 110, 2)
        h = max(o, c) + rng.uniform(0, 5) * (i % 4 != 0)   # every 4th day closes/opens at the high
        l = min(o, c) - rng.uniform(0, 5) * (i % 5 != 0)
        o, h, l, c = (round(v, 2) for v in (o, h, l, c))
        p = day_path(1, 700_000 + i, o, h, l, c)
        assert len(p) == TICKS + 1
        assert p[0] == pytest.approx(o, abs=0.006) and p[-1] == pytest.approx(c, abs=0.006)
        assert max(p) == pytest.approx(h, abs=0.006) and min(p) == pytest.approx(l, abs=0.006)


def test_path_is_reproducible_and_differs_by_day_and_stock():
    a = day_path(1, 738000, 100, 104, 97, 102)
    assert a == day_path(1, 738000, 100, 104, 97, 102)
    assert a != day_path(2, 738000, 100, 104, 97, 102)
    assert a != day_path(1, 738001, 100, 104, 97, 102)


def test_flat_day_stays_flat():
    assert set(day_path(1, 738000, 50, 50, 50, 50)) == {50.0}


def test_session_clock():
    assert tick_index(day(1, 9, 15)) == 0 and tick_index(day(1, 9, 20)) == 1 and tick_index(day(1, 15, 30)) == TICKS
    assert next_tick(day(1, 9, 15)) == day(1, 9, 20)
    assert next_tick(day(1, 15, 25)) == day(1, 15, 30)
    assert next_tick(day(1, 15, 30)) is None


def test_volume_is_u_shaped_and_complete():
    w = volume_weights()
    assert w.sum() == pytest.approx(1.0)
    assert w[0] == pytest.approx(3 * w[TICKS // 2], rel=0.01)  # the open is ~3x busier than midday


# ---------------------------------------------------------------------------
# No look-ahead mid-session
# ---------------------------------------------------------------------------
def test_quote_at_11am_only_shows_the_day_so_far(session):
    p = path_for(session, 2)
    k = tick_index(day(2, 11, 0))
    q = quotes(session, "TCS", "NSE", as_of=day(2, 11, 0))["data"]
    assert q["ltp"] == p[k]
    assert q["open"] == BARS[2][0]
    assert q["high"] == max(p[: k + 1]) and q["low"] == min(p[: k + 1])
    assert q["high"] <= BARS[2][1] and q["low"] >= BARS[2][2]
    assert q["prev_close"] == BARS[1][3]                       # yesterday's real close
    assert q["volume"] == int(VOLUME * volume_weights()[: k + 1].sum()) < VOLUME
    assert q["timestamp"] == day(2, 11, 0)


def test_at_the_close_the_partial_bar_equals_the_real_candle(session):
    c = session.scalar(select(Candle).where(Candle.timestamp == day(3)))
    bar = partial_bar(c, day(3, 15, 30))
    assert (bar.open, bar.high, bar.low, bar.close) == pytest.approx(BARS[3], abs=0.006)


def test_history_mid_session_has_a_growing_today_candle(session):
    rows = history(session, "TCS", "NSE", as_of=day(2, 11, 0))["data"]
    assert len(rows) == 2
    assert rows[0]["close"] == BARS[1][3]                       # yesterday: complete
    today = rows[1]
    assert today["timestamp"] == day(2)
    assert today["close"] == quotes(session, "TCS", "NSE", as_of=day(2, 11, 0))["data"]["ltp"]
    assert today["high"] <= BARS[2][1]


def test_ml_signals_ignore_the_unfinished_day(session):
    frame = candles_frame(session, as_of=day(2, 11, 0))
    assert list(frame["timestamp"]) == [day(1)]                 # only finished days
    assert len(candles_frame(session, as_of=day(2))) == 2       # daily mode: day 2 counts as closed


# ---------------------------------------------------------------------------
# Simulator in intraday mode
# ---------------------------------------------------------------------------
def test_start_intraday_opens_at_0915(session):
    start(session, day(1), intraday=True)
    assert get_clock(session) == day(1, 9, 15)
    assert settings(session)["intraday"] is True


def test_a_full_day_of_steps_then_the_next_open(session, user):
    start(session, day(1), intraday=True)
    run(session, TICKS)
    assert get_clock(session) == day(1, 15, 30)
    assert session.scalar(select(DailyPnl)) is None             # the day isn't closed yet
    step(session)
    assert get_clock(session) == day(2, 9, 15)                  # next trading day, at its open
    snap = session.scalar(select(DailyPnl))
    assert snap.trade_date == day(1).date()


def test_mis_has_real_intraday_pnl(session, user):
    """Buy at the open, auto squared off at 15:15: profit/loss is the move
    between those two times, not zero as with daily bars."""
    start(session, day(1), intraday=True)
    placeorder(session, user.id, "TCS", "NSE", "BUY", 100, product="MIS", as_of=get_clock(session))
    p = path_for(session, 1)
    run(session, tick_index(day(1, 15, 15)))                    # walk to 15:15
    assert get_clock(session) == day(1, 15, 15)
    pos = session.scalar(select(Position))
    assert pos.quantity == 0                                    # squared off automatically
    expected = (p[tick_index(day(1, 15, 15))] - p[0]) * 100
    assert pos.realised_pnl == pytest.approx(expected)
    assert expected != 0


def test_no_new_mis_after_square_off_time(session, user):
    start(session, day(1), intraday=True)
    run(session, tick_index(day(1, 15, 20)))
    result = placeorder(session, user.id, "TCS", "NSE", "BUY", 1, product="MIS", as_of=get_clock(session))
    assert "square-off" in result["message"]


def test_limit_order_fills_during_the_day(session, user):
    start(session, day(1), intraday=True)
    p = path_for(session, 1)
    limit = round(min(p) + 0.5, 2)                              # the path dips below this at some point
    oid = placeorder(session, user.id, "TCS", "NSE", "BUY", 10, pricetype="LIMIT", price=limit,
                     product="CNC", as_of=get_clock(session))["orderid"]
    first_cross = next(k for k, v in enumerate(p) if v <= limit)
    run(session, TICKS)
    order = session.scalar(select(Order).where(Order.orderid == oid))
    assert order.status == "complete"
    assert order.average_price == p[first_cross] <= limit       # filled at the first tick that crossed
    assert tick_index(order.updated_at) == first_cross


def test_large_order_fills_over_several_ticks(session, user):
    start(session, day(1), intraday=True)
    per_tick = int(VOLUME * volume_weights()[0]) // 10          # 10% of the opening tick's volume
    oid = placeorder(session, user.id, "TCS", "NSE", "BUY", per_tick * 3, product="CNC",
                     as_of=get_clock(session))["orderid"]
    order = session.scalar(select(Order).where(Order.orderid == oid))
    assert order.filled_quantity == per_tick                    # only the first tick's share
    run(session, 3)
    assert order.status == "complete"


def test_switching_intraday_off_mid_session_finishes_the_day(session):
    start(session, day(1), intraday=True)
    run(session, 10)
    update_settings(session, intraday=False)
    step(session)
    assert get_clock(session) == day(2)                         # back to daily steps (midnight)


def test_synthetic_mode_continues_intraday_after_the_data(session):
    start(session, day(5), intraday=True)
    update_settings(session, mode="synthetic")
    run(session, TICKS)                                          # to Fri 15:30
    step(session)                                                # beyond the data
    now = get_clock(session)
    assert now == datetime(2024, 1, 8, 9, 15)                   # Monday's open
    q = quotes(session, "TCS", "NSE", as_of=now)["data"]
    assert q["prev_close"] == BARS[5][3]
    assert q["timestamp"].time() == time(9, 15)
