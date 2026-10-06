"""Tests for src/trading/simulator.py.

The market: two stocks with DAILY bars on Mon 1 Jan - Fri 5 Jan 2024 and
Mon 8 Jan 2024 (the weekend has no bars, like real data). INFY has no bar on
3 Jan, like a row removed by cleaning.
"""

from datetime import date, datetime

import pytest
from sqlalchemy import func, select

from src import security
from src.config import STARTING_CASH
from src.db.models import Candle, DailyPnl, Instrument, Order, Trade
from src.db.session import get_engine, get_session_factory, init_db
from src.trading import create_user, funds, holdings, placeorder, positionbook
from src.trading.books import portfolio_value
from src.trading.simulator import get_clock, record_daily_pnl, reset, run, start, status, step

DAYS = [1, 2, 3, 4, 5, 8]
TCS_CLOSE = {1: 100, 2: 98, 3: 94, 4: 97, 5: 103, 8: 110}
INFY_CLOSE = {1: 50, 2: 51, 4: 52, 5: 53, 8: 54}  # no bar on the 3rd


def day(d: int) -> datetime:
    return datetime(2024, 1, d)


@pytest.fixture(autouse=True)
def fast_password_hashing(monkeypatch):
    monkeypatch.setattr(security, "ITERATIONS", 1000)


@pytest.fixture
def session():
    engine = get_engine("sqlite:///:memory:")
    init_db(engine)
    with get_session_factory(engine)() as s:
        for symbol, closes in (("TCS", TCS_CLOSE), ("INFY", INFY_CLOSE)):
            inst = Instrument(symbol=symbol, exchange="NSE", name=symbol, lot_size=1,
                              tick_size=0.05, is_active=True)
            s.add(inst)
            s.flush()
            for d, c in closes.items():
                s.add(Candle(instrument_id=inst.id, timestamp=day(d), open=c, high=c,
                             low=c, close=c, volume=1_000_000))
        s.commit()
        yield s


@pytest.fixture
def user(session):
    return create_user(session, "alice", "pw")


def buy(session, user, **kwargs):
    """Place an order at the simulated 'now' (as the app will do)."""
    args = dict(symbol="TCS", exchange="NSE", action="BUY", quantity=10,
                pricetype="MARKET", product="CNC", as_of=get_clock(session))
    args.update(kwargs)
    return placeorder(session, user.id, **args)


# ---------------------------------------------------------------------------
# The clock
# ---------------------------------------------------------------------------
def test_no_clock_before_start(session):
    assert get_clock(session) is None
    assert "not started" in step(session)["message"]


def test_start_snaps_to_next_trading_day(session):
    start(session, datetime(2023, 12, 30))  # a Saturday before the data
    assert get_clock(session) == day(1)
    start(session, day(6))                  # Saturday -> Monday the 8th
    assert get_clock(session) == day(8)


def test_start_after_data_ends_is_an_error(session):
    assert "No market data" in start(session, datetime(2025, 1, 1))["message"]


def test_step_moves_one_trading_day_and_skips_weekends(session):
    start(session, day(1))
    seen = [get_clock(session)]
    while step(session)["status"] == "success":
        seen.append(get_clock(session))
    # The 3rd is kept even though INFY has no bar that day (TCS does)
    assert seen == [day(d) for d in DAYS]


def test_end_of_data_keeps_clock_in_place(session):
    start(session, day(8))
    result = step(session)
    assert "End of data" in result["message"]
    assert get_clock(session) == day(8)


def test_run_stops_at_end_of_data(session):
    start(session, day(1))
    result = run(session, 100)
    assert result["data"]["steps"] == 5
    assert get_clock(session) == day(8)


def test_status(session):
    assert status(session)["data"]["current_time"] is None
    start(session, day(4))
    data = status(session)["data"]
    assert (data["current_time"], data["data_start"], data["data_end"]) == (day(4), day(1), day(8))
    assert data["bars_left"] == 2  # the 5th and the 8th


# ---------------------------------------------------------------------------
# Orders over several days
# ---------------------------------------------------------------------------
def test_limit_order_fills_on_a_later_day(session, user):
    start(session, day(1))
    orderid = buy(session, user, pricetype="LIMIT", price=95)["orderid"]
    order = session.scalar(select(Order).where(Order.orderid == orderid))

    step(session)  # 2nd: 98, too high
    assert order.status == "open"
    result = step(session)  # 3rd: 94, fills
    assert result["data"]["fills"] == 1
    assert order.status == "complete"
    assert order.average_price == 94


def test_no_look_ahead(session, user):
    """While the clock is on the 2nd, a market order must use the 2nd's price,
    even though later bars are in the database."""
    start(session, day(2))
    buy(session, user)
    trade = session.scalar(select(Trade))
    assert trade.price == 98
    assert trade.timestamp == day(2)


def test_mis_is_squared_off_at_day_end_but_cnc_is_kept(session, user):
    start(session, day(1))
    buy(session, user, product="MIS")
    buy(session, user, product="CNC", quantity=5)

    result = step(session)
    assert result["data"]["mis_squared_off"] == 1
    assert positionbook(session, user.id)["data"][0]["quantity"] == 0
    assert holdings(session, user.id)["data"]["holdings"][0]["quantity"] == 5


def test_missing_bar_uses_last_known_price(session, user):
    """INFY has no bar on the 3rd: its price on the 3rd is the 2nd's close."""
    start(session, day(1))
    buy(session, user, symbol="INFY")
    run(session, 2)  # now on the 3rd
    assert holdings(session, user.id, as_of=get_clock(session))["data"]["holdings"][0]["ltp"] == 51


# ---------------------------------------------------------------------------
# Daily P&L snapshots
# ---------------------------------------------------------------------------
def test_daily_snapshots_track_equity(session, user):
    start(session, day(1))
    buy(session, user)  # 10 TCS @ 100 on the 1st
    run(session, 3)     # closes the 1st, 2nd and 3rd; clock now on the 4th

    snaps = session.scalars(select(DailyPnl).order_by(DailyPnl.trade_date)).all()
    assert [s.trade_date for s in snaps] == [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)]

    paid = session.scalar(select(func.sum(Trade.fees)))
    # On the 3rd TCS closed at 94: 10 shares lost (94 - 100) x 10 = 60, plus charges
    third = snaps[-1]
    assert third.unrealised_pnl == pytest.approx(-60)
    assert third.equity == pytest.approx(STARTING_CASH - 60 - paid, abs=0.01)
    assert third.total_pnl == pytest.approx(-60 - paid, abs=0.01)


def test_snapshot_twice_on_one_day_does_not_duplicate(session, user):
    start(session, day(1))
    record_daily_pnl(session, day(1))
    record_daily_pnl(session, day(1))
    assert session.scalar(select(func.count()).select_from(DailyPnl)) == 1


def test_equity_after_round_trip(session, user):
    start(session, day(1))
    buy(session, user)                     # @100
    run(session, 4)                        # to the 5th: 103
    buy(session, user, action="SELL")      # sell @103

    value = portfolio_value(session, user.id, get_clock(session))
    paid = session.scalar(select(func.sum(Trade.fees)))
    assert value["realised_pnl"] == pytest.approx(30)
    assert value["equity"] == pytest.approx(STARTING_CASH + 30 - paid)
    assert value["holdings_value"] == 0


# ---------------------------------------------------------------------------
# Start / reset safety
# ---------------------------------------------------------------------------
def test_cannot_restart_while_trades_exist(session, user):
    start(session, day(1))
    buy(session, user)
    assert "reset" in start(session, day(1))["message"]


def test_reset_wipes_activity_and_restores_cash(session, user):
    start(session, day(1))
    buy(session, user)
    run(session, 2)

    reset(session)

    assert get_clock(session) is None
    assert session.scalar(select(func.count()).select_from(Order)) == 0
    assert session.scalar(select(func.count()).select_from(DailyPnl)) == 0
    assert funds(session, user.id)["data"]["availablecash"] == STARTING_CASH
    assert start(session, day(1))["status"] == "success"  # can start again
