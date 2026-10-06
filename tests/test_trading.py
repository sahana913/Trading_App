"""Tests for the trading engine in src/trading/.

Each test builds its own tiny market: an in-memory database, one instrument
(TCS) and candles added one at a time with bar(). Adding a bar moves the
simulated clock forward; calling match_orders() lets open orders react.
Bars are one minute apart starting 09:15, with plenty of volume unless a test
needs a partial fill.
"""

from datetime import datetime, timedelta

import pytest
from sqlalchemy import func, select

from src import security
from src.config import STARTING_CASH
from src.db.models import Candle, Instrument, Order, Position, Trade, User
from src.db.session import get_engine, get_session_factory, init_db
from src.trading import (
    cancelorder, create_user, funds, history, holdings, match_orders, modifyorder,
    orderbook, placeorder, positionbook, quotes, square_off_mis, tradebook,
)
from src.trading.charges import calculate_charges

T0 = datetime(2024, 1, 1, 9, 15)


def minute(n: int) -> datetime:
    return T0 + timedelta(minutes=n)


def fee(product, action, qty, price) -> float:
    return calculate_charges(product, action, qty, price)["total"]


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def fast_password_hashing(monkeypatch):
    """600,000 hash rounds per user is slow; tests don't need real security."""
    monkeypatch.setattr(security, "ITERATIONS", 1000)


@pytest.fixture
def session():
    engine = get_engine("sqlite:///:memory:")
    init_db(engine)
    with get_session_factory(engine)() as s:
        yield s


@pytest.fixture
def tcs(session) -> Instrument:
    inst = Instrument(symbol="TCS", exchange="NSE", name="TCS", lot_size=1, tick_size=0.05, is_active=True)
    session.add(inst)
    session.commit()
    return inst


@pytest.fixture
def user(session) -> User:
    return create_user(session, "alice", "pw")


def bar(session, inst, n, close, volume=1_000_000, at=None):
    """Add one candle where open = high = low = close (keeps the maths simple)."""
    session.add(Candle(instrument_id=inst.id, timestamp=at or minute(n), open=close,
                       high=close, low=close, close=close, volume=volume))
    session.commit()


def place(session, user, **kwargs) -> dict:
    """placeorder with sensible defaults: MARKET BUY 10 TCS, MIS."""
    args = dict(symbol="TCS", exchange="NSE", action="BUY", quantity=10,
                pricetype="MARKET", product="MIS")
    args.update(kwargs)
    return placeorder(session, user.id, **args)


def get_order(session, orderid) -> Order:
    return session.scalar(select(Order).where(Order.orderid == orderid))


def cash(session, user) -> dict:
    return funds(session, user.id)["data"]


def mis_position(session, user) -> Position:
    return session.scalar(select(Position).where(Position.user_id == user.id))


def total_fees(session, user) -> float:
    return session.scalar(select(func.sum(Trade.fees)).where(Trade.user_id == user.id)) or 0.0


# ---------------------------------------------------------------------------
# Accounts
# ---------------------------------------------------------------------------
def test_new_user_starts_with_ten_lakh(session, user):
    f = cash(session, user)
    assert f["availablecash"] == 1_000_000
    assert f["utiliseddebits"] == 0
    assert f["m2mrealized"] == 0
    assert f["m2munrealized"] == 0


# ---------------------------------------------------------------------------
# Fills by order type
# ---------------------------------------------------------------------------
def test_market_buy_cnc_fills_immediately(session, tcs, user):
    bar(session, tcs, 0, 100)
    result = place(session, user, product="CNC")

    assert result["status"] == "success"
    order = get_order(session, result["orderid"])
    assert order.status == "complete"
    assert order.average_price == 100

    h = holdings(session, user.id)["data"]["holdings"]
    assert h[0]["quantity"] == 10 and h[0]["average_price"] == 100

    f = cash(session, user)
    assert f["availablecash"] == pytest.approx(STARTING_CASH - 1000 - fee("CNC", "BUY", 10, 100))
    assert f["utiliseddebits"] == 0
    assert len(tradebook(session, user.id)["data"]) == 1


def test_limit_buy_waits_then_fills_when_price_drops(session, tcs, user):
    bar(session, tcs, 0, 100)
    result = place(session, user, pricetype="LIMIT", price=95, product="CNC")
    order = get_order(session, result["orderid"])
    assert order.status == "open"

    # Cash for 10 x 95 plus estimated charges is blocked while it waits
    block = 950 + fee("CNC", "BUY", 10, 95)
    assert cash(session, user)["utiliseddebits"] == pytest.approx(block)
    assert cash(session, user)["availablecash"] == pytest.approx(STARTING_CASH - block)

    bar(session, tcs, 1, 97)
    match_orders(session)
    assert order.status == "open"  # 97 is still above the limit

    bar(session, tcs, 2, 94)
    match_orders(session)
    assert order.status == "complete"
    assert order.average_price == 94  # filled at LTP, better than the limit
    assert cash(session, user)["utiliseddebits"] == 0
    assert cash(session, user)["availablecash"] == pytest.approx(
        STARTING_CASH - 940 - fee("CNC", "BUY", 10, 94))


def test_limit_buy_above_market_fills_now_at_ltp(session, tcs, user):
    bar(session, tcs, 0, 100)
    order = get_order(session, place(session, user, pricetype="LIMIT", price=105)["orderid"])
    assert order.status == "complete"
    assert order.average_price == 100


def test_limit_sell_fills_when_price_rises(session, tcs, user):
    bar(session, tcs, 0, 100)
    order = get_order(session, place(session, user, action="SELL", pricetype="LIMIT", price=110)["orderid"])
    assert order.status == "open"

    bar(session, tcs, 1, 111)
    match_orders(session)
    assert order.status == "complete"
    assert order.average_price == 111
    assert mis_position(session, user).quantity == -10  # MIS allows short selling


def test_sl_m_sell_works_as_a_stop_loss(session, tcs, user):
    bar(session, tcs, 0, 100)
    place(session, user)  # long 10 @ 100
    stop = get_order(session, place(session, user, action="SELL", pricetype="SL-M",
                                    trigger_price=95)["orderid"])
    assert stop.status == "open"

    bar(session, tcs, 1, 96)
    match_orders(session)
    assert stop.status == "open"  # trigger not reached

    bar(session, tcs, 2, 94)
    match_orders(session)
    assert stop.status == "complete"
    assert stop.average_price == 94
    pos = mis_position(session, user)
    assert pos.quantity == 0
    assert pos.realised_pnl == pytest.approx(-60)  # (94 - 100) x 10


def test_sl_buy_needs_trigger_hit_and_price_within_limit(session, tcs, user):
    bar(session, tcs, 0, 100)
    order = get_order(session, place(session, user, pricetype="SL", trigger_price=105, price=106)["orderid"])

    bar(session, tcs, 1, 104)
    match_orders(session)
    assert order.status == "open"  # not triggered yet

    bar(session, tcs, 2, 108)
    match_orders(session)
    assert order.status == "open"  # triggered, but above the 106 limit

    bar(session, tcs, 3, 105.5)
    match_orders(session)
    assert order.status == "complete"
    assert order.average_price == 105.5


def test_order_placed_in_the_past_uses_that_bars_price(session, tcs, user):
    bar(session, tcs, 0, 100)
    bar(session, tcs, 1, 110)
    order = get_order(session, place(session, user, as_of=minute(0))["orderid"])
    assert order.average_price == 100
    assert order.created_at == minute(0)


# ---------------------------------------------------------------------------
# Partial cases
# ---------------------------------------------------------------------------
def test_large_order_fills_over_several_bars(session, tcs, user):
    bar(session, tcs, 0, 100, volume=1000)  # at most 10% = 100 shares per bar
    order = get_order(session, place(session, user, quantity=250)["orderid"])
    assert (order.status, order.filled_quantity) == ("open", 100)

    match_orders(session)  # same bar again: its volume is already used up
    assert order.filled_quantity == 100

    bar(session, tcs, 1, 102, volume=1000)
    match_orders(session)
    assert order.filled_quantity == 200

    bar(session, tcs, 2, 104, volume=500)  # only 50 shares available
    match_orders(session)
    assert order.status == "complete"
    # Weighted average: (100x100 + 100x102 + 50x104) / 250
    assert order.average_price == pytest.approx(101.6)
    assert len(tradebook(session, user.id)["data"]) == 3
    # Only the position's 20% margin remains blocked
    assert cash(session, user)["utiliseddebits"] == pytest.approx(0.2 * 25400)


def test_cancel_partially_filled_order_keeps_the_filled_part(session, tcs, user):
    bar(session, tcs, 0, 100, volume=1000)
    order = get_order(session, place(session, user, quantity=300, pricetype="LIMIT",
                                     price=100, product="CNC")["orderid"])
    assert order.filled_quantity == 100

    assert cancelorder(session, user.id, order.orderid)["status"] == "success"
    assert order.status == "cancelled"
    assert order.filled_quantity == 100
    assert holdings(session, user.id)["data"]["holdings"][0]["quantity"] == 100
    f = cash(session, user)
    assert f["utiliseddebits"] == pytest.approx(0)  # unfilled block given back
    assert f["availablecash"] == pytest.approx(STARTING_CASH - 10_000 - fee("CNC", "BUY", 100, 100))


def test_partial_close_of_mis_position(session, tcs, user):
    bar(session, tcs, 0, 100)
    place(session, user)  # long 10 @ 100, margin 200
    bar(session, tcs, 1, 110)
    place(session, user, action="SELL", quantity=4)

    pos = mis_position(session, user)
    assert pos.quantity == 6
    assert pos.average_price == 100
    assert pos.realised_pnl == pytest.approx(40)  # (110 - 100) x 4
    assert pos.margin_used == pytest.approx(120)  # 6/10 of 200 still blocked
    assert cash(session, user)["utiliseddebits"] == pytest.approx(120)


def test_selling_more_than_long_flips_to_short(session, tcs, user):
    bar(session, tcs, 0, 100)
    place(session, user)  # long 10 @ 100
    bar(session, tcs, 1, 90)
    place(session, user, action="SELL", quantity=15)

    pos = mis_position(session, user)
    assert pos.quantity == -5
    assert pos.average_price == 90  # the new short starts at 90
    assert pos.realised_pnl == pytest.approx(-100)
    assert pos.margin_used == pytest.approx(5 * 90 * 0.2)


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "fields, message",
    [
        (dict(quantity=0), "greater than 0"),
        (dict(quantity=-5), "greater than 0"),
        (dict(quantity=2.5), "whole number"),
        (dict(quantity="10"), "whole number"),
        (dict(quantity=True), "whole number"),
        (dict(action="HOLD"), "Invalid action"),
        (dict(pricetype="STOP"), "Invalid pricetype"),
        (dict(product="NRML"), "Invalid product"),
        (dict(symbol="NOPE"), "Unknown"),
        (dict(pricetype="LIMIT", price=0), "price greater than 0"),
        (dict(pricetype="LIMIT", price=-1), "cannot be negative"),
        (dict(pricetype="SL", price=100, trigger_price=0), "trigger price greater than 0"),
        (dict(pricetype="SL-M"), "trigger price greater than 0"),
        (dict(pricetype="SL", price=99, trigger_price=100), "SL BUY"),
        (dict(pricetype="SL", action="SELL", price=101, trigger_price=100), "SL SELL"),
    ],
)
def test_invalid_input_is_an_error_and_nothing_is_saved(session, tcs, user, fields, message):
    bar(session, tcs, 0, 100)
    result = place(session, user, **fields)
    assert result["status"] == "error"
    assert message in result["message"]
    assert orderbook(session, user.id)["data"]["orders"] == []


def test_quantity_must_match_lot_size(session, tcs, user):
    tcs.lot_size = 50
    bar(session, tcs, 0, 100)
    assert "lot size" in place(session, user, quantity=25)["message"]
    assert place(session, user, quantity=50)["status"] == "success"


def test_inactive_user_cannot_trade(session, tcs, user):
    bar(session, tcs, 0, 100)
    user.is_active = False
    session.commit()
    assert "inactive" in place(session, user)["message"]


def test_no_price_data_is_an_error(session, tcs, user):
    assert "No price data" in place(session, user)["message"]


def test_insufficient_funds_rejects_and_keeps_cash(session, tcs, user):
    bar(session, tcs, 0, 100)
    result = place(session, user, quantity=10_001, product="CNC")  # ₹10,00,100 > ₹10,00,000

    assert result["status"] == "error"
    assert "Insufficient funds" in result["message"]
    # Rejected orders are kept so the user (and admin) can see why
    row = orderbook(session, user.id)["data"]["orders"][0]
    assert row["order_status"] == "rejected"
    assert row["orderid"] == result["orderid"]
    assert cash(session, user)["availablecash"] == STARTING_CASH


def test_mis_leverage_allows_bigger_orders_than_cnc(session, tcs, user):
    bar(session, tcs, 0, 100)
    # ₹40,00,000 of stock: MIS needs 20% = ₹8,00,000, CNC needs all of it
    assert place(session, user, quantity=40_000, product="MIS")["status"] == "success"
    assert place(session, user, quantity=40_000, product="CNC")["status"] == "error"


def test_cnc_sell_needs_holdings_minus_pending_sells(session, tcs, user):
    bar(session, tcs, 0, 100)
    assert "Insufficient holdings" in place(session, user, action="SELL", product="CNC")["message"]

    place(session, user, quantity=10, product="CNC")  # now hold 10
    place(session, user, action="SELL", quantity=6, product="CNC", pricetype="LIMIT", price=200)  # waits
    # Only 4 are left that aren't already promised to the open order
    assert "sell 4" in place(session, user, action="SELL", quantity=5, product="CNC")["message"]
    assert place(session, user, action="SELL", quantity=4, product="CNC")["status"] == "success"


def test_no_new_mis_orders_after_square_off_time(session, tcs, user):
    bar(session, tcs, 0, 100, at=datetime(2024, 1, 1, 15, 20))
    assert "square-off" in place(session, user, product="MIS")["message"]
    assert place(session, user, product="CNC")["status"] == "success"


# ---------------------------------------------------------------------------
# Cancellations
# ---------------------------------------------------------------------------
def test_cancel_releases_the_whole_block(session, tcs, user):
    bar(session, tcs, 0, 100)
    orderid = place(session, user, pricetype="LIMIT", price=90, product="CNC")["orderid"]
    assert cash(session, user)["utiliseddebits"] > 0

    assert cancelorder(session, user.id, orderid)["status"] == "success"
    assert get_order(session, orderid).status == "cancelled"
    assert cash(session, user)["availablecash"] == pytest.approx(STARTING_CASH)
    assert cash(session, user)["utiliseddebits"] == pytest.approx(0)


def test_cannot_cancel_finished_or_other_users_orders(session, tcs, user):
    bar(session, tcs, 0, 100)
    done = place(session, user)["orderid"]  # MARKET -> complete
    assert "complete" in cancelorder(session, user.id, done)["message"]

    waiting = place(session, user, pricetype="LIMIT", price=90)["orderid"]
    bob = create_user(session, "bob", "pw")
    assert "not found" in cancelorder(session, bob.id, waiting)["message"]

    cancelorder(session, user.id, waiting)
    assert "cancelled" in cancelorder(session, user.id, waiting)["message"]


# ---------------------------------------------------------------------------
# Modifications
# ---------------------------------------------------------------------------
def test_modify_to_marketable_price_fills_now(session, tcs, user):
    bar(session, tcs, 0, 100)
    orderid = place(session, user, pricetype="LIMIT", price=95)["orderid"]
    assert modifyorder(session, user.id, orderid, price=101)["status"] == "success"
    order = get_order(session, orderid)
    assert order.status == "complete"
    assert order.average_price == 100


def test_modify_quantity_resizes_the_block(session, tcs, user):
    bar(session, tcs, 0, 100)
    orderid = place(session, user, pricetype="LIMIT", price=95, product="CNC")["orderid"]
    modifyorder(session, user.id, orderid, quantity=20)
    assert cash(session, user)["utiliseddebits"] == pytest.approx(20 * 95 + fee("CNC", "BUY", 20, 95))


def test_unaffordable_modify_leaves_order_unchanged(session, tcs, user):
    bar(session, tcs, 0, 100)
    orderid = place(session, user, pricetype="LIMIT", price=95, product="CNC")["orderid"]
    before = cash(session, user)

    result = modifyorder(session, user.id, orderid, quantity=20_000)
    assert "Insufficient funds" in result["message"]
    assert get_order(session, orderid).quantity == 10
    assert cash(session, user) == before


def test_modify_quantity_must_exceed_filled(session, tcs, user):
    bar(session, tcs, 0, 100, volume=1000)
    orderid = place(session, user, quantity=300, pricetype="LIMIT", price=100, product="CNC")["orderid"]
    assert "already filled" in modifyorder(session, user.id, orderid, quantity=100)["message"]

    assert modifyorder(session, user.id, orderid, quantity=200)["status"] == "success"
    # 100 unfilled shares remain, so the block covers just those
    assert get_order(session, orderid).margin_blocked == pytest.approx(10_000 + fee("CNC", "BUY", 100, 100))


def test_cannot_modify_completed_order(session, tcs, user):
    bar(session, tcs, 0, 100)
    orderid = place(session, user)["orderid"]
    assert "only open orders" in modifyorder(session, user.id, orderid, quantity=20)["message"]


# ---------------------------------------------------------------------------
# Funds and P&L
# ---------------------------------------------------------------------------
def test_cnc_round_trip_pnl(session, tcs, user):
    bar(session, tcs, 0, 100)
    place(session, user, product="CNC")
    bar(session, tcs, 1, 120)
    place(session, user, action="SELL", product="CNC")

    f = cash(session, user)
    assert f["m2mrealized"] == pytest.approx(200)
    assert f["availablecash"] == pytest.approx(
        STARTING_CASH + 200 - fee("CNC", "BUY", 10, 100) - fee("CNC", "SELL", 10, 120))
    assert holdings(session, user.id)["data"]["holdings"] == []


def test_mis_long_unrealised_then_realised(session, tcs, user):
    bar(session, tcs, 0, 100)
    place(session, user, quantity=100)
    assert cash(session, user)["utiliseddebits"] == pytest.approx(2000)  # 20% of ₹10,000

    bar(session, tcs, 1, 105)
    assert cash(session, user)["m2munrealized"] == pytest.approx(500)
    row = positionbook(session, user.id)["data"][0]
    assert (row["quantity"], row["ltp"], row["pnl"]) == (100, 105, 500)

    place(session, user, action="SELL", quantity=100)
    f = cash(session, user)
    assert f["m2mrealized"] == pytest.approx(500)
    assert f["m2munrealized"] == 0
    assert f["utiliseddebits"] == pytest.approx(0)
    assert f["availablecash"] == pytest.approx(
        STARTING_CASH + 500 - fee("MIS", "BUY", 100, 100) - fee("MIS", "SELL", 100, 105))


def test_mis_short_loses_when_price_rises(session, tcs, user):
    bar(session, tcs, 0, 100)
    place(session, user, action="SELL")
    bar(session, tcs, 1, 110)
    assert positionbook(session, user.id)["data"][0]["unrealised_pnl"] == pytest.approx(-100)
    place(session, user, action="BUY")
    assert cash(session, user)["m2mrealized"] == pytest.approx(-100)


def test_holdings_statistics(session, tcs, user):
    bar(session, tcs, 0, 100)
    place(session, user, product="CNC")
    bar(session, tcs, 1, 110)
    stats = holdings(session, user.id)["data"]["statistics"]
    assert stats == {"totalholdingvalue": 1100, "totalinvvalue": 1000,
                     "totalprofitandloss": 100, "totalpnlpercentage": 10}


def test_money_is_never_created_or_lost(session, tcs, user):
    """After any mix of trades: free cash + blocked cash + cost of holdings
    = starting cash + realised P&L - all charges paid."""
    bar(session, tcs, 0, 100, volume=2000)
    place(session, user, quantity=50, product="CNC")
    place(session, user, quantity=300)                                  # partial MIS fill
    place(session, user, pricetype="LIMIT", price=90, product="CNC")    # waiting
    bar(session, tcs, 1, 107, volume=2000)
    match_orders(session)
    place(session, user, action="SELL", quantity=120)                   # MIS partial close
    place(session, user, action="SELL", quantity=20, product="CNC")
    bar(session, tcs, 2, 95, volume=2000)
    match_orders(session)

    f = cash(session, user)
    h = holdings(session, user.id)["data"]["statistics"]["totalinvvalue"]
    left = f["availablecash"] + f["utiliseddebits"] + h
    right = STARTING_CASH + f["m2mrealized"] - total_fees(session, user)
    assert left == pytest.approx(right, abs=0.05)  # display rounding only


# ---------------------------------------------------------------------------
# MIS square-off
# ---------------------------------------------------------------------------
def test_square_off_closes_mis_and_leaves_cnc(session, tcs, user):
    bar(session, tcs, 0, 100)
    place(session, user)                                               # MIS long 10
    place(session, user, quantity=5, product="CNC")                    # holding 5
    mis_wait = place(session, user, pricetype="LIMIT", price=50)["orderid"]
    cnc_wait = place(session, user, pricetype="LIMIT", price=50, product="CNC")["orderid"]

    bar(session, tcs, 1, 110)
    assert square_off_mis(session) == 1

    assert mis_position(session, user).quantity == 0
    assert mis_position(session, user).realised_pnl == pytest.approx(100)
    assert get_order(session, mis_wait).status == "cancelled"
    assert get_order(session, cnc_wait).status == "open"
    assert holdings(session, user.id)["data"]["holdings"][0]["quantity"] == 5
    # Only the waiting CNC order still has cash blocked
    assert cash(session, user)["utiliseddebits"] == pytest.approx(get_order(session, cnc_wait).margin_blocked)
    auto = [o for o in orderbook(session, user.id)["data"]["orders"] if o["strategy"] == "AUTO_SQUAREOFF"]
    assert len(auto) == 1 and auto[0]["order_status"] == "complete"


def test_match_orders_squares_off_at_market_close(session, tcs, user):
    bar(session, tcs, 0, 100, at=datetime(2024, 1, 1, 15, 0))
    place(session, user)
    match_orders(session, as_of=datetime(2024, 1, 1, 15, 0))
    assert mis_position(session, user).quantity == 10  # before 15:15: still open

    bar(session, tcs, 0, 102, at=datetime(2024, 1, 1, 15, 15))
    match_orders(session, as_of=datetime(2024, 1, 1, 15, 15))
    assert mis_position(session, user).quantity == 0
    assert mis_position(session, user).realised_pnl == pytest.approx(20)


# ---------------------------------------------------------------------------
# Market data and books
# ---------------------------------------------------------------------------
def test_quotes_respect_the_simulated_clock(session, tcs):
    for n, price in enumerate([100, 101, 102]):
        bar(session, tcs, n, price)
    assert quotes(session, "TCS", "NSE")["data"]["ltp"] == 102
    q = quotes(session, "tcs", "nse", as_of=minute(1))["data"]  # case-insensitive
    assert (q["ltp"], q["prev_close"]) == (101, 100)
    assert quotes(session, "NOPE", "NSE")["status"] == "error"


def test_history_date_range_is_inclusive(session, tcs):
    for day in (1, 2, 3):
        bar(session, tcs, 0, 100 + day, at=datetime(2024, 1, day, 9, 15))
    rows = history(session, "TCS", "NSE", start_date="2024-01-02", end_date="2024-01-02")["data"]
    assert [r["close"] for r in rows] == [102]
    assert len(history(session, "TCS", "NSE")["data"]) == 3


def test_books_only_show_your_own_orders(session, tcs, user):
    bar(session, tcs, 0, 100)
    bob = create_user(session, "bob", "pw")
    place(session, user)
    place(session, bob, action="SELL")
    place(session, user, pricetype="LIMIT", price=50)

    data = orderbook(session, user.id)["data"]
    assert len(data["orders"]) == 2
    assert data["statistics"]["total_completed_orders"] == 1
    assert data["statistics"]["total_open_orders"] == 1
    assert len(tradebook(session, bob.id)["data"]) == 1
