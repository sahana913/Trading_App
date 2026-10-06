"""
matching.py - Decide when open orders fill, and book each fill.

When does an order fill? (LTP = close of the current bar)
    MARKET          : always, at LTP
    LIMIT  BUY      : when LTP <= limit price, at LTP (never worse than the limit)
    LIMIT  SELL     : when LTP >= limit price, at LTP
    SL-M   BUY      : when LTP >= trigger price, at LTP
    SL-M   SELL     : when LTP <= trigger price, at LTP
    SL     BUY      : when LTP >= trigger AND LTP <= limit price, at LTP
    SL     SELL     : when LTP <= trigger AND LTP >= limit price, at LTP
Simplification: an SL order does not "remember" it was triggered. Both
conditions must hold on the same bar.

How much fills? At most MAX_VOLUME_PCT of the bar's volume per order per bar
(at least 1 share). The rest stays open and can fill on later bars. This is a
partial fill.

What a fill does to money (see accounts.py for available vs blocked cash):
    1. Release the slice of the order's blocked cash that belongs to the filled qty.
    2. CNC BUY : pay qty x price, add to holdings.
       CNC SELL: receive qty x price, reduce holdings, book realised P&L.
       MIS     : block margin for any NEW exposure; for exposure that is closed,
                 release its margin and book realised P&L. No full payment.
    3. Pay brokerage and charges from available cash.
    4. Record a Trade row and update the order's filled qty and average price.
"""

import uuid
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.config import MARGIN_RATE, MAX_VOLUME_PCT, SQUARE_OFF_TIME
from src.db.models import Fund, Holding, Order, Position, Trade
from src.trading.accounts import get_fund, release_from_order
from src.trading.charges import calculate_charges
from src.trading.market import latest_candle


def new_id() -> str:
    """A unique id for orders and trades, e.g. '3F9A0C1B7D2E4A68'."""
    return uuid.uuid4().hex[:16].upper()


def price_condition_met(order: Order, ltp: float) -> bool:
    """True if the order is allowed to fill at this LTP (table at the top of the file)."""
    buy = order.action == "BUY"
    if order.pricetype == "MARKET":
        return True
    if order.pricetype == "LIMIT":
        return ltp <= order.price if buy else ltp >= order.price
    if order.pricetype == "SL-M":
        return ltp >= order.trigger_price if buy else ltp <= order.trigger_price
    if order.pricetype == "SL":
        if buy:
            return order.trigger_price <= ltp <= order.price
        return order.price <= ltp <= order.trigger_price
    return False


def bar_capacity_left(session: Session, order: Order, bar_time: datetime, bar_volume: int) -> int:
    """How many more shares this order may fill on this bar."""
    cap = max(1, int(bar_volume * MAX_VOLUME_PCT))
    # Trades are stamped with their bar's time, so this sums what this
    # order already filled on this bar (if matching runs twice on one bar)
    used = session.scalar(
        select(func.coalesce(func.sum(Trade.quantity), 0)).where(
            Trade.order_id == order.id, Trade.timestamp == bar_time
        )
    )
    return cap - used


def try_fill(session: Session, order: Order, as_of: datetime | None = None) -> Trade | None:
    """Fill as much of one open order as the current bar allows. Does not commit."""
    if order.status != "open":
        return None
    bar = latest_candle(session, order.instrument_id, as_of)
    if bar is None or not price_condition_met(order, bar.close):
        return None

    remaining = order.quantity - order.filled_quantity
    qty = min(remaining, bar_capacity_left(session, order, bar.timestamp, bar.volume))
    if qty <= 0:
        return None
    return apply_fill(session, order, qty, bar.close, bar.timestamp)


# ---------------------------------------------------------------------------
# Booking a fill
# ---------------------------------------------------------------------------
def apply_fill(session: Session, order: Order, qty: int, price: float, when: datetime) -> Trade:
    """Book a fill of `qty` shares at `price`. Does not commit."""
    fund = get_fund(session, order.user_id)

    # 1. Release this fill's share of the order's blocked cash.
    #    E.g. 30 of 100 remaining shares fill -> release 30% of the block.
    #    On the final fill release everything left, so no float crumbs remain.
    remaining = order.quantity - order.filled_quantity
    share = order.margin_blocked if qty == remaining else order.margin_blocked * qty / remaining
    release_from_order(fund, order, share)

    # 2. Move shares and money
    #    (pnl is None if this fill only opened/added to a position)
    if order.product == "CNC":
        pnl = _fill_cnc(session, order, fund, qty, price)
    else:
        pnl = _fill_mis(session, order, fund, qty, price, when)

    # 3. Pay brokerage and taxes
    fees = calculate_charges(order.product, order.action, qty, price)["total"]
    fund.available_cash -= fees

    # 4. Update the order: weighted average of all its fills so far
    filled = order.filled_quantity + qty
    order.average_price = (order.average_price * order.filled_quantity + price * qty) / filled
    order.filled_quantity = filled
    order.updated_at = when
    if filled == order.quantity:
        order.status = "complete"

    trade = Trade(
        tradeid=new_id(), order_id=order.id, user_id=order.user_id,
        instrument_id=order.instrument_id, action=order.action,
        quantity=qty, price=price, fees=fees, realised_pnl=pnl, timestamp=when,
    )
    session.add(trade)
    session.flush()
    return trade


def _fill_cnc(session: Session, order: Order, fund: Fund, qty: int, price: float) -> float | None:
    """Delivery: pay in full for buys; sells come out of holdings.
    Returns the realised P&L of a sell, or None for a buy."""
    holding = session.scalar(
        select(Holding).where(Holding.user_id == order.user_id,
                              Holding.instrument_id == order.instrument_id)
    )
    if order.action == "BUY":
        if holding is None:
            holding = Holding(user_id=order.user_id, instrument_id=order.instrument_id,
                              quantity=0, average_price=0.0)
            session.add(holding)
        total = holding.quantity + qty
        holding.average_price = (holding.average_price * holding.quantity + price * qty) / total
        holding.quantity = total
        fund.available_cash -= qty * price
        return None
    else:
        # placeorder already checked that enough shares are held
        pnl = (price - holding.average_price) * qty
        holding.quantity -= qty
        fund.available_cash += qty * price
        fund.realised_pnl += pnl
        if holding.quantity == 0:
            session.delete(holding)  # sold everything
        return pnl


def _fill_mis(session: Session, order: Order, fund: Fund, qty: int, price: float,
              when: datetime) -> float | None:
    """Intraday: first close any opposite exposure, then open new exposure.
    Returns the realised P&L of the closing part, or None if nothing was closed."""
    pos = session.scalar(
        select(Position).where(Position.user_id == order.user_id,
                               Position.instrument_id == order.instrument_id,
                               Position.product == "MIS")
    )
    if pos is None:
        pos = Position(user_id=order.user_id, instrument_id=order.instrument_id, product="MIS",
                       quantity=0, average_price=0.0, realised_pnl=0.0, margin_used=0.0)
        session.add(pos)
    pos.updated_at = when
    sign = 1 if order.action == "BUY" else -1  # +1 adds to longs, -1 adds to shorts
    pnl = None

    # --- a) Closing part: the order goes against the current position ---
    if pos.quantity != 0 and (pos.quantity > 0) != (sign > 0):
        closing = min(qty, abs(pos.quantity))
        direction = 1 if pos.quantity > 0 else -1  # +1 closing a long, -1 closing a short
        pnl = (price - pos.average_price) * closing * direction
        # Free the same share of the position's margin as the share being closed
        freed = pos.margin_used * closing / abs(pos.quantity)
        pos.margin_used -= freed
        fund.used_margin -= freed
        fund.available_cash += freed + pnl
        pos.realised_pnl += pnl
        fund.realised_pnl += pnl
        pos.quantity -= closing * direction
        if pos.quantity == 0:
            pos.average_price = 0.0
        qty -= closing

    # --- b) Opening part: whatever is left adds exposure (or flips the side) ---
    if qty > 0:
        size = abs(pos.quantity)
        pos.average_price = (pos.average_price * size + price * qty) / (size + qty)
        pos.quantity += sign * qty
        margin = qty * price * MARGIN_RATE["MIS"]
        pos.margin_used += margin
        fund.used_margin += margin
        fund.available_cash -= margin
    return pnl


# ---------------------------------------------------------------------------
# Running the market
# ---------------------------------------------------------------------------
def match_orders(session: Session, as_of: datetime | None = None) -> list[Trade]:
    """Check every open order against the current price and fill what can fill.

    Call this each time the simulated clock moves. At or after SQUARE_OFF_TIME
    it also squares off all MIS positions.
    """
    open_orders = session.scalars(
        select(Order).where(Order.status == "open").order_by(Order.created_at, Order.id)
    ).all()
    trades = []
    for order in open_orders:  # oldest first: first come, first served
        trade = try_fill(session, order, as_of)
        if trade is not None:
            trades.append(trade)
    session.commit()

    if as_of is not None and as_of.time() >= SQUARE_OFF_TIME:
        square_off_mis(session, as_of)
    return trades


def square_off_mis(session: Session, as_of: datetime | None = None) -> int:
    """Market close for intraday: cancel open MIS orders and close every MIS
    position at the current price. Returns how many positions were closed.

    The broker does this for you, so it skips the funds check and the volume
    cap: the position must be closed no matter what.
    With daily bars (no time of day), the simulator calls this at each day's end.
    """
    # 1. Cancel waiting MIS orders and give back their blocked cash
    for order in session.scalars(select(Order).where(Order.status == "open", Order.product == "MIS")):
        release_from_order(get_fund(session, order.user_id), order, order.margin_blocked)
        order.status = "cancelled"
        order.rejection_reason = "Cancelled at MIS square-off"

    # 2. Close each open MIS position with an opposite MARKET order
    closed = 0
    for pos in session.scalars(select(Position).where(Position.product == "MIS", Position.quantity != 0)).all():
        bar = latest_candle(session, pos.instrument_id, as_of)
        if bar is None:
            continue  # no price to close at
        order = Order(
            orderid=new_id(), user_id=pos.user_id, instrument_id=pos.instrument_id,
            strategy="AUTO_SQUAREOFF", action="SELL" if pos.quantity > 0 else "BUY",
            pricetype="MARKET", product="MIS", quantity=abs(pos.quantity),
            price=0.0, trigger_price=0.0, status="open", filled_quantity=0,
            average_price=0.0, margin_blocked=0.0,
            created_at=bar.timestamp, updated_at=bar.timestamp,
        )
        session.add(order)
        session.flush()
        apply_fill(session, order, order.quantity, bar.close, bar.timestamp)
        closed += 1

    session.commit()
    return closed
