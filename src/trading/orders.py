"""
orders.py - OpenAlgo's placeorder, modifyorder and cancelorder.

ORDER LIFECYCLE
===============

  placeorder()
      |
      |-- invalid input (bad symbol, quantity, price...) --> error, NOTHING saved
      |
      |-- not enough cash / shares ----------------------> saved as "rejected"
      |                                                   (shows in the orderbook)
      v
   "open"   cash for the order is BLOCKED (moved from available to used margin)
      |     and we immediately try to match it against the current price:
      |     a MARKET order fills now; a LIMIT/SL order fills now only if the
      |     price already allows it.
      |
      |-- modifyorder() : change qty/price/trigger; the block is recalculated
      |                   and matching is tried again. Stays "open".
      |
      |-- match_orders() on each new bar --> fills some or all of the order
      |        part filled: still "open", filled_quantity < quantity
      |        all filled : "complete"; its whole block has been released and
      |                     replaced by the real cost (see matching.py)
      |
      |-- cancelorder() : "cancelled"; the unfilled part's block is released.
      |                   Shares already filled stay filled.
      |
      '-- MIS square-off (market close): open MIS orders become "cancelled"

"complete", "cancelled" and "rejected" are final: such orders can't change.

Differences from OpenAlgo: we take a user_id instead of an API key, and
modifyorder only needs the fields you want to change.
"""

from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.config import MARGIN_RATE, SQUARE_OFF_TIME
from src.db.models import ACTIONS, PRICETYPES, Holding, Instrument, Order, Position, User
from src.trading.accounts import block_for_order, get_fund, release_from_order
from src.trading.charges import calculate_charges
from src.trading.market import error, get_instrument, latest_candle, success
from src.trading.matching import new_id, try_fill

SUPPORTED_PRODUCTS = ("MIS", "CNC")  # NRML (F&O) is not simulated


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def check_order_fields(action: str, pricetype: str, quantity, price: float,
                       trigger_price: float, lot_size: int) -> str | None:
    """Return an error message, or None if the fields make sense."""
    if action not in ACTIONS:
        return f"Invalid action '{action}'. Use BUY or SELL"
    if pricetype not in PRICETYPES:
        return f"Invalid pricetype '{pricetype}'. Use one of {', '.join(PRICETYPES)}"
    # bool is a subclass of int in Python, so True would sneak through without this
    if isinstance(quantity, bool) or not isinstance(quantity, int):
        return "Quantity must be a whole number"
    if quantity <= 0:
        return "Quantity must be greater than 0"
    if quantity % lot_size != 0:
        return f"Quantity must be a multiple of the lot size ({lot_size})"
    if price < 0 or trigger_price < 0:
        return "Price and trigger price cannot be negative"
    if pricetype in ("LIMIT", "SL") and price <= 0:
        return f"{pricetype} orders need a price greater than 0"
    if pricetype in ("SL", "SL-M") and trigger_price <= 0:
        return f"{pricetype} orders need a trigger price greater than 0"
    if pricetype == "SL":
        # A stop-loss BUY triggers on the way up, so its limit must be at or above
        # the trigger; a SELL triggers on the way down, so its limit is at or below
        if action == "BUY" and price < trigger_price:
            return "For SL BUY the price must be >= trigger price"
        if action == "SELL" and price > trigger_price:
            return "For SL SELL the price must be <= trigger price"
    return None


def estimate_price(pricetype: str, price: float, trigger_price: float, ltp: float) -> float:
    """Best guess of the fill price, used to size the cash block."""
    if pricetype in ("LIMIT", "SL"):
        return price  # it can't fill worse than this
    if pricetype == "SL-M":
        return trigger_price  # it fills around the trigger
    return ltp  # MARKET


def mis_position_qty(session: Session, user_id: int, instrument_id: int) -> int:
    """Current net MIS quantity (+ long, - short, 0 flat)."""
    qty = session.scalar(
        select(Position.quantity).where(Position.user_id == user_id,
                                        Position.instrument_id == instrument_id,
                                        Position.product == "MIS")
    )
    return qty or 0


def required_block(session: Session, user_id: int, instrument_id: int, product: str,
                   action: str, quantity: int, est_price: float) -> float:
    """Cash to block for an order: margin for new exposure + estimated charges."""
    charges = calculate_charges(product, action, quantity, est_price)["total"]
    if product == "CNC":
        # Buying delivery needs the full amount; selling shares you own needs nothing
        return quantity * est_price + charges if action == "BUY" else 0.0

    # MIS: only the part that ADDS exposure needs margin. If you are long 10
    # and sell 15, 10 just close the long and only 5 open a new short.
    current = mis_position_qty(session, user_id, instrument_id)
    signed = quantity if action == "BUY" else -quantity
    if current == 0 or (current > 0) == (signed > 0):
        opening = quantity
    else:
        opening = max(0, quantity - abs(current))
    return opening * est_price * MARGIN_RATE["MIS"] + charges


def sellable_cnc_qty(session: Session, user_id: int, instrument_id: int,
                     exclude: Order | None = None) -> int:
    """Shares held minus shares already promised to open CNC SELL orders."""
    held = session.scalar(
        select(Holding.quantity).where(Holding.user_id == user_id,
                                       Holding.instrument_id == instrument_id)
    ) or 0
    query = select(func.coalesce(func.sum(Order.quantity - Order.filled_quantity), 0)).where(
        Order.user_id == user_id, Order.instrument_id == instrument_id,
        Order.product == "CNC", Order.action == "SELL", Order.status == "open",
    )
    if exclude is not None:
        query = query.where(Order.id != exclude.id)
    return held - session.scalar(query)


def _active_user(session: Session, user_id: int) -> User | None:
    user = session.get(User, user_id)
    return user if user is not None and user.is_active else None


def _users_order(session: Session, user_id: int, orderid: str) -> Order | None:
    """Look up an order, but only if it belongs to this user."""
    return session.scalar(select(Order).where(Order.orderid == orderid, Order.user_id == user_id))


# ---------------------------------------------------------------------------
# OpenAlgo API
# ---------------------------------------------------------------------------
def placeorder(session: Session, user_id: int, symbol: str, exchange: str, action: str,
               quantity: int, pricetype: str = "MARKET", product: str = "MIS",
               price: float = 0.0, trigger_price: float = 0.0, strategy: str | None = None,
               as_of: datetime | None = None) -> dict:
    """Place an order. Returns {"status": "success", "orderid": ...} or an error."""
    action, pricetype, product = action.upper(), pricetype.upper(), product.upper()

    # --- 1. Input checks: problems here are errors and nothing is saved ---
    if _active_user(session, user_id) is None:
        return error("User not found or inactive")
    if product not in SUPPORTED_PRODUCTS:
        return error(f"Invalid product '{product}'. Use MIS or CNC")
    inst: Instrument | None = get_instrument(session, symbol, exchange)
    if inst is None or not inst.is_active:
        return error(f"Unknown or inactive symbol {symbol} on {exchange}")
    problem = check_order_fields(action, pricetype, quantity, price, trigger_price, inst.lot_size)
    if problem:
        return error(problem)
    bar = latest_candle(session, inst.id, as_of)
    if bar is None:
        return error(f"No price data for {symbol} yet")
    now = bar.timestamp  # simulated time = time of the current bar
    if product == "MIS" and now.time() >= SQUARE_OFF_TIME:
        return error("MIS orders are not accepted after square-off time")

    order = Order(
        orderid=new_id(), user_id=user_id, instrument_id=inst.id, strategy=strategy,
        action=action, pricetype=pricetype, product=product, quantity=quantity,
        price=price, trigger_price=trigger_price, status="open", filled_quantity=0,
        average_price=0.0, margin_blocked=0.0, created_at=now, updated_at=now,
    )

    # --- 2. Money checks: problems here save the order as "rejected" ---
    fund = get_fund(session, user_id)
    block = required_block(session, user_id, inst.id, product, action, quantity,
                           estimate_price(pricetype, price, trigger_price, bar.close))
    reason = None
    if product == "CNC" and action == "SELL":
        can_sell = sellable_cnc_qty(session, user_id, inst.id)
        if quantity > can_sell:
            reason = f"Insufficient holdings: you can sell {can_sell} {inst.symbol}"
    elif block > fund.available_cash:
        reason = f"Insufficient funds: need ₹{block:,.2f}, available ₹{fund.available_cash:,.2f}"

    session.add(order)
    if reason:
        order.status = "rejected"
        order.rejection_reason = reason
        session.commit()
        return error(reason, orderid=order.orderid)

    # --- 3. Accepted: block the cash, then try to fill straight away ---
    block_for_order(fund, order, block)
    session.flush()  # gives order.id, needed by the trade row
    try_fill(session, order, as_of)
    session.commit()
    return success(orderid=order.orderid)


def modifyorder(session: Session, user_id: int, orderid: str, quantity: int | None = None,
                price: float | None = None, trigger_price: float | None = None,
                pricetype: str | None = None, as_of: datetime | None = None) -> dict:
    """Change an open order. Fields left as None keep their current value.
    If the new version can't be afforded, the old order stays unchanged."""
    order = _users_order(session, user_id, orderid)
    if order is None:
        return error(f"Order {orderid} not found")
    if order.status != "open":
        return error(f"Order is {order.status}; only open orders can be modified")

    new_qty = order.quantity if quantity is None else quantity
    new_price = order.price if price is None else price
    new_trigger = order.trigger_price if trigger_price is None else trigger_price
    new_type = order.pricetype if pricetype is None else pricetype.upper()

    problem = check_order_fields(order.action, new_type, new_qty, new_price, new_trigger,
                                 order.instrument.lot_size)
    if problem:
        return error(problem)
    if new_qty <= order.filled_quantity:
        return error(f"Quantity must be more than the {order.filled_quantity} already filled")

    bar = latest_candle(session, order.instrument_id, as_of)
    remaining = new_qty - order.filled_quantity
    fund = get_fund(session, user_id)

    if order.product == "CNC" and order.action == "SELL":
        can_sell = sellable_cnc_qty(session, user_id, order.instrument_id, exclude=order)
        if remaining > can_sell:
            return error(f"Insufficient holdings: you can sell {can_sell}")
    new_block = required_block(session, user_id, order.instrument_id, order.product, order.action,
                               remaining, estimate_price(new_type, new_price, new_trigger, bar.close))
    # The order's current block will be given back, so it counts as available
    if new_block > fund.available_cash + order.margin_blocked:
        return error(f"Insufficient funds: need ₹{new_block:,.2f}")

    # Swap the old block for the new one, then apply the changes
    release_from_order(fund, order, order.margin_blocked)
    block_for_order(fund, order, new_block)
    order.quantity, order.price = new_qty, new_price
    order.trigger_price, order.pricetype = new_trigger, new_type
    order.updated_at = bar.timestamp

    try_fill(session, order, as_of)  # the new price may be fillable right away
    session.commit()
    return success(orderid=order.orderid)


def cancelorder(session: Session, user_id: int, orderid: str) -> dict:
    """Cancel the unfilled part of an open order and release its blocked cash."""
    order = _users_order(session, user_id, orderid)
    if order is None:
        return error(f"Order {orderid} not found")
    if order.status != "open":
        return error(f"Order is {order.status}; only open orders can be cancelled")

    release_from_order(get_fund(session, user_id), order, order.margin_blocked)
    order.status = "cancelled"
    session.commit()
    return success(orderid=order.orderid)
