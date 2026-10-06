"""
books.py - Read-only views, in OpenAlgo's response shapes:
orderbook, tradebook, positionbook, holdings, funds.

Unrealised P&L is never stored. It is worked out here from the latest price:
    long : (LTP - average price) x quantity
    short: same formula; quantity is negative, so a falling price is a profit
"""

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.models import Holding, Order, Position, Trade, User
from src.trading.accounts import get_fund
from src.trading.market import error, get_ltp, success


def _r(x: float) -> float:
    """Round money to paise for display."""
    return round(x, 2)


def orderbook(session: Session, user_id: int) -> dict:
    """All of the user's orders (newest first) plus counts by status."""
    if session.get(User, user_id) is None:
        return error("User not found")
    orders = session.scalars(
        select(Order).where(Order.user_id == user_id).order_by(Order.created_at.desc(), Order.id.desc())
    ).all()
    rows = [
        {
            "orderid": o.orderid,
            "symbol": o.instrument.symbol,
            "exchange": o.instrument.exchange,
            "action": o.action,
            "pricetype": o.pricetype,
            "product": o.product,
            "quantity": o.quantity,
            "filled_quantity": o.filled_quantity,
            "price": o.price,
            "trigger_price": o.trigger_price,
            "average_price": _r(o.average_price),
            "order_status": o.status,
            "rejection_reason": o.rejection_reason,
            "strategy": o.strategy,
            "timestamp": o.created_at,
        }
        for o in orders
    ]
    statistics = {
        "total_buy_orders": sum(o.action == "BUY" for o in orders),
        "total_sell_orders": sum(o.action == "SELL" for o in orders),
        "total_open_orders": sum(o.status == "open" for o in orders),
        "total_completed_orders": sum(o.status == "complete" for o in orders),
        "total_rejected_orders": sum(o.status == "rejected" for o in orders),
        "total_cancelled_orders": sum(o.status == "cancelled" for o in orders),
    }
    return success({"orders": rows, "statistics": statistics})


def tradebook(session: Session, user_id: int) -> dict:
    """Every fill, newest first."""
    if session.get(User, user_id) is None:
        return error("User not found")
    trades = session.scalars(
        select(Trade).where(Trade.user_id == user_id).order_by(Trade.timestamp.desc(), Trade.id.desc())
    ).all()
    return success([
        {
            "tradeid": t.tradeid,
            "orderid": t.order.orderid,
            "symbol": t.instrument.symbol,
            "exchange": t.instrument.exchange,
            "action": t.action,
            "product": t.order.product,
            "quantity": t.quantity,
            "average_price": t.price,
            "trade_value": _r(t.quantity * t.price),
            "fees": t.fees,
            "timestamp": t.timestamp,
        }
        for t in trades
    ])


def _mis_unrealised(session: Session, user_id: int, as_of: datetime | None) -> float:
    """Total unrealised P&L of the user's open MIS positions."""
    total = 0.0
    for p in session.scalars(select(Position).where(Position.user_id == user_id, Position.quantity != 0)):
        total += (get_ltp(session, p.instrument_id, as_of) - p.average_price) * p.quantity
    return total


def positionbook(session: Session, user_id: int, as_of: datetime | None = None) -> dict:
    """Intraday (MIS) positions, including closed ones (quantity 0) with their realised P&L."""
    if session.get(User, user_id) is None:
        return error("User not found")
    rows = []
    for p in session.scalars(select(Position).where(Position.user_id == user_id).order_by(Position.id)):
        ltp = get_ltp(session, p.instrument_id, as_of)
        unrealised = (ltp - p.average_price) * p.quantity
        rows.append({
            "symbol": p.instrument.symbol,
            "exchange": p.instrument.exchange,
            "product": p.product,
            "quantity": p.quantity,
            "average_price": _r(p.average_price),
            "ltp": ltp,
            "realised_pnl": _r(p.realised_pnl),
            "unrealised_pnl": _r(unrealised),
            "pnl": _r(p.realised_pnl + unrealised),
        })
    return success(rows)


def holdings(session: Session, user_id: int, as_of: datetime | None = None) -> dict:
    """Delivery (CNC) shares with their current value and P&L."""
    if session.get(User, user_id) is None:
        return error("User not found")
    rows = []
    total_value = total_cost = 0.0
    for h in session.scalars(select(Holding).where(Holding.user_id == user_id).order_by(Holding.id)):
        ltp = get_ltp(session, h.instrument_id, as_of)
        cost = h.average_price * h.quantity
        value = ltp * h.quantity
        rows.append({
            "symbol": h.instrument.symbol,
            "exchange": h.instrument.exchange,
            "product": "CNC",
            "quantity": h.quantity,
            "average_price": _r(h.average_price),
            "ltp": ltp,
            "pnl": _r(value - cost),
            "pnlpercent": _r((value - cost) / cost * 100),
        })
        total_value += value
        total_cost += cost
    pnl = total_value - total_cost
    statistics = {
        "totalholdingvalue": _r(total_value),
        "totalinvvalue": _r(total_cost),
        "totalprofitandloss": _r(pnl),
        "totalpnlpercentage": _r(pnl / total_cost * 100) if total_cost else 0.0,
    }
    return success({"holdings": rows, "statistics": statistics})


def funds(session: Session, user_id: int, as_of: datetime | None = None) -> dict:
    """Cash summary, using OpenAlgo's field names.

    availablecash  : free cash for new orders
    utiliseddebits : cash blocked by open orders and open MIS positions
    m2mrealized    : profit/loss already booked (before charges)
    m2munrealized  : profit/loss on open MIS positions at the current price
    """
    if session.get(User, user_id) is None:
        return error("User not found")
    fund = get_fund(session, user_id)
    return success({
        "availablecash": _r(fund.available_cash),
        "collateral": 0.0,  # no pledged shares in this simulator
        "m2mrealized": _r(fund.realised_pnl),
        "m2munrealized": _r(_mis_unrealised(session, user_id, as_of)),
        "utiliseddebits": _r(fund.used_margin),
    })


def portfolio_value(session: Session, user_id: int, as_of: datetime | None = None) -> dict:
    """Everything the user owns, valued at the current price.

    equity = free cash + blocked cash + market value of holdings + unrealised MIS P&L
    (Blocked cash still belongs to the user; it is just reserved.)
    """
    fund = get_fund(session, user_id)
    holdings_cost = holdings_value = 0.0
    for h in session.scalars(select(Holding).where(Holding.user_id == user_id)):
        holdings_cost += h.average_price * h.quantity
        holdings_value += get_ltp(session, h.instrument_id, as_of) * h.quantity
    mis_unrealised = _mis_unrealised(session, user_id, as_of)
    return {
        "cash": fund.available_cash,
        "blocked": fund.used_margin,
        "holdings_value": holdings_value,
        "realised_pnl": fund.realised_pnl,
        "unrealised_pnl": (holdings_value - holdings_cost) + mis_unrealised,
        "equity": fund.available_cash + fund.used_margin + holdings_value + mis_unrealised,
    }
