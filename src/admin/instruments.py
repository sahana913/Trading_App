"""instruments.py - Add, edit and remove tradable instruments."""

import re

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.admin.audit import log_action, require_admin
from src.db.models import Candle, Holding, Instrument, Order, Position
from src.trading.market import error, success
from src.trading.simulator import get_clock

SYMBOL_PATTERN = re.compile(r"^[A-Z0-9&\-]{1,20}$")  # e.g. RELIANCE, M&M, BAJAJ-AUTO
EDITABLE = ("name", "lot_size", "tick_size", "is_active")


def instruments_frame(session: Session) -> pd.DataFrame:
    """All instruments with how much data and trading each has."""
    rows = []
    for i in session.scalars(select(Instrument).order_by(Instrument.symbol)):
        count = lambda model: session.scalar(  # noqa: E731
            select(func.count()).select_from(model).where(model.instrument_id == i.id))
        rows.append({"id": i.id, "symbol": i.symbol, "exchange": i.exchange, "name": i.name,
                     "lot_size": i.lot_size, "tick_size": i.tick_size, "active": i.is_active,
                     "candles": count(Candle), "orders": count(Order)})
    return pd.DataFrame(rows, columns=["id", "symbol", "exchange", "name", "lot_size", "tick_size",
                                       "active", "candles", "orders"])


def _check(lot_size: int, tick_size: float) -> str | None:
    if not isinstance(lot_size, int) or lot_size < 1:
        return "Lot size must be a whole number of at least 1"
    if not tick_size > 0:
        return "Tick size must be greater than 0"
    return None


def add_instrument(session: Session, admin_id: int, symbol: str, exchange: str = "NSE", name: str = "",
                   lot_size: int = 1, tick_size: float = 0.05, start_price: float | None = None) -> dict:
    """Add an instrument. With start_price, a first (synthetic) candle is
    created at the market date so it can be traded straight away; without
    one it has no prices until data is loaded for it."""
    require_admin(session, admin_id)
    symbol, exchange = symbol.strip().upper(), exchange.strip().upper()
    if not SYMBOL_PATTERN.match(symbol):
        return error("Symbol must be 1-20 capital letters, digits, & or -")
    problem = _check(lot_size, tick_size)
    if problem:
        return error(problem)
    if session.scalar(select(Instrument).where(Instrument.symbol == symbol, Instrument.exchange == exchange)):
        return error(f"{symbol} already exists on {exchange}")
    clock = get_clock(session)
    if start_price is not None and (start_price <= 0 or clock is None):
        return error("A start price must be > 0 and needs a started market (it is placed at the market date)")

    inst = Instrument(symbol=symbol, exchange=exchange, name=name.strip() or symbol,
                      lot_size=lot_size, tick_size=tick_size, is_active=True)
    session.add(inst)
    session.flush()
    if start_price is not None:
        session.add(Candle(instrument_id=inst.id, timestamp=clock, open=start_price, high=start_price,
                           low=start_price, close=start_price, volume=100_000, is_synthetic=True))
    log_action(session, admin_id, "instrument_add", details={"symbol": symbol, "exchange": exchange,
                                                             "start_price": start_price})
    session.commit()
    return success({"id": inst.id})


def update_instrument(session: Session, admin_id: int, instrument_id: int, **changes) -> dict:
    """Edit name / lot_size / tick_size / is_active. Logs old -> new."""
    require_admin(session, admin_id)
    inst = session.get(Instrument, instrument_id)
    if inst is None:
        return error("Instrument not found")
    unknown = set(changes) - set(EDITABLE)
    if unknown:
        return error(f"Can't edit {sorted(unknown)}")
    problem = _check(changes.get("lot_size", inst.lot_size), changes.get("tick_size", inst.tick_size))
    if problem:
        return error(problem)
    diff = {k: {"from": getattr(inst, k), "to": v} for k, v in changes.items() if getattr(inst, k) != v}
    for key, value in changes.items():
        setattr(inst, key, value)
    if diff:
        log_action(session, admin_id, "instrument_edit", details={"symbol": inst.symbol, **diff})
    session.commit()
    return success()


def remove_instrument(session: Session, admin_id: int, instrument_id: int) -> dict:
    """Delete an unused instrument. One with prices, orders or holdings is
    only DEACTIVATED (no new orders), so the history stays intact."""
    require_admin(session, admin_id)
    inst = session.get(Instrument, instrument_id)
    if inst is None:
        return error("Instrument not found")
    used = any(session.scalar(select(func.count()).select_from(m).where(m.instrument_id == inst.id))
               for m in (Candle, Order, Holding, Position))
    if used:
        inst.is_active = False
        log_action(session, admin_id, "instrument_deactivate", details={"symbol": inst.symbol})
        outcome = "deactivated"
    else:
        session.delete(inst)
        log_action(session, admin_id, "instrument_delete", details={"symbol": inst.symbol})
        outcome = "deleted"
    session.commit()
    return success({"outcome": outcome})
