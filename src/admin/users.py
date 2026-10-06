"""users.py - Look after trader accounts."""

from datetime import datetime

import pandas as pd
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from src.admin.audit import log_action, require_admin
from src.config import STARTING_CASH
from src.db.models import DailyPnl, Holding, Order, Position, Trade, User
from src.trading.accounts import get_fund
from src.trading.books import portfolio_value
from src.trading.market import error, success

MAX_TOP_UP = 10_000_000.0  # ₹1 crore per top-up, to catch typing mistakes


def search_users(session: Session, query: str = "") -> pd.DataFrame:
    """Users whose name contains `query` (any case), with their cash and status."""
    stmt = select(User).order_by(User.username)
    if query.strip():
        stmt = stmt.where(func.lower(User.username).contains(query.strip().lower()))
    rows = []
    for u in session.scalars(stmt):
        fund = get_fund(session, u.id) if u.role == "user" else None
        rows.append({"id": u.id, "username": u.username, "role": u.role, "active": u.is_active,
                     "created": u.created_at, "cash": fund.available_cash if fund else None,
                     "orders": session.scalar(select(func.count()).select_from(Order).where(Order.user_id == u.id))})
    session.commit()  # get_fund may have created a funds row for an older account
    return pd.DataFrame(rows, columns=["id", "username", "role", "active", "created", "cash", "orders"])


def set_active(session: Session, admin_id: int, user_id: int, active: bool) -> dict:
    """Enable or disable an account. Disabled users can't log in (and are
    logged out on their next click)."""
    admin = require_admin(session, admin_id)
    user = session.get(User, user_id)
    if user is None:
        return error("User not found")
    if user.id == admin.id and not active:
        return error("You can't disable your own account")
    user.is_active = active
    log_action(session, admin_id, "enable_user" if active else "disable_user", user.id,
               {"username": user.username})
    session.commit()
    return success()


def top_up(session: Session, admin_id: int, user_id: int, amount: float) -> dict:
    """Add virtual cash. It counts as new CAPITAL, not profit: both available
    cash and the opening balance go up, so total P&L (equity - opening
    balance) is unchanged. (Daily P&L charts will show that day as a jump.)"""
    require_admin(session, admin_id)
    user = session.get(User, user_id)
    if user is None or user.role != "user":
        return error("Only trader accounts have balances")
    if not 0 < amount <= MAX_TOP_UP:
        return error(f"Amount must be between ₹1 and ₹{MAX_TOP_UP:,.0f}")
    fund = get_fund(session, user.id)
    fund.available_cash += amount
    fund.opening_balance += amount
    log_action(session, admin_id, "top_up", user.id, {"amount": amount})
    session.commit()
    return success({"available_cash": fund.available_cash})


def reset_account(session: Session, admin_id: int, user_id: int) -> dict:
    """Wipe ONE trader's orders, trades, positions, holdings and P&L history,
    and give them the starting cash again. Other users are untouched."""
    require_admin(session, admin_id)
    user = session.get(User, user_id)
    if user is None or user.role != "user":
        return error("Only trader accounts can be reset")
    counts = {}
    for table in (Trade, Order, Position, Holding, DailyPnl):  # children before parents
        counts[table.__tablename__] = session.execute(delete(table).where(table.user_id == user.id)).rowcount
    fund = get_fund(session, user.id)
    fund.opening_balance = fund.available_cash = STARTING_CASH
    fund.used_margin = fund.realised_pnl = 0.0
    log_action(session, admin_id, "reset_account", user.id, {"deleted": counts})
    session.commit()
    return success(counts)


def user_snapshot(session: Session, user_id: int, as_of: datetime | None) -> dict:
    """Headline numbers for one trader (used on the Users page)."""
    fund = get_fund(session, user_id)
    value = portfolio_value(session, user_id, as_of)
    return {"equity": value["equity"], "cash": fund.available_cash, "blocked": fund.used_margin,
            "opening_balance": fund.opening_balance, "total_pnl": value["equity"] - fund.opening_balance,
            "realised_pnl": fund.realised_pnl, "unrealised_pnl": value["unrealised_pnl"]}
