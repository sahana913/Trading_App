"""
accounts.py - Users, their cash account, and moving cash in and out of "blocked".

Cash model (all amounts in ₹):
    available_cash : free to use for new orders
    used_margin    : blocked, either by open orders or by open MIS positions
Blocking moves money from available_cash to used_margin; releasing moves it
back. Neither changes the total, so blocking never "spends" anything.
"""

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.config import STARTING_CASH
from src.db.models import Fund, Order, User
from src.security import hash_password


def create_user(session: Session, username: str, password: str, role: str = "user") -> User:
    """Create a user together with a funds row holding the starting cash."""
    user = User(username=username, password_hash=hash_password(password), role=role, is_active=True)
    session.add(user)
    session.flush()  # gives user.id
    session.add(new_fund(user.id))
    session.commit()
    return user


def new_fund(user_id: int) -> Fund:
    """A fresh account with ₹10,00,000 and nothing blocked."""
    return Fund(
        user_id=user_id,
        opening_balance=STARTING_CASH,
        available_cash=STARTING_CASH,
        used_margin=0.0,
        realised_pnl=0.0,
    )


def get_fund(session: Session, user_id: int) -> Fund:
    """The user's funds row. Users created another way (e.g. the seeded
    admin) get one with the starting cash the first time they need it."""
    fund = session.scalar(select(Fund).where(Fund.user_id == user_id))
    if fund is None:
        fund = new_fund(user_id)
        session.add(fund)
        session.flush()
    return fund


def block_for_order(fund: Fund, order: Order, amount: float) -> None:
    """Block `amount` of free cash for an order that is waiting to fill."""
    fund.available_cash -= amount
    fund.used_margin += amount
    order.margin_blocked += amount


def release_from_order(fund: Fund, order: Order, amount: float) -> None:
    """Give back `amount` of an order's blocked cash (on fill, modify or cancel)."""
    fund.available_cash += amount
    fund.used_margin -= amount
    order.margin_blocked -= amount
