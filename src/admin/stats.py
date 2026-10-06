"""
stats.py - Platform-wide numbers for the Overview and Leaderboard pages.

  active today        traders who placed at least one order on the market date
  traded volume       sum of quantity x price of fills (₹)
  platform equity     sum of every trader's equity
  platform P&L        platform equity - sum of opening balances (after charges)
"""

from datetime import datetime, timedelta

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.analytics.metrics import summary
from src.db.models import DailyPnl, Order, Trade, User
from src.trading.accounts import get_fund
from src.trading.books import portfolio_value


def _traders(session: Session) -> list[User]:
    return session.scalars(select(User).where(User.role == "user").order_by(User.id)).all()


def overview(session: Session, as_of: datetime | None) -> dict:
    traders = _traders(session)
    day_start = as_of.replace(hour=0, minute=0, second=0, microsecond=0) if as_of else None
    day_end = day_start + timedelta(days=1) if day_start else None

    def today(column):
        return [column >= day_start, column < day_end] if day_start else [False]

    equity = opening = 0.0
    for u in traders:
        equity += portfolio_value(session, u.id, as_of)["equity"]
        opening += get_fund(session, u.id).opening_balance
    session.commit()  # get_fund may create missing funds rows

    return {
        "total_users": len(traders),
        "disabled_users": sum(not u.is_active for u in traders),
        "active_today": session.scalar(select(func.count(func.distinct(Order.user_id))).where(*today(Order.created_at))),
        "orders_today": session.scalar(select(func.count()).select_from(Order).where(*today(Order.created_at))),
        "traded_value_today": session.scalar(
            select(func.coalesce(func.sum(Trade.quantity * Trade.price), 0.0)).where(*today(Trade.timestamp))),
        "traded_value_total": session.scalar(select(func.coalesce(func.sum(Trade.quantity * Trade.price), 0.0))),
        "platform_equity": equity,
        "platform_pnl": equity - opening,
    }


def activity_over_time(session: Session) -> pd.DataFrame:
    """Per market day: orders, active traders, traded value, platform equity."""
    orders = pd.DataFrame(session.execute(select(Order.created_at, Order.user_id)).all(),
                          columns=["time", "user_id"])
    trades = pd.DataFrame(session.execute(select(Trade.timestamp, Trade.quantity * Trade.price)).all(),
                          columns=["time", "value"])
    equity = pd.DataFrame(session.execute(
        select(DailyPnl.trade_date, func.sum(DailyPnl.equity)).group_by(DailyPnl.trade_date)).all(),
        columns=["date", "platform_equity"])

    frames = []
    if not orders.empty:
        orders["date"] = pd.to_datetime(orders["time"]).dt.date
        frames.append(orders.groupby("date").agg(orders=("user_id", "size"), active_traders=("user_id", "nunique")))
    if not trades.empty:
        trades["date"] = pd.to_datetime(trades["time"]).dt.date
        frames.append(trades.groupby("date").agg(traded_value=("value", "sum")))
    if not equity.empty:
        frames.append(equity.set_index("date"))
    if not frames:
        return pd.DataFrame(columns=["date", "orders", "active_traders", "traded_value", "platform_equity"])
    out = pd.concat(frames, axis=1).sort_index()
    for col in ("orders", "active_traders", "traded_value"):
        out[col] = out[col].fillna(0) if col in out else 0
    return out.reset_index(names="date")


def leaderboard(session: Session, as_of: datetime | None) -> pd.DataFrame:
    """Every trader with their results; sort it however the page needs."""
    rows = []
    for u in _traders(session):
        k = summary(session, u.id, as_of)
        rows.append({"username": u.username, "active": u.is_active, "equity": k["equity"],
                     "total_pnl": k["total_pnl"], "return": k["total_return"], "sharpe": k["sharpe"],
                     "max_drawdown": k["max_drawdown"], "win_rate": k["win_rate"],
                     "trades": k["trades"], "days": k["days"]})
    session.commit()
    df = pd.DataFrame(rows, columns=["username", "active", "equity", "total_pnl", "return", "sharpe",
                                     "max_drawdown", "win_rate", "trades", "days"])
    return df.sort_values("total_pnl", ascending=False).reset_index(drop=True)

