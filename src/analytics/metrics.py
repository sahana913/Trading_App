"""
metrics.py - Portfolio and trade statistics.

Two kinds of functions:
  * pure maths on pandas Series (drawdown, sharpe_ratio, trade_stats...): no
    database, easy to test with hand-made numbers;
  * loaders (equity_curve, pnl_by_symbol, summary) that read the database and
    feed the pure functions.

Definitions used everywhere (you should be able to say these in a viva):
  equity        cash + blocked cash + holdings at market price + open MIS P&L
  daily return  today's equity / yesterday's equity - 1
  drawdown      how far equity is below its highest point so far (e.g. -5%)
  Sharpe ratio  average daily return / std of daily returns x sqrt(252),
                with a 0% risk-free rate. Above 1 is good, above 2 very good.
  win rate      share of CLOSING trades that made money (before charges)
"""

from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.db.models import DailyPnl, Holding, Instrument, Position, Trade
from src.trading.accounts import get_fund
from src.trading.books import portfolio_value
from src.trading.market import get_ltp

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Pure maths
# ---------------------------------------------------------------------------
def daily_returns(equity: pd.Series, start_value: float) -> pd.Series:
    """Return of each day. Day 1 is measured against the starting cash."""
    previous = equity.shift(1)
    previous.iloc[0] = start_value
    return equity / previous - 1


def drawdown(equity: pd.Series, start_value: float) -> pd.Series:
    """Fraction below the running peak (0 = at a new high, -0.1 = 10% below).
    The starting cash counts as the first peak."""
    peak = equity.cummax().clip(lower=start_value)
    return equity / peak - 1


def max_drawdown(equity: pd.Series, start_value: float) -> float:
    """The worst drawdown ever seen (a negative number, or 0.0)."""
    if equity.empty:
        return 0.0
    return float(drawdown(equity, start_value).min())


def sharpe_ratio(returns: pd.Series) -> float | None:
    """Annualised Sharpe ratio, or None if it can't be measured
    (fewer than 2 days, or returns that never change)."""
    returns = returns.dropna()
    if len(returns) < 2:
        return None
    std = returns.std()
    if std == 0 or np.isnan(std):
        return None
    return float(returns.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR))


def trade_stats(realised: pd.Series, fees: pd.Series) -> dict:
    """Statistics over the trades.

    realised : realised P&L per trade; NaN/None for trades that closed nothing
    fees     : charges per trade (all trades pay charges)
    """
    closed = realised.dropna()
    wins, losses = closed[closed > 0], closed[closed < 0]
    gross_loss = -losses.sum()
    return {
        "trades": int(len(realised)),
        "closed_trades": int(len(closed)),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "win_rate": float(len(wins) / len(closed)) if len(closed) else None,
        "avg_win": float(wins.mean()) if len(wins) else None,
        "avg_loss": float(losses.mean()) if len(losses) else None,
        # money won per rupee lost; None when nothing was lost yet
        "profit_factor": float(wins.sum() / gross_loss) if gross_loss > 0 else None,
        "total_charges": float(fees.sum()),
    }


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def equity_curve(session: Session, user_id: int, as_of: datetime | None = None) -> pd.DataFrame:
    """One row per day: date, equity, realised_pnl, unrealised_pnl, day_pnl,
    daily_return, drawdown.

    Days come from the daily_pnl snapshots the simulator saves at each day's
    close. If as_of is given, a live row for that day is added (today has no
    snapshot yet because the day isn't over).
    """
    snaps = session.scalars(
        select(DailyPnl).where(DailyPnl.user_id == user_id).order_by(DailyPnl.trade_date)
    ).all()
    rows = [{"date": s.trade_date, "equity": s.equity, "realised_pnl": s.realised_pnl,
             "unrealised_pnl": s.unrealised_pnl} for s in snaps]

    if as_of is not None:
        live = portfolio_value(session, user_id, as_of)
        rows = [r for r in rows if r["date"] != as_of.date()]  # live value replaces a same-day snapshot
        rows.append({"date": as_of.date(), "equity": live["equity"],
                     "realised_pnl": live["realised_pnl"], "unrealised_pnl": live["unrealised_pnl"]})

    df = pd.DataFrame(rows, columns=["date", "equity", "realised_pnl", "unrealised_pnl"])
    if df.empty:
        return df.assign(day_pnl=[], daily_return=[], drawdown=[])

    start = get_fund(session, user_id).opening_balance
    df["day_pnl"] = df["equity"].diff()
    df.loc[0, "day_pnl"] = df.loc[0, "equity"] - start
    df["daily_return"] = daily_returns(df["equity"], start)
    df["drawdown"] = drawdown(df["equity"], start)
    return df


def trades_frame(session: Session, user_id: int) -> pd.DataFrame:
    """All of the user's fills as a table, oldest first."""
    rows = session.execute(
        select(Trade.timestamp, Instrument.symbol, Trade.action, Trade.quantity,
               Trade.price, Trade.fees, Trade.realised_pnl)
        .join(Instrument, Trade.instrument_id == Instrument.id)
        .where(Trade.user_id == user_id)
        .order_by(Trade.timestamp, Trade.id)
    ).all()
    return pd.DataFrame(rows, columns=["timestamp", "symbol", "action", "quantity",
                                       "price", "fees", "realised_pnl"])


def pnl_by_symbol(session: Session, user_id: int, as_of: datetime | None = None) -> pd.DataFrame:
    """Realised + unrealised P&L per stock, biggest absolute total first."""
    realised = dict(session.execute(
        select(Instrument.symbol, func.sum(Trade.realised_pnl))
        .join(Instrument, Trade.instrument_id == Instrument.id)
        .where(Trade.user_id == user_id, Trade.realised_pnl.is_not(None))
        .group_by(Instrument.symbol)
    ).all())

    unrealised: dict[str, float] = {}
    open_lots = [*session.scalars(select(Holding).where(Holding.user_id == user_id)),
                 *session.scalars(select(Position).where(Position.user_id == user_id, Position.quantity != 0))]
    for lot in open_lots:
        gain = (get_ltp(session, lot.instrument_id, as_of) - lot.average_price) * lot.quantity
        unrealised[lot.instrument.symbol] = unrealised.get(lot.instrument.symbol, 0.0) + gain

    symbols = sorted(set(realised) | set(unrealised))
    df = pd.DataFrame({
        "symbol": symbols,
        "realised_pnl": [realised.get(s, 0.0) for s in symbols],
        "unrealised_pnl": [unrealised.get(s, 0.0) for s in symbols],
    })
    df["total_pnl"] = df["realised_pnl"] + df["unrealised_pnl"]
    return df.reindex(df["total_pnl"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def summary(session: Session, user_id: int, as_of: datetime | None = None) -> dict:
    """All headline numbers for the dashboard in one dict."""
    start = get_fund(session, user_id).opening_balance
    curve = equity_curve(session, user_id, as_of)
    trades = trades_frame(session, user_id)
    live = portfolio_value(session, user_id, as_of)

    stats = trade_stats(trades["realised_pnl"].astype(float), trades["fees"])
    return {
        "equity": live["equity"],
        "total_pnl": live["equity"] - start,  # after charges
        "total_return": live["equity"] / start - 1,
        "realised_pnl": live["realised_pnl"],
        "unrealised_pnl": live["unrealised_pnl"],
        "sharpe": sharpe_ratio(curve["daily_return"]) if not curve.empty else None,
        "max_drawdown": max_drawdown(curve["equity"], start) if not curve.empty else 0.0,
        "days": len(curve),
        **stats,
    }
