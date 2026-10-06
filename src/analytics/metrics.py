"""
metrics.py - Portfolio and trade statistics.

Two kinds of functions:
  * pure maths on pandas Series (drawdown, sharpe_ratio, trade_stats...): no
    database, easy to test with hand-made numbers;
  * loaders (equity_curve, pnl_by_symbol, summary) that read the database and
    feed the pure functions.

Every formula used (you should be able to say these in a viva):

  equity_t       = cash + blocked cash + holdings x price_t + open MIS P&L
  day P&L_t      = equity_t - equity_(t-1)          (day 1: equity_1 - starting cash)
  daily return_t = equity_t / equity_(t-1) - 1
  total P&L      = equity_now - starting cash        (after all charges)

  drawdown_t     = equity_t / max(starting cash, equity_1..equity_t) - 1
                   e.g. peak 110, now 99 -> 99/110 - 1 = -10%
  max drawdown   = the smallest (most negative) drawdown_t

  Sharpe ratio   = mean(daily return) / std(daily return) x sqrt(252)
                   (risk-free rate taken as 0; sqrt(252) turns a daily ratio into
                   a yearly one because there are ~252 trading days a year)

  win rate       = winning closing trades / all closing trades
  profit factor  = sum of winning trades' P&L / |sum of losing trades' P&L|
                   (above 1 = the strategy made more than it lost)

  VaR 95%        = -(5th percentile of daily returns)
                   "On 95% of days the loss was no bigger than this."
                   Historical method: no bell-curve assumption, just the
                   actual past days. 5th percentile uses numpy's default
                   linear interpolation between the two nearest values.
  CVaR 95%       = -(average of the daily returns at or below that 5th percentile)
                   "On the worst 5% of days, the average loss was this."
                   Always >= VaR, because it averages the tail beyond it.
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
RISK_CONFIDENCE = 0.95  # VaR / CVaR level
MIN_DAYS_FOR_VAR = 20   # with fewer days the "worst 5%" is less than one day


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


def historical_var(returns: pd.Series, confidence: float = RISK_CONFIDENCE) -> float | None:
    """Value at Risk as a positive fraction (0.021 = 2.1% of equity), or None
    if there are fewer than MIN_DAYS_FOR_VAR days."""
    returns = returns.dropna()
    if len(returns) < MIN_DAYS_FOR_VAR:
        return None
    cutoff = np.percentile(returns, (1 - confidence) * 100)  # 5th percentile
    return float(-cutoff)  # a loss, reported as a positive number


def historical_cvar(returns: pd.Series, confidence: float = RISK_CONFIDENCE) -> float | None:
    """Conditional VaR (expected shortfall): average loss on the worst days."""
    returns = returns.dropna()
    if len(returns) < MIN_DAYS_FOR_VAR:
        return None
    cutoff = np.percentile(returns, (1 - confidence) * 100)
    tail = returns[returns <= cutoff]  # the worst ~5% of days
    return float(-tail.mean())


def calendar_frame(dates: pd.Series, values: pd.Series) -> pd.DataFrame:
    """Reshape daily values into a calendar grid for the heatmap.

    Rows = weekday (Mon..Fri), columns = the Monday of each week, cells = the
    day's value (NaN where there was no trading, e.g. a holiday).
    """
    df = pd.DataFrame({"date": pd.to_datetime(dates), "value": values.to_numpy()})
    df["weekday"] = df["date"].dt.dayofweek                          # 0 = Monday
    df["week"] = df["date"] - pd.to_timedelta(df["weekday"], unit="D")  # that week's Monday
    grid = df.pivot_table(index="weekday", columns="week", values="value", aggfunc="sum")
    return grid.reindex(range(5))  # always Mon-Fri, even if a weekday never traded


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
    returns = curve["daily_return"] if not curve.empty else pd.Series(dtype=float)
    var, cvar = historical_var(returns), historical_cvar(returns)
    return {
        "equity": live["equity"],
        "total_pnl": live["equity"] - start,  # after charges
        "total_return": live["equity"] / start - 1,
        "day_pnl": float(curve["day_pnl"].iloc[-1]) if not curve.empty else 0.0,
        # VaR/CVaR as a share of equity, and in rupees at today's equity
        "var_95": var,
        "cvar_95": cvar,
        "var_95_amount": var * live["equity"] if var is not None else None,
        "cvar_95_amount": cvar * live["equity"] if cvar is not None else None,
        "realised_pnl": live["realised_pnl"],
        "unrealised_pnl": live["unrealised_pnl"],
        "sharpe": sharpe_ratio(curve["daily_return"]) if not curve.empty else None,
        "max_drawdown": max_drawdown(curve["equity"], start) if not curve.empty else 0.0,
        "days": len(curve),
        **stats,
    }
