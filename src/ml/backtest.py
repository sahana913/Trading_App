"""
backtest.py - Would trading the signal have made money? Compared with simply
buying every stock and holding it (buy-and-hold), after transaction costs.

Rules (long-only, delivery, one position per stock):
  * at day t's close: if P(up) > 0.5 hold the stock tomorrow, else hold cash
  * strategy return for tomorrow = position_t x next_return_t
  * every change of position (cash -> stock or stock -> cash) costs
    COST_PER_SIDE of the traded value; the last open position pays an exit cost
  * buy-and-hold: in the stock every day, paying one entry and one exit cost
  * portfolio: equal weight across stocks = average of their daily returns

  equity_t       = (1 + r_1) x (1 + r_2) x ... x (1 + r_t)   starting at 1.0
  total return   = equity_last - 1
  Sharpe, max drawdown: same formulas as src/analytics/metrics.py (reused)
  exposure       = share of days invested
"""

import numpy as np
import pandas as pd

from src.analytics.metrics import max_drawdown, sharpe_ratio
from src.ml.evaluate import THRESHOLD
from src.trading.charges import calculate_charges


def cost_per_side() -> float:
    """Charges for one buy or sell as a share of the traded value, from the
    same charges table the trading engine uses (CNC, ₹1,00,000 trade)."""
    buy = calculate_charges("CNC", "BUY", 100, 1000.0)["total"]
    sell = calculate_charges("CNC", "SELL", 100, 1000.0)["total"]
    return (buy + sell) / 2 / 100_000  # about 0.11%


def daily_strategy_returns(next_return: pd.Series, position: pd.Series, cost: float) -> pd.Series:
    """Returns of ONE stock, rows in time order.

    position[t] is decided at day t's close and earns next_return[t].
    """
    position = position.astype(float).to_numpy()
    previous = np.concatenate([[0.0], position[:-1]])  # start in cash
    trades = np.abs(position - previous)                # 1 whenever we buy or sell
    trades[-1] += position[-1]                          # close any open position at the end
    return pd.Series(position * next_return.to_numpy() - trades * cost, index=next_return.index)


def summarise(daily: pd.Series) -> dict:
    """Headline numbers for a series of daily portfolio returns."""
    equity = (1 + daily).cumprod()
    return {
        "total_return": float(equity.iloc[-1] - 1),
        "sharpe": sharpe_ratio(daily),
        "max_drawdown": max_drawdown(equity, 1.0),
        "days": int(len(daily)),
    }


def _positions_and_returns(rows: pd.DataFrame, proba_up: np.ndarray, cost: float) -> pd.DataFrame:
    """Per stock and day: position, strategy return and buy-and-hold return."""
    df = rows[["symbol", "timestamp", "next_return"]].copy()
    df["position"] = (np.asarray(proba_up) > THRESHOLD).astype(float)
    df = df.sort_values(["symbol", "timestamp"])
    strat, hold = [], []
    for _, g in df.groupby("symbol"):
        strat.append(daily_strategy_returns(g["next_return"], g["position"], cost))
        hold.append(daily_strategy_returns(g["next_return"], pd.Series(1.0, index=g.index), cost))
    df["strategy"] = pd.concat(strat)
    df["hold"] = pd.concat(hold)
    return df


def equity_curves(rows: pd.DataFrame, proba_up: np.ndarray, cost: float | None = None) -> pd.DataFrame:
    """Daily equity (starting at 1.0) of the strategy and of buy-and-hold.
    Columns: timestamp, strategy, buy_and_hold. Equal weight across stocks."""
    cost = cost_per_side() if cost is None else cost
    by_day = _positions_and_returns(rows, proba_up, cost).groupby("timestamp")[["strategy", "hold"]].mean()
    return pd.DataFrame({"timestamp": by_day.index,
                         "strategy": (1 + by_day["strategy"]).cumprod().to_numpy(),
                         "buy_and_hold": (1 + by_day["hold"]).cumprod().to_numpy()})


def backtest(rows: pd.DataFrame, proba_up: np.ndarray, cost: float | None = None) -> dict:
    """rows: symbol, timestamp, next_return (the test period). Returns
    {"strategy": {...}, "buy_and_hold": {...}, "cost_per_side": ...}."""
    cost = cost_per_side() if cost is None else cost
    df = _positions_and_returns(rows, proba_up, cost)

    # Equal-weight portfolio: average across stocks for each day
    by_day = df.groupby("timestamp")[["strategy", "hold"]].mean()
    # Every buy and every sell, including closing the last position (as costed above)
    trades = int(df.groupby("symbol")["position"].apply(
        lambda p: np.abs(np.diff(np.concatenate([[0.0], p.to_numpy(), [0.0]]))).sum()).sum())
    return {
        "strategy": {**summarise(by_day["strategy"]), "trades": trades,
                     "exposure": float(df["position"].mean())},
        "buy_and_hold": {**summarise(by_day["hold"]), "trades": int(2 * df["symbol"].nunique()),
                         "exposure": 1.0},
        "cost_per_side": cost,
    }
