"""
anomaly.py - Spot users whose trading behaviour is unusual compared with
everyone else, for the admin app. Unsupervised: there are no labels saying
"this user is a problem", so the model only finds the odd ones out.

Behaviour features, one row per trader:
  trades_per_day        trades / days the user has been in the market
                        (days = their daily P&L snapshots, at least 1)
  avg_trade_value       average  quantity x price  of their trades, in ₹
  max_trade_share       biggest single trade / starting cash
                        (0.5 = one trade worth half the account)
  pnl_volatility        standard deviation of their daily equity returns
                        (how wildly their account swings day to day)
  worst_day             their worst daily return (e.g. -0.08 = lost 8% in a day)
  rejection_rate        rejected orders / all orders
                        (many rejections = trying to trade beyond their means)

Two checks, because each catches what the other misses:

1. IsolationForest (the ML part): builds many random trees that keep
   splitting the data at random. Unusual points are isolated after only a
   few splits, normal points need many; the average number of splits becomes
   the anomaly score. It is good at users who are unusual in a COMBINATION
   of ways. With only a handful of users, though, its trees are shallow and
   it can miss someone extreme on a single measure.

2. Extreme-value rule: flag anyone more than Z_LIMIT robust z-scores from the
   typical trader on any one measure.
     robust z = (value - median) / (1.4826 x MAD),  MAD = median absolute deviation
   Median/MAD are used instead of mean/std so one extreme user can't hide
   itself by inflating the average (1.4826 makes MAD comparable to a std).

Thresholds were calibrated by simulation (300 groups of 9 similar traders
plus one making a single trade worth 90% of the account):
  normal traders: forest score up to 0.61, max robust z up to 15
  extreme trader: forest score 0.50-0.55 (!), max robust z 31-80+
So with ~10 users the forest alone could NOT tell the extreme trader apart,
while the z rule always could. FOREST_THRESHOLD and Z_LIMIT sit just above
what normal traders reached by chance. The forest gets better with more users.

Output columns:
  anomaly_score  IsolationForest score, higher = more unusual (-score_samples())
  max_abs_z      the user's most extreme robust z-score
  is_anomaly     flagged by either check
  flagged_by     "pattern" (forest), "extreme value" (rule), "both" or ""
  reason         the measure furthest from typical, in plain words
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.analytics.metrics import equity_curve
from src.db.models import DailyPnl, Fund, Order, Trade, User
from src.ml.models import SEED

BEHAVIOUR_FEATURES = ["trades_per_day", "avg_trade_value", "max_trade_share",
                      "pnl_volatility", "worst_day", "rejection_rate"]
MIN_USERS = 5            # below this, "unusual compared with others" means little
FOREST_THRESHOLD = 0.62  # flag by pattern above this score (normal max in simulation: 0.61)
Z_LIMIT = 20.0           # flag by extreme value above this robust z (normal max in simulation: 15)

LABELS = {
    "trades_per_day": "trades per day",
    "avg_trade_value": "average trade size",
    "max_trade_share": "largest trade vs account",
    "pnl_volatility": "daily P&L swings",
    "worst_day": "worst daily loss",
    "rejection_rate": "rejected-order rate",
}


def behaviour_frame(session: Session) -> pd.DataFrame:
    """One row per trader (role "user") who has placed at least one order."""
    rows = []
    users = session.scalars(select(User).where(User.role == "user").order_by(User.id)).all()
    for user in users:
        orders = session.scalar(select(func.count()).select_from(Order).where(Order.user_id == user.id))
        if not orders:
            continue
        rejected = session.scalar(select(func.count()).select_from(Order).where(
            Order.user_id == user.id, Order.status == "rejected"))
        values = [q * p for q, p in session.execute(
            select(Trade.quantity, Trade.price).where(Trade.user_id == user.id))]
        days = session.scalar(select(func.count()).select_from(DailyPnl).where(DailyPnl.user_id == user.id))
        start = session.scalar(select(Fund.opening_balance).where(Fund.user_id == user.id)) or 1.0
        returns = equity_curve(session, user.id)["daily_return"] if days else pd.Series(dtype=float)

        rows.append({
            "user_id": user.id,
            "username": user.username,
            "is_active": user.is_active,
            "trades": len(values),
            "trades_per_day": len(values) / max(days, 1),
            "avg_trade_value": float(np.mean(values)) if values else 0.0,
            "max_trade_share": max(values) / start if values else 0.0,
            "pnl_volatility": float(returns.std()) if len(returns) > 1 else 0.0,
            "worst_day": float(min(returns.min(), 0.0)) if len(returns) else 0.0,
            "rejection_rate": rejected / orders,
        })
    return pd.DataFrame(rows, columns=["user_id", "username", "is_active", "trades", *BEHAVIOUR_FEATURES])


def robust_z(df: pd.DataFrame) -> pd.DataFrame:
    """(value - median) / (1.4826 x MAD) per column. 1.4826 makes MAD comparable
    to a standard deviation. If MAD is 0 (most users identical), fall back to
    the standard deviation; if that is 0 too, the column is all zeros."""
    median = df.median()
    mad = (df - median).abs().median() * 1.4826
    scale = mad.where(mad > 0, df.std(ddof=0)).replace(0, np.nan)
    return ((df - median) / scale).fillna(0.0)


def show(feature: str, value: float) -> str:
    """Readable value: ₹ for money, % for shares, 2 decimals for counts."""
    if feature == "avg_trade_value":
        return f"₹{value:,.0f}"
    if feature == "trades_per_day":
        return f"{value:.2f}"
    return f"{value:.2%}"  # shares and returns


def explain(row: pd.Series, z: pd.Series, medians: pd.Series) -> str:
    """'largest trade vs account is far higher than usual (90.00% vs typical 5.00%)'"""
    feature = z.abs().idxmax()
    direction = "higher" if z[feature] > 0 else "lower"
    if feature == "worst_day":  # more negative = a bigger loss
        direction = "bigger" if z[feature] < 0 else "smaller"
    return (f"{LABELS[feature]} is far {direction} than usual "
            f"({show(feature, row[feature])} vs typical {show(feature, medians[feature])})")


def detect_anomalies(behaviour: pd.DataFrame, seed: int = SEED, min_users: int = MIN_USERS,
                     forest_threshold: float = FOREST_THRESHOLD, z_limit: float = Z_LIMIT) -> pd.DataFrame:
    """Add anomaly_score, max_abs_z, is_anomaly, flagged_by and reason columns,
    flagged users first. With fewer than min_users traders nobody is flagged."""
    out = behaviour.copy()
    if len(out) < min_users:
        out["anomaly_score"] = np.nan
        out["max_abs_z"] = np.nan
        out["is_anomaly"] = False
        out["flagged_by"] = ""
        out["reason"] = f"Need at least {min_users} active traders to compare"
        return out

    X = out[BEHAVIOUR_FEATURES].to_numpy(dtype=float)
    # Put ₹ amounts and rates on the same scale. RobustScaler uses the median
    # and inter-quartile range, so one extreme trader can't stretch the scale
    # and make itself look normal (StandardScaler's mean/std would let it).
    X = RobustScaler().fit_transform(X)
    forest = IsolationForest(n_estimators=300, random_state=seed).fit(X)
    out["anomaly_score"] = -forest.score_samples(X)   # flip the sign: higher = more unusual
    by_forest = (out["anomaly_score"] >= forest_threshold).to_numpy()

    z = robust_z(out[BEHAVIOUR_FEATURES])
    out["max_abs_z"] = z.abs().max(axis=1)
    by_rule = (out["max_abs_z"] >= z_limit).to_numpy()

    out["is_anomaly"] = by_forest | by_rule
    out["flagged_by"] = np.select([by_forest & by_rule, by_forest, by_rule],
                                  ["both", "pattern", "extreme value"], default="")
    medians = out[BEHAVIOUR_FEATURES].median()
    out["reason"] = [explain(out.loc[i], z.loc[i], medians) if out.loc[i, "is_anomaly"]
                     else "Within the normal range" for i in out.index]
    return (out.sort_values(["is_anomaly", "anomaly_score"], ascending=[False, False])
            .reset_index(drop=True))
