"""Tests for src/analytics: the maths with hand-made numbers, then the loaders
on a small simulated market (daily bars, 1-8 Jan 2024)."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select

from src.analytics import charts
from src.analytics.metrics import (
    daily_returns, drawdown, equity_curve, max_drawdown, pnl_by_symbol, sharpe_ratio,
    summary, trade_stats,
)
from src.config import STARTING_CASH
from src.db.models import Candle, Instrument, Trade
from src.db.session import get_engine, get_session_factory, init_db
from src.trading import create_user, placeorder
from src.trading.simulator import get_clock, run, start

# ---------------------------------------------------------------------------
# Pure maths
# ---------------------------------------------------------------------------


def test_daily_returns_start_from_starting_cash():
    r = daily_returns(pd.Series([1010.0, 1000.0, 1050.0]), start_value=1000)
    assert r.tolist() == pytest.approx([0.01, 1000 / 1010 - 1, 0.05])


def test_drawdown_measures_distance_below_peak():
    assert drawdown(pd.Series([1100.0, 990.0, 1210.0]), 1000).tolist() == pytest.approx([0, -0.1, 0])
    # Losing from day one: the starting cash is the first peak
    assert drawdown(pd.Series([900.0, 950.0]), 1000).tolist() == pytest.approx([-0.1, -0.05])


def test_max_drawdown():
    assert max_drawdown(pd.Series([1100.0, 990.0, 1210.0, 968.0]), 1000) == pytest.approx(-0.2)
    assert max_drawdown(pd.Series([], dtype=float), 1000) == 0.0


def test_sharpe_ratio():
    r = pd.Series([0.01, -0.01, 0.02, 0.0])
    expected = r.mean() / r.std() * np.sqrt(252)
    assert sharpe_ratio(r) == pytest.approx(expected)
    assert sharpe_ratio(pd.Series([0.01])) is None             # too few days
    assert sharpe_ratio(pd.Series([0.01, 0.01, 0.01])) is None  # no variation


def test_trade_stats():
    # None = fill that closed nothing; 0 = break-even close (still a closed trade)
    realised = pd.Series([None, 100, -50, 0, None, 30], dtype=float)
    fees = pd.Series([1.0] * 6)
    s = trade_stats(realised, fees)
    assert (s["trades"], s["closed_trades"], s["wins"], s["losses"]) == (6, 4, 2, 1)
    assert s["win_rate"] == 0.5
    assert s["avg_win"] == 65 and s["avg_loss"] == -50
    assert s["profit_factor"] == pytest.approx(130 / 50)
    assert s["total_charges"] == 6


def test_trade_stats_with_no_trades():
    s = trade_stats(pd.Series([], dtype=float), pd.Series([], dtype=float))
    assert s["win_rate"] is None and s["profit_factor"] is None and s["trades"] == 0


# ---------------------------------------------------------------------------
# A small simulated market
# ---------------------------------------------------------------------------
TCS = {1: 100, 2: 98, 3: 94, 4: 97, 5: 103, 8: 110}
INFY = {1: 50, 2: 51, 3: 52, 4: 53, 5: 54, 8: 55}


def day(d: int) -> datetime:
    return datetime(2024, 1, d)


@pytest.fixture
def session():
    engine = get_engine("sqlite:///:memory:")
    init_db(engine)
    with get_session_factory(engine)() as s:
        for symbol, closes in (("TCS", TCS), ("INFY", INFY)):
            inst = Instrument(symbol=symbol, exchange="NSE", name=symbol, lot_size=1,
                              tick_size=0.05, is_active=True)
            s.add(inst)
            s.flush()
            for d, c in closes.items():
                s.add(Candle(instrument_id=inst.id, timestamp=day(d), open=c, high=c,
                             low=c, close=c, volume=1_000_000))
        s.commit()
        yield s


@pytest.fixture
def user(session):
    return create_user(session, "alice", "password1")


def order(session, user, **kwargs):
    args = dict(symbol="TCS", exchange="NSE", action="BUY", quantity=10, pricetype="MARKET",
                product="CNC", as_of=get_clock(session))
    args.update(kwargs)
    result = placeorder(session, user.id, **args)
    assert result["status"] == "success", result
    return result


def test_trade_records_its_realised_pnl(session, user):
    start(session, day(1))
    order(session, user)                                    # buy 10 @ 100
    run(session, 4)                                         # to the 5th: 103
    order(session, user, action="SELL", quantity=4)         # +12
    order(session, user, action="SELL", quantity=6)         # +18
    pnl = session.scalars(select(Trade.realised_pnl).order_by(Trade.id)).all()
    assert pnl == [None, pytest.approx(12), pytest.approx(18)]


def test_break_even_close_is_zero_not_none(session, user):
    start(session, day(1))
    order(session, user)
    order(session, user, action="SELL")  # same day, same price
    assert session.scalars(select(Trade.realised_pnl).order_by(Trade.id)).all() == [None, 0]


def test_equity_curve_has_snapshots_plus_live_today(session, user):
    start(session, day(1))
    order(session, user)  # 10 TCS @ 100
    run(session, 3)       # snapshots for the 1st-3rd; clock on the 4th

    curve = equity_curve(session, user.id, as_of=get_clock(session))
    assert len(curve) == 4
    assert curve["date"].iloc[-1] == day(4).date()
    # The daily P&L bars add up to the total change since the start
    assert curve["day_pnl"].sum() == pytest.approx(curve["equity"].iloc[-1] - STARTING_CASH)
    assert (curve["drawdown"] <= 0).all()
    # Worst point: the 3rd, when TCS was at 94 (6 below cost on 10 shares, plus charges)
    assert curve["drawdown"].idxmin() == 2


def test_equity_curve_empty_for_new_user(session, user):
    assert equity_curve(session, user.id).empty


def test_pnl_by_symbol_combines_realised_and_unrealised(session, user):
    start(session, day(1))
    order(session, user)                          # TCS @ 100
    order(session, user, symbol="INFY", quantity=20)  # INFY @ 50
    run(session, 4)                               # 5th: TCS 103, INFY 54
    order(session, user, action="SELL")           # TCS realised +30

    df = pnl_by_symbol(session, user.id, as_of=get_clock(session)).set_index("symbol")
    assert df.loc["TCS", "realised_pnl"] == pytest.approx(30)
    assert df.loc["TCS", "unrealised_pnl"] == 0
    assert df.loc["INFY", "unrealised_pnl"] == pytest.approx(80)   # (54 - 50) x 20
    assert df.index[0] == "INFY"  # biggest total first


def test_summary(session, user):
    start(session, day(1))
    order(session, user)
    run(session, 4)
    order(session, user, action="SELL")  # +30 before charges

    s = summary(session, user.id, as_of=get_clock(session))
    fees = sum(session.scalars(select(Trade.fees)).all())
    assert s["realised_pnl"] == pytest.approx(30)
    assert s["total_pnl"] == pytest.approx(30 - fees)
    assert s["total_return"] == pytest.approx((30 - fees) / STARTING_CASH)
    assert (s["closed_trades"], s["win_rate"]) == (1, 1.0)
    assert s["days"] == 5
    assert s["max_drawdown"] < 0  # TCS dipped to 94 on the way
    assert s["sharpe"] is not None


def test_charts_build_without_errors(session, user):
    start(session, day(1))
    order(session, user)
    run(session, 3)
    curve = equity_curve(session, user.id, as_of=get_clock(session))
    bars = pd.DataFrame({"timestamp": [day(1), day(2)], "open": [1, 2], "high": [2, 3],
                         "low": [0.5, 1.5], "close": [1.5, 2.5]})
    figures = [
        charts.equity_chart(curve, STARTING_CASH),
        charts.drawdown_chart(curve),
        charts.daily_pnl_chart(curve),
        charts.pnl_by_symbol_chart(pnl_by_symbol(session, user.id, as_of=get_clock(session))),
        charts.candle_chart(bars, "TCS"),
    ]
    assert all(len(f.data) == 1 for f in figures)
    # Loss days are drawn red, gain days blue
    colours = figures[2].data[0].marker.color
    assert set(colours) <= {charts.GAIN, charts.LOSS}
