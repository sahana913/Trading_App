"""Tests for the admin service layer (src/admin) and the simulator's market
controls: permission checks, audit rows, and hand-checked numbers."""

from datetime import datetime, timedelta

import pytest
from sqlalchemy import func, select

from src.admin import audit, instruments, market, mlops, stats, users
from src.config import STARTING_CASH
from src.db.models import AdminLog, Candle, Fund, Instrument, ModelRegistry, Order, Trade, User
from src.db.session import get_engine, get_session_factory, init_db
from src.trading import create_user, placeorder
from src.trading.simulator import get_clock, settings, start, status, step, tick

DAYS = [datetime(2024, 1, d) for d in (1, 2, 3, 4, 5, 8)]  # Mon-Fri + next Mon
CLOSES = [100, 102, 101, 104, 103, 105]


@pytest.fixture
def session():
    engine = get_engine("sqlite:///:memory:")
    init_db(engine)
    with get_session_factory(engine)() as s:
        inst = Instrument(symbol="TCS", exchange="NSE", name="TCS", lot_size=1, tick_size=0.05, is_active=True)
        s.add(inst)
        s.flush()
        for day, c in zip(DAYS, CLOSES):
            s.add(Candle(instrument_id=inst.id, timestamp=day, open=c, high=c, low=c, close=c, volume=1_000_000))
        s.commit()
        yield s


@pytest.fixture
def admin(session):
    return create_user(session, "boss", "password1", role="admin")


@pytest.fixture
def trader(session):
    return create_user(session, "alice", "password1")


def actions(session) -> list[str]:
    return list(session.scalars(select(AdminLog.action).order_by(AdminLog.id)))


def buy(session, user, qty=10):
    return placeorder(session, user.id, "TCS", "NSE", "BUY", qty, product="CNC", as_of=get_clock(session))


# ---------------------------------------------------------------------------
# Non-admins are blocked at the function level
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("call", [
    lambda s, uid, tid: users.set_active(s, uid, tid, False),
    lambda s, uid, tid: users.top_up(s, uid, tid, 1000),
    lambda s, uid, tid: users.reset_account(s, uid, tid),
    lambda s, uid, tid: market.start_market(s, uid, DAYS[0]),
    lambda s, uid, tid: market.set_running(s, uid, True),
    lambda s, uid, tid: market.change_settings(s, uid, speed_seconds=2),
    lambda s, uid, tid: market.step_once(s, uid),
    lambda s, uid, tid: market.reset_market(s, uid),
    lambda s, uid, tid: instruments.add_instrument(s, uid, "NEW"),
    lambda s, uid, tid: instruments.update_instrument(s, uid, 1, lot_size=5),
    lambda s, uid, tid: instruments.remove_instrument(s, uid, 1),
    lambda s, uid, tid: mlops.retrain(s, uid),
    lambda s, uid, tid: mlops.activate_version(s, uid, 1),
])
def test_traders_cannot_call_admin_actions(session, trader, call):
    with pytest.raises(PermissionError):
        call(session, trader.id, trader.id)
    assert actions(session) == []  # nothing logged, nothing done


def test_disabled_admin_is_blocked_too(session, admin, trader):
    admin.is_active = False
    session.commit()
    with pytest.raises(PermissionError):
        users.top_up(session, admin.id, trader.id, 1000)


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------
def test_search_users(session, admin, trader):
    create_user(session, "alicia", "password1")
    assert list(users.search_users(session, "ALI")["username"]) == ["alice", "alicia"]
    assert len(users.search_users(session)) == 3


def test_enable_disable_with_audit(session, admin, trader):
    assert users.set_active(session, admin.id, trader.id, False)["status"] == "success"
    assert session.get(User, trader.id).is_active is False
    users.set_active(session, admin.id, trader.id, True)
    assert actions(session) == ["disable_user", "enable_user"]
    log = session.scalar(select(AdminLog).order_by(AdminLog.id))
    assert log.target_user_id == trader.id and log.admin_user_id == admin.id


def test_admin_cannot_disable_themselves(session, admin):
    assert "own account" in users.set_active(session, admin.id, admin.id, False)["message"]


def test_top_up_is_capital_not_profit(session, admin, trader):
    users.top_up(session, admin.id, trader.id, 50_000)
    fund = session.scalar(select(Fund).where(Fund.user_id == trader.id))
    assert fund.available_cash == STARTING_CASH + 50_000
    assert fund.opening_balance == STARTING_CASH + 50_000  # so total P&L stays 0
    assert users.user_snapshot(session, trader.id, None)["total_pnl"] == 0
    assert "between" in users.top_up(session, admin.id, trader.id, -5)["message"]
    assert actions(session) == ["top_up"]


def test_reset_account_only_touches_that_user(session, admin, trader):
    bob = create_user(session, "bob", "password1")
    start(session, DAYS[0])
    buy(session, trader)
    buy(session, bob)
    users.reset_account(session, admin.id, trader.id)

    owners = set(session.scalars(select(Order.user_id)))
    assert owners == {bob.id}
    fund = session.scalar(select(Fund).where(Fund.user_id == trader.id))
    assert (fund.available_cash, fund.realised_pnl) == (STARTING_CASH, 0)
    assert actions(session) == ["reset_account"]


# ---------------------------------------------------------------------------
# Overview and leaderboard
# ---------------------------------------------------------------------------
def test_overview_numbers_by_hand(session, admin, trader):
    bob = create_user(session, "bob", "password1")
    start(session, DAYS[0])
    buy(session, trader, 10)                       # 10 x 100 = 1,000 traded today
    step(session)                                  # day 2: price 102
    buy(session, bob, 5)                           # 5 x 102 = 510 traded today
    o = stats.overview(session, get_clock(session))
    assert o["total_users"] == 2                   # admins don't count
    assert (o["active_today"], o["orders_today"]) == (1, 1)
    assert o["traded_value_today"] == pytest.approx(510)
    assert o["traded_value_total"] == pytest.approx(1510)
    # alice gained (102 - 100) x 10 = 20 before charges; bob 0; both paid charges
    fees = session.scalar(select(func.sum(Trade.fees)))
    assert o["platform_pnl"] == pytest.approx(20 - fees)


def test_activity_over_time(session, admin, trader):
    start(session, DAYS[0])
    buy(session, trader)
    step(session)
    buy(session, trader)
    df = stats.activity_over_time(session)
    assert list(df["orders"]) == [1, 1]
    assert df["traded_value"].tolist() == pytest.approx([1000, 1020])


def test_leaderboard_sorted_by_pnl(session, admin, trader):
    bob = create_user(session, "bob", "password1")
    start(session, DAYS[0])
    buy(session, trader, 100)  # price rises: alice ahead
    step(session)
    board = stats.leaderboard(session, get_clock(session))
    assert list(board["username"]) == ["alice", "bob"]
    assert board.loc[0, "total_pnl"] > 0       # alice: +200 before charges
    assert board.loc[1, "total_pnl"] == 0      # bob never traded


# ---------------------------------------------------------------------------
# Market control
# ---------------------------------------------------------------------------
def test_settings_validation_and_logging(session, admin):
    assert "Start the market" in market.change_settings(session, admin.id, speed_seconds=2)["message"]
    market.start_market(session, admin.id, DAYS[0])
    assert market.change_settings(session, admin.id, speed_seconds=2, volatility=1.5)["status"] == "success"
    assert (settings(session)["speed_seconds"], settings(session)["volatility"]) == (2, 1.5)
    assert "Speed must be" in market.change_settings(session, admin.id, speed_seconds=0)["message"]
    assert "Mode must be" in market.change_settings(session, admin.id, mode="fast")["message"]
    log = session.scalar(select(AdminLog).where(AdminLog.action == "market_settings"))
    assert log.details["speed_seconds"] == {"from": 5.0, "to": 2}
    assert actions(session) == ["market_start", "market_settings"]


def test_tick_respects_running_and_speed(session, admin):
    market.start_market(session, admin.id, DAYS[0])
    t0 = datetime(2030, 1, 1, 12, 0, 0)
    assert tick(session, t0) is None                       # paused: nothing happens
    market.change_settings(session, admin.id, speed_seconds=5)
    market.set_running(session, admin.id, True)
    assert tick(session, t0)["status"] == "success"         # first tick is immediate
    assert get_clock(session) == DAYS[1]
    assert tick(session, t0 + timedelta(seconds=3)) is None  # only 3 s later: not due
    tick(session, t0 + timedelta(seconds=5))
    assert get_clock(session) == DAYS[2]


def test_replay_pauses_itself_at_end_of_data(session, admin):
    market.start_market(session, admin.id, DAYS[-1])
    market.set_running(session, admin.id, True)
    assert "End of data" in tick(session, datetime(2030, 1, 1))["message"]
    assert settings(session)["is_running"] is False


def test_synthetic_mode_continues_after_history(session, admin):
    market.start_market(session, admin.id, DAYS[-2])
    market.change_settings(session, admin.id, mode="synthetic")
    step(session)                                    # Fri -> Mon 8th: still real history
    assert get_clock(session) == DAYS[-1]
    assert step(session)["status"] == "success"      # beyond the data: invented
    assert get_clock(session) == datetime(2024, 1, 9)
    bar = session.scalar(select(Candle).where(Candle.timestamp == datetime(2024, 1, 9)))
    assert bar.is_synthetic and bar.low <= min(bar.open, bar.close) <= max(bar.open, bar.close) <= bar.high
    assert status(session)["data"]["real_data_end"] == DAYS[-1]


def test_synthetic_bars_are_reproducible(session, admin):
    market.start_market(session, admin.id, DAYS[-1])
    market.change_settings(session, admin.id, mode="synthetic")
    step(session)
    first = session.scalar(select(Candle.close).where(Candle.is_synthetic.is_(True)))
    market.reset_market(session, admin.id)               # deletes synthetic bars
    assert session.scalar(select(func.count()).select_from(Candle).where(Candle.is_synthetic.is_(True))) == 0
    market.start_market(session, admin.id, DAYS[-1])
    market.change_settings(session, admin.id, mode="synthetic")
    step(session)
    assert session.scalar(select(Candle.close).where(Candle.is_synthetic.is_(True))) == first


def test_higher_volatility_moves_prices_more():
    import numpy as np
    from src.trading.synthetic import make_bar
    calm = [make_bar(100, 0.01, 1e5, np.random.default_rng(i))["close"] for i in range(400)]
    wild = [make_bar(100, 0.03, 1e5, np.random.default_rng(i))["close"] for i in range(400)]
    assert np.std(np.log(np.array(wild) / 100)) == pytest.approx(3 * np.std(np.log(np.array(calm) / 100)), rel=0.05)


# ---------------------------------------------------------------------------
# Instruments
# ---------------------------------------------------------------------------
def test_add_instrument_with_start_price_is_tradable(session, admin, trader):
    start(session, DAYS[0])
    assert instruments.add_instrument(session, admin.id, "infy", name="Infosys", start_price=1500)["status"] == "success"
    assert buy(session, trader)["status"] == "success"
    result = placeorder(session, trader.id, "INFY", "NSE", "BUY", 1, product="CNC", as_of=get_clock(session))
    assert result["status"] == "success"
    assert "already exists" in instruments.add_instrument(session, admin.id, "INFY")["message"]
    assert "Symbol must be" in instruments.add_instrument(session, admin.id, "bad symbol!")["message"]


def test_edit_instrument(session, admin):
    inst_id = session.scalar(select(Instrument.id))
    instruments.update_instrument(session, admin.id, inst_id, lot_size=25, name="Tata Consultancy")
    inst = session.get(Instrument, inst_id)
    assert (inst.lot_size, inst.name) == (25, "Tata Consultancy")
    assert "Lot size" in instruments.update_instrument(session, admin.id, inst_id, lot_size=0)["message"]
    log = session.scalar(select(AdminLog).where(AdminLog.action == "instrument_edit"))
    assert log.details["lot_size"] == {"from": 1, "to": 25}


def test_remove_deletes_unused_but_deactivates_used(session, admin):
    instruments.add_instrument(session, admin.id, "NEWCO")
    new_id = session.scalar(select(Instrument.id).where(Instrument.symbol == "NEWCO"))
    assert instruments.remove_instrument(session, admin.id, new_id)["data"]["outcome"] == "deleted"
    tcs_id = session.scalar(select(Instrument.id).where(Instrument.symbol == "TCS"))
    assert instruments.remove_instrument(session, admin.id, tcs_id)["data"]["outcome"] == "deactivated"
    assert session.get(Instrument, tcs_id).is_active is False  # history kept, no new orders


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------
def test_audit_frame_shows_names(session, admin, trader):
    users.top_up(session, admin.id, trader.id, 100)
    users.set_active(session, admin.id, trader.id, False)
    df = audit.audit_frame(session)
    assert list(df["action"]) == ["disable_user", "top_up"]  # newest first
    assert set(df["admin"]) == {"boss"} and set(df["target_user"]) == {"alice"}
    assert list(audit.audit_frame(session, action="top_up")["details"]) == [{"amount": 100}]


# ---------------------------------------------------------------------------
# ML Ops
# ---------------------------------------------------------------------------
def test_retrain_and_activate_versions(admin, tmp_path):
    from tests.test_ml import random_walk
    from tests.test_ml_insights import load_candles

    engine = get_engine("sqlite:///:memory:")
    init_db(engine)
    with get_session_factory(engine)() as s:
        boss = create_user(s, "boss2", "password1", role="admin")
        load_candles(s, random_walk(symbols=("AAA", "BBB"), days=300))
        out = {"models_dir": tmp_path / "models", "card_path": tmp_path / "card.md"}
        assert mlops.retrain(s, boss.id, **out)["data"]["version"] == 1
        assert mlops.retrain(s, boss.id, **out)["data"]["version"] == 2

        table = mlops.registry_frame(s)
        assert list(table["version"]) == [2, 1]          # newest first
        assert list(table["active"]) == [True, False]
        assert table["test_auc"].notna().all()

        mlops.activate_version(s, boss.id, 1)
        assert list(mlops.registry_frame(s)["active"]) == [False, True]
        assert "not found" in mlops.activate_version(s, boss.id, 99)["message"]
        assert [a for a in s.scalars(select(AdminLog.action).order_by(AdminLog.id))] ==             ["model_retrain", "model_retrain", "model_activate"]


def test_background_ticker_advances_a_running_market(tmp_path):
    """The real heartbeat thread (src/admin/ticker.py) moves the clock by itself.
    (The exact speed rule is tested with fake times in test_tick_respects_running_and_speed;
    here we only wait, up to a generous timeout, for the thread to do its job.)"""
    import time

    from src.admin.ticker import start_ticker

    engine = get_engine(f"sqlite:///{(tmp_path / 'tick.db').as_posix()}")  # a file: shared across threads
    init_db(engine)
    factory = get_session_factory(engine)
    with factory() as s:
        inst = Instrument(symbol="TCS", exchange="NSE", name="TCS", lot_size=1, tick_size=0.05, is_active=True)
        s.add(inst)
        s.flush()
        for day, c in zip(DAYS, CLOSES):
            s.add(Candle(instrument_id=inst.id, timestamp=day, open=c, high=c, low=c, close=c, volume=1_000_000))
        s.commit()
        boss = create_user(s, "boss", "password1", role="admin")
        market.start_market(s, boss.id, DAYS[0])
        market.change_settings(s, boss.id, speed_seconds=0.5)
        market.set_running(s, boss.id, True)

    stop = start_ticker(factory)
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            with factory() as s:
                if not settings(s)["is_running"]:
                    break  # replayed to the last bar and paused itself
            time.sleep(0.2)
    finally:
        stop.set()

    with factory() as s:
        assert get_clock(s) == DAYS[-1]                 # it walked through every day on its own
        assert settings(s)["is_running"] is False       # and paused at the end of the data
