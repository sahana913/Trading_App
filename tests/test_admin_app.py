"""Click through the real admin app with AppTest: non-admins are blocked,
every page renders, and UI actions change data and leave an audit row."""

from datetime import datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import select
from streamlit.testing.v1 import AppTest

from src.db.models import AdminLog, Candle, Instrument, User
from src.db.session import get_engine, get_session_factory, init_db
from src.trading import create_user, placeorder
from src.trading.simulator import get_clock, settings, start, step

ADMIN_APP = str(Path(__file__).resolve().parent.parent / "app" / "admin" / "admin_app.py")
PAGES = ["pages/overview.py", "pages/users.py", "pages/leaderboard.py", "pages/market.py",
         "pages/instruments.py", "pages/mlops.py", "pages/audit.py"]
FIRST_DAY = datetime(2024, 1, 1)


@pytest.fixture
def admin_db(tmp_path, monkeypatch):
    url = f"sqlite:///{(tmp_path / 'admin.db').as_posix()}"
    monkeypatch.setenv("PAPER_TRADING_DB_URL", url)
    engine = get_engine(url)
    init_db(engine)
    factory = get_session_factory(engine)
    with factory() as s:
        inst = Instrument(symbol="TCS", exchange="NSE", name="TCS", lot_size=1, tick_size=0.05, is_active=True)
        s.add(inst)
        s.flush()
        for d in range(10):
            c = 3500.0 + 10 * d
            s.add(Candle(instrument_id=inst.id, timestamp=FIRST_DAY + timedelta(days=d), open=c, high=c,
                         low=c, close=c, volume=1_000_000))
        s.commit()
        create_user(s, "boss", "password1", role="admin")
        create_user(s, "alice", "password1")
        create_user(s, "carol", "password1")
    return factory


def run(at: AppTest) -> AppTest:
    at.run(timeout=60)
    assert not at.exception, at.exception
    return at


def log_in(username: str) -> AppTest:
    at = run(AppTest.from_file(ADMIN_APP))
    at.text_input(key="login_username").input(username)
    at.text_input(key="login_password").input("password1")
    at.button(key="login_submit").click()
    return run(at)


def audit_actions(factory) -> list[str]:
    with factory() as s:
        return list(s.scalars(select(AdminLog.action).order_by(AdminLog.id)))


# ---------------------------------------------------------------------------
# Non-admins are blocked
# ---------------------------------------------------------------------------
def test_logged_out_visitor_only_has_the_login_page(admin_db):
    at = run(AppTest.from_file(ADMIN_APP))
    assert at.title[0].value == "Admin Dashboard"
    assert len(at.tabs) == 0 or [t.label for t in at.tabs] == ["Log in"]  # no Register tab here
    for page in PAGES:  # admin pages are not even registered
        with pytest.raises(ValueError):
            at.switch_page(page)


def test_trader_is_blocked(admin_db):
    at = log_in("alice")
    assert "admin accounts only" in at.error[0].value
    assert not any(t.value == "Overview" for t in at.title)
    for page in PAGES:
        with pytest.raises(ValueError):
            at.switch_page(page)
    assert audit_actions(admin_db) == []  # a trader's visit is not an admin login


def test_disabled_admin_cannot_log_in(admin_db):
    with admin_db() as s:
        s.scalar(select(User).where(User.username == "boss")).is_active = False
        s.commit()
    at = log_in("boss")
    assert "disabled" in at.error[0].value


def test_admin_demoted_mid_session_is_logged_out(admin_db):
    at = log_in("boss")
    assert at.title[0].value == "Overview"
    with admin_db() as s:
        s.scalar(select(User).where(User.username == "boss")).role = "user"
        s.commit()
    at = run(at)
    assert "session ended" in at.warning[0].value
    assert at.title[0].value == "Admin Dashboard"  # back to the login page


# ---------------------------------------------------------------------------
# Admin pages
# ---------------------------------------------------------------------------
def test_admin_login_is_audited(admin_db):
    log_in("boss")
    assert audit_actions(admin_db) == ["login"]


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_with_activity(admin_db, page):
    with admin_db() as s:
        start(s, FIRST_DAY)
        alice = s.scalar(select(User).where(User.username == "alice"))
        placeorder(s, alice.id, "TCS", "NSE", "BUY", 10, product="CNC", as_of=get_clock(s))
        step(s)
    at = log_in("boss")
    at.switch_page(page)
    at = run(at)
    assert at.title[0].value in {"Overview", "Users", "Leaderboard", "Market control", "Instruments",
                                 "ML Ops", "Audit log"}


def test_disable_user_from_the_users_page(admin_db):
    at = log_in("boss")
    at.switch_page("pages/users.py")
    at = run(at)
    at.selectbox(key="user_pick").set_value("carol")
    at = run(at)
    at.button(key="toggle_active").click()
    run(at)
    with admin_db() as s:
        assert s.scalar(select(User).where(User.username == "carol")).is_active is False
    assert audit_actions(admin_db) == ["login", "disable_user"]


def test_start_market_and_change_settings_from_the_market_page(admin_db):
    at = log_in("boss")
    at.switch_page("pages/market.py")
    at = run(at)
    at.button(key="admin_start").click()
    at = run(at)
    with admin_db() as s:
        assert get_clock(s) == FIRST_DAY.replace(hour=9, minute=15)   # intraday is the default: opens 09:15
        assert settings(s)["intraday"] is True

    at.radio(key="mode").set_value("synthetic")
    at.slider(key="speed").set_value(2.0)
    at.button(key="save_settings").click()
    at = run(at)
    at.button(key="resume").click()
    run(at)
    with admin_db() as s:
        cfg = settings(s)
    assert (cfg["mode"], cfg["speed_seconds"], cfg["is_running"], cfg["intraday"]) == ("synthetic", 2.0, True, True)
    assert audit_actions(admin_db) == ["login", "market_start", "market_settings", "market_resume"]
