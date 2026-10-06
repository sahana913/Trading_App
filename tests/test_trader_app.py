"""Click through the real trader app with Streamlit's AppTest (no browser).

The app reads its database from PAPER_TRADING_DB_URL, which we point at a
temporary file holding two stocks with 10 days of daily candles.
"""

from datetime import datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import select
from streamlit.testing.v1 import AppTest

from src.db.models import Candle, Instrument, Order, User
from src.db.session import get_engine, get_session_factory, init_db
from src.trading.accounts import create_user
from src.trading.simulator import get_clock

TRADER_APP = str(Path(__file__).resolve().parent.parent / "app" / "trader_app.py")
FIRST_DAY = datetime(2024, 1, 1)


@pytest.fixture
def app_db(tmp_path, monkeypatch):
    url = f"sqlite:///{(tmp_path / 'app.db').as_posix()}"
    monkeypatch.setenv("PAPER_TRADING_DB_URL", url)
    engine = get_engine(url)
    init_db(engine)
    factory = get_session_factory(engine)
    with factory() as s:
        for symbol, base in (("INFY", 1500.0), ("TCS", 3500.0)):
            inst = Instrument(symbol=symbol, exchange="NSE", name=symbol, lot_size=1,
                              tick_size=0.05, is_active=True)
            s.add(inst)
            s.flush()
            for d in range(10):
                c = base + 10 * d
                s.add(Candle(instrument_id=inst.id, timestamp=FIRST_DAY + timedelta(days=d),
                             open=c - 5, high=c + 10, low=c - 10, close=c, volume=1_000_000))
        s.commit()
        create_user(s, "alice", "password1")
    return factory


def run(at: AppTest) -> AppTest:
    at.run(timeout=30)  # charts make the first run slower than the 3 s default
    assert not at.exception, at.exception
    return at


def logged_in_app() -> AppTest:
    at = run(AppTest.from_file(TRADER_APP))
    at.text_input(key="login_username").input("alice")
    at.text_input(key="login_password").input("password1")
    at.button(key="login_submit").click()
    return run(at)


def start_market(at: AppTest) -> AppTest:
    at.date_input(key="start_date").set_value(FIRST_DAY.date())
    at.button(key="start_market").click()
    return run(at)


def submit_order(at: AppTest, qty: int, product="CNC", pricetype="MARKET", price=None,
                 action="BUY") -> AppTest:
    at.selectbox(key="symbol").set_value("TCS")
    at.radio(key="o_action").set_value(action)
    at.radio(key="o_product").set_value(product)
    at.selectbox(key="o_type").set_value(pricetype)
    at.number_input(key="o_qty").set_value(qty)
    if price is not None:
        at.number_input(key="o_price").set_value(price)
    at.button(key="o_submit").click()
    return run(at)


def orders_in_db(factory) -> list[Order]:
    with factory() as s:
        return s.scalars(select(Order).order_by(Order.id)).all()


def test_market_must_be_started_first(app_db):
    at = logged_in_app()
    assert "hasn't started" in at.info[0].value
    assert len(at.tabs) == 0


def test_start_market_and_buy(app_db):
    at = start_market(logged_in_app())
    assert at.success[0].value == "Market started"
    assert [t.label for t in at.tabs] == ["Trade", "Orders", "Portfolio", "Analytics"]

    at = submit_order(at, qty=10)
    assert "BUY 10 TCS accepted" in at.success[0].value
    (order,) = orders_in_db(app_db)
    assert (order.status, order.filled_quantity, order.average_price) == ("complete", 10, 3500.0)


def test_rejected_order_shows_reason(app_db):
    at = submit_order(start_market(logged_in_app()), qty=1000)  # ₹35 lakh > ₹10 lakh
    assert "Insufficient funds" in at.error[0].value


def test_cancel_open_limit_order(app_db):
    at = submit_order(start_market(logged_in_app()), qty=5, pricetype="LIMIT", price=3000.0)
    assert orders_in_db(app_db)[0].status == "open"

    at.button(key="cancel_order").click()
    at = run(at)
    assert "cancelled" in at.success[0].value
    assert orders_in_db(app_db)[0].status == "cancelled"


def test_next_day_moves_clock_and_fills_waiting_order(app_db):
    # TCS closes at 3500 today and 3510 tomorrow
    at = submit_order(start_market(logged_in_app()), qty=5)          # buy 5 @ 3500
    at = submit_order(at, qty=5, action="SELL", pricetype="LIMIT", price=3505.0)
    assert orders_in_db(app_db)[1].status == "open"  # 3500 < 3505: waits

    at.button(key="next_day").click()
    at = run(at)
    with app_db() as s:
        assert get_clock(s) == FIRST_DAY + timedelta(days=1)
    assert "Moved to 02 Jan 2024 · 1 order(s) filled" in at.success[0].value
    sell = orders_in_db(app_db)[1]
    assert (sell.status, sell.average_price) == ("complete", 3510.0)


def test_all_tabs_render_after_a_few_days(app_db):
    at = submit_order(start_market(logged_in_app()), qty=10)
    at.number_input(key="run_days").set_value(5)
    at.button(key="run_many").click()
    at = run(at)  # every tab, including the Analytics charts, renders without error
    assert "Ran 5 day(s)" in at.success[0].value
    labels = [m.label for m in at.metric]
    for label in ("Equity", "Sharpe ratio", "Max drawdown", "Win rate", "Invested", "Current value"):
        assert label in labels
    assert len(at.get("plotly_chart")) >= 4  # candles + analytics charts


def test_disabled_trader_is_logged_out(app_db):
    at = start_market(logged_in_app())
    with app_db() as s:
        s.scalar(select(User).where(User.username == "alice")).is_active = False
        s.commit()
    at = run(at)
    assert "session ended" in at.warning[0].value
