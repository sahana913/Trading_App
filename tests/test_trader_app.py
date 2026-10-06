"""Click through the real multipage trader app with Streamlit's AppTest (no browser).

The app reads its database from PAPER_TRADING_DB_URL, which we point at a
temporary file holding two stocks with 10 days of daily candles (TCS closes
3500, 3510, 3520, ... and INFY 1500, 1510, ...).
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
from src.ui import BANNER

TRADER_APP = str(Path(__file__).resolve().parent.parent / "app" / "trader" / "app.py")
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
    at.run(timeout=30)  # charts make a run slower than AppTest's 3 s default
    assert not at.exception, at.exception
    return at


def banner_shown(at: AppTest) -> bool:
    """Note: after at.switch_page(), AppTest only reports the page file's own
    elements, not those drawn by app.py, so we check the banner on pages
    reached without switching. (Checked in a real browser: it shows on all.)"""
    return any(BANNER in h.proto.body for h in at.get("html"))


def logged_in_app() -> AppTest:
    at = run(AppTest.from_file(TRADER_APP))
    at.text_input(key="login_username").input("alice")
    at.text_input(key="login_password").input("password1")
    at.button(key="login_submit").click()
    return run(at)


def started_app() -> AppTest:
    at = logged_in_app()
    at.date_input(key="start_date").set_value(FIRST_DAY.date())
    at.button(key="start_market").click()
    return run(at)


def submit_order(at: AppTest, qty: int, action="BUY", pricetype="MARKET", price=None,
                 product="CNC") -> AppTest:
    at.selectbox(key="o_symbol").set_value("TCS")
    at.radio(key="o_action").set_value(action)
    at.number_input(key="o_qty").set_value(qty)
    at.selectbox(key="o_type").set_value(pricetype)
    if price is not None:
        at.number_input(key="o_price").set_value(price)
    at.radio(key="o_product").set_value(product)
    at.button(key="o_submit").click()
    return run(at)


def orders_in_db(factory) -> list[Order]:
    with factory() as s:
        return s.scalars(select(Order).order_by(Order.id)).all()


# ---------------------------------------------------------------------------
def test_logged_out_visitor_sees_only_login_with_banner(app_db):
    at = run(AppTest.from_file(TRADER_APP))
    assert at.title[0].value == "Paper Trading"
    assert at.text_input(key="login_username") is not None
    assert banner_shown(at)


def test_terminal_is_the_landing_page(app_db):
    at = started_app()
    assert at.title[0].value == "Trading Terminal"
    assert banner_shown(at)
    assert at.success[0].value == "Market started"


def test_market_must_be_started_first(app_db):
    at = logged_in_app()
    assert "hasn't started" in at.info[0].value


def test_watchlist_shows_prices_and_direction(app_db):
    at = started_app()
    at.sidebar.button(key="next_day").click()  # day 2: every stock is up 10
    at = run(at)
    table = at.dataframe[0].value  # the watchlist (a Styler's underlying data)
    assert list(table["Symbol"]) == ["INFY", "TCS"]
    assert table.set_index("Symbol").loc["TCS", "LTP"] == 3510
    assert table.set_index("Symbol").loc["TCS", "Change %"] == pytest.approx(10 / 3500 * 100)


def test_chart_has_candles_moving_averages_and_volume(app_db):
    at = started_app()
    chart = at.get("plotly_chart")[0]
    traces = [t["type"] for t in __import__("json").loads(chart.proto.spec)["data"]]
    assert traces == ["candlestick", "scatter", "scatter", "bar"]


def test_buy_from_the_terminal(app_db):
    at = submit_order(started_app(), qty=10)
    assert "BUY 10 TCS accepted" in at.success[-1].value
    (order,) = orders_in_db(app_db)
    assert (order.status, order.filled_quantity, order.average_price) == ("complete", 10, 3500.0)


def test_rejection_is_shown(app_db):
    at = submit_order(started_app(), qty=1000)  # ₹35 lakh > ₹10 lakh
    assert "Rejected: Insufficient funds" in at.error[0].value


def test_cancel_on_the_orders_page(app_db):
    at = submit_order(started_app(), qty=5, pricetype="LIMIT", price=3000.0)
    assert orders_in_db(app_db)[0].status == "open"

    at.switch_page("pages/orders.py")
    at = run(at)
    assert at.title[0].value == "Orders"
    at.button(key="cancel_order").click()
    at = run(at)
    # (The "Order ... cancelled" message is drawn by app.py, which AppTest
    # doesn't report after switch_page; see banner_shown.)
    assert orders_in_db(app_db)[0].status == "cancelled"


def test_next_day_fills_a_waiting_order(app_db):
    at = submit_order(started_app(), qty=5)                                    # buy 5 @ 3500
    at = submit_order(at, qty=5, action="SELL", pricetype="LIMIT", price=3505.0)
    assert orders_in_db(app_db)[1].status == "open"                           # 3500 < 3505: waits

    at.sidebar.button(key="next_day").click()
    at = run(at)
    with app_db() as s:
        assert get_clock(s) == FIRST_DAY + timedelta(days=1)
    assert "Moved to 02 Jan 2024 · 1 order(s) filled" in at.success[0].value
    sell = orders_in_db(app_db)[1]
    assert (sell.status, sell.average_price) == ("complete", 3510.0)


@pytest.mark.parametrize("page, title, metric", [
    ("pages/trades.py", "Trades", "Turnover"),
    ("pages/positions.py", "Positions", "Unrealised P&L"),
    ("pages/holdings.py", "Holdings", "Current value"),
    ("pages/funds.py", "Funds", "Available cash"),
    ("pages/analytics.py", "Analytics", "Profit factor"),
    ("pages/analytics.py", "Analytics", "VaR 95% (1 day)"),
])
def test_other_pages_render_after_trading(app_db, page, title, metric):
    at = submit_order(started_app(), qty=10)                       # CNC holding
    at = submit_order(at, qty=4, product="MIS")                    # an intraday position
    at.sidebar.number_input(key="run_days").set_value(5)
    at.sidebar.button(key="run_many").click()
    at = run(at)
    at = submit_order(at, qty=2, product="MIS")                    # open MIS on the current day
    at.switch_page(page)
    at = run(at)
    assert at.title[0].value == title
    assert metric in [m.label for m in at.metric]


@pytest.mark.parametrize("page, key", [("pages/orders.py", "orders_csv"),
                                       ("pages/trades.py", "trades_csv")])
def test_csv_downloads(app_db, page, key):
    at = submit_order(started_app(), qty=10)
    at.switch_page(page)
    at = run(at)
    (button,) = [b for b in at.get("download_button") if b.proto.id.endswith(key)]
    assert button.proto.label == "Download CSV"


def test_disabled_trader_is_logged_out(app_db):
    at = started_app()
    with app_db() as s:
        s.scalar(select(User).where(User.username == "alice")).is_active = False
        s.commit()
    at = run(at)
    assert "session ended" in at.warning[0].value
    assert at.title[0].value == "Paper Trading"  # back on the login page
