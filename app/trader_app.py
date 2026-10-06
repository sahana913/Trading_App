"""
trader_app.py - The trader's Streamlit app.

Tabs: Trade (quote, chart, order form) | Orders | Portfolio | Analytics.
The sidebar shows the simulated market date and, if allowed in config,
buttons to move it forward.

Run from the project root (PowerShell):
    .venv\\Scripts\\python.exe -m streamlit run app/trader_app.py

How Streamlit works, in one line: the whole script re-runs from top to bottom
on every click, and st.session_state keeps values between runs.
"""

import sys
from datetime import datetime
from pathlib import Path

# Streamlit only puts app/ on the import path; add the project root so
# "from src... import" works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402  (imports must come after the path fix)
import streamlit as st  # noqa: E402
from sqlalchemy import select  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from src.analytics import charts  # noqa: E402
from src.analytics.metrics import equity_curve, pnl_by_symbol, summary  # noqa: E402
from src.auth import db, require_role  # noqa: E402
from src.config import CHART_BARS, TRADER_CAN_MOVE_CLOCK  # noqa: E402
from src.db.models import PRICETYPES, Instrument  # noqa: E402
from src.trading import (  # noqa: E402
    cancelorder, funds, history, holdings, modifyorder, orderbook, placeorder,
    positionbook, quotes, tradebook,
)
from src.trading.accounts import get_fund  # noqa: E402
from src.trading.simulator import get_clock, run, start, status, step  # noqa: E402

st.set_page_config(page_title="Paper Trading", page_icon="📈", layout="wide")
user = require_role("user", "Paper Trading", allow_register=True)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def money(x: float | None) -> str:
    return "–" if x is None else f"₹{x:,.2f}"


def pct(x: float | None) -> str:
    return "–" if x is None else f"{x:.2%}"


def flash(kind: str, message: str) -> None:
    """Save a message, then re-run the page so every number is refreshed.
    The message is shown once at the top of the next run."""
    st.session_state["flash"] = (kind, message)
    st.rerun()


def show_flash() -> None:
    if "flash" in st.session_state:
        kind, message = st.session_state.pop("flash")
        getattr(st, kind)(message)  # st.success(...) / st.error(...)


def report(result: dict, ok_message: str) -> None:
    """Turn an engine response into a flash message."""
    if result["status"] == "success":
        flash("success", ok_message)
    else:
        flash("error", result["message"])


# ---------------------------------------------------------------------------
# Sidebar: the market clock
# ---------------------------------------------------------------------------
def market_sidebar(s: Session, clock: datetime | None) -> None:
    with st.sidebar:
        st.subheader("Market")
        if clock is not None:
            st.metric("Market date", f"{clock:%a %d %b %Y}")
        if not TRADER_CAN_MOVE_CLOCK:
            return

        if clock is None:
            info = status(s)["data"]
            first = st.date_input("Start date", value=info["data_start"].date(),
                                  min_value=info["data_start"].date(),
                                  max_value=info["data_end"].date(), key="start_date")
            if st.button("Start market", key="start_market", type="primary"):
                report(start(s, datetime.combine(first, datetime.min.time())), "Market started")
            return

        if st.button("Next day ▶", key="next_day", type="primary", width="stretch"):
            r = step(s)
            if r["status"] == "success":
                flash("success", f"Moved to {r['data']['to']:%d %b %Y} · {r['data']['fills']} order(s) filled")
            else:
                flash("error", r["message"])
        days = st.number_input("Days", min_value=1, max_value=250, value=5, key="run_days")
        if st.button(f"Run {days} days ⏩", key="run_many", width="stretch"):
            r = run(s, int(days))["data"]
            flash("success", f"Ran {r['steps']} day(s) · {r['fills']} order(s) filled")
        st.caption("Moving the clock closes MIS positions at the day's close and "
                   "fills waiting orders at the new day's prices.")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
def trade_tab(s: Session, clock: datetime) -> None:
    # Plain strings as options: database objects are re-created on every run
    instruments = {i.symbol: i for i in s.scalars(
        select(Instrument).where(Instrument.is_active).order_by(Instrument.symbol))}
    left, right = st.columns([2, 1], gap="large")

    with left:
        inst = instruments[st.selectbox("Stock", list(instruments), key="symbol")]
        q = quotes(s, inst.symbol, inst.exchange, as_of=clock)
        if q["status"] == "error":
            st.warning(q["message"])
            return
        q = q["data"]
        change = q["ltp"] - q["prev_close"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Last price", money(q["ltp"]), f"{change:+,.2f} ({change / q['prev_close']:+.2%})")
        c2.metric("Open", money(q["open"]))
        c3.metric("High / Low", f"{q['high']:,.2f} / {q['low']:,.2f}")
        c4.metric("Volume", f"{q['volume']:,}")

        bars = pd.DataFrame(history(s, inst.symbol, inst.exchange, as_of=clock)["data"]).tail(CHART_BARS)
        st.plotly_chart(charts.candle_chart(bars, inst.symbol), width="stretch")

    with right:
        st.subheader("Place order")
        with st.form("order_form"):
            action = st.radio("Side", ["BUY", "SELL"], horizontal=True, key="o_action")
            product = st.radio("Product", ["CNC", "MIS"], horizontal=True, key="o_product",
                               help="CNC = delivery (kept overnight). MIS = intraday, "
                                    "5x leverage, closed automatically at the day's close.")
            pricetype = st.selectbox("Order type", PRICETYPES, key="o_type",
                                     help="MARKET fills now. LIMIT waits for your price. "
                                          "SL / SL-M wait for the trigger price (stop-loss).")
            quantity = st.number_input("Quantity", min_value=1, value=1, step=1, key="o_qty")
            price = st.number_input("Limit price (LIMIT, SL)", min_value=0.0, value=float(q["ltp"]),
                                    step=0.05, format="%.2f", key="o_price")
            trigger = st.number_input("Trigger price (SL, SL-M)", min_value=0.0, value=float(q["ltp"]),
                                      step=0.05, format="%.2f", key="o_trigger")
            st.caption(f"Approx. value at last price: {money(quantity * q['ltp'])}")
            if st.form_submit_button("Submit order", type="primary", key="o_submit", width="stretch"):
                result = placeorder(
                    s, user["id"], inst.symbol, inst.exchange, action, int(quantity),
                    pricetype=pricetype, product=product,
                    # send only the prices this order type uses
                    price=price if pricetype in ("LIMIT", "SL") else 0.0,
                    trigger_price=trigger if pricetype in ("SL", "SL-M") else 0.0,
                    as_of=clock,
                )
                report(result, f"{action} {quantity} {inst.symbol} accepted (order {result.get('orderid')})")


def orders_tab(s: Session, clock: datetime) -> None:
    book = orderbook(s, user["id"])["data"]
    orders = pd.DataFrame(book["orders"])
    stats = book["statistics"]
    c = st.columns(4)
    c[0].metric("Open", stats["total_open_orders"])
    c[1].metric("Completed", stats["total_completed_orders"])
    c[2].metric("Cancelled", stats["total_cancelled_orders"])
    c[3].metric("Rejected", stats["total_rejected_orders"])
    if orders.empty:
        st.info("No orders yet. Place one in the Trade tab.")
        return

    open_orders = orders[orders["order_status"] == "open"]
    if not open_orders.empty:
        st.subheader("Open orders")
        labels = {
            r.orderid: f"{r.action} {r.quantity} {r.symbol} {r.pricetype}"
                       f"{f' @ {r.price:,.2f}' if r.price else ''}"
                       f"{f' trigger {r.trigger_price:,.2f}' if r.trigger_price else ''}"
                       f" · filled {r.filled_quantity}"
            for r in open_orders.itertuples()
        }
        oid = st.selectbox("Order", list(labels), format_func=labels.get, key="open_order")
        row = open_orders.set_index("orderid").loc[oid]
        col_modify, col_cancel = st.columns([3, 1])
        with col_modify, st.form("modify_form"):
            m1, m2, m3 = st.columns(3)
            new_qty = m1.number_input("Quantity", min_value=1, value=int(row["quantity"]), key="m_qty")
            new_price = m2.number_input("Limit price", min_value=0.0, value=float(row["price"]),
                                        step=0.05, format="%.2f", key="m_price")
            new_trigger = m3.number_input("Trigger", min_value=0.0, value=float(row["trigger_price"]),
                                          step=0.05, format="%.2f", key="m_trigger")
            if st.form_submit_button("Modify", key="m_submit"):
                report(modifyorder(s, user["id"], oid, quantity=int(new_qty), price=new_price,
                                   trigger_price=new_trigger, as_of=clock), f"Order {oid} modified")
        if col_cancel.button("Cancel order", key="cancel_order", width="stretch"):
            report(cancelorder(s, user["id"], oid), f"Order {oid} cancelled")

    st.subheader("Order book")
    orders["timestamp"] = pd.to_datetime(orders["timestamp"]).dt.date
    st.dataframe(
        orders[["timestamp", "symbol", "action", "product", "pricetype", "quantity", "filled_quantity",
                "price", "trigger_price", "average_price", "order_status", "rejection_reason", "orderid"]],
        hide_index=True, width="stretch",
        column_config={"timestamp": "Date", "filled_quantity": "Filled", "order_status": "Status",
                       "rejection_reason": "Note", "average_price": st.column_config.NumberColumn(
                           "Avg price", format="%.2f")},
    )

    trades = pd.DataFrame(tradebook(s, user["id"])["data"])
    if not trades.empty:
        st.subheader("Trade book")
        trades["timestamp"] = pd.to_datetime(trades["timestamp"]).dt.date
        st.dataframe(trades[["timestamp", "symbol", "action", "product", "quantity", "average_price",
                             "trade_value", "fees", "orderid"]],
                     hide_index=True, width="stretch",
                     column_config={"timestamp": "Date", "average_price": "Price",
                                    "trade_value": st.column_config.NumberColumn("Value", format="%.2f")})


def portfolio_tab(s: Session, clock: datetime) -> None:
    f = funds(s, user["id"], as_of=clock)["data"]
    c = st.columns(4)
    c[0].metric("Available cash", money(f["availablecash"]))
    c[1].metric("Blocked (margin)", money(f["utiliseddebits"]))
    c[2].metric("Realised P&L", money(f["m2mrealized"]))
    c[3].metric("Unrealised MIS P&L", money(f["m2munrealized"]))

    h = holdings(s, user["id"], as_of=clock)["data"]
    st.subheader("Holdings (CNC)")
    if h["holdings"]:
        hs = h["statistics"]
        c = st.columns(3)
        c[0].metric("Invested", money(hs["totalinvvalue"]))
        c[1].metric("Current value", money(hs["totalholdingvalue"]))
        c[2].metric("P&L", money(hs["totalprofitandloss"]), f"{hs['totalpnlpercentage']:+.2f}%")
        st.dataframe(pd.DataFrame(h["holdings"]), hide_index=True, width="stretch",
                     column_config={"pnlpercent": st.column_config.NumberColumn("P&L %", format="%.2f%%")})
    else:
        st.info("No holdings. Buy with product CNC to keep shares overnight.")

    positions = positionbook(s, user["id"], as_of=clock)["data"]
    if positions:
        st.subheader("Intraday positions (MIS)")
        st.dataframe(pd.DataFrame(positions), hide_index=True, width="stretch")


def analytics_tab(s: Session, clock: datetime) -> None:
    k = summary(s, user["id"], as_of=clock)
    c = st.columns(6)
    c[0].metric("Equity", money(k["equity"]))
    c[1].metric("Total P&L", money(k["total_pnl"]), pct(k["total_return"]),
                help="Change in account value since the start, after all charges")
    c[2].metric("Sharpe ratio", "–" if k["sharpe"] is None else f"{k['sharpe']:.2f}",
                help="Return per unit of risk, annualised. Needs at least 2 days.")
    c[3].metric("Max drawdown", pct(k["max_drawdown"]), help="Worst fall from a previous high")
    c[4].metric("Win rate", pct(k["win_rate"]),
                help=f"{k['wins']} winning of {k['closed_trades']} closing trades (before charges)")
    c[5].metric("Charges paid", money(k["total_charges"]))

    curve = equity_curve(s, user["id"], as_of=clock)
    if len(curve) < 2:
        st.info("Charts appear after the market has moved at least one day (use “Next day”).")
        return
    start_cash = get_fund(s, user["id"]).opening_balance
    left, right = st.columns(2)
    left.plotly_chart(charts.equity_chart(curve, start_cash), width="stretch")
    right.plotly_chart(charts.drawdown_chart(curve), width="stretch")
    left.plotly_chart(charts.daily_pnl_chart(curve), width="stretch")
    by_symbol = pnl_by_symbol(s, user["id"], as_of=clock)
    if not by_symbol.empty:
        right.plotly_chart(charts.pnl_by_symbol_chart(by_symbol), width="stretch")

    with st.expander("Data table"):  # the same numbers as the charts, for reading exact values
        st.dataframe(curve, hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
with db()() as s:
    clock = get_clock(s)
    market_sidebar(s, clock)
    st.title("Paper Trading")
    show_flash()

    if clock is None:
        st.info("The market hasn't started yet. "
                + ("Pick a start date in the sidebar and press **Start market**."
                   if TRADER_CAN_MOVE_CLOCK else "Ask an administrator to start it."))
        st.stop()

    f = funds(s, user["id"], as_of=clock)["data"]
    k = summary(s, user["id"], as_of=clock)
    c = st.columns(3)
    c[0].metric("Equity", money(k["equity"]))
    c[1].metric("Available cash", money(f["availablecash"]))
    c[2].metric("Total P&L", money(k["total_pnl"]), pct(k["total_return"]))

    t_trade, t_orders, t_portfolio, t_analytics = st.tabs(["Trade", "Orders", "Portfolio", "Analytics"])
    with t_trade:
        trade_tab(s, clock)
    with t_orders:
        orders_tab(s, clock)
    with t_portfolio:
        portfolio_tab(s, clock)
    with t_analytics:
        analytics_tab(s, clock)
