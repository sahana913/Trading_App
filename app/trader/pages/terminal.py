"""
Trading Terminal: live watchlist | price chart | order form.

The watchlist and the chart are "fragments": pieces of the page that re-run
on their own every WATCHLIST_REFRESH_SECONDS without redrawing the rest, so
the order form you're typing into is not reset. Prices change whenever the
simulated clock moves; in intraday mode today's candle grows step by step.
"""

import pandas as pd
import streamlit as st
from sqlalchemy import select

from src import ui
from src.analytics import animated, charts
from src.auth import current_user, db
from src.config import CHART_BARS, MA_WINDOWS, SECTORS, WATCHLIST_REFRESH_SECONDS
from src.db.models import PRICETYPES, Instrument
from src.trading import history, placeorder, quotes
from src.trading.accounts import get_fund
from src.ml.predict import market_snapshot
from src.trading.simulator import get_clock

user = current_user()
st.title("Trading Terminal")

with db()() as s:
    clock = ui.require_clock(s)
    # {symbol: exchange}, plain strings (database objects don't survive re-runs)
    symbols = {i.symbol: i.exchange for i in s.scalars(
        select(Instrument).where(Instrument.is_active).order_by(Instrument.symbol))}


def colour_change(value: float) -> str:
    """CSS for one watchlist cell: green up, red down."""
    if value > 0:
        return f"color: {charts.GAIN}"
    if value < 0:
        return f"color: {charts.LOSS}"
    return ""


@st.fragment(run_every=WATCHLIST_REFRESH_SECONDS)
def watchlist() -> None:
    with db()() as s:
        now = get_clock(s)
        rows = []
        for symbol, exchange in symbols.items():
            q = quotes(s, symbol, exchange, as_of=now)
            if q["status"] == "error":
                continue  # no price yet for this stock
            q = q["data"]
            change_pct = (q["ltp"] / q["prev_close"] - 1) * 100  # vs previous close
            rows.append({"Symbol": symbol, "LTP": q["ltp"], "Change %": change_pct})

    table = pd.DataFrame(rows)
    styled = (
        table.style
        .format({"LTP": "₹{:,.2f}", "Change %": lambda v: f"{ui.arrow(v)} {v:+.2f}%"})
        .map(colour_change, subset=["Change %"])
    )
    st.subheader("Watchlist")
    st.caption(f"{ui.market_time(now)} · refreshes every {WATCHLIST_REFRESH_SECONDS}s")
    st.dataframe(styled, hide_index=True, width="stretch", height=38 + 35 * len(table))


@st.fragment(run_every=WATCHLIST_REFRESH_SECONDS)
def live_tape() -> None:
    with db()() as s:
        snapshot = market_snapshot(s, get_clock(s), SECTORS)
    if not snapshot.empty:
        ui.ticker_tape(snapshot)


live_tape()
col_watch, col_chart, col_order = st.columns([1.35, 2.25, 1.1], gap="medium")

with col_watch:
    watchlist()



@st.fragment(run_every=WATCHLIST_REFRESH_SECONDS)
def live_chart() -> None:
    tab_chart, tab_heat = st.tabs(["Chart", "Market heatmap"])
    with tab_chart:
        chart_symbol = st.selectbox("Chart", list(symbols), key="chart_symbol")
        with db()() as s:
            bars = pd.DataFrame(history(s, chart_symbol, symbols[chart_symbol], as_of=get_clock(s))["data"])
        if bars.empty:
            st.warning(f"No price data for {chart_symbol} yet.")
        else:
            # Pass the full history: moving averages need the bars before the visible window
            st.plotly_chart(charts.price_volume_chart(bars, chart_symbol, MA_WINDOWS, CHART_BARS),
                            width="stretch", key="price_chart")
    with tab_heat:
        with db()() as s:
            snapshot = market_snapshot(s, get_clock(s), SECTORS)
        if snapshot.empty:
            st.info("No prices yet.")
        else:
            st.plotly_chart(animated.market_heatmap(snapshot), width="stretch", key="heatmap")
            st.caption("Grouped by sector. Tile size = money traded today; colour = change since the "
                       "previous close. Hover a tile for details.")


with col_chart:
    live_chart()
chart_symbol = st.session_state.get("chart_symbol", next(iter(symbols)))  # the order form follows the chart

with col_order:
    st.subheader("Place order")
    with db()() as s:
        st.metric("Available cash", ui.money(get_fund(s, user["id"]).available_cash))

    with st.form("order_form"):
        symbol = st.selectbox("Symbol", list(symbols), index=list(symbols).index(chart_symbol),
                              key="o_symbol")
        action = st.radio("Side", ["BUY", "SELL"], horizontal=True, key="o_action")
        quantity = st.number_input("Quantity", min_value=1, value=1, step=1, key="o_qty")
        pricetype = st.selectbox("Order type", PRICETYPES, key="o_type",
                                 help="MARKET fills now at the last price. LIMIT waits for your "
                                      "price. SL / SL-M wait for the trigger price (stop-loss).")
        price = st.number_input("Limit price (LIMIT, SL)", min_value=0.0, step=0.05,
                                format="%.2f", key="o_price")
        trigger = st.number_input("Trigger price (SL, SL-M)", min_value=0.0, step=0.05,
                                  format="%.2f", key="o_trigger")
        product = st.radio("Product", ["CNC", "MIS"], horizontal=True, key="o_product",
                           help="CNC = delivery, kept overnight. MIS = intraday with 5x "
                                "leverage, closed automatically at the day's close.")
        submitted = st.form_submit_button("Submit order", type="primary", key="o_submit",
                                          width="stretch")

    if submitted:
        with db()() as s:
            result = placeorder(
                s, user["id"], symbol, symbols[symbol], action, int(quantity),
                pricetype=pricetype, product=product,
                # send only the prices this order type uses
                price=price if pricetype in ("LIMIT", "SL") else 0.0,
                trigger_price=trigger if pricetype in ("SL", "SL-M") else 0.0,
                as_of=clock,
            )
        if result["status"] == "success":
            st.success(f"✅ {action} {quantity} {symbol} accepted · order {result['orderid']}")
        else:
            st.error(f"❌ Rejected: {result['message']}")
