"""
Trading Terminal: ticker tape, then watchlist | chart | order ticket.

The tape, watchlist and chart are "fragments": pieces of the page that re-run
on their own every WATCHLIST_REFRESH_SECONDS without redrawing the rest.
Prices change whenever the simulated clock moves; in intraday mode today's
candle grows step by step.

The order ticket is not a form, so it reacts as you type: the estimate
(value, cash blocked, cash left) comes from the engine's own margin rules
(orders.required_block, read-only), and the submit button says exactly what
it will do and is green for BUY, red for SELL.
"""

import pandas as pd
import streamlit as st
from sqlalchemy import select

from src import ui
from src.analytics import animated, charts
from src.auth import current_user, db
from src.config import CHART_BARS, MA_WINDOWS, SECTORS, WATCHLIST_REFRESH_SECONDS
from src.db.models import PRICETYPES, Instrument
from src.ml.predict import market_snapshot
from src.trading import history, placeorder, quotes
from src.trading.accounts import get_fund
from src.trading.orders import estimate_price, required_block, sellable_cnc_qty
from src.trading.simulator import get_clock

user = current_user()

with db()() as s:
    clock = ui.require_clock(s)
    chips = ui.market_chips(s)
    # {symbol: exchange}, plain strings (database objects don't survive re-runs)
    instruments = s.scalars(select(Instrument).where(Instrument.is_active).order_by(Instrument.symbol)).all()
    symbols = {i.symbol: i.exchange for i in instruments}
    inst_ids = {i.symbol: i.id for i in instruments}

ui.page_header("Trading Terminal", "Watch, chart and trade. Prices refresh every few seconds while the market runs.",
               chips)


def colour_change(value: float) -> str:
    """CSS for one watchlist cell: green up, red down."""
    if value > 0:
        return f"color: {charts.GAIN}"
    if value < 0:
        return f"color: {charts.LOSS}"
    return ""


@st.fragment(run_every=WATCHLIST_REFRESH_SECONDS)
def live_tape() -> None:
    with db()() as s:
        snapshot = market_snapshot(s, get_clock(s), SECTORS)
    if not snapshot.empty:
        ui.ticker_tape(snapshot)


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
            rows.append({"Symbol": symbol, "LTP (₹)": q["ltp"], "Change %": change_pct})

    table = pd.DataFrame(rows)
    styled = (
        table.style
        .format({"LTP (₹)": "{:,.2f}", "Change %": lambda v: f"{ui.arrow(v)}{abs(v):.2f}%"})  # the arrow + colour carry the sign
        .map(colour_change, subset=["Change %"])
    )
    st.subheader("Watchlist")
    st.caption(f"{ui.market_time(now)} · {len(table)} stocks")
    st.dataframe(styled, hide_index=True, width="stretch", height=38 + 35 * len(table),
                 column_config={"Symbol": st.column_config.Column(width=100),
                                "LTP (₹)": st.column_config.Column(width=72),
                                "Change %": st.column_config.Column("Chg %", width=64)})


@st.fragment(run_every=WATCHLIST_REFRESH_SECONDS)
def live_chart() -> None:
    tab_chart, tab_heat = st.tabs(["Chart", "Market heatmap"])
    with tab_chart:
        chart_symbol = st.selectbox("Chart", list(symbols), key="chart_symbol", label_visibility="collapsed")
        with db()() as s:
            bars = pd.DataFrame(history(s, chart_symbol, symbols[chart_symbol], as_of=get_clock(s))["data"])
        if bars.empty:
            ui.empty_state(f"No prices for {chart_symbol} yet", "The chart appears once the market has data for it.")
        else:
            # Pass the full history: moving averages need the bars before the visible window
            st.plotly_chart(charts.price_volume_chart(bars, chart_symbol, MA_WINDOWS, CHART_BARS),
                            width="stretch", key="price_chart")
    with tab_heat:
        with db()() as s:
            snapshot = market_snapshot(s, get_clock(s), SECTORS)
        if snapshot.empty:
            ui.empty_state("No prices yet", "The heatmap fills in once the market has data.")
        else:
            st.plotly_chart(animated.market_heatmap(snapshot), width="stretch", key="heatmap")
            st.caption("Grouped by sector. Tile size = money traded today; colour = change since the "
                       "previous close. Hover a tile for details.")


def order_ticket() -> None:
    chart_symbol = st.session_state.get("chart_symbol", next(iter(symbols)))  # the ticket follows the chart
    st.subheader("Order ticket")
    symbol = st.selectbox("Symbol", list(symbols), index=list(symbols).index(chart_symbol), key="o_symbol")
    action = st.radio("Side", ["BUY", "SELL"], horizontal=True, key="o_action")
    c1, c2 = st.columns([1, 1.3])  # order type needs a little more room
    quantity = c1.number_input("Quantity", min_value=1, value=1, step=1, key="o_qty")
    pricetype = c2.selectbox("Order type", PRICETYPES, key="o_type",
                             help="MARKET fills now at the last price. LIMIT waits for your "
                                  "price. SL / SL-M wait for the trigger price (stop-loss).")
    c1, c2 = st.columns(2)
    price = c1.number_input("Limit price", min_value=0.0, step=0.05, format="%.2f", key="o_price",
                            help="Used by LIMIT and SL orders")
    trigger = c2.number_input("Trigger price", min_value=0.0, step=0.05, format="%.2f", key="o_trigger",
                              help="Used by SL and SL-M orders")
    product = st.radio("Product", ["CNC", "MIS"], horizontal=True, key="o_product",
                       help="CNC = delivery, kept overnight. MIS = intraday with 5x "
                            "leverage, closed automatically at 15:15 (or the day's close).")

    # --- Estimate, using the engine's own rules (nothing is changed here) ---
    with db()() as s:
        now = get_clock(s)
        q = quotes(s, symbol, symbols[symbol], as_of=now)
        cash = get_fund(s, user["id"]).available_cash
        if q["status"] == "success":
            ltp = q["data"]["ltp"]
            est = estimate_price(pricetype, price, trigger, ltp) or ltp
            block = required_block(s, user["id"], inst_ids[symbol], product, action, int(quantity), est)
            can_sell = sellable_cnc_qty(s, user["id"], inst_ids[symbol]) if (product, action) == ("CNC", "SELL") else None
            lines = [("Last price", ui.money(ltp), ""), ("Est. value", ui.money(est * quantity), ""),
                     ("Cash blocked", ui.money(block), ""), ("Cash left after", ui.money(cash - block),
                                                            "warn" if block > cash else "")]
            if can_sell is not None:
                lines.append(("Shares you can sell", f"{can_sell:,}", "warn" if quantity > can_sell else ""))
            st.html('<div class="pd-est">' + "".join(
                f'<div><span>{k}</span><span class="{tone}">{v}</span></div>' for k, v, tone in lines) + "</div>")
            if block > cash:
                st.caption("⚠ Not enough cash for this order: it would be rejected.")

    ui.button_tone("o_submit", "up" if action == "BUY" else "down")
    label = f"{'Buy' if action == 'BUY' else 'Sell'} {quantity:,} {symbol}"
    if st.button(label, type="primary", key="o_submit", width="stretch"):
        with db()() as s:
            result = placeorder(
                s, user["id"], symbol, symbols[symbol], action, int(quantity),
                pricetype=pricetype, product=product,
                # send only the prices this order type uses
                price=price if pricetype in ("LIMIT", "SL") else 0.0,
                trigger_price=trigger if pricetype in ("SL", "SL-M") else 0.0,
                as_of=get_clock(s),
            )
        if result["status"] == "success":
            st.success(f"✅ {action} {quantity} {symbol} accepted · order {result['orderid']}")
        else:
            st.error(f"❌ Rejected: {result['message']}")


live_tape()
col_watch, col_chart, col_order = st.columns([1.6, 1.8, 1.38], gap="small")
with col_watch, st.container(border=True):
    watchlist()
with col_chart, st.container(border=True):
    live_chart()
with col_order, st.container(border=True):
    order_ticket()
