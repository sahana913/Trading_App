"""Overview: the platform at a glance, plus activity over time."""

import streamlit as st

from src import ui
from src.admin.stats import activity_over_time, equity_history, overview
from src.analytics import animated, charts
from src.auth import db
from src.trading.simulator import get_clock

st.title("Overview")

with db()() as s:
    clock = get_clock(s)
    o = overview(s, clock)
    history = activity_over_time(s)
    race = equity_history(s)

st.caption(f"Market date: **{clock:%d %b %Y}**" if clock else "The market hasn't started yet.")
c = st.columns(3)
c[0].metric("Traders", o["total_users"], help=f"{o['disabled_users']} disabled")
c[1].metric("Active today", o["active_today"], help="Traders who placed an order on the market date")
c[2].metric("Orders today", o["orders_today"])
c = st.columns(3)
c[0].metric("Traded today", ui.money(o["traded_value_today"]), help="Sum of quantity × price of today's fills")
c[1].metric("Traded in total", ui.money(o["traded_value_total"]))
c[2].metric("Platform P&L", ui.money(o["platform_pnl"]),
            help=f"All traders' equity ({ui.money(o['platform_equity'])}) − their opening balances, after charges")

if race["date"].nunique() >= 2:
    st.plotly_chart(animated.leaderboard_race(race), width="stretch", key="race")
    st.caption("Each trader's account value at the end of every day. Press Play to watch the ranking change.")

if history.empty:
    st.info("Charts appear once traders start placing orders.")
    st.stop()

left, right = st.columns(2)
left.plotly_chart(charts.bar_over_time(history, "orders", "Orders per day", prefix=""), width="stretch")
right.plotly_chart(charts.bar_over_time(history, "traded_value", "Traded value per day", prefix="₹"),
                   width="stretch")
left.plotly_chart(charts.bar_over_time(history, "active_traders", "Active traders per day", prefix=""),
                  width="stretch")
equity = history.dropna(subset=["platform_equity"]) if "platform_equity" in history else history.iloc[0:0]
if not equity.empty:
    right.plotly_chart(charts.line_over_time(equity, "platform_equity", "Platform equity (end of day)"),
                       width="stretch")
