"""Analytics dashboard: KPI cards, risk (VaR / CVaR) and P&L charts.
Every formula is written out at the top of src/analytics/metrics.py."""

import streamlit as st

from src import ui
from src.analytics import charts
from src.analytics.metrics import (
    MIN_DAYS_FOR_VAR, calendar_frame, equity_curve, pnl_by_symbol, summary,
)
from src.auth import current_user, db
from src.trading.accounts import get_fund

user = current_user()
st.title("Analytics")

with db()() as s:
    clock = ui.require_clock(s)
    k = summary(s, user["id"], as_of=clock)
    curve = equity_curve(s, user["id"], as_of=clock)
    by_symbol = pnl_by_symbol(s, user["id"], as_of=clock)
    start_cash = get_fund(s, user["id"]).opening_balance

# --- KPI cards: performance, then risk (two rows of four fit narrow screens) --
not_enough = f" Needs at least {MIN_DAYS_FOR_VAR} days of history."
c = st.columns(4)
c[0].metric("Total P&L", ui.money(k["total_pnl"]), ui.pct(k["total_return"]),
            help=f"Equity now − starting cash, after all charges ({ui.money(k['total_charges'])} so far)")
c[1].metric("Day P&L", ui.money(k["day_pnl"]), help="Equity today − equity at yesterday's close")
c[2].metric("Win rate", ui.pct(k["win_rate"]),
            help=f"Winning ÷ all closing trades: {k['wins']} of {k['closed_trades']} (before charges)")
c[3].metric("Profit factor", "–" if k["profit_factor"] is None else f"{k['profit_factor']:.2f}",
            help="Money won ÷ money lost on closing trades. Above 1 = made more than lost. "
                 "Shown once at least one trade has lost.")

c = st.columns(4)
c[0].metric("Max drawdown", ui.pct(k["max_drawdown"]), help="Worst fall from a previous high")
c[1].metric("Sharpe ratio", "–" if k["sharpe"] is None else f"{k['sharpe']:.2f}",
            help="mean ÷ std of daily returns × √252. Unreliable over short periods: "
                 "a few good weeks can show a huge number.")
c[2].metric("VaR 95% (1 day)", ui.money(k["var_95_amount"]),
            None if k["var_95"] is None else f"{k['var_95']:.2%} of equity",
            delta_color="off", delta_arrow="off",
            help="On 95% of past days the loss was no bigger than this."
                 + ("" if k["var_95"] is not None else not_enough))
c[3].metric("CVaR 95% (1 day)", ui.money(k["cvar_95_amount"]),
            None if k["cvar_95"] is None else f"{k['cvar_95']:.2%} of equity",
            delta_color="off", delta_arrow="off",
            help="Average loss on the worst 5% of past days."
                 + ("" if k["cvar_95"] is not None else not_enough))

if len(curve) < 2:
    st.info("Charts appear after the market has moved at least one day (use “Next day”).")
    st.stop()

# --- Charts ----------------------------------------------------------------
left, right = st.columns(2)
left.plotly_chart(charts.equity_chart(curve, start_cash), width="stretch")
right.plotly_chart(charts.drawdown_chart(curve), width="stretch")
st.plotly_chart(charts.pnl_calendar_chart(calendar_frame(curve["date"], curve["day_pnl"])),
                width="stretch")
if not by_symbol.empty:
    st.plotly_chart(charts.pnl_by_symbol_chart(by_symbol), width="stretch")

with st.expander("Data table"):  # the same numbers as the charts, for exact values
    st.dataframe(curve, hide_index=True, width="stretch")
