"""Leaderboard: top traders by P&L or by Sharpe ratio."""

import streamlit as st

from src import ui
from src.admin.stats import leaderboard
from src.analytics import charts
from src.auth import db
from src.trading.simulator import get_clock

ui.page_header("Leaderboard", "Who is doing best: by money made, or by return per unit of risk.")

with db()() as s:
    board = leaderboard(s, get_clock(s))

if board.empty:
    ui.empty_state("No traders yet", "Traders appear here once they register.")
    st.stop()

by = st.radio("Rank by", ["Total P&L", "Sharpe ratio"], horizontal=True, key="rank_by",
              help="P&L = money made after charges. Sharpe = return per unit of risk; it needs at least "
                   "2 days and is unreliable over short periods.")
column, fmt = ("total_pnl", ",.0f") if by == "Total P&L" else ("sharpe", ".2f")
ranked = board.sort_values(column, ascending=False, na_position="last").reset_index(drop=True)
ranked.insert(0, "rank", range(1, len(ranked) + 1))

st.plotly_chart(charts.leaderboard_chart(ranked, column, f"Top 10 by {by}", fmt), width="stretch")
st.dataframe(ranked, hide_index=True, width="stretch", column_config={
    "equity": st.column_config.NumberColumn("Equity", format="localized"),
    "total_pnl": st.column_config.NumberColumn("Total P&L", format="localized"),
    "return": st.column_config.NumberColumn("Return", format="percent"),
    "sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f"),
    "max_drawdown": st.column_config.NumberColumn("Max drawdown", format="percent"),
    "win_rate": st.column_config.NumberColumn("Win rate", format="percent"),
})
