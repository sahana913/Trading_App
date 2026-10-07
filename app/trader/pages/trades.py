"""Trades: every fill, with totals and a CSV download."""

import pandas as pd
import streamlit as st

from src import ui
from src.analytics.metrics import trades_frame
from src.auth import current_user, db
from src.trading import tradebook

user = current_user()
ui.page_header("Trades", "Every fill, with what it cost and what it booked.")

with db()() as s:
    clock = ui.require_clock(s)
    book = pd.DataFrame(tradebook(s, user["id"])["data"])
    pnl = trades_frame(s, user["id"])  # same fills, with each one's realised P&L

if book.empty:
    ui.empty_state("No trades yet", "Orders become trades when they fill. Place one in the Terminal.")
    st.stop()

c = st.columns(4)
c[0].metric("Trades", len(book))
c[1].metric("Turnover", ui.money(book["trade_value"].sum()), help="Total value bought + sold")
c[2].metric("Realised P&L", ui.money(pnl["realised_pnl"].sum()),
            help="Sum of profit/loss booked by closing trades, before charges")
c[3].metric("Charges", ui.money(book["fees"].sum()))

# tradebook is newest first; trades_frame oldest first, so line them up by reversing
book["realised_pnl"] = pnl["realised_pnl"].iloc[::-1].astype(float).round(2).to_numpy()  # None -> NaN
book["timestamp"] = pd.to_datetime(book["timestamp"]).dt.date
columns = ["timestamp", "symbol", "action", "product", "quantity", "average_price",
           "trade_value", "fees", "realised_pnl", "orderid", "tradeid"]

head, button = st.columns([4, 1], vertical_alignment="bottom")
head.subheader("Trade book")
with button:
    ui.csv_download(book[columns], "Download CSV", f"trades_{clock:%Y%m%d}.csv", key="trades_csv")
st.dataframe(
    book[columns], hide_index=True, width="stretch",
    column_config={"timestamp": "Date", "average_price": "Price",
                   "trade_value": st.column_config.NumberColumn("Value", format="%.2f"),
                   "realised_pnl": st.column_config.NumberColumn(
                       "Realised P&L", format="%.2f",
                       help="Empty for fills that only opened or added to a position")},
)
