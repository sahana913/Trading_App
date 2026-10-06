"""Funds: where the money is.

  equity = available cash + blocked cash + holdings value + unrealised MIS P&L
  blocked cash = margin held for open orders and open MIS positions;
                 it still belongs to you and comes back when they close.
"""

import pandas as pd
import streamlit as st

from src import ui
from src.analytics.metrics import trades_frame
from src.auth import current_user, db
from src.trading import funds
from src.trading.accounts import get_fund
from src.trading.books import portfolio_value

user = current_user()
st.title("Funds")

with db()() as s:
    clock = ui.require_clock(s)
    f = funds(s, user["id"], as_of=clock)["data"]
    value = portfolio_value(s, user["id"], as_of=clock)
    opening = get_fund(s, user["id"]).opening_balance
    charges = trades_frame(s, user["id"])["fees"].sum()

c = st.columns(4)
c[0].metric("Available cash", ui.money(f["availablecash"]), help="Free to use for new orders")
c[1].metric("Blocked (margin)", ui.money(f["utiliseddebits"]),
            help="Held for open orders and open MIS positions")
c[2].metric("Realised P&L", ui.money(f["m2mrealized"]), help="Profit already booked, before charges")
c[3].metric("Unrealised MIS P&L", ui.money(f["m2munrealized"]))

st.subheader("Account value")
breakdown = pd.DataFrame([
    ("Available cash", value["cash"]),
    ("+ Blocked cash", value["blocked"]),
    ("+ Holdings at last price", value["holdings_value"]),
    ("+ Unrealised MIS P&L", value["equity"] - value["cash"] - value["blocked"] - value["holdings_value"]),
    ("= Equity", value["equity"]),
    ("Starting cash", opening),
    ("Total P&L (equity − starting cash)", value["equity"] - opening),
    ("of which charges paid", -charges),
], columns=["Item", "Amount (₹)"])
st.dataframe(breakdown, hide_index=True, width="stretch",
             column_config={"Amount (₹)": st.column_config.NumberColumn(format="localized")})
