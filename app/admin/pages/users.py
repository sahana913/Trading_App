"""Users: find any account, look at its portfolio and trades, and manage it."""

import pandas as pd
import streamlit as st

from src import ui
from src.admin.users import reset_account, search_users, set_active, top_up, user_snapshot
from src.auth import current_user, db
from src.trading import holdings, orderbook, positionbook, tradebook
from src.trading.simulator import get_clock

admin = current_user()
ui.page_header("Users", "Find any account, look at its portfolio, and manage it.")

query = st.text_input("Search by username", key="user_search", placeholder="e.g. ali")
with db()() as s:
    found = search_users(s, query)
st.dataframe(found.drop(columns=["id"]), hide_index=True, width="stretch",
             column_config={"active": st.column_config.CheckboxColumn("Active"),
                            "cash": st.column_config.NumberColumn("Cash", format="localized")})
if found.empty:
    st.stop()

names = dict(zip(found["username"], found["id"]))
chosen = st.selectbox("Account", list(names), key="user_pick")
uid = int(names[chosen])
row = found.set_index("username").loc[chosen]

st.divider()
st.subheader(f"{chosen} · {row['role']} · {'active' if row['active'] else 'DISABLED'}")

# --- Manage ---------------------------------------------------------------
c1, c2, c3 = st.columns(3)
with c1:
    label = "Disable account" if row["active"] else "Enable account"
    if st.button(label, key="toggle_active", width="stretch", disabled=uid == admin["id"],
                 help="You can't disable yourself" if uid == admin["id"] else None):
        with db()() as s:
            ui.report(set_active(s, admin["id"], uid, not row["active"]), f"{chosen}: {label.lower()}d")
if row["role"] == "user":
    with c2, st.form("top_up_form"):
        amount = st.number_input("Top up (₹)", min_value=1.0, value=100_000.0, step=10_000.0, key="top_up_amount")
        if st.form_submit_button("Add funds", key="top_up_submit", width="stretch"):
            with db()() as s:
                ui.report(top_up(s, admin["id"], uid, float(amount)), f"Added {ui.money(amount)} to {chosen}")
    with c3:
        sure = st.checkbox("I understand this deletes all of this user's orders and trades", key="reset_sure")
        if st.button("Reset account", key="reset_account", width="stretch", disabled=not sure, type="primary"):
            with db()() as s:
                ui.report(reset_account(s, admin["id"], uid), f"{chosen} reset to starting cash")

if row["role"] != "user":
    ui.empty_state("No portfolio", "Admin accounts don't trade, so there is nothing to show here.", mark="A")
    st.stop()

# --- Portfolio -------------------------------------------------------------
with db()() as s:
    clock = get_clock(s)
    snap = user_snapshot(s, uid, clock)
    h = holdings(s, uid, as_of=clock)["data"]
    positions = positionbook(s, uid, as_of=clock)["data"]
    trades = pd.DataFrame(tradebook(s, uid)["data"])
    orders = pd.DataFrame(orderbook(s, uid)["data"]["orders"])

c = st.columns(4)
c[0].metric("Equity", ui.money(snap["equity"]))
c[1].metric("Total P&L", ui.money(snap["total_pnl"]), help="Equity − opening balance (top-ups included)")
c[2].metric("Cash", ui.money(snap["cash"]))
c[3].metric("Blocked", ui.money(snap["blocked"]))

tab_h, tab_p, tab_t, tab_o = st.tabs(["Holdings", "Positions", "Trades", "Orders"])
for tab, df, empty_text in ((tab_h, pd.DataFrame(h["holdings"]), "No holdings."),
                             (tab_p, pd.DataFrame(positions), "No intraday positions."),
                             (tab_t, trades, "No trades."),
                             (tab_o, orders, "No orders.")):
    with tab:
        if df.empty:
            st.caption(empty_text)
        else:
            st.dataframe(df, hide_index=True, width="stretch")
