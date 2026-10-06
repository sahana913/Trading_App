"""Instruments: add, edit and remove what traders can trade."""

import streamlit as st

from src import ui
from src.admin.instruments import add_instrument, instruments_frame, remove_instrument, update_instrument
from src.auth import current_user, db

admin = current_user()
st.title("Instruments")

with db()() as s:
    table = instruments_frame(s)
st.dataframe(table.drop(columns=["id"]), hide_index=True, width="stretch",
             column_config={"active": st.column_config.CheckboxColumn("Active")})

add_tab, edit_tab, remove_tab = st.tabs(["Add", "Edit", "Remove"])

with add_tab, st.form("add_instrument"):
    c = st.columns(3)
    symbol = c[0].text_input("Symbol", key="new_symbol", placeholder="e.g. WIPRO")
    exchange = c[1].text_input("Exchange", value="NSE", key="new_exchange")
    name = c[2].text_input("Name", key="new_name")
    c = st.columns(3)
    lot = c[0].number_input("Lot size", min_value=1, value=1, step=1, key="new_lot")
    tick = c[1].number_input("Tick size", min_value=0.01, value=0.05, step=0.01, format="%.2f", key="new_tick")
    price = c[2].number_input("Start price (optional)", min_value=0.0, value=0.0, step=1.0, key="new_price",
                              help="Creates a first price at the market date so it can be traded now. "
                                   "Leave 0 if you will load real data for it.")
    if st.form_submit_button("Add instrument", type="primary", key="add_submit"):
        with db()() as s:
            ui.report(add_instrument(s, admin["id"], symbol, exchange, name, int(lot), float(tick),
                                     float(price) if price > 0 else None), f"Added {symbol.upper()}")

if table.empty:
    st.stop()
labels = {int(r.id): f"{r.symbol} ({r.exchange})" for r in table.itertuples()}

with edit_tab:
    iid = st.selectbox("Instrument", list(labels), format_func=labels.get, key="edit_pick")
    row = table.set_index("id").loc[iid]
    with st.form("edit_instrument"):
        c = st.columns(4)
        name = c[0].text_input("Name", value=row["name"] or "", key="edit_name")
        lot = c[1].number_input("Lot size", min_value=1, value=int(row["lot_size"]), step=1, key="edit_lot")
        tick = c[2].number_input("Tick size", min_value=0.01, value=float(row["tick_size"]), step=0.01,
                                 format="%.2f", key="edit_tick")
        active = c[3].checkbox("Active (tradable)", value=bool(row["active"]), key="edit_active")
        if st.form_submit_button("Save changes", key="edit_submit"):
            with db()() as s:
                ui.report(update_instrument(s, admin["id"], iid, name=name, lot_size=int(lot),
                                            tick_size=float(tick), is_active=active), f"Saved {labels[iid]}")

with remove_tab:
    iid = st.selectbox("Instrument", list(labels), format_func=labels.get, key="remove_pick")
    st.caption("An instrument with prices, orders or holdings is only deactivated (no new orders), so its "
               "history stays intact. An unused one is deleted.")
    if st.button("Remove", key="remove_submit", type="primary"):
        with db()() as s:
            result = remove_instrument(s, admin["id"], iid)
        ui.report(result, f"{labels[iid]} {result.get('data', {}).get('outcome', '')}")
