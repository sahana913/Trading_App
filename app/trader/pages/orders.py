"""Orders: open orders (modify / cancel) and the full order book, with CSV download."""

import pandas as pd
import streamlit as st

from src import ui
from src.auth import current_user, db
from src.trading import cancelorder, modifyorder, orderbook

user = current_user()
st.title("Orders")

with db()() as s:
    clock = ui.require_clock(s)
    book = orderbook(s, user["id"])["data"]

    stats = book["statistics"]
    c = st.columns(4)
    c[0].metric("Open", stats["total_open_orders"])
    c[1].metric("Completed", stats["total_completed_orders"])
    c[2].metric("Cancelled", stats["total_cancelled_orders"])
    c[3].metric("Rejected", stats["total_rejected_orders"])

    orders = pd.DataFrame(book["orders"])
    if orders.empty:
        st.info("No orders yet. Place one in the Terminal.")
        st.stop()

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
                ui.report(modifyorder(s, user["id"], oid, quantity=int(new_qty), price=new_price,
                                      trigger_price=new_trigger, as_of=clock), f"Order {oid} modified")
        if col_cancel.button("Cancel order", key="cancel_order", width="stretch"):
            ui.report(cancelorder(s, user["id"], oid), f"Order {oid} cancelled")

orders["timestamp"] = pd.to_datetime(orders["timestamp"]).dt.date
columns = ["timestamp", "symbol", "action", "product", "pricetype", "quantity", "filled_quantity",
           "price", "trigger_price", "average_price", "order_status", "rejection_reason", "orderid"]

head, button = st.columns([4, 1], vertical_alignment="bottom")
head.subheader("Order book")
with button:
    ui.csv_download(orders[columns], "Download CSV", f"orders_{clock:%Y%m%d}.csv", key="orders_csv")
st.dataframe(
    orders[columns], hide_index=True, width="stretch",
    column_config={"timestamp": "Date", "filled_quantity": "Filled", "order_status": "Status",
                   "rejection_reason": "Note",
                   "average_price": st.column_config.NumberColumn("Avg price", format="%.2f")},
)
