"""Audit log: every admin action, newest first, with CSV download."""

import json

import streamlit as st

from src import ui
from src.admin.audit import audit_frame
from src.auth import db

st.title("Audit log")

with db()() as s:
    everything = audit_frame(s)

if everything.empty:
    st.info("No admin actions recorded yet.")
    st.stop()

choices = ["All actions", *sorted(everything["action"].unique())]
picked = st.selectbox("Action", choices, key="audit_filter")
shown = everything if picked == "All actions" else everything[everything["action"] == picked]
shown = shown.assign(details=shown["details"].map(lambda d: json.dumps(d, default=str) if d else ""))

head, button = st.columns([4, 1], vertical_alignment="bottom")
head.caption(f"{len(shown)} of {len(everything)} entries. Rows are only ever added, never edited.")
with button:
    ui.csv_download(shown, "Download CSV", "audit_log.csv", key="audit_csv")
st.dataframe(shown, hide_index=True, width="stretch",
             column_config={"time": st.column_config.DatetimeColumn("Time", format="YYYY-MM-DD HH:mm:ss"),
                            "target_user": "Target user"})
