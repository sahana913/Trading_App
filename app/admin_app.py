"""
admin_app.py - The administrator's Streamlit app (dashboard comes in a later step).
Only "admin" accounts get in; there is no self-registration here.

Run from the project root (PowerShell):
    .venv\\Scripts\\python.exe -m streamlit run app/admin_app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # see trader_app.py

import streamlit as st  # noqa: E402

from src.auth import require_role  # noqa: E402

st.set_page_config(page_title="Admin Dashboard", page_icon="🛠️")

user = require_role("admin", "Admin Dashboard")

st.title(f"Admin: {user['username']}")
st.info("You are logged in as an administrator. The dashboard arrives in a later step.")
