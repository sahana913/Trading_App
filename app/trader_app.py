"""
trader_app.py - The trader's Streamlit app (trading screens come in a later step).

Run from the project root (PowerShell):
    .venv\\Scripts\\python.exe -m streamlit run app/trader_app.py
"""

import sys
from pathlib import Path

# Streamlit only puts app/ on the import path; add the project root so
# "from src... import" works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402  (must come after the path fix)

from src.auth import require_role  # noqa: E402

st.set_page_config(page_title="Paper Trading", page_icon="📈")

user = require_role("user", "Paper Trading", allow_register=True)

st.title(f"Welcome, {user['username']}")
st.info("You are logged in. Trading screens arrive in a later step.")
