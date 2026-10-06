"""
app.py - Entry point of the trader app (multipage).

Run from the project root (PowerShell); it opens on port 8501:
    .venv\\Scripts\\python.exe -m streamlit run app/trader/app.py

How it works: Streamlit runs THIS file on every click. It draws what every
page shares (style, banner, sidebar), decides which pages the visitor may
see, then runs the chosen page file from pages/:
    logged out -> only "Log in"
    logged in  -> Terminal, Orders, Trades, Positions, Holdings, Funds, Analytics, AI Insights
"""

import sys
from pathlib import Path

# Streamlit only puts this file's folder on the import path; add the project
# root so "from src... import" works here and in every page
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402  (must come after the path fix)

from src import ui  # noqa: E402
from src.auth import account_sidebar, block_wrong_role, db, session_user  # noqa: E402

st.set_page_config(page_title="Paper Trading", page_icon="📈", layout="wide")

user = session_user()
if user is None:
    page = st.navigation([st.Page("pages/login.py", title="Log in", icon=":material/login:")])
else:
    block_wrong_role(user, "user")  # admins use the admin app
    account_sidebar(user)
    page = st.navigation({
        "Trade": [
            st.Page("pages/terminal.py", title="Terminal", icon=":material/candlestick_chart:", default=True),
            st.Page("pages/orders.py", title="Orders", icon=":material/receipt_long:"),
            st.Page("pages/trades.py", title="Trades", icon=":material/swap_horiz:"),
        ],
        "Portfolio": [
            st.Page("pages/positions.py", title="Positions", icon=":material/show_chart:"),
            st.Page("pages/holdings.py", title="Holdings", icon=":material/inventory_2:"),
            st.Page("pages/funds.py", title="Funds", icon=":material/account_balance_wallet:"),
        ],
        "Insights": [
            st.Page("pages/analytics.py", title="Analytics", icon=":material/monitoring:"),
            st.Page("pages/ai_insights.py", title="AI Insights", icon=":material/psychology:"),
        ],
    })
    with db()() as s:
        ui.market_sidebar(s)

# Shared by every page: drawn here, just before the chosen page runs
ui.apply_style()
ui.banner()  # on every page, logged in or not
if user is not None:
    ui.show_flash()  # e.g. "Order cancelled", left by the previous click
page.run()
