"""
main.py - PaperDesk: ONE app, ONE address, for traders and admins.

Run from the project root (PowerShell); it opens on http://localhost:8501:
    .venv\\Scripts\\python.exe -m streamlit run app/main.py
(or simply: powershell -ExecutionPolicy Bypass -File .\\run.ps1)

How it works: Streamlit runs THIS file on every click. It decides which pages
the visitor may see from who is logged in, draws what every page shares, then
runs the chosen page file:

    logged out  -> Log in / Register                    (login.py)
    trader      -> Terminal, Orders, Trades, Positions,  (trader/pages/*.py)
                   Holdings, Funds, Analytics, AI Insights
    admin       -> Overview, Users, Leaderboard, Market   (admin/pages/*.py)
                   control, Instruments, ML Ops, Audit log

A page that isn't in the visitor's menu doesn't exist for them: a trader can't
reach an admin page even by typing its address. On top of that, every admin
action in src/admin/ re-checks the role and writes the audit log.
"""

import os
import sys
from pathlib import Path

# Streamlit only puts this file's folder on the import path; add the project
# root so "from src... import" works here and in every page
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402  (must come after the path fix)

from src import ui  # noqa: E402
from src.admin.ticker import start_ticker  # noqa: E402
from src.auth import account_sidebar, db, session_user  # noqa: E402
from src.config import DB_URL  # noqa: E402

ASSETS = Path(__file__).resolve().parent / "assets"
TRADER_PAGES = {
    "Trade": [
        st.Page("trader/pages/terminal.py", title="Terminal", icon=":material/candlestick_chart:", default=True),
        st.Page("trader/pages/orders.py", title="Orders", icon=":material/receipt_long:"),
        st.Page("trader/pages/trades.py", title="Trades", icon=":material/swap_horiz:"),
    ],
    "Portfolio": [
        st.Page("trader/pages/positions.py", title="Positions", icon=":material/show_chart:"),
        st.Page("trader/pages/holdings.py", title="Holdings", icon=":material/inventory_2:"),
        st.Page("trader/pages/funds.py", title="Funds", icon=":material/account_balance_wallet:"),
    ],
    "Insights": [
        st.Page("trader/pages/analytics.py", title="Analytics", icon=":material/monitoring:"),
        st.Page("trader/pages/ai_insights.py", title="AI Insights", icon=":material/psychology:"),
    ],
}
ADMIN_PAGES = {
    "Platform": [
        st.Page("admin/pages/overview.py", title="Overview", icon=":material/dashboard:", default=True),
        st.Page("admin/pages/users.py", title="Users", icon=":material/group:"),
        st.Page("admin/pages/leaderboard.py", title="Leaderboard", icon=":material/emoji_events:"),
    ],
    "Operations": [
        st.Page("admin/pages/market.py", title="Market control", icon=":material/tune:"),
        st.Page("admin/pages/instruments.py", title="Instruments", icon=":material/list_alt:"),
        st.Page("admin/pages/mlops.py", title="ML Ops", icon=":material/model_training:"),
    ],
    "Governance": [
        st.Page("admin/pages/audit.py", title="Audit log", icon=":material/history:"),
    ],
}


@st.cache_resource
def market_heartbeat(db_url: str):
    """Start the background thread that advances a running market, ONCE per
    server process (cache_resource) and database. See src/admin/ticker.py."""
    return start_ticker(db())


if os.environ.get("PAPER_TRADING_TICKER", "on") == "on":  # tests (and run.ps1's old simulator window) switch it off
    market_heartbeat(os.environ.get("PAPER_TRADING_DB_URL", DB_URL))

user = session_user()
is_admin = user is not None and user["role"] == "admin"

st.set_page_config(page_title="PaperDesk Admin" if is_admin else "PaperDesk", layout="wide",
                   page_icon=str(ASSETS / ("logo_icon_admin.svg" if is_admin else "logo_icon.svg")))
st.logo(str(ASSETS / ("logo_admin.svg" if is_admin else "logo.svg")), size="large",
        icon_image=str(ASSETS / ("logo_icon_admin.svg" if is_admin else "logo_icon.svg")))

if user is None:
    page = st.navigation([st.Page("login.py", title="Log in or register", icon=":material/login:")])
elif is_admin:
    page = st.navigation(ADMIN_PAGES)
    with st.sidebar:
        ui.live_clock()               # live market card
    account_sidebar(user)             # signed-in name + Log out
else:
    page = st.navigation(TRADER_PAGES)
    with db()() as s:
        ui.market_sidebar(s)          # market card + clock buttons
    with st.sidebar:
        ui.account_card(user["id"])   # equity, cash, P&L (live)
    account_sidebar(user)

# Shared by every page: drawn here, just before the chosen page runs.
# Admin pages wear violet, trading pages saffron, so you always know where you are.
ui.apply_style("admin" if is_admin else "trader")
ui.banner()
if user is not None:
    ui.show_flash()  # e.g. "Order cancelled", left by the previous click
page.run()
