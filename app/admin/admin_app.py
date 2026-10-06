"""
admin_app.py - Entry point of the ADMIN app (multipage, admin accounts only).

Run from the project root (PowerShell). Port 8502, next to the trader app on 8501:
    .venv\\Scripts\\python.exe -m streamlit run app/admin/admin_app.py --server.port 8502

Access:
    logged out          -> only the Log in page exists
    logged in, trader   -> blocked with an error (no admin pages are registered)
    logged in, admin    -> all pages below
Every change an admin makes goes through src/admin/, which re-checks the
role and writes the audit log.
"""

import os
import sys
from pathlib import Path

# Streamlit only puts this file's folder on the import path; add the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from src import ui  # noqa: E402
from src.admin.ticker import start_ticker  # noqa: E402
from src.auth import account_sidebar, block_wrong_role, db, session_user  # noqa: E402
from src.config import DB_URL  # noqa: E402
from src.trading.simulator import settings, get_clock  # noqa: E402

st.set_page_config(page_title="Admin Dashboard", page_icon="🛠️", layout="wide")


@st.cache_resource
def market_heartbeat(db_url: str):
    """Start the background thread that advances a running market, ONCE per
    server process (cache_resource) and database. See src/admin/ticker.py."""
    return start_ticker(db())


user = session_user()
if user is None:
    page = st.navigation([st.Page("pages/login.py", title="Log in", icon=":material/login:")])
else:
    block_wrong_role(user, "admin")  # traders stop here: no admin pages are ever registered
    account_sidebar(user)
    if os.environ.get("PAPER_TRADING_TICKER", "on") == "on":  # tests switch it off
        market_heartbeat(os.environ.get("PAPER_TRADING_DB_URL", DB_URL))
    page = st.navigation({
        "Platform": [
            st.Page("pages/overview.py", title="Overview", icon=":material/dashboard:", default=True),
            st.Page("pages/users.py", title="Users", icon=":material/group:"),
            st.Page("pages/leaderboard.py", title="Leaderboard", icon=":material/emoji_events:"),
        ],
        "Operations": [
            st.Page("pages/market.py", title="Market control", icon=":material/tune:"),
            st.Page("pages/instruments.py", title="Instruments", icon=":material/list_alt:"),
            st.Page("pages/mlops.py", title="ML Ops", icon=":material/model_training:"),
        ],
        "Governance": [
            st.Page("pages/audit.py", title="Audit log", icon=":material/history:"),
        ],
    })
    with db()() as s, st.sidebar:
        clock = get_clock(s)
        cfg = settings(s)
        st.subheader("Market")
        st.write(f"**{clock:%d %b %Y}**" if clock else "Not started")
        st.caption(f"{'▶ running' if cfg['is_running'] else '⏸ paused'} · {cfg['mode']} mode")

# Shared by every page: drawn here, just before the chosen page runs
ui.apply_style()
ui.banner()
if user is not None:
    ui.show_flash()
page.run()
