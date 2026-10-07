"""Log in or register: one page for everyone. Traders and admins log in here;
new traders register here. After success, main.py re-runs and shows the pages
for that person's role. Admin logins are written to the audit log."""

import streamlit as st

from src import ui
from src.admin.audit import log_action
from src.auth import login_page


def record_login(session, user) -> None:
    """Audit trail: every admin login is recorded (trader logins are not)."""
    if user.role == "admin":
        log_action(session, user.id, "login")
        session.commit()


hero, form = st.columns([1.15, 1], gap="large")
with hero:
    ui.login_hero("Trade the NSE with ₹10,00,000 of virtual cash",
                  "Real 2022–2025 prices from 10 large NSE stocks, replayed in 5-minute steps. "
                  "Educational only, not financial advice.",
                  ["Market, limit and stop-loss orders", "Live P&L, VaR and drawdown",
                   "AI signals that explain themselves"])
with form:
    login_page(ui.BRAND, allow_register=True, on_login=record_login)
    st.caption("**New here?** Open the **Register** tab above to create a trading account with "
               "₹10,00,000 of virtual cash. Admins log in here too; the admin pages open after login.")
