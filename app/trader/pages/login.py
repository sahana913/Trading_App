"""Log in / register. After success, app.py re-runs and shows the trading pages."""

from src.auth import login_page

login_page("Paper Trading", allow_register=True)
