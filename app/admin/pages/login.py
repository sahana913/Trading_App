"""Admin log in (no registration: admin accounts come from the seed command).
Successful admin logins are written to the audit log."""

from src.admin.audit import log_action
from src.auth import login_page


def record_login(session, user) -> None:
    if user.role == "admin":  # a trader trying this app is blocked right after
        log_action(session, user.id, "login")
        session.commit()


login_page("Admin Dashboard", allow_register=False, on_login=record_login)
