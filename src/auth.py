"""
auth.py - Register, log in, log out and role checks for both Streamlit apps.

Two layers:
  1. Plain functions (register_user, authenticate, has_role) that only need a
     database session. These hold all the rules and are what the tests check.
  2. Streamlit helpers (require_role, login_page, logout) that draw the forms
     and remember who is logged in using st.session_state.

Single-page app: one line at the top:
    user = require_role("admin", "Admin Dashboard")
If nobody is logged in, the login form is shown and the rest of the page is
not run (st.stop()). Otherwise `user` is {"id", "username", "role"}.

Multipage app (app/trader/app.py): call session_user() and offer only the
login page while it returns None.
"""

import os
import re
from functools import cache

import streamlit as st
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from src.config import DB_URL
from src.db.models import ROLES, User
from src.db.session import get_engine, get_session_factory, init_db
from src.security import MAX_PASSWORD_BYTES, hash_password, needs_rehash, verify_password
from src.trading.accounts import create_user

USERNAME_PATTERN = re.compile(r"^[a-z0-9_]{3,30}$")  # 3-30 lower-case letters, digits, _
MIN_PASSWORD_LENGTH = 8
SESSION_KEY = "auth_user"  # where the logged-in user lives in st.session_state

# Used when the username doesn't exist, so a failed login takes as long as a
# real password check. Otherwise the response time would reveal which
# usernames exist.
_DUMMY_HASH = hash_password("not-a-real-password")  # made once, when the module loads
LOGIN_FAILED = "Invalid username or password"


# ---------------------------------------------------------------------------
# 1. Rules (no Streamlit)
# ---------------------------------------------------------------------------
def normalise_username(username: str) -> str:
    """'  Alice ' -> 'alice', so names are unique regardless of case."""
    return username.strip().lower()


def check_new_credentials(username: str, password: str, confirm: str) -> str | None:
    """Return an error message for a bad sign-up, or None if it is fine."""
    if not USERNAME_PATTERN.match(username):
        return "Username must be 3-30 characters: letters, digits or _"
    if len(password) < MIN_PASSWORD_LENGTH:
        return f"Password must be at least {MIN_PASSWORD_LENGTH} characters"
    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        return f"Password must be at most {MAX_PASSWORD_BYTES} bytes"
    if password != confirm:
        return "Passwords do not match"
    return None


def register_user(session: Session, username: str, password: str,
                  confirm: str) -> tuple[User | None, str | None]:
    """Create a normal (role "user") account with ₹10,00,000 virtual cash.

    Returns (user, None) on success or (None, error message).
    Admins can't be created here; they come from the seed command.
    """
    username = normalise_username(username)
    problem = check_new_credentials(username, password, confirm)
    if problem:
        return None, problem
    if session.scalar(select(User).where(func.lower(User.username) == username)):
        return None, "That username is taken"
    return create_user(session, username, password, role="user"), None


def authenticate(session: Session, username: str, password: str) -> tuple[User | None, str | None]:
    """Check a login. Returns (user, None) or (None, error message)."""
    username = normalise_username(username)
    user = session.scalar(select(User).where(func.lower(User.username) == username))

    if user is None:
        verify_password(password, _DUMMY_HASH)  # same time cost as a real check
        return None, LOGIN_FAILED
    if not verify_password(password, user.password_hash):
        return None, LOGIN_FAILED  # same message: don't reveal the user exists
    # Only after a correct password, so this doesn't reveal disabled accounts to guessers
    if not user.is_active:
        return None, "This account is disabled. Contact an administrator."

    # Old PBKDF2 hash (or old bcrypt cost)? We know the password right now,
    # so store a fresh bcrypt hash.
    if needs_rehash(user.password_hash):
        user.password_hash = hash_password(password)
        session.commit()
    return user, None


def has_role(user_role: str, required: str) -> bool:
    """True if the user's role matches the page's required role.

    Roles are kept separate on purpose: admins manage the platform from the
    admin app and don't trade in the trader app.
    """
    if required not in ROLES:
        raise ValueError(f"Unknown role '{required}'")
    return user_role == required


# ---------------------------------------------------------------------------
# 2. Streamlit helpers
# ---------------------------------------------------------------------------
@cache
def _session_factory(url: str) -> sessionmaker[Session]:
    """One engine per database URL for the whole process (not per page refresh)."""
    engine = get_engine(url)
    init_db(engine)
    return get_session_factory(engine)


def db() -> sessionmaker[Session]:
    """Session factory for the app's database. Use: `with db()() as s: ...`.
    The PAPER_TRADING_DB_URL environment variable overrides the database
    (the tests use it to point at a temporary file)."""
    return _session_factory(os.environ.get("PAPER_TRADING_DB_URL", DB_URL))


def current_user() -> dict | None:
    return st.session_state.get(SESSION_KEY)


def _remember(user: User) -> None:
    # Store plain values, not the User object: database objects can't be
    # safely reused after their session closes
    st.session_state[SESSION_KEY] = {"id": user.id, "username": user.username, "role": user.role}


def logout() -> None:
    st.session_state.pop(SESSION_KEY, None)


def _still_valid(info: dict) -> bool:
    """Re-check the database on every page run, so an admin disabling an
    account (or changing its role) takes effect immediately."""
    with db()() as s:
        user = s.get(User, info["id"])
        return user is not None and user.is_active and user.role == info["role"]


def login_page(title: str, allow_register: bool, on_login=None) -> None:
    """Draw the login form (and optionally a register tab).
    on_login(session, user), if given, runs after a successful login (the
    admin app uses it to write the login to the audit log)."""
    st.title(title)
    tabs = st.tabs(["Log in", "Register"] if allow_register else ["Log in"])

    with tabs[0], st.form("login_form"):
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.form_submit_button("Log in", key="login_submit"):
            with db()() as s:
                user, problem = authenticate(s, username, password)
                if problem:
                    st.error(problem)
                else:
                    if on_login is not None:
                        on_login(s, user)
                    _remember(user)
                    st.rerun()  # draw the page again, now logged in

    if allow_register:
        with tabs[1], st.form("register_form"):
            username = st.text_input("Choose a username", key="reg_username")
            password = st.text_input("Password", type="password", key="reg_password")
            confirm = st.text_input("Confirm password", type="password", key="reg_confirm")
            if st.form_submit_button("Create account", key="reg_submit"):
                with db()() as s:
                    user, problem = register_user(s, username, password, confirm)
                    if problem:
                        st.error(problem)
                    else:
                        _remember(user)  # log straight in after registering
                        st.rerun()


def session_user() -> dict | None:
    """The logged-in user, re-checked against the database; None if logged out."""
    info = current_user()
    if info is not None and not _still_valid(info):
        logout()
        st.warning("Your session ended because your account changed. Please log in again.")
        info = None
    return info


def block_wrong_role(info: dict, role: str) -> None:
    """Stop the page if the user's role doesn't match (with a way to log out)."""
    if not has_role(info["role"], role):
        st.error(f"This app is for {role} accounts only. You are logged in as a {info['role']}.")
        st.button("Log out", on_click=logout, key="wrong_role_logout")
        st.stop()


def account_sidebar(info: dict) -> None:
    with st.sidebar:
        st.write(f"Signed in as **{info['username']}** ({info['role']})")
        st.button("Log out", on_click=logout, key="logout")


def require_role(role: str, title: str, allow_register: bool = False) -> dict:
    """Gatekeeper for a single-page app. Returns the logged-in user or stops the page."""
    info = session_user()
    if info is None:
        login_page(title, allow_register)
        st.stop()  # nothing below require_role() runs until someone logs in
    block_wrong_role(info, role)
    account_sidebar(info)
    return info
