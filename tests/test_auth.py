"""Tests for src/security.py (hashing) and src/auth.py (register, login, roles).

The last section drives the real Streamlit apps with Streamlit's AppTest,
against a temporary database file.
"""

import hashlib
from pathlib import Path

import pytest
from sqlalchemy import select
from streamlit.testing.v1 import AppTest

from src import security
from src.auth import LOGIN_FAILED, authenticate, has_role, register_user
from src.db.models import Fund, User
from src.db.session import get_engine, get_session_factory, init_db
from src.security import hash_password, needs_rehash, verify_password
from src.trading.accounts import create_user

APP_DIR = Path(__file__).resolve().parent.parent / "app"
TRADER_APP = str(APP_DIR / "trader_app.py")
ADMIN_APP = str(APP_DIR / "admin_app.py")


@pytest.fixture
def session():
    engine = get_engine("sqlite:///:memory:")
    init_db(engine)
    with get_session_factory(engine)() as s:
        yield s


def legacy_pbkdf2_hash(password: str) -> str:
    """A hash in the format the project used before bcrypt."""
    salt = b"0123456789abcdef"
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 1000)
    return f"pbkdf2_sha256$1000${salt.hex()}${digest.hex()}"


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------
def test_bcrypt_hash_and_verify():
    stored = hash_password("correct horse")
    assert stored.startswith("$2b$")          # bcrypt format
    assert "correct horse" not in stored      # the password itself is never stored
    assert verify_password("correct horse", stored)
    assert not verify_password("wrong horse", stored)


def test_same_password_gives_different_hashes():
    assert hash_password("same-password") != hash_password("same-password")  # random salt


def test_empty_or_too_long_password_cannot_be_hashed():
    with pytest.raises(ValueError):
        hash_password("")
    with pytest.raises(ValueError):
        hash_password("x" * 73)  # bcrypt only uses 72 bytes


def test_legacy_pbkdf2_hash_still_verifies_and_needs_rehash():
    old = legacy_pbkdf2_hash("old-password")
    assert verify_password("old-password", old)
    assert not verify_password("nope", old)
    assert needs_rehash(old)
    assert not needs_rehash(hash_password("x"))


def test_needs_rehash_when_cost_changes(monkeypatch):
    stored = hash_password("pw")  # cost 4 in tests
    monkeypatch.setattr(security, "BCRYPT_ROUNDS", 5)
    assert needs_rehash(stored)


def test_unknown_hash_format_never_matches():
    assert not verify_password("anything", "plain-text-password")


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------
def test_register_creates_user_with_starting_cash(session):
    user, problem = register_user(session, "  Alice ", "password1", "password1")
    assert problem is None
    assert user.username == "alice"  # trimmed and lower-cased
    assert user.role == "user"       # self-registration never makes admins
    assert user.is_active
    fund = session.scalar(select(Fund).where(Fund.user_id == user.id))
    assert fund.available_cash == 1_000_000


@pytest.mark.parametrize(
    "username, password, confirm, message",
    [
        ("al", "password1", "password1", "Username"),
        ("alice smith", "password1", "password1", "Username"),
        ("alice!", "password1", "password1", "Username"),
        ("alice", "short", "short", "at least 8"),
        ("alice", "password1", "password2", "do not match"),
        ("alice", "x" * 73, "x" * 73, "at most 72"),
    ],
)
def test_register_rejects_bad_input(session, username, password, confirm, message):
    user, problem = register_user(session, username, password, confirm)
    assert user is None
    assert message in problem


def test_register_rejects_taken_username_in_any_case(session):
    register_user(session, "alice", "password1", "password1")
    user, problem = register_user(session, "ALICE", "password1", "password1")
    assert user is None and "taken" in problem


# ---------------------------------------------------------------------------
# Log in
# ---------------------------------------------------------------------------
def test_login_success_is_case_insensitive(session):
    register_user(session, "alice", "password1", "password1")
    user, problem = authenticate(session, "Alice", "password1")
    assert problem is None and user.username == "alice"


def test_wrong_password_and_unknown_user_get_the_same_message(session):
    register_user(session, "alice", "password1", "password1")
    assert authenticate(session, "alice", "wrong-pass") == (None, LOGIN_FAILED)
    assert authenticate(session, "nobody", "password1") == (None, LOGIN_FAILED)


def test_disabled_user_cannot_log_in(session):
    user, _ = register_user(session, "alice", "password1", "password1")
    user.is_active = False
    session.commit()

    _, problem = authenticate(session, "alice", "password1")
    assert "disabled" in problem
    # A wrong password still gets the generic message, so guessers learn nothing
    assert authenticate(session, "alice", "wrong-pass") == (None, LOGIN_FAILED)


def test_login_upgrades_legacy_hash_to_bcrypt(session):
    session.add(User(username="admin", password_hash=legacy_pbkdf2_hash("old-password"),
                     role="admin", is_active=True))
    session.commit()

    user, problem = authenticate(session, "admin", "old-password")
    assert problem is None
    assert user.password_hash.startswith("$2b$")
    assert authenticate(session, "admin", "old-password")[1] is None  # still works


# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------
def test_role_checks():
    assert has_role("admin", "admin")
    assert has_role("user", "user")
    assert not has_role("user", "admin")   # traders can't open the admin app
    assert not has_role("admin", "user")   # admins don't trade
    with pytest.raises(ValueError):
        has_role("user", "superuser")


# ---------------------------------------------------------------------------
# The real Streamlit apps (AppTest runs a script without a browser)
# ---------------------------------------------------------------------------
@pytest.fixture
def app_db(tmp_path, monkeypatch):
    """A temporary database with one trader and one admin, used by the apps."""
    url = f"sqlite:///{(tmp_path / 'app.db').as_posix()}"
    monkeypatch.setenv("PAPER_TRADING_DB_URL", url)
    engine = get_engine(url)
    init_db(engine)
    with get_session_factory(engine)() as s:
        create_user(s, "alice", "password1")
        create_user(s, "boss", "password1", role="admin")
    return get_session_factory(engine)


def log_in(at: AppTest, username: str, password: str) -> AppTest:
    at.text_input(key="login_username").input(username)
    at.text_input(key="login_password").input(password)
    at.button(key="login_submit").click()
    return at.run(timeout=30)


def test_trader_app_login_and_logout(app_db):
    at = AppTest.from_file(TRADER_APP).run(timeout=30)
    assert at.title[0].value == "Paper Trading"  # login page

    log_in(at, "alice", "password1")
    assert "Signed in as **alice**" in at.sidebar.markdown[0].value

    at.sidebar.button(key="logout").click()
    at.run(timeout=30)
    assert at.title[0].value == "Paper Trading"  # back to the login page


def test_trader_app_wrong_password_shows_error(app_db):
    at = log_in(AppTest.from_file(TRADER_APP).run(timeout=30), "alice", "wrong-pass")
    assert at.error[0].value == LOGIN_FAILED


def test_trader_app_register_logs_straight_in(app_db):
    at = AppTest.from_file(TRADER_APP).run(timeout=30)
    at.text_input(key="reg_username").input("carol")
    at.text_input(key="reg_password").input("password1")
    at.text_input(key="reg_confirm").input("password1")
    at.button(key="reg_submit").click()
    at.run(timeout=30)
    assert "Signed in as **carol**" in at.sidebar.markdown[0].value


def test_admin_app_blocks_traders_and_admits_admins(app_db):
    at = log_in(AppTest.from_file(ADMIN_APP).run(timeout=30), "alice", "password1")
    assert "admin accounts only" in at.error[0].value

    at = log_in(AppTest.from_file(ADMIN_APP).run(timeout=30), "boss", "password1")
    assert at.title[0].value == "Admin: boss"


def test_disabling_a_user_ends_their_session(app_db):
    at = log_in(AppTest.from_file(TRADER_APP).run(timeout=30), "alice", "password1")
    assert "Signed in as **alice**" in at.sidebar.markdown[0].value

    with app_db() as s:  # an admin disables alice while she is logged in
        s.scalar(select(User).where(User.username == "alice")).is_active = False
        s.commit()

    at.run(timeout=30)  # her next click
    assert "session ended" in at.warning[0].value
    assert at.title[0].value == "Paper Trading"
