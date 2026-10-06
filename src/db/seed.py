"""
seed.py - Fill the database from data/processed/candles.parquet and create
one admin user. Safe to run again: instruments and the admin are reused, and
each instrument's candles are replaced rather than duplicated.

Run from the project root (PowerShell), AFTER the pipeline:
    $env:ADMIN_PASSWORD = "choose-a-password"
    .venv\\Scripts\\python.exe -m src.db.seed
    # optional: --admin-username someone   (default: admin)

If ADMIN_PASSWORD is not set, you are asked to type the password.
"""

import argparse
import getpass
import os

import pandas as pd
from sqlalchemy import delete, insert, select
from sqlalchemy.orm import Session

from src.config import CANDLES_PARQUET
from src.data.validate import validate_candles
from src.db.models import AdminLog, Candle, Instrument, User
from src.db.session import get_engine, get_session_factory, init_db
from src.security import hash_password


def get_or_create_instrument(session: Session, symbol: str, exchange: str) -> Instrument:
    """Return the existing instrument row, or add a new one."""
    inst = session.scalar(
        select(Instrument).where(Instrument.symbol == symbol, Instrument.exchange == exchange)
    )
    if inst is None:
        inst = Instrument(symbol=symbol, exchange=exchange, name=symbol)
        session.add(inst)
        session.flush()  # sends the INSERT now so inst.id is filled in
    return inst


def load_candles(session: Session, candles: pd.DataFrame) -> tuple[int, int]:
    """Insert instruments and candles. Returns (instrument count, candle count)."""
    n_candles = 0
    groups = candles.groupby(["symbol", "exchange"], sort=True)
    for (symbol, exchange), bars in groups:
        inst = get_or_create_instrument(session, symbol, exchange)

        # Replace this instrument's candles so re-running never duplicates them
        session.execute(delete(Candle).where(Candle.instrument_id == inst.id))

        rows = bars[["timestamp", "open", "high", "low", "close", "volume"]].to_dict("records")
        for row in rows:
            row["instrument_id"] = inst.id
            row["timestamp"] = row["timestamp"].to_pydatetime()  # pandas -> plain datetime
        # One bulk INSERT per instrument is far faster than adding objects one by one
        session.execute(insert(Candle), rows)
        n_candles += len(rows)
    return groups.ngroups, n_candles


def ensure_admin(session: Session, username: str, password: str) -> tuple[User, bool]:
    """Create the admin user if missing. Returns (user, created?)."""
    user = session.scalar(select(User).where(User.username == username))
    if user is not None:
        if user.role != "admin":
            raise ValueError(f"User '{username}' exists but is not an admin")
        return user, False
    user = User(username=username, password_hash=hash_password(password), role="admin", is_active=True)
    session.add(user)
    session.flush()
    return user, True


def seed(session: Session, candles: pd.DataFrame, admin_username: str, admin_password: str) -> dict:
    """Run the whole seed inside the caller's session. Returns a summary dict."""
    validate_candles(candles)  # refuse to load data that isn't clean
    admin, created = ensure_admin(session, admin_username, admin_password)
    n_instruments, n_candles = load_candles(session, candles)
    summary = {
        "admin_created": created,
        "instruments": n_instruments,
        "candles": n_candles,
    }
    session.add(AdminLog(admin_user_id=admin.id, action="seed", details=summary))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the paper-trading database.")
    parser.add_argument("--admin-username", default="admin")
    args = parser.parse_args()

    if not CANDLES_PARQUET.exists():
        raise SystemExit(f"{CANDLES_PARQUET} not found. Run: python -m src.data.pipeline")
    candles = pd.read_parquet(CANDLES_PARQUET)

    engine = get_engine()
    init_db(engine)
    factory = get_session_factory(engine)

    with factory() as session:
        exists = session.scalar(select(User).where(User.username == args.admin_username))
        # Only ask for a password if we actually need to create the admin
        password = ""
        if exists is None:
            password = os.environ.get("ADMIN_PASSWORD") or getpass.getpass("Admin password: ")
        summary = seed(session, candles, args.admin_username, password)
        session.commit()  # everything is saved together, or nothing is

    print(f"Seed complete: {summary}")


if __name__ == "__main__":
    main()
