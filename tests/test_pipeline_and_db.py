"""End-to-end checks: raw CSV -> parquet -> seeded database.

Everything runs in pytest's tmp_path folder and an in-memory database,
so the real data/ and db/ folders are never touched.
"""

import pandas as pd
import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from src.data.pipeline import run_pipeline
from src.db.models import AdminLog, Candle, Instrument, Order, User
from src.db.seed import seed
from src.db.session import get_engine, get_session_factory, init_db
from src.security import hash_password, verify_password


@pytest.fixture
def session():
    engine = get_engine("sqlite:///:memory:")
    init_db(engine)
    with get_session_factory(engine)() as s:
        yield s


def test_pipeline_writes_clean_parquet(tmp_path, raw_candles):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    # Add one duplicate so we can see it removed
    pd.concat([raw_candles, raw_candles.iloc[[0]]]).to_csv(raw_dir / "prices.csv", index=False)
    out = tmp_path / "processed" / "candles.parquet"

    clean = run_pipeline(raw_dir, out)

    assert out.exists()
    assert len(clean) == 6
    pd.testing.assert_frame_equal(pd.read_parquet(out), clean)


def test_password_hashing():
    stored = hash_password("secret123")
    assert "secret123" not in stored
    assert verify_password("secret123", stored)
    assert not verify_password("wrong", stored)
    assert hash_password("secret123") != stored  # new random salt each time


def test_seed_loads_data_and_is_safe_to_rerun(session, raw_candles):
    from src.data.clean import clean_candles

    candles, _ = clean_candles(raw_candles)

    first = seed(session, candles, "admin", "pw")
    second = seed(session, candles, "admin", "pw")  # second run must not duplicate
    session.commit()

    assert first == {"admin_created": True, "instruments": 2, "candles": 6}
    assert second["admin_created"] is False
    assert session.scalar(select(func.count()).select_from(Instrument)) == 2
    assert session.scalar(select(func.count()).select_from(Candle)) == 6
    assert session.scalar(select(func.count()).select_from(AdminLog)) == 2

    admin = session.scalar(select(User).where(User.username == "admin"))
    assert admin.role == "admin" and admin.is_active
    assert verify_password("pw", admin.password_hash)


def test_database_rejects_invalid_values(session):
    session.add(User(username="bob", password_hash="x", role="superuser"))
    with pytest.raises(IntegrityError):
        session.commit()  # role must be 'user' or 'admin'


def test_foreign_keys_are_enforced(session):
    session.add(Order(orderid="1", user_id=999, instrument_id=999, action="BUY",
                      pricetype="MARKET", product="CNC", quantity=1))
    with pytest.raises(IntegrityError):
        session.commit()  # user 999 and instrument 999 don't exist
