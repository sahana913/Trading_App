"""
session.py - Create the database engine, the tables, and sessions.
"""

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from src.config import DB_PATH, DB_URL
from src.db.models import Base


def get_engine(url: str = DB_URL) -> Engine:
    """Return an engine (the object that talks to the database file)."""
    if url == DB_URL:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)  # make sure db/ exists
    engine = create_engine(url)

    # SQLite ignores FOREIGN KEY rules unless this is switched on for every
    # new connection. Without it, an order could point at a user that doesn't exist.
    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return engine


def init_db(engine: Engine) -> None:
    """Create any tables that don't exist yet (existing tables are left alone)."""
    Base.metadata.create_all(engine)


def get_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Return a factory; call it to get a session: `with factory() as s: ...`."""
    return sessionmaker(bind=engine)
