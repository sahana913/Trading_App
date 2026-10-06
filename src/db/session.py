"""
session.py - Create the database engine, the tables, and sessions.
"""

from sqlalchemy import Engine, create_engine, event, inspect, text
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
    """Create missing tables, then add any columns that newer code expects."""
    Base.metadata.create_all(engine)  # only creates tables that don't exist yet
    add_missing_columns(engine)


def add_missing_columns(engine: Engine) -> list[str]:
    """A tiny migration tool: if a model has a column the database table
    lacks (because the code was updated after the database was created),
    add it with ALTER TABLE. Returns the "table.column" names it added.

    It only ever ADDS nullable columns; it never changes or drops anything.
    """
    added = []
    db_columns = inspect(engine)
    for table in Base.metadata.sorted_tables:
        existing = {c["name"] for c in db_columns.get_columns(table.name)}
        for column in table.columns:
            if column.name in existing:
                continue
            col_type = column.type.compile(dialect=engine.dialect)  # e.g. FLOAT
            with engine.begin() as conn:
                conn.execute(text(f'ALTER TABLE {table.name} ADD COLUMN "{column.name}" {col_type}'))
            added.append(f"{table.name}.{column.name}")
    return added


def get_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Return a factory; call it to get a session: `with factory() as s: ...`."""
    return sessionmaker(bind=engine)
