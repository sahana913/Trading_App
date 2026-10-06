"""
models.py - Every database table, as SQLAlchemy classes.

Field values follow OpenAlgo's API conventions so the simulated broker can
return the same shapes OpenAlgo does:
    action    : BUY / SELL
    pricetype : MARKET / LIMIT / SL / SL-M
    product   : CNC (delivery) / MIS (intraday) / NRML (normal F&O)
    status    : open / complete / cancelled / rejected

Money is stored as Float. That is fine for a paper-trading project; a real
broker would use exact decimals.
"""

from datetime import date, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Parent class for all tables; Base.metadata knows about every table."""


# Allowed values, reused in CHECK constraints so the database itself
# rejects anything else (e.g. action="buy" or role="superuser").
ROLES = ("user", "admin")
ACTIONS = ("BUY", "SELL")
PRICETYPES = ("MARKET", "LIMIT", "SL", "SL-M")
PRODUCTS = ("CNC", "MIS", "NRML")
ORDER_STATUSES = ("open", "complete", "cancelled", "rejected")


def _in(column: str, values: tuple) -> str:
    """Build SQL like: action IN ('BUY', 'SELL')"""
    quoted = ", ".join(f"'{v}'" for v in values)
    return f"{column} IN ({quoted})"


# ---------------------------------------------------------------------------
# Users and money
# ---------------------------------------------------------------------------
class User(Base):
    __tablename__ = "users"
    __table_args__ = (CheckConstraint(_in("role", ROLES), name="ck_users_role"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True)
    password_hash: Mapped[str] = mapped_column(String(200))  # never the plain password
    role: Mapped[str] = mapped_column(String(10), default="user")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    # uselist=False: each user has exactly one funds row
    funds: Mapped["Fund"] = relationship(back_populates="user", uselist=False)


class Fund(Base):
    """Cash account of one user (OpenAlgo 'funds')."""

    __tablename__ = "funds"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), unique=True)
    opening_balance: Mapped[float] = mapped_column(Float)
    available_cash: Mapped[float] = mapped_column(Float)
    used_margin: Mapped[float] = mapped_column(Float, default=0.0)  # blocked by open orders
    realised_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)

    user: Mapped[User] = relationship(back_populates="funds")


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------
class Instrument(Base):
    """A tradable symbol on an exchange, e.g. RELIANCE on NSE."""

    __tablename__ = "instruments"
    __table_args__ = (UniqueConstraint("symbol", "exchange", name="uq_instrument"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(50))
    exchange: Mapped[str] = mapped_column(String(10))
    name: Mapped[str | None] = mapped_column(String(100))
    lot_size: Mapped[int] = mapped_column(Integer, default=1)
    tick_size: Mapped[float] = mapped_column(Float, default=0.05)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class Candle(Base):
    """One OHLCV bar. Used by 'history', 'quotes' and the market simulator."""

    __tablename__ = "candles"
    __table_args__ = (
        # One bar per instrument per time; this also creates an index
        # that makes "candles for RELIANCE between two dates" fast
        UniqueConstraint("instrument_id", "timestamp", name="uq_candle"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    instrument_id: Mapped[int] = mapped_column(ForeignKey("instruments.id"))
    timestamp: Mapped[datetime] = mapped_column(DateTime)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[int] = mapped_column(Integer)


# ---------------------------------------------------------------------------
# Trading
# ---------------------------------------------------------------------------
class Order(Base):
    """Every order ever placed (OpenAlgo 'orderbook')."""

    __tablename__ = "orders"
    __table_args__ = (
        CheckConstraint(_in("action", ACTIONS), name="ck_orders_action"),
        CheckConstraint(_in("pricetype", PRICETYPES), name="ck_orders_pricetype"),
        CheckConstraint(_in("product", PRODUCTS), name="ck_orders_product"),
        CheckConstraint(_in("status", ORDER_STATUSES), name="ck_orders_status"),
        CheckConstraint("quantity > 0", name="ck_orders_quantity"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    orderid: Mapped[str] = mapped_column(String(30), unique=True)  # id shown to the user
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    instrument_id: Mapped[int] = mapped_column(ForeignKey("instruments.id"))
    strategy: Mapped[str | None] = mapped_column(String(50))
    action: Mapped[str] = mapped_column(String(4))
    pricetype: Mapped[str] = mapped_column(String(6))
    product: Mapped[str] = mapped_column(String(4))
    quantity: Mapped[int] = mapped_column(Integer)
    price: Mapped[float] = mapped_column(Float, default=0.0)          # limit price
    trigger_price: Mapped[float] = mapped_column(Float, default=0.0)  # for SL / SL-M
    status: Mapped[str] = mapped_column(String(10), default="open")
    filled_quantity: Mapped[int] = mapped_column(Integer, default=0)
    average_price: Mapped[float] = mapped_column(Float, default=0.0)
    # Cash still blocked for the UNFILLED part of this order (0 once done)
    margin_blocked: Mapped[float] = mapped_column(Float, default=0.0)
    rejection_reason: Mapped[str | None] = mapped_column(String(200))  # also used for cancel reasons
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)

    instrument: Mapped[Instrument] = relationship()


class Trade(Base):
    """One fill of an order (OpenAlgo 'tradebook')."""

    __tablename__ = "trades"
    __table_args__ = (
        CheckConstraint(_in("action", ACTIONS), name="ck_trades_action"),
        CheckConstraint("quantity > 0", name="ck_trades_quantity"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    tradeid: Mapped[str] = mapped_column(String(30), unique=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id"))
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    instrument_id: Mapped[int] = mapped_column(ForeignKey("instruments.id"))
    action: Mapped[str] = mapped_column(String(4))
    quantity: Mapped[int] = mapped_column(Integer)
    price: Mapped[float] = mapped_column(Float)
    fees: Mapped[float] = mapped_column(Float, default=0.0)  # brokerage + taxes for this fill
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    instrument: Mapped[Instrument] = relationship()
    order: Mapped[Order] = relationship()


class Position(Base):
    """Net open quantity per user, instrument and product (OpenAlgo 'positionbook').

    quantity > 0 means long, < 0 means short, 0 means closed.
    Unrealised P&L is not stored: it is computed live from the latest price.
    """

    __tablename__ = "positions"
    __table_args__ = (
        UniqueConstraint("user_id", "instrument_id", "product", name="uq_position"),
        CheckConstraint(_in("product", PRODUCTS), name="ck_positions_product"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    instrument_id: Mapped[int] = mapped_column(ForeignKey("instruments.id"))
    product: Mapped[str] = mapped_column(String(4))
    quantity: Mapped[int] = mapped_column(Integer, default=0)
    average_price: Mapped[float] = mapped_column(Float, default=0.0)
    realised_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    # Cash blocked to keep this position open (MIS margin)
    margin_used: Mapped[float] = mapped_column(Float, default=0.0)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)

    instrument: Mapped[Instrument] = relationship()


class Holding(Base):
    """Delivery (CNC) shares owned long-term (OpenAlgo 'holdings')."""

    __tablename__ = "holdings"
    __table_args__ = (
        UniqueConstraint("user_id", "instrument_id", name="uq_holding"),
        CheckConstraint("quantity >= 0", name="ck_holdings_quantity"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    instrument_id: Mapped[int] = mapped_column(ForeignKey("instruments.id"))
    quantity: Mapped[int] = mapped_column(Integer)
    average_price: Mapped[float] = mapped_column(Float)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)

    instrument: Mapped[Instrument] = relationship()


# ---------------------------------------------------------------------------
# Analytics, admin and ML
# ---------------------------------------------------------------------------
class DailyPnl(Base):
    """End-of-day snapshot per user; feeds the equity curve and drawdown charts."""

    __tablename__ = "daily_pnl"
    __table_args__ = (UniqueConstraint("user_id", "trade_date", name="uq_daily_pnl"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    # Named trade_date, not date, so it doesn't hide the `date` type imported above
    trade_date: Mapped[date] = mapped_column(Date)
    realised_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    unrealised_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    total_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    equity: Mapped[float] = mapped_column(Float)  # cash + market value of positions


class AdminLog(Base):
    """Audit trail: which admin did what, to whom, and when."""

    __tablename__ = "admin_log"

    id: Mapped[int] = mapped_column(primary_key=True)
    admin_user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    action: Mapped[str] = mapped_column(String(50))  # e.g. "deactivate_user", "seed"
    target_user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    details: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class ModelRegistry(Base):
    """Every trained ML model: where the file is, how it was trained, how it scored."""

    __tablename__ = "model_registry"
    __table_args__ = (UniqueConstraint("name", "version", name="uq_model_version"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(50))
    version: Mapped[int] = mapped_column(Integer)
    algorithm: Mapped[str] = mapped_column(String(50))  # e.g. "LightGBM"
    file_path: Mapped[str] = mapped_column(String(300))
    features: Mapped[list | None] = mapped_column(JSON)  # list of feature column names
    params: Mapped[dict | None] = mapped_column(JSON)    # hyper-parameters
    metrics: Mapped[dict | None] = mapped_column(JSON)   # e.g. {"accuracy": 0.54}
    train_start: Mapped[datetime | None] = mapped_column(DateTime)
    train_end: Mapped[datetime | None] = mapped_column(DateTime)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)  # the one the app uses
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
