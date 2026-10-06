"""
simulator.py - Replay historical candles as if the market were live.

How a simulated day works
-------------------------
The clock always points at one bar time (for daily data: one trading day).
While the clock is on day D, traders place orders with as_of=D, so they see
D's prices and nothing later (no look-ahead).

step() then ends day D and opens the next one:
    1. square_off_mis(D)    close every intraday (MIS) position at D's close
    2. record_daily_pnl(D)  save each user's equity snapshot for D
    3. clock -> next trading day D+1
    4. match_orders(D+1)    waiting LIMIT / SL orders react to D+1's price

Note on daily bars: an MIS trade opened on day D is closed at D's close, the
same price it was bought at, so MIS only becomes interesting with intraday
data. CNC (delivery) trades span days and work fully.

Command line (PowerShell, from the project root):
    .venv\\Scripts\\python.exe -m src.trading.simulator status
    .venv\\Scripts\\python.exe -m src.trading.simulator start 2024-01-01
    .venv\\Scripts\\python.exe -m src.trading.simulator step
    .venv\\Scripts\\python.exe -m src.trading.simulator run 20
    .venv\\Scripts\\python.exe -m src.trading.simulator reset --yes
"""

import argparse
from datetime import datetime

from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import Session

from src.config import STARTING_CASH
from src.db.models import Candle, DailyPnl, Fund, Holding, Order, Position, SimClock, Trade
from src.trading.books import portfolio_value
from src.trading.market import error, success
from src.trading.matching import match_orders, square_off_mis

CLOCK_ID = 1  # the one and only clock row


# ---------------------------------------------------------------------------
# Reading the clock and the calendar
# ---------------------------------------------------------------------------
def get_clock(session: Session) -> datetime | None:
    """Current simulated time, or None if no simulation has been started."""
    clock = session.get(SimClock, CLOCK_ID)
    return clock.current_time if clock else None


def next_bar_time(session: Session, after: datetime) -> datetime | None:
    """The first bar time strictly after `after`, across ALL instruments.
    Weekends and holidays have no bars, so they are skipped automatically."""
    return session.scalar(select(func.min(Candle.timestamp)).where(Candle.timestamp > after))


def first_bar_on_or_after(session: Session, when: datetime) -> datetime | None:
    return session.scalar(select(func.min(Candle.timestamp)).where(Candle.timestamp >= when))


def status(session: Session) -> dict:
    """Where the clock is and how much data is left."""
    now = get_clock(session)
    first, last = session.execute(select(func.min(Candle.timestamp), func.max(Candle.timestamp))).one()
    days_left = None
    if now is not None:
        days_left = session.scalar(
            select(func.count(func.distinct(Candle.timestamp))).where(Candle.timestamp > now)
        )
    return success({"current_time": now, "data_start": first, "data_end": last, "bars_left": days_left})


# ---------------------------------------------------------------------------
# Changing the clock
# ---------------------------------------------------------------------------
def start(session: Session, when: datetime) -> dict:
    """Start a simulation on the first trading day on or after `when`.

    Refused if trades already exist: moving the clock back in time would let
    old orders fill at prices from "before" they were placed. Use reset() first.
    """
    if session.scalar(select(func.count()).select_from(Order)) > 0:
        return error("Orders already exist. Run reset() before starting a new simulation")
    first = first_bar_on_or_after(session, when)
    if first is None:
        return error(f"No market data on or after {when:%Y-%m-%d}")

    clock = session.get(SimClock, CLOCK_ID)
    if clock is None:
        clock = SimClock(id=CLOCK_ID, current_time=first, started_at=first)
        session.add(clock)
    clock.current_time = first
    clock.started_at = first
    session.commit()
    return success({"current_time": first})


def record_daily_pnl(session: Session, as_of: datetime) -> int:
    """Save (or overwrite) one equity snapshot per user for the day `as_of`.

    The values are running totals since the start, e.g. realised_pnl is all
    profit booked so far. The P&L of a single day is today's equity minus
    yesterday's, which the analytics step will compute.
    """
    day = as_of.date()
    count = 0
    for fund in session.scalars(select(Fund)).all():
        value = portfolio_value(session, fund.user_id, as_of)
        snap = session.scalar(
            select(DailyPnl).where(DailyPnl.user_id == fund.user_id, DailyPnl.trade_date == day)
        )
        if snap is None:  # running it twice for one day updates, never duplicates
            snap = DailyPnl(user_id=fund.user_id, trade_date=day)
            session.add(snap)
        snap.realised_pnl = round(value["realised_pnl"], 2)
        snap.unrealised_pnl = round(value["unrealised_pnl"], 2)
        snap.equity = round(value["equity"], 2)
        snap.total_pnl = round(value["equity"] - fund.opening_balance, 2)  # includes charges paid
        count += 1
    session.commit()
    return count


def step(session: Session) -> dict:
    """End the current day and move to the next trading day (see top of file)."""
    now = get_clock(session)
    if now is None:
        return error("Simulation not started. Call start() first")
    nxt = next_bar_time(session, now)
    if nxt is None:
        return error(f"End of data: {now:%Y-%m-%d} is the last bar")

    # 1-2. Close the current day
    squared_off = square_off_mis(session, as_of=now)
    record_daily_pnl(session, as_of=now)

    # 3. Move the clock
    session.get(SimClock, CLOCK_ID).current_time = nxt
    session.commit()

    # 4. Let waiting orders see the new prices
    fills = match_orders(session, as_of=nxt)
    return success({"from": now, "to": nxt, "fills": len(fills), "mis_squared_off": squared_off})


def run(session: Session, days: int) -> dict:
    """Call step() up to `days` times; stops early at the end of the data."""
    steps = fills = 0
    for _ in range(days):
        result = step(session)
        if result["status"] == "error":
            break
        steps += 1
        fills += result["data"]["fills"]
    return success({"steps": steps, "fills": fills, "current_time": get_clock(session)})


def reset(session: Session) -> dict:
    """Wipe ALL trading activity and give every user the starting cash again.

    Users, instruments and candles are kept. This cannot be undone.
    """
    # Children before parents, so no foreign key points at a deleted row
    for table in (Trade, Order, Position, Holding, DailyPnl, SimClock):
        session.execute(delete(table))
    session.execute(update(Fund).values(
        opening_balance=STARTING_CASH, available_cash=STARTING_CASH,
        used_margin=0.0, realised_pnl=0.0,
    ))
    session.commit()
    return success()


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def main() -> None:
    from src.db.session import get_engine, get_session_factory, init_db

    parser = argparse.ArgumentParser(description="Control the market simulator.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    p_start = sub.add_parser("start")
    p_start.add_argument("date", help="YYYY-MM-DD")
    sub.add_parser("step")
    p_run = sub.add_parser("run")
    p_run.add_argument("days", type=int)
    p_reset = sub.add_parser("reset")
    p_reset.add_argument("--yes", action="store_true", help="confirm wiping all trades")
    args = parser.parse_args()

    engine = get_engine()
    init_db(engine)  # creates the sim_clock table on an older database
    with get_session_factory(engine)() as session:
        if args.command == "status":
            result = status(session)
        elif args.command == "start":
            result = start(session, datetime.fromisoformat(args.date))
        elif args.command == "step":
            result = step(session)
        elif args.command == "run":
            result = run(session, args.days)
        elif not args.yes:
            raise SystemExit("reset deletes every order and trade. Add --yes to confirm.")
        else:
            result = reset(session)
    show(result)


def show(result: dict) -> None:
    """Print a result as readable lines, with dates as YYYY-MM-DD."""
    if result["status"] == "error":
        print("ERROR:", result["message"])
        return
    for key, value in (result.get("data") or {}).items():
        if isinstance(value, datetime):
            value = f"{value:%Y-%m-%d}"
        print(f"{key:<16} {value}")
    if not result.get("data"):
        print("done")


if __name__ == "__main__":
    main()
