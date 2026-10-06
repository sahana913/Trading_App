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

Modes (admin "Market control"):
  replay     step to the next historical bar; stop at the end of the data
  synthetic  same while history lasts; after the last real bar, invent the
             next day's bars (synthetic.py) so the market never runs out.
             Real history is never overwritten.
Auto-advance: when is_running is on, tick() steps once every speed_seconds
of real time. The admin app calls tick() from a background thread.

Intraday mode (setting "intraday"): instead of jumping a whole day, the clock
moves 09:15 -> 09:20 -> ... -> 15:30 in 5-minute steps, with prices taken
from a path inside each real daily candle (intraday.py). MIS positions are
squared off at 15:15 by match_orders; at 15:30 the next step closes the day
(snapshot) and opens the next trading day at 09:15.

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
from datetime import datetime, timedelta

from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import Session

from src.config import STARTING_CASH
from src.db.models import Candle, DailyPnl, Fund, Holding, Order, Position, SimClock, Trade
from src.trading.books import portfolio_value
from src.trading.market import error, success
from src.trading.matching import match_orders, square_off_mis
from src.trading.intraday import SESSION_OPEN, day_start, is_intraday, next_tick, session_close
from src.trading.synthetic import generate_next_day

CLOCK_ID = 1  # the one and only clock row
MODES = ("replay", "synthetic")
DEFAULTS = {"mode": "replay", "is_running": False, "speed_seconds": 5.0, "volatility": 1.0, "intraday": False}
SPEED_RANGE = (0.5, 60.0)       # real seconds per simulated bar
VOLATILITY_RANGE = (0.1, 5.0)   # multiplier for synthetic bars


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
    real_end = session.scalar(select(func.max(Candle.timestamp)).where(Candle.is_synthetic.is_not(True)))
    return success({"current_time": now, "data_start": first, "data_end": last, "real_data_end": real_end,
                    "bars_left": days_left, **settings(session)})


def settings(session: Session) -> dict:
    """Market-control settings, with defaults for anything not set yet."""
    clock = session.get(SimClock, CLOCK_ID)
    values = {k: getattr(clock, k, None) if clock else None for k in DEFAULTS}
    return {k: DEFAULTS[k] if v is None else v for k, v in values.items()}


def update_settings(session: Session, **changes) -> dict:
    """Change mode / is_running / speed_seconds / volatility (validated).
    Needs a started simulation (the settings live on the clock row)."""
    clock = session.get(SimClock, CLOCK_ID)
    if clock is None:
        return error("Start the market first")
    if "mode" in changes and changes["mode"] not in MODES:
        return error(f"Mode must be one of {MODES}")
    if "speed_seconds" in changes and not SPEED_RANGE[0] <= changes["speed_seconds"] <= SPEED_RANGE[1]:
        return error(f"Speed must be {SPEED_RANGE[0]}-{SPEED_RANGE[1]} seconds per bar")
    if "volatility" in changes and not VOLATILITY_RANGE[0] <= changes["volatility"] <= VOLATILITY_RANGE[1]:
        return error(f"Volatility must be {VOLATILITY_RANGE[0]}-{VOLATILITY_RANGE[1]}x")
    unknown = set(changes) - set(DEFAULTS)
    if unknown:
        return error(f"Unknown setting(s): {sorted(unknown)}")
    for key, value in changes.items():
        setattr(clock, key, value)
    if changes.get("is_running"):
        clock.last_tick_at = None  # first automatic step happens straight away
    session.commit()
    return success(settings(session))


# ---------------------------------------------------------------------------
# Changing the clock
# ---------------------------------------------------------------------------
def start(session: Session, when: datetime, intraday: bool = False) -> dict:
    """Start a simulation on the first trading day on or after `when`.
    intraday=True starts at that day's 09:15 open and steps in 5-minute ticks.

    Refused if trades already exist: moving the clock back in time would let
    old orders fill at prices from "before" they were placed. Use reset() first.
    """
    if session.scalar(select(func.count()).select_from(Order)) > 0:
        return error("Orders already exist. Run reset() before starting a new simulation")
    first = first_bar_on_or_after(session, when)
    if first is None:
        return error(f"No market data on or after {when:%Y-%m-%d}")

    if intraday:
        first = datetime.combine(first.date(), SESSION_OPEN)  # open of the first day
    clock = session.get(SimClock, CLOCK_ID)
    if clock is None:
        clock = SimClock(id=CLOCK_ID, current_time=first, started_at=first)
        session.add(clock)
    clock.current_time = first
    clock.started_at = first
    clock.intraday = intraday
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


def _move_clock(session: Session, to: datetime) -> list:
    """Set the clock and let waiting orders react to the new prices."""
    session.get(SimClock, CLOCK_ID).current_time = to
    session.commit()
    return match_orders(session, as_of=to)


def step(session: Session) -> dict:
    """Move the market forward one step (see top of file):
    intraday mode mid-session -> 5 minutes; otherwise -> close today, open the next day."""
    now = get_clock(session)
    if now is None:
        return error("Simulation not started. Call start() first")
    cfg = settings(session)

    # Intraday, still trading: just move 5 minutes
    if cfg["intraday"] and is_intraday(now) and next_tick(now) is not None:
        nxt = next_tick(now)
        fills = _move_clock(session, nxt)
        return success({"from": now, "to": nxt, "fills": len(fills), "mis_squared_off": 0})

    # Intraday switched OFF mid-session: play the rest of today to the close first
    if is_intraday(now) and next_tick(now) is not None:
        _move_clock(session, session_close(now))
        now = session_close(now)

    # --- End of day: from here on it's the daily logic, keyed on the date ---
    nxt = next_bar_time(session, day_start(now))
    if nxt is None and settings(session)["mode"] == "synthetic":
        # History has run out: invent the next day (after closing today below)
        nxt = "synthetic"
    if nxt is None:
        return error(f"End of data: {now:%Y-%m-%d} is the last bar")

    # 1-2. Close the current day
    squared_off = square_off_mis(session, as_of=now)
    record_daily_pnl(session, as_of=now)

    # 3. Move the clock (creating tomorrow's bars first in synthetic mode)
    if nxt == "synthetic":
        nxt = generate_next_day(session, day_start(now), cfg["volatility"])
    if cfg["intraday"]:
        nxt = datetime.combine(nxt.date(), SESSION_OPEN)  # the next day starts at its open

    # 4. Let waiting orders see the new prices
    fills = _move_clock(session, nxt)
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


def tick(session: Session, wall_now: datetime | None = None) -> dict | None:
    """Auto-advance: step once if the market is running and speed_seconds of
    real time have passed since the last automatic step. Returns the step
    result, or None if nothing was due. At the end of the data (replay mode)
    the market is paused automatically."""
    clock = session.get(SimClock, CLOCK_ID)
    cfg = settings(session)
    if clock is None or not cfg["is_running"]:
        return None
    wall_now = wall_now or datetime.now()
    if clock.last_tick_at and wall_now - clock.last_tick_at < timedelta(seconds=cfg["speed_seconds"]):
        return None  # not due yet
    result = step(session)
    clock = session.get(SimClock, CLOCK_ID)
    clock.last_tick_at = wall_now
    if result["status"] == "error":
        clock.is_running = False  # nothing more to replay: pause
    session.commit()
    return result


def reset(session: Session) -> dict:
    """Wipe ALL trading activity and give every user the starting cash again.

    Users, instruments and REAL candles are kept; synthetic candles are
    deleted (they belong to the simulation that is being wiped). Cannot be undone.
    """
    # Children before parents, so no foreign key points at a deleted row
    for table in (Trade, Order, Position, Holding, DailyPnl, SimClock):
        session.execute(delete(table))
    session.execute(delete(Candle).where(Candle.is_synthetic.is_(True)))
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
