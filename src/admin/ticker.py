"""
ticker.py - The background "heartbeat" that moves the market while it is
running (Market control -> Start).

A daemon thread wakes up every POLL_SECONDS and calls simulator.tick(),
which only steps when the market is running AND speed_seconds have passed.
Two ways to run it:
  * inside the PaperDesk app (default): starts with app/main.py, stops with it;
  * on its own, as a separate "simulator" process:
        .venv\\Scripts\\python.exe -m src.admin.ticker
    then start the app with PAPER_TRADING_TICKER=off, so the market is never
    advanced twice.
A daemon thread stops with the program.
"""

import logging
import os
import threading
import time

from sqlalchemy.orm import Session, sessionmaker

from src.trading.simulator import tick

POLL_SECONDS = 0.5
log = logging.getLogger("ticker")


def _loop(factory: sessionmaker[Session], stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            with factory() as session:
                tick(session)
        except Exception:  # never let one bad tick kill the heartbeat
            log.exception("market tick failed")
        stop.wait(POLL_SECONDS)


def start_ticker(factory: sessionmaker[Session]) -> threading.Event:
    """Start the heartbeat thread. Returns an Event; .set() it to stop."""
    stop = threading.Event()
    threading.Thread(target=_loop, args=(factory, stop), name="market-ticker", daemon=True).start()
    return stop


def main() -> None:
    """Run the heartbeat until Ctrl+C. Start / pause the market and set its
    speed from the admin Market control page."""
    from src.config import DB_URL
    from src.db.session import get_engine, get_session_factory, init_db

    engine = get_engine(os.environ.get("PAPER_TRADING_DB_URL", DB_URL))
    init_db(engine)
    stop = start_ticker(get_session_factory(engine))
    print("Market simulator running. Use Market control (admin pages) to start / pause it.")
    print("Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop.set()
        print("Simulator stopped.")


if __name__ == "__main__":
    main()
