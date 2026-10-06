"""
ui.py - Streamlit pieces shared by every page: the CSS, the disclaimer
banner, money formatting, one-time messages, and the market-clock sidebar.
"""

from datetime import datetime

import streamlit as st
from sqlalchemy.orm import Session

from src.config import TRADER_CAN_MOVE_CLOCK, WATCHLIST_REFRESH_SECONDS
from src.trading.intraday import is_intraday
from src.trading.simulator import get_clock, run, settings, start, status, step

BANNER = "Paper trading — simulated data, educational only, not financial advice"

# Small, readable CSS. Colours match .streamlit/config.toml (dark theme).
CSS = """
<style>
  /* Space above the page content. Streamlit's top bar (Deploy / menu) is
     fixed and about 3.75rem tall, so anything less hides the banner under it. */
  .block-container { padding-top: 4.5rem; }

  /* the disclaimer banner shown on every page */
  .pt-banner {
    background: #2a2412; border: 1px solid #6b5a1e; color: #f1d98a;
    border-radius: 8px; padding: 0.45rem 0.9rem; margin-bottom: 0.8rem;
    font-size: 0.88rem;
  }

  /* metric cards: subtle border so numbers group visually */
  [data-testid="stMetric"] {
    background: #1a1d24; border: 1px solid #2b2f38;
    border-radius: 10px; padding: 0.6rem 0.9rem;
  }

  /* smaller metric numbers so they fit narrow cards without "..." */
  [data-testid="stMetricValue"] { font-size: 1.45rem; }

  /* digits line up in columns (prices, P&L) */
  [data-testid="stMetricValue"], [data-testid="stDataFrame"] {
    font-variant-numeric: tabular-nums;
  }
</style>
"""


def apply_style() -> None:
    st.html(CSS)


def banner() -> None:
    st.html(f'<div class="pt-banner">⚠️ {BANNER}</div>')


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def money(x: float | None) -> str:
    """₹1,234.50 or −₹1,234.50 (the minus goes before the ₹, as in accounting)."""
    if x is None:
        return "–"
    return f"−₹{-x:,.2f}" if x < 0 else f"₹{x:,.2f}"


def pct(x: float | None) -> str:
    return "–" if x is None else f"{x:.2%}"


def market_time(clock: datetime) -> str:
    """'02 Jan 2024' in daily mode, '02 Jan 2024 · 11:05' during an intraday session."""
    return f"{clock:%d %b %Y} · {clock:%H:%M}" if is_intraday(clock) else f"{clock:%d %b %Y}"


def arrow(x: float) -> str:
    """▲ up, ▼ down, • unchanged: direction is never shown by colour alone."""
    return "▲" if x > 0 else "▼" if x < 0 else "•"


# ---------------------------------------------------------------------------
# One-time messages that survive a page re-run
# ---------------------------------------------------------------------------
def flash(kind: str, message: str) -> None:
    """Save a message, then re-run so every number on screen is refreshed.
    show_flash() displays it once on the next run."""
    st.session_state["flash"] = (kind, message)
    st.rerun()


def show_flash() -> None:
    if "flash" in st.session_state:
        kind, message = st.session_state.pop("flash")
        getattr(st, kind)(message)  # st.success(...) or st.error(...)


def report(result: dict, ok_message: str) -> None:
    """Turn a trading-engine response into a flash message."""
    if result["status"] == "success":
        flash("success", ok_message)
    else:
        flash("error", result["message"])


# ---------------------------------------------------------------------------
# Market clock
# ---------------------------------------------------------------------------
def require_clock(s: Session) -> datetime:
    """The simulated 'now'. Stops the page with a hint if the market hasn't started."""
    clock = get_clock(s)
    if clock is None:
        st.info("The market hasn't started yet. "
                + ("Pick a start date in the sidebar and press **Start market**."
                   if TRADER_CAN_MOVE_CLOCK else "Ask an administrator to start it."))
        st.stop()
    return clock


@st.fragment(run_every=WATCHLIST_REFRESH_SECONDS)
def live_clock() -> None:
    """The market date/time card. A fragment, so it keeps up with a running
    market without the whole page reloading."""
    from src.auth import db  # imported here: auth is only needed once the page runs

    with db()() as s:
        clock = get_clock(s)
    if clock is not None:
        st.metric("Market time" if is_intraday(clock) else "Market date", market_time(clock), help=f"{clock:%A}")


def market_sidebar(s: Session) -> None:
    """Market date, plus buttons to move the clock if config allows it."""
    clock = get_clock(s)
    with st.sidebar:
        st.subheader("Market")
        live_clock()
        if not TRADER_CAN_MOVE_CLOCK:
            return

        if clock is None:
            info = status(s)["data"]
            first = st.date_input("Start date", value=info["data_start"].date(),
                                  min_value=info["data_start"].date(),
                                  max_value=info["data_end"].date(), key="start_date")
            if st.button("Start market", key="start_market", type="primary", width="stretch"):
                report(start(s, datetime.combine(first, datetime.min.time())), "Market started")
            return

        intraday = settings(s)["intraday"]
        unit = "step" if intraday else "day"
        if st.button("Next 5 min ▶" if intraday else "Next day ▶", key="next_day", type="primary", width="stretch"):
            r = step(s)
            if r["status"] == "success":
                flash("success", f"Moved to {market_time(r['data']['to'])} · {r['data']['fills']} order(s) filled")
            else:
                flash("error", r["message"])
        days = st.number_input("Steps" if intraday else "Days", min_value=1, max_value=250, value=5, key="run_days")
        if st.button(f"Run {days} {unit}s ⏩", key="run_many", width="stretch"):
            r = run(s, int(days))["data"]
            flash("success", f"Ran {r['steps']} {unit}(s) · {r['fills']} order(s) filled")
        if intraday:
            st.caption("Intraday mode: prices move in 5-minute steps from 09:15 to 15:30; "
                       "MIS positions close automatically at 15:15.")
        else:
            st.caption("Moving the clock closes MIS positions at the day's close and "
                       "fills waiting orders at the new day's prices.")


def csv_download(df, label: str, filename: str, key: str) -> None:
    """A button that downloads `df` as a CSV file (opens in Excel)."""
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=filename,
                       mime="text/csv", key=key, icon=":material/download:")
