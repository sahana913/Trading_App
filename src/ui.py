"""
ui.py - The PaperDesk look ("Night Desk") and the pieces every page shares.

Design tokens (CSS variables, set once in apply_style):
  --ground  page background        --panel / --panel-2  cards / insets
  --line    hairlines              --ink / --muted      text / secondary text
  --accent  brand colour: saffron amber on trading pages, violet on admin pages
  --data    blue, for charts and selections
  --up / --down   green / red, ONLY for money moving
Fonts come from .streamlit/config.toml: Plus Jakarta Sans (headings and text)
and JetBrains Mono (numbers only, so prices line up in columns).

Components: page_header, live_clock (market card), account_card, empty_state,
login_hero, order_status_style, button_tone, ticker_tape, plus the formatting
and one-time-message helpers the pages already use.
"""

import time
from datetime import datetime

import streamlit as st
from sqlalchemy.orm import Session

from src.config import TRADER_CAN_MOVE_CLOCK, WATCHLIST_REFRESH_SECONDS
from src.trading.intraday import SESSION_CLOSE, SESSION_OPEN, TICKS, is_intraday, tick_index
from src.trading.simulator import get_clock, run, settings, start, status, step

BRAND = "PaperDesk"
BANNER = "Paper trading — simulated data, educational only, not financial advice"
ACCENTS = {"trader": "#f2a93b", "admin": "#9b8cf2"}  # saffron amber / violet
ACCENT_INK = {"trader": "#1a1205", "admin": "#15112b"}  # dark text that reads on the accent

CSS = """
<style>
  :root {
    --ground: #0c1018; --panel: #131925; --panel-2: #1a2231; --line: #253044;
    --ink: #e8ebf2; --muted: #8b94a8;
    --accent: __ACCENT__; --accent-ink: __ACCENT_INK__;
    --data: #4c8ee6; --up: #22a55a; --down: #e34948;
    --mono: "JetBrains Mono", Consolas, monospace;
    --sans: "Plus Jakarta Sans", "Segoe UI", system-ui, sans-serif;
  }

  /* ---------- Page frame ---------- */
  /* Streamlit's top bar is fixed and ~3.75rem tall: keep content below it */
  .block-container { padding-top: 4.5rem; }
  h1 { font-weight: 800 !important; letter-spacing: -0.025em; }
  h2, h3 { font-weight: 700 !important; letter-spacing: -0.015em; }

  .pt-banner { font: 500 0.82rem/1.3 var(--sans); color: #f3d38a; background: #241d0c;
               border: 1px solid #5a4818; border-radius: 7px; padding: 0.5rem 0.8rem; margin-bottom: 0.4rem; }

  /* page header: title + subtitle on the left, status chips on the right */
  .pd-sub { color: var(--muted); font-size: 0.9rem; margin: -0.6rem 0 0.4rem; }
  .pd-chips { display: flex; gap: 0.4rem; flex-wrap: wrap; justify-content: flex-end; padding-bottom: 1.1rem; }
  .pd-chip { font: 600 0.74rem/1 var(--sans); padding: 0.42rem 0.65rem; border-radius: 99px;
             border: 1px solid var(--line); color: var(--muted); white-space: nowrap; }
  .pd-chip.hot { color: var(--accent); border-color: color-mix(in srgb, var(--accent) 55%, transparent); }
  .pd-chip.up { color: var(--up); border-color: color-mix(in srgb, var(--up) 55%, transparent); }

  /* ---------- Metric tiles ---------- */
  [data-testid="stMetric"] { background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
                             padding: 0.75rem 0.9rem 0.6rem; }
  [data-testid="stMetricLabel"] p { font: 600 0.72rem/1.2 var(--sans) !important; letter-spacing: 0.06em;
                                    text-transform: uppercase; color: var(--muted); }
  [data-testid="stMetricValue"] { font-family: var(--mono); font-size: 1.4rem; font-weight: 600;
                                  font-variant-numeric: tabular-nums; }
  [data-testid="stMetricDelta"] { font-family: var(--mono); }
  /* Tiles in one row share a height, even when only some have a sparkline:
     the columns already stretch, so let the tile fill its column */
  [data-testid="stColumn"] > [data-testid="stVerticalBlock"] { height: 100%; }
  [data-testid="stColumn"] [data-testid="stElementContainer"]:has(> [data-testid="stMetric"]) { flex: 1 1 auto; }
  [data-testid="stElementContainer"] > [data-testid="stMetric"] { height: 100%; }
  h3 { font-size: 1.45rem !important; }

  /* ---------- Buttons ---------- */
  [data-testid^="stBaseButton-primary"] { background: var(--accent) !important; border-color: var(--accent) !important;
                                          color: var(--accent-ink) !important; font-weight: 700; }
  [data-testid^="stBaseButton-primary"]:hover { filter: brightness(1.08); }
  [data-testid^="stBaseButton-secondary"]:hover { border-color: var(--accent) !important; color: var(--accent) !important; }

  /* ---------- Sidebar ---------- */
  [data-testid="stSidebarNav"] a[aria-current="page"],
  [data-testid="stSidebarNavLink"][aria-current="page"] {
    background: color-mix(in srgb, var(--accent) 16%, transparent) !important;
    box-shadow: inset 3px 0 0 var(--accent); }
  [data-testid="stSidebarNav"] a[aria-current="page"] span { color: var(--accent) !important; font-weight: 600; }
  [data-testid="stSidebarNavSeparator"] { border-color: var(--line); }

  /* market card */
  .pd-card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 0.75rem 0.85rem; }
  .pd-k { font: 600 0.66rem/1 var(--sans); letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
  .pd-clock { font: 600 1.45rem/1.15 var(--mono); margin: 0.45rem 0 0.1rem; font-variant-numeric: tabular-nums; }
  .pd-date { color: var(--muted); font-size: 0.8rem; }
  .pd-bar { height: 5px; border-radius: 99px; background: var(--panel-2); overflow: hidden; margin-top: 0.65rem; }
  .pd-bar span { display: block; height: 100%; border-radius: 99px; background: var(--accent); transition: width 0.6s ease; }
  .pd-ends { display: flex; justify-content: space-between; font: 500 0.64rem/1.9 var(--sans); color: var(--muted); }
  .pd-acct { display: grid; grid-template-columns: auto 1fr; gap: 0.35rem 0.6rem; align-items: baseline; }
  .pd-acct .v { font: 600 0.95rem/1.2 var(--mono); text-align: right; font-variant-numeric: tabular-nums; }
  .pd-acct .v.up { color: var(--up); } .pd-acct .v.down { color: var(--down); }

  .live-badge { display: inline-flex; align-items: center; gap: 0.45rem; font: 700 0.68rem/1 var(--sans);
                letter-spacing: 0.1em; color: var(--up); }
  .live-badge.paused { color: var(--muted); }
  .live-dot { width: 8px; height: 8px; border-radius: 50%; background: currentColor; animation: pulse 1.6s infinite; }
  .paused .live-dot { animation: none; }
  @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(34,165,90,0.55); } 70% { box-shadow: 0 0 0 8px rgba(34,165,90,0); }
                     100% { box-shadow: 0 0 0 0 rgba(34,165,90,0); } }

  /* ---------- Empty state ---------- */
  .pd-empty { border: 1px dashed var(--line); border-radius: 10px; padding: 1.1rem 1.2rem; display: flex; gap: 0.9rem;
              align-items: center; margin: 0.4rem 0 0.8rem; }
  .pd-empty .em { flex: none; width: 40px; height: 40px; border-radius: 10px; background: var(--panel-2); display: grid;
                  place-items: center; color: var(--accent); font: 700 1.1rem var(--mono); }
  .pd-empty b { display: block; color: var(--ink); } .pd-empty span { color: var(--muted); font-size: 0.88rem; }

  /* ---------- Login hero ---------- */
  .pd-hero { min-height: 440px; border-radius: 14px; border: 1px solid var(--line); padding: 1.6rem;
             display: flex; flex-direction: column; justify-content: space-between; gap: 1.2rem;
             background: radial-gradient(120% 90% at 0% 100%, color-mix(in srgb, var(--accent) 20%, transparent), transparent 60%), #0f141e; }
  .pd-hero h2 { font: 800 2.2rem/1.12 var(--sans); letter-spacing: -0.03em; margin: 0; max-width: 15ch; color: var(--ink); }
  .pd-hero p { color: var(--muted); margin: 0; max-width: 46ch; }
  .pd-hero ul { margin: 0; padding: 0; list-style: none; display: grid; gap: 0.45rem; }
  .pd-hero li { font: 500 0.95rem/1.4 var(--sans); color: var(--ink); }
  .pd-hero li::before { content: "▸ "; color: var(--accent); }

  /* ---------- Ticker tape ---------- */
  .tape { overflow: hidden; white-space: nowrap; border: 1px solid var(--line); border-radius: 8px;
          background: var(--panel); padding: 0.55rem 0; margin-bottom: 0.4rem;
          -webkit-mask-image: linear-gradient(90deg, transparent, #000 4%, #000 96%, transparent);
                  mask-image: linear-gradient(90deg, transparent, #000 4%, #000 96%, transparent); }
  .tape-track { display: inline-flex; gap: 2.2rem; padding-left: 1rem; animation: tape linear infinite; }
  .tape-item { font: 500 0.82rem/1 var(--mono); color: var(--ink); }
  .tape-item b { letter-spacing: 0.04em; margin-right: 0.4rem; }
  .tape-up { color: var(--up); } .tape-down { color: var(--down); }
  @keyframes tape { to { transform: translateX(-50%); } }

  /* ---------- Order ticket estimate ---------- */
  .pd-est { border-top: 1px dashed var(--line); padding-top: 0.55rem; margin: 0.2rem 0 0.6rem; display: grid; gap: 0.3rem;
            font: 400 0.8rem/1.2 var(--mono); }
  .pd-est div { display: flex; justify-content: space-between; }
  .pd-est span:first-child { color: var(--muted); font-family: var(--sans); }
  .pd-est .warn { color: var(--down); }

  /* ---------- Motion ---------- */
  [data-testid="stMetric"], [data-testid="stPlotlyChart"] { transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease; }
  [data-testid="stMetric"]:hover { transform: translateY(-2px); border-color: var(--accent);
                                   box-shadow: 0 6px 18px color-mix(in srgb, var(--accent) 18%, transparent); }
  @media (prefers-reduced-motion: reduce) {
    .tape-track, .live-dot { animation: none !important; }
    [data-testid="stMetric"], [data-testid="stPlotlyChart"], .pd-bar span { transition: none; }
  }
  [data-testid="stDataFrame"] { font-variant-numeric: tabular-nums; }
</style>
"""


def apply_style(app: str = "trader") -> None:
    """Inject the design tokens and CSS. app="admin" switches the accent to violet."""
    st.html(CSS.replace("__ACCENT__", ACCENTS[app]).replace("__ACCENT_INK__", ACCENT_INK[app]))


def banner() -> None:
    st.html(f'<div class="pt-banner">⚠ {BANNER}</div>')


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
# Components
# ---------------------------------------------------------------------------
def page_header(title: str, subtitle: str | None = None, chips: list[tuple[str, str]] | None = None) -> None:
    """Title (a real st.title, so tests and screen readers see it), an optional
    one-line subtitle, and status chips on the right: [(text, "hot"/"up"/"")]."""
    left, right = st.columns([3, 2], vertical_alignment="bottom")
    with left:
        st.title(title)
        if subtitle:
            st.html(f'<p class="pd-sub">{subtitle}</p>')
    if chips:
        with right:
            st.html('<div class="pd-chips">' + "".join(
                f'<span class="pd-chip {tone}">{text}</span>' for text, tone in chips) + "</div>")


def market_chips(s: Session) -> list[tuple[str, str]]:
    """Status chips for the page header: exchange/mode, replay/synthetic, speed."""
    cfg = settings(s)
    chips = [("NSE · Intraday" if cfg["intraday"] else "NSE · Daily", "hot"), (cfg["mode"].capitalize(), "")]
    if cfg["is_running"]:
        chips.append((f"Running · {cfg['speed_seconds']:g} s/step", "up"))
    return chips


def empty_state(title: str, text: str, mark: str = "0") -> None:
    """A designed 'nothing here yet' block that says what will appear and how."""
    st.html(f'<div class="pd-empty"><div class="em">{mark}</div><div><b>{title}</b><span>{text}</span></div></div>')


def login_hero(headline: str, text: str, points: list[str]) -> None:
    """The brand panel beside the login form."""
    items = "".join(f"<li>{p}</li>" for p in points)
    st.html(f'<div class="pd-hero"><div><h2>{headline}</h2></div><ul>{items}</ul><p>{text}</p></div>')


def order_status_style(value: str) -> str:
    """Pandas Styler rule: colour each order status like a pill."""
    colours = {"open": ("#4c8ee6", "rgba(76,142,230,0.14)"), "complete": ("#22a55a", "rgba(34,165,90,0.14)"),
               "cancelled": ("#8b94a8", "rgba(139,148,168,0.12)"), "rejected": ("#e34948", "rgba(227,73,72,0.14)")}
    fg, bg = colours.get(str(value), ("", ""))
    return f"color: {fg}; background-color: {bg}; font-weight: 600" if fg else ""


def button_tone(key: str, tone: str) -> None:
    """Colour one button by its widget key (Streamlit gives each widget a
    `st-key-<key>` CSS class): tone "up" = green, "down" = red."""
    colour = {"up": "#22a55a", "down": "#e34948"}[tone]
    st.html(f"<style>.st-key-{key} button {{ background: {colour} !important; border-color: {colour} !important;"
            f" color: #fff !important; font-weight: 700; box-shadow: 0 6px 18px {colour}40; }}</style>")


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
# Market clock and account (sidebar)
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


def market_card_html(clock: datetime | None, running: bool, intraday: bool) -> str:
    """The market card: LIVE/PAUSED badge, big clock, date, and in intraday
    mode a session bar from the 09:15 open to the 15:30 close."""
    if clock is None:
        return '<div class="pd-card"><span class="pd-k">Market</span><div class="pd-date">Not started</div></div>'
    badge = (f'<span class="live-badge{"" if running else " paused"}"><span class="live-dot"></span>'
             f'{"LIVE" if running else "PAUSED"}{" · INTRADAY" if intraday else ""}</span>')
    if is_intraday(clock):
        done = tick_index(clock) / TICKS * 100
        big, sub = f"{clock:%H:%M}", f"{clock:%a %d %b %Y}"
        bar = (f'<div class="pd-bar"><span style="width:{done:.0f}%"></span></div>'
               f'<div class="pd-ends"><span>{SESSION_OPEN:%H:%M}</span><span>sq-off 15:15</span>'
               f'<span>{SESSION_CLOSE:%H:%M}</span></div>')
    else:
        big, sub, bar = f"{clock:%d %b %Y}", f"{clock:%A} · daily steps", ""
    return f'<div class="pd-card">{badge}<div class="pd-clock">{big}</div><div class="pd-date">{sub}</div>{bar}</div>'


@st.fragment(run_every=WATCHLIST_REFRESH_SECONDS)
def live_clock() -> None:
    """The market card, refreshed on its own so it keeps up with a running market."""
    from src.auth import db  # imported here: auth is only needed once the page runs

    with db()() as s:
        cfg = settings(s)
        clock = get_clock(s)
    st.html(market_card_html(clock, cfg["is_running"], cfg["intraday"]))


@st.fragment(run_every=WATCHLIST_REFRESH_SECONDS)
def account_card(user_id: int) -> None:
    """Equity, cash and total P&L for the signed-in trader, kept live."""
    from src.auth import db
    from src.trading.accounts import get_fund
    from src.trading.books import portfolio_value

    with db()() as s:
        clock = get_clock(s)
        value = portfolio_value(s, user_id, clock)
        opening = get_fund(s, user_id).opening_balance
    pnl = value["equity"] - opening
    st.html(f'<div class="pd-card pd-acct"><span class="pd-k">Equity</span><span class="v">{money(value["equity"])}</span>'
            f'<span class="pd-k">Cash</span><span class="v">{money(value["cash"])}</span>'
            f'<span class="pd-k">Total P&amp;L</span><span class="v {"up" if pnl >= 0 else "down"}">{money(pnl)}</span></div>')


def market_sidebar(s: Session) -> None:
    """Market card, plus buttons to move the clock if config allows it."""
    clock = get_clock(s)
    with st.sidebar:
        live_clock()
        if not TRADER_CAN_MOVE_CLOCK:
            return

        if clock is None:
            info = status(s)["data"]
            if info["data_start"] is None:  # a fresh database without any prices yet
                st.caption("No market data loaded yet. Run the data pipeline and the seed command.")
                return
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
            st.caption("Prices move in 5-minute steps from 09:15 to 15:30; MIS positions close at 15:15.")
        else:
            st.caption("Moving the clock closes MIS positions at the day's close and "
                       "fills waiting orders at the new day's prices.")


def ticker_tape(snapshot, seconds: float = 60.0) -> None:
    """A scrolling strip of prices (symbol, last price, ▲/▼ change).

    The strip is redrawn every few seconds with fresh prices. To stop it
    jumping back to the start each time, the animation is given a negative
    delay equal to how far into its loop it should already be, worked out
    from the wall clock, so the scroll continues smoothly across redraws.
    """
    items = "".join(
        f'<span class="tape-item"><b>{r.symbol}</b>₹{r.ltp:,.2f} '
        f'<span class="{"tape-up" if r.change_pct >= 0 else "tape-down"}">'
        f'{arrow(r.change_pct)} {r.change_pct:+.2f}%</span></span>'
        for r in snapshot.itertuples())
    offset = time.time() % seconds  # where in the loop we are right now
    st.html(f'<div class="tape"><div class="tape-track" style="animation-duration:{seconds}s;'
            f'animation-delay:-{offset:.2f}s">{items}{items}</div></div>')  # twice, for a seamless loop


def csv_download(df, label: str, filename: str, key: str) -> None:
    """A button that downloads `df` as a CSV file (opens in Excel)."""
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=filename,
                       mime="text/csv", key=key, icon=":material/download:")
