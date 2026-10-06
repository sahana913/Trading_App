"""Market control: start / pause, replay vs synthetic, speed and volatility."""

from datetime import datetime

import streamlit as st

from src import ui
from src.admin import market
from src.auth import current_user, db
from src.trading.simulator import SPEED_RANGE, VOLATILITY_RANGE, status
from src.ui import market_time

admin = current_user()
st.title("Market control")


@st.fragment(run_every=2)
def live_status() -> None:
    """Re-reads the clock every 2 s, so you can watch a running market move."""
    with db()() as s:
        info = status(s)["data"]
    c = st.columns(4)
    c[0].metric("Market time", market_time(info["current_time"]) if info["current_time"] else "–")
    c[1].metric("State", "▶ Running" if info["is_running"] else "⏸ Paused")
    c[2].metric("Mode", info["mode"].capitalize() + (" · intraday" if info["intraday"] else ""))
    c[3].metric("History left", "–" if info["bars_left"] is None else f"{info['bars_left']} days",
                help=f"Real data ends {info['real_data_end']:%d %b %Y}" if info["real_data_end"] else None)


with db()() as s:
    info = status(s)["data"]

if info["current_time"] is None:
    st.info("The market hasn't started. Choose the first trading day.")
    with st.form("start_form"):
        first = st.date_input("Start date", value=info["data_start"].date(), min_value=info["data_start"].date(),
                              max_value=info["data_end"].date(), key="admin_start_date")
        intraday_start = st.toggle("Intraday ticks (5-minute steps, 09:15 to 15:30)", value=True,
                                   key="admin_start_intraday",
                                   help="Prices move during the day and MIS trades get real P&L. Off = one step per day.")
        if st.form_submit_button("Start market", type="primary", key="admin_start"):
            with db()() as s:
                ui.report(market.start_market(s, admin["id"], datetime.combine(first, datetime.min.time()),
                                              intraday=intraday_start), "Market started")
    st.stop()

live_status()

# --- Run / pause / step -------------------------------------------------------
st.subheader("Run")
c = st.columns(4)
if info["is_running"]:
    if c[0].button("⏸ Pause", key="pause", type="primary", width="stretch"):
        with db()() as s:
            ui.report(market.set_running(s, admin["id"], False), "Market paused")
else:
    if c[0].button("▶ Start", key="resume", type="primary", width="stretch",
                   help="Advance one trading day every 'speed' seconds while this admin app is running"):
        with db()() as s:
            ui.report(market.set_running(s, admin["id"], True), "Market running")
unit = "5-min step" if info["intraday"] else "day"
if c[1].button(f"Step 1 {unit}", key="step_one", width="stretch"):
    with db()() as s:
        ui.report(market.step_once(s, admin["id"], 1), f"Moved forward 1 {unit}")
days = c[2].number_input("Steps", min_value=1, max_value=250, value=5, key="admin_days", label_visibility="collapsed")
if c[3].button(f"Step {days} {unit}s", key="step_many", width="stretch"):
    with db()() as s:
        ui.report(market.step_once(s, admin["id"], int(days)), f"Moved forward {days} {unit}s")

# --- Settings -----------------------------------------------------------------
st.subheader("Settings")
with st.form("settings_form"):
    mode = st.radio("Mode", ["replay", "synthetic"], index=["replay", "synthetic"].index(info["mode"]),
                    horizontal=True, key="mode",
                    help="replay = the real historical prices, stopping at the end of the data. "
                         "synthetic = after the real data ends, invent new prices (random walk) so the market "
                         "keeps going. Real history is never overwritten.")
    intraday = st.toggle("Intraday ticks (5-minute steps)", value=bool(info["intraday"]), key="intraday",
                         help="Switching off mid-session plays the rest of today to the close first.")
    speed = st.slider("Speed: seconds per step", *SPEED_RANGE, value=float(info["speed_seconds"]), step=0.5,
                      key="speed", help="One step is 5 minutes in intraday mode, one day otherwise. "
                                        "At 1 s per step a trading day takes 75 s.")
    vol = st.slider("Volatility (synthetic prices only)", *VOLATILITY_RANGE, value=float(info["volatility"]),
                    step=0.1, key="volatility",
                    help="1.0 = each stock moves as much as it did recently; 2.0 = twice as wild. "
                         "Historical replay can't be changed.")
    if st.form_submit_button("Save settings", key="save_settings"):
        with db()() as s:
            ui.report(market.change_settings(s, admin["id"], mode=mode, speed_seconds=float(speed),
                                             volatility=float(vol), intraday=bool(intraday)), "Settings saved")
if info["mode"] == "synthetic" and info["bars_left"]:
    st.caption(f"Synthetic prices begin after the real data ends ({info['real_data_end']:%d %b %Y}); "
               "replay continues until then.")

# --- Danger zone --------------------------------------------------------------
with st.expander("Reset the whole market"):
    st.warning("Deletes EVERY trader's orders, trades, positions, holdings and P&L history, removes "
               "synthetic prices and stops the clock. Accounts get the starting cash back.")
    sure = st.checkbox("I understand", key="reset_market_sure")
    if st.button("Reset market", key="reset_market", disabled=not sure, type="primary"):
        with db()() as s:
            ui.report(market.reset_market(s, admin["id"]), "Market reset")
