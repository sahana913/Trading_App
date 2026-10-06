"""Positions: intraday (MIS) positions with live P&L.

The table is a fragment that re-runs every few seconds, so it follows the
market clock without reloading the page.

  unrealised P&L = (last price - average price) x quantity
                   (quantity is negative for a short, so a falling price is a gain)
  P&L            = realised + unrealised
"""

import pandas as pd
import streamlit as st

from src import ui
from src.auth import current_user, db
from src.config import WATCHLIST_REFRESH_SECONDS
from src.trading import positionbook
from src.trading.simulator import get_clock

user = current_user()
st.title("Positions")
st.caption("Intraday (MIS) positions. They are closed automatically at 15:15 (intraday mode) or at "
           "the day's close (daily mode), so they only exist during the day they were opened.")

with db()() as s:
    ui.require_clock(s)


@st.fragment(run_every=WATCHLIST_REFRESH_SECONDS)
def live_positions() -> None:
    with db()() as s:
        now = get_clock(s)
        rows = positionbook(s, user["id"], as_of=now)["data"]
    if not rows:
        st.info("No intraday positions. Place an order with product MIS to open one.")
        return

    df = pd.DataFrame(rows)
    open_now = df[df["quantity"] != 0]
    c = st.columns(3)
    c[0].metric("Open positions", len(open_now))
    c[1].metric("Unrealised P&L", ui.money(df["unrealised_pnl"].sum()))
    c[2].metric("Realised P&L", ui.money(df["realised_pnl"].sum()))
    st.caption(f"Prices as of {ui.market_time(now)} · refreshes every {WATCHLIST_REFRESH_SECONDS}s")
    st.dataframe(df, hide_index=True, width="stretch",
                 column_config={"average_price": "Avg price", "ltp": "Last price",
                                "realised_pnl": "Realised", "unrealised_pnl": "Unrealised",
                                "pnl": st.column_config.NumberColumn("P&L", format="%.2f")})


live_positions()
