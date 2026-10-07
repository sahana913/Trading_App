"""Holdings: delivery (CNC) shares, valued live.

  invested       = average price x quantity
  current value  = last price x quantity
  P&L            = current value - invested
  P&L %          = P&L / invested x 100
"""

import pandas as pd
import streamlit as st

from src import ui
from src.auth import current_user, db
from src.config import WATCHLIST_REFRESH_SECONDS
from src.trading import holdings
from src.trading.simulator import get_clock

user = current_user()
ui.page_header("Holdings", "Delivery (CNC) shares, valued at the latest price.")

with db()() as s:
    ui.require_clock(s)


@st.fragment(run_every=WATCHLIST_REFRESH_SECONDS)
def live_holdings() -> None:
    with db()() as s:
        now = get_clock(s)
        data = holdings(s, user["id"], as_of=now)["data"]
    if not data["holdings"]:
        ui.empty_state("No holdings yet", "Buy with product CNC to keep shares overnight. They appear here with live value and P&L.")
        return

    stats = data["statistics"]
    c = st.columns(3)
    c[0].metric("Invested", ui.money(stats["totalinvvalue"]))
    c[1].metric("Current value", ui.money(stats["totalholdingvalue"]))
    c[2].metric("P&L", ui.money(stats["totalprofitandloss"]), f"{stats['totalpnlpercentage']:+.2f}%")
    st.caption(f"Prices as of {ui.market_time(now)} · refreshes every {WATCHLIST_REFRESH_SECONDS}s")
    st.dataframe(pd.DataFrame(data["holdings"]), hide_index=True, width="stretch",
                 column_config={"average_price": "Avg price", "ltp": "Last price",
                                "pnlpercent": st.column_config.NumberColumn("P&L %", format="%.2f%%")})


live_holdings()
