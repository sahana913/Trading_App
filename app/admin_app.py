"""
admin_app.py - The administrator's Streamlit app (dashboard comes in a later step).
Only "admin" accounts get in; there is no self-registration here.

Run from the project root (PowerShell). Port 8502 so it can run next to the
trader app (8501):
    .venv\\Scripts\\python.exe -m streamlit run app/admin_app.py --server.port 8502
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # see app/trader/app.py

import streamlit as st  # noqa: E402

from src import ui  # noqa: E402
from src.auth import db, require_role  # noqa: E402
from src.ml.anomaly import BEHAVIOUR_FEATURES, MIN_USERS, behaviour_frame, detect_anomalies  # noqa: E402

st.set_page_config(page_title="Admin Dashboard", page_icon="🛠️", layout="wide")
ui.apply_style()
ui.banner()

user = require_role("admin", "Admin Dashboard")

st.title(f"Admin: {user['username']}")
st.info("The full admin dashboard arrives in a later step.")

# --- Unusual trading behaviour (IsolationForest, see src/ml/anomaly.py) -----
st.subheader("Unusual trading behaviour")
st.caption("Compares every trader's activity (trades per day, trade size, P&L swings, rejected "
           "orders). An IsolationForest flags unusual *patterns*; a robust z-score rule flags any "
           "single *extreme value*. It points at accounts to look at; it does not prove anything is wrong.")
with db()() as s:
    result = detect_anomalies(behaviour_frame(s))

if result.empty:
    st.info("No trader has placed an order yet.")
elif len(result) < MIN_USERS:
    st.info(f"{len(result)} active trader(s). The detector needs at least {MIN_USERS} to compare.")
else:
    flagged = result[result["is_anomaly"]]
    c = st.columns(2)
    c[0].metric("Traders analysed", len(result))
    c[1].metric("Flagged as unusual", len(flagged))
    st.dataframe(
        result[["username", "is_anomaly", "flagged_by", "reason", "anomaly_score", "max_abs_z", "trades",
                *BEHAVIOUR_FEATURES]],
        hide_index=True, width="stretch",
        column_config={
            "is_anomaly": st.column_config.CheckboxColumn("Flagged"),
            "flagged_by": st.column_config.TextColumn(
                "Flagged by", help="pattern = IsolationForest; extreme value = one measure far from typical"),
            "max_abs_z": st.column_config.NumberColumn("Max robust z", format="%.1f"),
            "anomaly_score": st.column_config.ProgressColumn("Score", min_value=0.0, max_value=1.0,
                                                             format="%.2f", help="Higher = more unusual"),
            "avg_trade_value": st.column_config.NumberColumn(format="₹%.0f"),
            "max_trade_share": st.column_config.NumberColumn(format="percent"),
            "pnl_volatility": st.column_config.NumberColumn(format="percent"),
            "worst_day": st.column_config.NumberColumn(format="percent"),
            "rejection_rate": st.column_config.NumberColumn(format="percent"),
            "trades_per_day": st.column_config.NumberColumn(format="%.2f"),
        },
    )
