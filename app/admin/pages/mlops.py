"""ML Ops: model registry, retraining, comparing versions, anomaly flags."""

import streamlit as st

from src import ui
from src.admin.mlops import activate_version, registry_frame, retrain
from src.analytics import charts
from src.auth import current_user, db
from src.ml.anomaly import BEHAVIOUR_FEATURES, MIN_USERS, behaviour_frame, detect_anomalies

admin = current_user()
ui.page_header("ML Ops", "Model versions, retraining, and unusual trading behaviour.")

# --- Registry -----------------------------------------------------------------
st.subheader("Model registry")
with db()() as s:
    registry = registry_frame(s)

c = st.columns([1, 1, 2])
if c[0].button("Retrain now", key="retrain", type="primary", width="stretch",
               help="Trains every model on the real candles in the database (about 10-20 s) and makes the "
                    "best one the new active version."):
    with st.spinner("Training models…"), db()() as s:
        result = retrain(s, admin["id"])
    st.cache_data.clear()  # pages that cached the old model's backtest must recompute
    ui.report(result, f"Trained version {result.get('data', {}).get('version')}")

if registry.empty:
    ui.empty_state("No models yet", "Press Retrain now, or run python -m src.ml.train.")
else:
    with c[1], st.form("activate_form", border=False):
        version = st.selectbox("Version", list(registry["version"]), key="activate_pick",
                               label_visibility="collapsed")
        if st.form_submit_button("Make active", key="activate_submit", width="stretch"):
            with db()() as s:
                ui.report(activate_version(s, admin["id"], int(version)), f"Version {version} is now active")
    st.dataframe(registry, hide_index=True, width="stretch", column_config={
        "active": st.column_config.CheckboxColumn("Active"),
        "val_auc": st.column_config.NumberColumn("Val AUC", format="%.3f"),
        "test_auc": st.column_config.NumberColumn("Test AUC", format="%.3f"),
        "test_accuracy": st.column_config.NumberColumn("Test accuracy", format="percent"),
        "backtest_return": st.column_config.NumberColumn("Backtest", format="percent"),
        "buy_and_hold": st.column_config.NumberColumn("Buy & hold", format="percent"),
    })
    st.plotly_chart(charts.model_versions_chart(registry), width="stretch")
    st.caption("Validation AUC chose each model; test AUC is the honest score on data it never saw. "
               "Versions trained on the same data with the same seed should match exactly.")

# --- Anomalies ------------------------------------------------------------------
st.subheader("Unusual trading behaviour")
st.caption("Compares every trader's activity (trades per day, trade size, P&L swings, rejected orders). An "
           "IsolationForest flags unusual *patterns*; a robust z-score rule flags any single *extreme value*. "
           "It points at accounts to look at; it does not prove anything is wrong.")
with db()() as s:
    flags = detect_anomalies(behaviour_frame(s))

if flags.empty:
    ui.empty_state("Nothing to analyse yet", "No trader has placed an order.")
elif len(flags) < MIN_USERS:
    st.info(f"{len(flags)} active trader(s). The detector needs at least {MIN_USERS} to compare.")
else:
    c = st.columns(2)
    c[0].metric("Traders analysed", len(flags))
    c[1].metric("Flagged as unusual", int(flags["is_anomaly"].sum()))
    st.dataframe(
        flags[["username", "is_anomaly", "flagged_by", "reason", "anomaly_score", "max_abs_z", "trades",
               *BEHAVIOUR_FEATURES]],
        hide_index=True, width="stretch",
        column_config={
            "is_anomaly": st.column_config.CheckboxColumn("Flagged"),
            "flagged_by": st.column_config.TextColumn(
                "Flagged by", help="pattern = IsolationForest; extreme value = one measure far from typical"),
            "anomaly_score": st.column_config.ProgressColumn("Score", min_value=0.0, max_value=1.0, format="%.2f"),
            "max_abs_z": st.column_config.NumberColumn("Max robust z", format="%.1f"),
            "avg_trade_value": st.column_config.NumberColumn(format="₹%.0f"),
            "max_trade_share": st.column_config.NumberColumn(format="percent"),
            "pnl_volatility": st.column_config.NumberColumn(format="percent"),
            "worst_day": st.column_config.NumberColumn(format="percent"),
            "rejection_rate": st.column_config.NumberColumn(format="percent"),
            "trades_per_day": st.column_config.NumberColumn(format="%.2f"),
        },
    )
