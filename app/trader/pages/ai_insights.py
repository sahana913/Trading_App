"""AI Insights: the active ML model's signals, how good it really is, and why.

Signals use only candles up to the market clock, so the model never sees the
future. Everything here is educational, not advice.
"""

import pandas as pd
import streamlit as st

from src import ui
from src.analytics import charts
from src.auth import db
from src.config import PROJECT_ROOT
from src.ml.predict import (
    active_model, backtest_equity_curves, feature_importance, latest_signals, load_artifact,
)

st.title("AI Insights")
st.warning("**Educational only, not financial advice.** These are experimental signals from a "
           "model trained on past prices. Next-day direction is close to random, and the model's "
           "own test results below show it does not reliably beat simply buying and holding.",
           icon="🎓")


@st.cache_data(show_spinner="Re-running the backtest…")
def cached_curves(registry_id: int, model_path: str) -> pd.DataFrame:
    """The backtest only depends on the model, so compute it once per model
    (the id + file path identify it; the arguments are only the cache key)."""
    with db()() as s:
        entry = active_model(s)
        return backtest_equity_curves(s, load_artifact(entry))


with db()() as s:
    clock = ui.require_clock(s)
    entry = active_model(s)
    if entry is None:
        st.info("No model trained yet. Run `.venv\\Scripts\\python.exe -m src.ml.train`.")
        st.stop()
    artifact = load_artifact(entry)
    signals = latest_signals(s, artifact, as_of=clock)
    # copy what we need while the session is open
    info = {"version": entry.version, "algorithm": entry.algorithm, "id": entry.id, "path": entry.file_path,
            "train_start": entry.train_start, "train_end": entry.train_end, "metrics": entry.metrics}

metrics = info["metrics"]
chosen = artifact["model_key"]
st.caption(f"Model: **{info['algorithm']}** (v{info['version']}) · trained on "
           f"{info['train_start']:%d %b %Y} – {info['train_end']:%d %b %Y} · chosen by validation ROC-AUC")

if clock <= info["train_end"]:
    st.info(f"The market date ({clock:%d %b %Y}) is inside the model's **training period**. "
            "It has already seen these prices, so its signals here look better than they "
            "really are. Signals after the training period are the honest ones.", icon="ℹ️")

# --- Signals -----------------------------------------------------------------
st.subheader("Today's signals")
st.caption(f"Based on candles up to {clock:%d %b %Y}. P(up) = the model's probability that the "
           "next close is higher; it says UP above 50%.")
table = signals.rename(columns={"symbol": "Symbol", "close": "Close", "proba_up": "P(up)"})
table["Signal"] = table["signal"].map({"UP": "▲ UP", "DOWN": "▼ DOWN"}).fillna("needs 50+ days of data")
styled = table[["Symbol", "Close", "P(up)", "Signal"]].style.map(
    lambda v: f"color: {charts.GAIN}" if v == "▲ UP" else f"color: {charts.LOSS}" if v == "▼ DOWN" else "",
    subset=["Signal"])
st.dataframe(styled, hide_index=True, width="stretch",
             column_config={"Close": st.column_config.NumberColumn(format="₹%.2f"),
                            "P(up)": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0,
                                                                     format="percent")})

# --- How good is it? ---------------------------------------------------------
st.subheader("How good is the model? (test period it never saw)")
test = metrics["all_models"]["test"]
best_baseline = max(test["majority"]["roc_auc"], test["yesterday"]["roc_auc"])
bt = metrics["backtest"]
c = st.columns(3)
c[0].metric("Test ROC-AUC", f"{test[chosen]['roc_auc']:.3f}",
            f"{test[chosen]['roc_auc'] - best_baseline:+.3f} vs best baseline", delta_arrow="off",
            delta_color="normal",
            help="0.5 = coin flip, 1.0 = perfect. Compared with the better of the two baselines.")
c[1].metric("Test accuracy", f"{test[chosen]['accuracy']:.1%}",
            f"{test[chosen]['accuracy'] - test['majority']['accuracy']:+.1%} vs always-majority",
            delta_arrow="off", help="Share of days where the predicted direction was right")
c[2].metric("Backtest return", f"{bt['strategy']['total_return']:+.2%}",
            f"buy & hold {bt['buy_and_hold']['total_return']:+.2%}", delta_arrow="off",
            delta_color="off", help="Trading the signal over the test period, after costs")

left, right = st.columns(2)
left.plotly_chart(charts.model_auc_chart(test, chosen), width="stretch")
with right:
    st.markdown("**All models on the test period**")
    scores = pd.DataFrame(test).T[["accuracy", "precision", "recall", "f1", "roc_auc"]]
    st.dataframe(scores.style.format("{:.3f}"), width="stretch")
    st.caption("`majority` always predicts the most common direction; `yesterday` repeats today's "
               "direction. A model is only useful if it beats both.")

importance = feature_importance(artifact["model"])
if importance is not None:
    st.plotly_chart(charts.feature_importance_chart(importance), width="stretch")

curves = cached_curves(info["id"], info["path"])
st.plotly_chart(charts.backtest_chart(curves, chosen), width="stretch")
st.caption(f"Test period {curves['timestamp'].iloc[0]:%d %b %Y} – {curves['timestamp'].iloc[-1]:%d %b %Y}. "
           "Long-only: hold a stock the next day when P(up) > 50%, else cash; equal weight across "
           f"stocks; {bt['cost_per_side']:.2%} cost per buy or sell.")

card = PROJECT_ROOT / "reports" / "model_card.md"
if card.exists():
    with st.expander("Full model card"):
        st.markdown(card.read_text(encoding="utf-8"))
