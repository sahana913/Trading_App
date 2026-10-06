"""Tests for using the model in the apps (src/ml/predict.py), the anomaly
detector (src/ml/anomaly.py), and the AI Insights / admin pages."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sqlalchemy import select
from streamlit.testing.v1 import AppTest

from src.db.models import Candle, Instrument, User
from src.db.session import get_engine, get_session_factory, init_db
from src.ml.anomaly import BEHAVIOUR_FEATURES, behaviour_frame, detect_anomalies, robust_z
from src.ml.features import build_features
from src.ml.models import make_models
from src.ml.predict import (
    active_model, backtest_equity_curves, feature_importance, latest_signals, load_artifact,
)
from src.ml.train import run_training
from src.trading import create_user, placeorder
from src.trading.simulator import get_clock, run, start
from tests.test_ml import random_walk

APP_DIR = Path(__file__).resolve().parent.parent / "app"


def load_candles(session, candles: pd.DataFrame) -> None:
    """Put a candles DataFrame into the database, one instrument per symbol."""
    for symbol, bars in candles.groupby("symbol"):
        inst = Instrument(symbol=symbol, exchange="NSE", name=symbol, lot_size=1, tick_size=0.05, is_active=True)
        session.add(inst)
        session.flush()
        session.add_all(Candle(instrument_id=inst.id, timestamp=r.timestamp.to_pydatetime(), open=r.open,
                               high=r.high, low=r.low, close=r.close, volume=int(r.volume))
                        for r in bars.itertuples())
    session.commit()


@pytest.fixture
def trained(tmp_path):
    """A database with 3 stocks x 400 days and a trained, registered model."""
    url = f"sqlite:///{(tmp_path / 'ml.db').as_posix()}"
    engine = get_engine(url)
    init_db(engine)
    factory = get_session_factory(engine)
    candles = random_walk(symbols=("AAA", "BBB", "CCC"), days=400)
    with factory() as s:
        load_candles(s, candles)
        run_training(candles, s, models_dir=tmp_path / "models", card_path=tmp_path / "card.md", cv_folds=3)
    return {"url": url, "factory": factory, "candles": candles}


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------
def test_signals_use_only_data_up_to_the_clock(trained):
    candles = trained["candles"]
    as_of = pd.Timestamp(np.sort(candles["timestamp"].unique())[200]).to_pydatetime()
    with trained["factory"]() as s:
        artifact = load_artifact(active_model(s))
        signals = latest_signals(s, artifact, as_of)

    assert list(signals["symbol"]) == ["AAA", "BBB", "CCC"]
    assert (signals["timestamp"] == as_of).all()  # the bar used is "today", never later
    # Same probabilities as predicting from features of data that STOPS at as_of
    cut = build_features(candles[candles["timestamp"] <= as_of]).groupby("symbol").tail(1)
    expected = artifact["model"].predict_proba(cut[artifact["features"]])[:, 1]
    assert signals["proba_up"].to_numpy() == pytest.approx(expected)
    assert set(signals["signal"]) <= {"UP", "DOWN"}
    assert ((signals["proba_up"] > 0.5) == (signals["signal"] == "UP")).all()


def test_no_signal_without_enough_history(trained):
    early = pd.Timestamp(np.sort(trained["candles"]["timestamp"].unique())[20]).to_pydatetime()
    with trained["factory"]() as s:
        signals = latest_signals(s, load_artifact(active_model(s)), early)
    assert signals["signal"].isna().all() and signals["proba_up"].isna().all()


def test_feature_importance_shares():
    candles = random_walk(days=200)
    from src.ml.features import FEATURE_COLUMNS, usable_rows
    rows = usable_rows(build_features(candles))
    X, y = rows[FEATURE_COLUMNS], rows["target"]
    models = make_models()
    for name in ("lightgbm", "logistic"):
        share = feature_importance(models[name].fit(X, y))
        assert share.sum() == pytest.approx(1.0)
        assert list(share) == sorted(share, reverse=True)  # biggest first
    assert feature_importance(DummyClassifier().fit(X, y)) is None


def test_backtest_curve_matches_registry_numbers(trained):
    with trained["factory"]() as s:
        entry = active_model(s)
        curves = backtest_equity_curves(s, load_artifact(entry))
        recorded = entry.metrics["backtest"]
    assert curves["strategy"].iloc[-1] - 1 == pytest.approx(recorded["strategy"]["total_return"])
    assert curves["buy_and_hold"].iloc[-1] - 1 == pytest.approx(recorded["buy_and_hold"]["total_return"])


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------
def behaviour(n_normal: int = 9, seed: int = 0) -> pd.DataFrame:
    """n_normal similar traders plus one who makes huge trades."""
    rng = np.random.default_rng(seed)
    rows = [{"username": f"user{i}", "trades_per_day": rng.normal(2, 0.3),
             "avg_trade_value": rng.normal(20_000, 2_000), "max_trade_share": rng.normal(0.05, 0.01),
             "pnl_volatility": rng.normal(0.01, 0.002), "worst_day": rng.normal(-0.02, 0.004),
             "rejection_rate": rng.normal(0.05, 0.01)} for i in range(n_normal)]
    rows.append({"username": "whale", "trades_per_day": 2.1, "avg_trade_value": 21_000,
                 "max_trade_share": 0.90, "pnl_volatility": 0.011, "worst_day": -0.021,
                 "rejection_rate": 0.05})
    return pd.DataFrame(rows)


def test_robust_z_by_hand():
    # median 3; |x - 3| = 2, 1, 0, 1, 97 -> MAD 1 -> scale 1.4826
    z = robust_z(pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 100.0]}))["x"]
    assert z.iloc[2] == 0
    assert z.iloc[4] == pytest.approx(97 / 1.4826)


@pytest.mark.parametrize("seed", range(5))
def test_extreme_value_is_flagged_and_explained(seed):
    """One trade worth 90% of the account: caught by the extreme-value rule."""
    result = detect_anomalies(behaviour(seed=seed))
    top = result.iloc[0]
    assert top["username"] == "whale" and bool(top["is_anomaly"])   # flagged users come first
    assert top["flagged_by"] in ("extreme value", "both")
    assert "largest trade vs account" in top["reason"]
    assert result["is_anomaly"].sum() == 1                          # nobody else


@pytest.mark.parametrize("seed", range(5))
def test_unusual_combination_is_flagged_by_the_forest(seed):
    """Moderately unusual on EVERY measure at once: no single value is extreme,
    but IsolationForest isolates the pattern."""
    b = behaviour(seed=seed).iloc[:-1].copy()
    b.loc[len(b)] = {"username": "reckless", "trades_per_day": 3.5, "avg_trade_value": 28_000,
                     "max_trade_share": 0.09, "pnl_volatility": 0.018, "worst_day": -0.04,
                     "rejection_rate": 0.10}
    result = detect_anomalies(b)
    flagged = result[result["is_anomaly"]]
    assert list(flagged["username"]) == ["reckless"]
    assert flagged.iloc[0]["flagged_by"] in ("pattern", "both")


@pytest.mark.parametrize("seed", range(5))
def test_nobody_flagged_when_everyone_is_normal(seed):
    assert not detect_anomalies(behaviour(seed=seed).iloc[:-1])["is_anomaly"].any()


def test_detector_is_reproducible():
    a, b = detect_anomalies(behaviour()), detect_anomalies(behaviour())
    pd.testing.assert_frame_equal(a, b)


def test_detector_needs_enough_users():
    result = detect_anomalies(behaviour(n_normal=3))  # 4 users < 5
    assert not result["is_anomaly"].any()
    assert "at least 5" in result["reason"].iloc[0]


def test_behaviour_features_by_hand(tmp_path):
    engine = get_engine("sqlite:///:memory:")
    init_db(engine)
    with get_session_factory(engine)() as s:
        load_candles(s, random_walk(symbols=("AAA",), days=30))
        alice = create_user(s, "alice", "password1")
        create_user(s, "bob", "password1")                   # never trades: left out
        create_user(s, "boss", "password1", role="admin")    # admins are left out
        start(s, datetime(2023, 1, 2))
        price = s.scalar(select(Candle.close).order_by(Candle.timestamp))
        for qty in (10, 5):
            placeorder(s, alice.id, "AAA", "NSE", "BUY", qty, product="CNC", as_of=get_clock(s))
        placeorder(s, alice.id, "AAA", "NSE", "BUY", 100_000, product="CNC", as_of=get_clock(s))  # rejected
        run(s, 2)  # two daily snapshots

        df = behaviour_frame(s)
    assert list(df["username"]) == ["alice"]
    row = df.iloc[0]
    assert row["trades"] == 2
    assert row["trades_per_day"] == pytest.approx(2 / 2)
    assert row["avg_trade_value"] == pytest.approx(7.5 * price)            # (10 + 5) / 2 shares
    assert row["max_trade_share"] == pytest.approx(10 * price / 1_000_000)
    assert row["rejection_rate"] == pytest.approx(1 / 3)
    assert set(BEHAVIOUR_FEATURES) <= set(df.columns)


# ---------------------------------------------------------------------------
# The pages
# ---------------------------------------------------------------------------
def test_ai_insights_page(trained, monkeypatch):
    monkeypatch.setenv("PAPER_TRADING_DB_URL", trained["url"])
    with trained["factory"]() as s:
        create_user(s, "alice", "password1")
        start(s, datetime(2024, 3, 1))
    at = AppTest.from_file(str(APP_DIR / "trader" / "app.py")).run(timeout=30)
    at.text_input(key="login_username").input("alice")
    at.text_input(key="login_password").input("password1")
    at.button(key="login_submit").click()
    at.run(timeout=30)
    at.switch_page("pages/ai_insights.py")
    at.run(timeout=60)
    assert not at.exception, at.exception
    assert at.title[0].value == "AI Insights"
    assert "not financial advice" in at.warning[0].value
    signals = at.dataframe[0].value
    assert list(signals["Symbol"]) == ["AAA", "BBB", "CCC"]
    assert {m.label for m in at.metric} >= {"Test ROC-AUC", "Test accuracy", "Backtest return"}
    assert len(at.get("plotly_chart")) >= 2  # AUC chart + backtest (+ importance if LightGBM)
