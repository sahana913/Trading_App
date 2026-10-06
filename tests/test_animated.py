"""Tests for the animated visuals (src/analytics/animated.py) and the signal
explanations (src/ml/predict.explain_signal)."""

import numpy as np
import pandas as pd
import pytest

from src.analytics import animated
from src.ml.features import FEATURE_COLUMNS, build_features, usable_rows
from src.ml.models import make_models
from tests.test_ml import random_walk


def logit(p: float) -> float:
    return float(np.log(p / (1 - p)))


# ---------------------------------------------------------------------------
# Animations open on the finished picture
# ---------------------------------------------------------------------------
def test_equity_replay_starts_complete_and_replays():
    curve = pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=100),
                          "equity": 1_000_000 + np.arange(100) * 250.0})
    fig = animated.equity_replay(curve, 1_000_000)
    assert len(fig.data[0].x) == 100                     # opens with the whole curve
    assert 2 <= len(fig.frames) <= 60                    # at most 60 frames, however long the history
    assert len(fig.frames[-1].data[0].x) == 100          # the last frame is the full curve again
    assert fig.layout.updatemenus[0].buttons[0].label == "▶ Play"


def test_leaderboard_race_reorders_and_keeps_colours():
    dates = pd.bdate_range("2024-01-01", periods=3)
    # bob leads on day 1, alice overtakes by day 3
    daily = pd.DataFrame([("alice", dates[0], 1.00e6), ("bob", dates[0], 1.02e6),
                          ("alice", dates[1], 1.03e6), ("bob", dates[1], 1.02e6),
                          ("alice", dates[2], 1.05e6), ("bob", dates[2], 1.01e6)],
                         columns=["username", "date", "equity"])
    fig = animated.leaderboard_race(daily)
    assert len(fig.frames) == 3
    assert list(fig.frames[0].data[0].y) == ["alice", "bob"]   # bars bottom-to-top: leader drawn last (on top)
    assert list(fig.frames[2].data[0].y) == ["bob", "alice"]
    colour = lambda frame, user: frame.data[0].marker.color[list(frame.data[0].y).index(user)]  # noqa: E731
    assert colour(fig.frames[0], "alice") == colour(fig.frames[2], "alice")  # colour follows the trader


# ---------------------------------------------------------------------------
# Forecast fan maths
# ---------------------------------------------------------------------------
def test_simulated_paths_start_today_and_have_no_drift():
    paths = animated.simulate_paths(100.0, 0.02, days=20, sims=20_000, seed=1)
    assert paths.shape == (20_000, 21)
    assert (paths[:, 0] == 100).all()
    assert paths[:, -1].mean() == pytest.approx(100, rel=0.01)       # the average future price = today's
    assert np.log(paths[:, -1] / 100).std() == pytest.approx(0.02 * np.sqrt(20), rel=0.03)  # spread grows with sqrt(time)


def test_simulated_paths_are_reproducible():
    a = animated.simulate_paths(50.0, 0.01, 10, 100, seed=3)
    assert (a == animated.simulate_paths(50.0, 0.01, 10, 100, seed=3)).all()


def test_forecast_fan_bands_are_ordered():
    candles = random_walk(symbols=("AAA",), days=120)
    fig, info = animated.forecast_fan(candles[["timestamp", "close"]], "AAA")
    assert info["p5"] < info["p50"] < info["p95"]
    assert info["last"] == pytest.approx(candles["close"].iloc[-1])
    assert len(fig.frames) == 20                          # one frame per future trading day


# ---------------------------------------------------------------------------
# Explanations add up exactly
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["lightgbm", "logistic"])
def test_explanation_adds_up_to_the_prediction(name, tmp_path):
    """base + sum of contributions = the model's log-odds, so the waterfall
    ends exactly at the predicted probability."""
    from src.db.session import get_engine, get_session_factory, init_db
    from src.ml.predict import explain_signal
    from tests.test_ml_insights import load_candles

    candles = random_walk(symbols=("AAA", "BBB"), days=200)
    rows = usable_rows(build_features(candles))
    model = make_models()[name].fit(rows[FEATURE_COLUMNS], rows["target"])
    artifact = {"model": model, "features": FEATURE_COLUMNS}

    engine = get_engine("sqlite:///:memory:")
    init_db(engine)
    with get_session_factory(engine)() as s:
        load_candles(s, candles)
        table, base = explain_signal(s, artifact, "AAA", None)
    last = build_features(candles).query("symbol == 'AAA'").tail(1)[FEATURE_COLUMNS]
    p = model.predict_proba(last)[0, 1]
    assert base + table["contribution"].sum() == pytest.approx(logit(p), abs=1e-6)
    assert list(table["contribution"].abs()) == sorted(table["contribution"].abs(), reverse=True)
    fig = animated.signal_waterfall(table, base, "AAA")
    assert fig.data[0].text[-1] == f"{p:.1%}"               # the chart ends at the model's probability


def test_baselines_have_no_explanation():
    from sklearn.dummy import DummyClassifier
    from src.ml.predict import explain_signal

    candles = random_walk(days=120)
    rows = usable_rows(build_features(candles))
    artifact = {"model": DummyClassifier().fit(rows[FEATURE_COLUMNS], rows["target"]), "features": FEATURE_COLUMNS}

    from src.db.session import get_engine, get_session_factory, init_db
    from tests.test_ml_insights import load_candles
    engine = get_engine("sqlite:///:memory:")
    init_db(engine)
    with get_session_factory(engine)() as s:
        load_candles(s, candles)
        assert explain_signal(s, artifact, "AAA", None) is None


# ---------------------------------------------------------------------------
# Correlation network
# ---------------------------------------------------------------------------
def test_correlated_stocks_settle_closer_together():
    names = ["A", "B", "C", "D"]
    corr = pd.DataFrame([[1, 0.8, 0.1, 0.1], [0.8, 1, 0.1, 0.1], [0.1, 0.1, 1, 0.7], [0.1, 0.1, 0.7, 1]],
                        index=names, columns=names)
    final = animated.force_layout(corr, threshold=0.5)[-1]
    dist = lambda i, j: np.linalg.norm(final[i] - final[j])  # noqa: E731
    assert dist(0, 1) < dist(0, 2) and dist(2, 3) < dist(1, 3)   # pairs end up close, clusters apart
    again = animated.force_layout(corr, threshold=0.5)[-1]
    assert np.allclose(final, again)                              # same seed, same picture


def test_network_and_heatmap_build():
    candles = random_walk(symbols=("AAA", "BBB", "CCC"), days=80)
    wide = candles.pivot_table(index="timestamp", columns="symbol", values="close")
    fig = animated.correlation_network(wide.pct_change().dropna(), {"AAA": "Banks"}, threshold=-1.0)
    assert len(fig.frames) > 10
    snapshot = pd.DataFrame({"symbol": ["AAA", "BBB"], "sector": ["Banks", "IT"], "ltp": [10.0, 20.0],
                             "change_pct": [1.2, -0.4], "turnover": [100.0, 50.0]})
    heat = animated.market_heatmap(snapshot)
    assert heat.data[0].type == "treemap"
