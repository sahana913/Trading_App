"""Tests for src/ml: no look-ahead in features, honest splits, hand-checked
metrics and backtest maths, and a full reproducible training run."""

import joblib
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select

from src.db.models import ModelRegistry
from src.db.session import get_engine, get_session_factory, init_db
from src.ml.backtest import backtest, daily_strategy_returns
from src.ml.evaluate import classification_metrics
from src.ml.features import FEATURE_COLUMNS, TARGET_COLUMN, build_features, rsi, usable_rows
from src.ml.models import YesterdayDirection
from src.ml.split import time_split, walk_forward_folds
from src.ml.train import run_training


def random_walk(symbols=("AAA", "BBB"), days=260, seed=0) -> pd.DataFrame:
    """Fake daily candles: a random walk per stock on business days."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=days)
    frames = []
    for i, sym in enumerate(symbols):
        close = 100 * (i + 1) * np.exp(np.cumsum(rng.normal(0, 0.015, days)))
        frames.append(pd.DataFrame({
            "symbol": sym, "timestamp": dates, "open": close, "high": close * 1.01,
            "low": close * 0.99, "close": close, "volume": rng.integers(1e5, 1e6, days),
        }))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# No look-ahead
# ---------------------------------------------------------------------------
def features_up_to(feats: pd.DataFrame, cutoff) -> pd.DataFrame:
    rows = feats[feats["timestamp"] <= cutoff].sort_values(["symbol", "timestamp"])
    return rows[FEATURE_COLUMNS].reset_index(drop=True)


def test_features_never_use_the_future():
    """For many days t: features computed from data that STOPS at t must equal
    the features computed from the full history. If any feature peeked at a
    later day, the two would differ."""
    candles = random_walk()
    full = build_features(candles)
    dates = np.sort(candles["timestamp"].unique())
    for cutoff in dates[5::11]:  # ~23 different cut days, from very early to the end
        cut = build_features(candles[candles["timestamp"] <= cutoff])
        pd.testing.assert_frame_equal(features_up_to(cut, cutoff), features_up_to(full, cutoff),
                                      check_exact=False, rtol=1e-9)


def test_changing_future_prices_does_not_change_past_features():
    candles = random_walk()
    cutoff = candles["timestamp"].sort_values().unique()[150]
    shocked = candles.copy()
    later = shocked["timestamp"] > cutoff
    shocked.loc[later, ["open", "high", "low", "close"]] *= 3   # a wild future
    shocked.loc[later, "volume"] *= 10
    pd.testing.assert_frame_equal(features_up_to(build_features(candles), cutoff),
                                  features_up_to(build_features(shocked), cutoff))


def test_stocks_do_not_leak_into_each_other():
    candles = random_walk()
    alone = build_features(candles[candles["symbol"] == "AAA"])
    together = build_features(candles)
    pd.testing.assert_frame_equal(
        alone[FEATURE_COLUMNS].reset_index(drop=True),
        together[together["symbol"] == "AAA"][FEATURE_COLUMNS].reset_index(drop=True))


def test_answers_are_not_features():
    assert TARGET_COLUMN not in FEATURE_COLUMNS
    assert "next_return" not in FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# Hand-checkable feature and target values
# ---------------------------------------------------------------------------
def tiny(closes) -> pd.DataFrame:
    n = len(closes)
    return pd.DataFrame({"symbol": "X", "timestamp": pd.bdate_range("2024-01-01", periods=n),
                         "open": closes, "high": closes, "low": closes, "close": closes,
                         "volume": [100.0] * n})


def test_target_is_next_day_direction():
    f = build_features(tiny([100.0, 101.0, 100.0, 100.0, 102.0]))
    assert f["next_return"].iloc[:4].tolist() == pytest.approx([0.01, 100 / 101 - 1, 0.0, 0.02])
    # unchanged (0%) counts as "not up"; the last day has no tomorrow
    assert f["target"].iloc[:4].tolist() == [1.0, 0.0, 0.0, 1.0]
    assert np.isnan(f["target"].iloc[4])


def test_lagged_returns_and_ma_ratio():
    f = build_features(tiny([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    # day 5 (close 6): today's return 6/5-1, yesterday's 5/4-1, two days ago 4/3-1
    assert f.loc[5, ["ret_1", "ret_2", "ret_3"]].tolist() == pytest.approx([0.2, 0.25, 1 / 3])
    # 5-day average of 2..6 is 4, so close 6 is 50% above it
    assert f.loc[5, "ma_ratio_5"] == pytest.approx(0.5)


def test_rsi_extremes():
    up = pd.Series(np.arange(1.0, 41.0))      # only gains
    down = pd.Series(np.arange(40.0, 0.0, -1))  # only losses
    assert rsi(up).iloc[-1] == 100
    assert rsi(down).iloc[-1] == pytest.approx(0)
    assert rsi(up).iloc[:14].isna().all()      # not enough history yet


def test_usable_rows_drop_warm_up_and_last_day():
    candles = random_walk(days=120)
    rows = usable_rows(build_features(candles))
    assert rows[FEATURE_COLUMNS + [TARGET_COLUMN]].notna().all().all()
    # 50-day average needs 50 days; the last day has no target
    assert len(rows) == 2 * (120 - 49 - 1)


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def test_time_split_is_ordered_with_a_gap():
    rows = usable_rows(build_features(random_walk()))
    train, val, test = time_split(rows)
    dates = np.sort(rows["timestamp"].unique())
    assert train["timestamp"].max() < val["timestamp"].min() < val["timestamp"].max() < test["timestamp"].min()
    # exactly one date skipped between parts
    assert np.searchsorted(dates, val["timestamp"].min()) - np.searchsorted(dates, train["timestamp"].max()) == 2
    assert np.searchsorted(dates, test["timestamp"].min()) - np.searchsorted(dates, val["timestamp"].max()) == 2
    # a date's stocks are never split across parts
    for a, b in ((train, val), (val, test), (train, test)):
        assert not set(a["timestamp"]) & set(b["timestamp"])


def test_walk_forward_folds_only_test_the_future():
    rows = usable_rows(build_features(random_walk()))
    folds = walk_forward_folds(rows, n_splits=4)
    assert len(folds) == 4
    previous_train = 0
    for fit_rows, test_rows in folds:
        assert rows.loc[fit_rows, "timestamp"].max() < rows.loc[test_rows, "timestamp"].min()
        assert len(fit_rows) > previous_train  # the training window grows
        previous_train = len(fit_rows)


# ---------------------------------------------------------------------------
# Models, metrics, backtest (hand-checked)
# ---------------------------------------------------------------------------
def test_yesterday_baseline():
    X = pd.DataFrame({"ret_1": [0.02, -0.01, 0.0]})
    assert YesterdayDirection().fit(X).predict(X).tolist() == [1.0, 0.0, 0.0]


def test_classification_metrics_by_hand():
    # predictions at 0.5: up, up, down, up   vs truth: up, down, up, up
    m = classification_metrics([1, 0, 1, 1], [0.9, 0.6, 0.4, 0.8])
    assert m["accuracy"] == 0.5                       # 2 of 4 right
    assert m["precision"] == pytest.approx(2 / 3)     # said up 3 times, right twice
    assert m["recall"] == pytest.approx(2 / 3)        # 3 real ups, caught 2
    assert m["f1"] == pytest.approx(2 / 3)
    # pairs (up-day score > down-day score): 0.9>0.6 yes, 0.4>0.6 no, 0.8>0.6 yes
    assert m["roc_auc"] == pytest.approx(2 / 3)


def test_strategy_returns_with_costs_by_hand():
    r = daily_strategy_returns(pd.Series([0.01, -0.02, 0.03]), pd.Series([1, 0, 1]), cost=0.001)
    # day 1: buy (cost) and earn 1%; day 2: sell (cost), in cash;
    # day 3: buy (cost), earn 3%, then the final exit (cost)
    assert r.tolist() == pytest.approx([0.009, -0.001, 0.028])


def test_always_invested_equals_buy_and_hold():
    rows = usable_rows(build_features(random_walk()))
    result = backtest(rows, np.ones(len(rows)), cost=0.001)
    assert result["strategy"]["total_return"] == pytest.approx(result["buy_and_hold"]["total_return"])
    assert result["strategy"]["trades"] == result["buy_and_hold"]["trades"] == 4  # in + out, 2 stocks


def test_never_invested_makes_nothing():
    rows = usable_rows(build_features(random_walk()))
    result = backtest(rows, np.zeros(len(rows)), cost=0.001)
    assert result["strategy"]["total_return"] == 0 and result["strategy"]["trades"] == 0


# ---------------------------------------------------------------------------
# Full training run
# ---------------------------------------------------------------------------
@pytest.fixture
def session():
    engine = get_engine("sqlite:///:memory:")
    init_db(engine)
    with get_session_factory(engine)() as s:
        yield s


def test_training_saves_registers_and_is_reproducible(session, tmp_path):
    candles = random_walk(symbols=("AAA", "BBB", "CCC"), days=400)
    first = run_training(candles, session, models_dir=tmp_path / "models",
                         card_path=tmp_path / "model_card.md", cv_folds=3)
    second = run_training(candles, session, models_dir=tmp_path / "models",
                          card_path=tmp_path / "model_card.md", cv_folds=3)

    # Same data + same seed -> identical results
    assert first["best"] == second["best"]
    assert first["test"] == second["test"]
    assert first["cv"] == second["cv"]

    # Registry: two versions, only the newest active, with features/dates/seed
    rows = session.scalars(select(ModelRegistry).order_by(ModelRegistry.version)).all()
    assert [r.version for r in rows] == [1, 2]
    assert [r.is_active for r in rows] == [False, True]
    latest = rows[-1]
    assert latest.features == FEATURE_COLUMNS
    assert latest.params["seed"] == 42
    assert latest.train_start < latest.train_end
    assert set(latest.metrics) >= {"validation", "test", "cv", "backtest"}

    # The saved file loads and predicts
    saved_path = tmp_path / "models" / f"direction_v2_{second['best']}.joblib"
    assert latest.file_path == saved_path.as_posix()  # outside the project: stored as-is
    saved = joblib.load(saved_path)
    rows_ = usable_rows(build_features(candles)).tail(5)
    proba = saved["model"].predict_proba(rows_[saved["features"]])[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()

    # The model card has the key sections
    card = (tmp_path / "model_card.md").read_text(encoding="utf-8")
    for heading in ("## Data", "### Test (held out)", "## Backtest on the test period",
                    "## Honest interpretation", "Not financial advice"):
        assert heading in card
