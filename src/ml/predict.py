"""
predict.py - Use the active model from model_registry inside the apps.

  latest_signals    P(up) for every stock at the simulated "now" (no look-ahead:
                    only candles up to the market clock are used)
  feature_importance  which inputs the model relies on most
  backtest_equity_curves  the model's backtest over its test period, day by day
"""

from datetime import datetime
from functools import cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.config import PROJECT_ROOT
from src.db.models import Candle, Instrument, ModelRegistry
from src.ml.backtest import equity_curves
from src.ml.evaluate import THRESHOLD
from src.ml.features import FEATURE_COLUMNS, build_features, usable_rows
from src.ml.report import MODEL_NAME
from src.ml.split import time_split


def active_model(session: Session) -> ModelRegistry | None:
    """The registry row marked active (the one the app should use)."""
    return session.scalar(select(ModelRegistry).where(ModelRegistry.name == MODEL_NAME,
                                                      ModelRegistry.is_active))


def model_file(entry: ModelRegistry) -> Path:
    path = Path(entry.file_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


@cache
def _load(path: str) -> dict:
    """Read a .joblib file once per process (loading is slow-ish; models don't change)."""
    return joblib.load(path)


def load_artifact(entry: ModelRegistry) -> dict:
    """{"model": fitted model, "features": [...], ...} as saved by report.save_model."""
    return _load(str(model_file(entry)))


def candles_frame(session: Session, as_of: datetime | None = None,
                  include_synthetic: bool = True) -> pd.DataFrame:
    """All candles (optionally up to as_of) as a DataFrame with a symbol column."""
    query = (select(Instrument.symbol, Candle.timestamp, Candle.open, Candle.high, Candle.low,
                    Candle.close, Candle.volume)
             .join(Instrument, Candle.instrument_id == Instrument.id))
    if not include_synthetic:
        query = query.where(Candle.is_synthetic.is_not(True))  # real history only
    if as_of is not None:
        query = query.where(Candle.timestamp <= as_of)  # the model must not see the future
    return pd.DataFrame(session.execute(query).all(),
                        columns=["symbol", "timestamp", "open", "high", "low", "close", "volume"])


def latest_signals(session: Session, artifact: dict, as_of: datetime | None) -> pd.DataFrame:
    """One row per stock: symbol, timestamp (the bar used), close, proba_up,
    signal ("UP" / "DOWN" / None if not enough history yet)."""
    feats = build_features(candles_frame(session, as_of))
    latest = feats.groupby("symbol").tail(1).reset_index(drop=True)  # each stock's newest bar
    ready = latest[artifact["features"]].notna().all(axis=1)
    latest["proba_up"] = np.nan
    if ready.any():
        latest.loc[ready, "proba_up"] = artifact["model"].predict_proba(
            latest.loc[ready, artifact["features"]])[:, 1]
    latest["signal"] = np.where(~ready, None, np.where(latest["proba_up"] > THRESHOLD, "UP", "DOWN"))
    return latest[["symbol", "timestamp", "close", "proba_up", "signal"]]


def feature_importance(model) -> pd.Series | None:
    """Share of importance per feature (sums to 1), biggest first.

    LightGBM: total "gain" = how much each feature improved the trees' splits.
    Logistic regression: size of each coefficient. Features were standardised
    first, so a bigger |coefficient| means a bigger effect per standard deviation.
    Baselines learn nothing, so they have none (None).
    """
    if isinstance(model, LGBMClassifier):
        raw = model.booster_.feature_importance(importance_type="gain")
    elif isinstance(model, Pipeline):
        raw = np.abs(model[-1].coef_[0])
    else:
        return None
    share = pd.Series(raw / raw.sum(), index=FEATURE_COLUMNS)
    return share.sort_values(ascending=False)


def backtest_equity_curves(session: Session, artifact: dict) -> pd.DataFrame:
    """Re-run the backtest over the model's test period (the same split as
    training used), returning daily equity for the strategy and buy-and-hold."""
    rows = usable_rows(build_features(candles_frame(session)))
    _, _, test = time_split(rows)
    proba = artifact["model"].predict_proba(test[artifact["features"]])[:, 1]
    return equity_curves(test, proba)
