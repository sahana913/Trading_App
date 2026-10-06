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
from src.trading.intraday import day_start, is_intraday


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
    if is_intraday(as_of):
        # Mid-session: today's daily candle isn't finished, so the model only
        # sees completed days (its features are built on daily closes)
        query = query.where(Candle.timestamp != day_start(as_of))
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


# Plain-English names and display formats for the explanation chart
FEATURE_LABELS = {
    "ret_1": ("Today's return", "pct"), "ret_2": ("Yesterday's return", "pct"),
    "ret_3": ("Return 2 days ago", "pct"), "ret_4": ("Return 3 days ago", "pct"),
    "ret_5": ("Return 4 days ago", "pct"), "ret_5d": ("5-day return", "pct"), "ret_20d": ("20-day return", "pct"),
    "ma_ratio_5": ("Price vs 5-day average", "pct"), "ma_ratio_10": ("Price vs 10-day average", "pct"),
    "ma_ratio_20": ("Price vs 20-day average", "pct"), "ma_ratio_50": ("Price vs 50-day average", "pct"),
    "rsi_14": ("RSI (14)", "num"), "macd": ("MACD", "pct"), "macd_signal": ("MACD signal", "pct"),
    "macd_hist": ("MACD histogram", "pct"), "bb_position": ("Bollinger position", "num"),
    "vol_10": ("10-day volatility", "pct"), "vol_20": ("20-day volatility", "pct"),
    "volume_change": ("Volume change", "pct"), "volume_ratio_20": ("Volume vs 20-day avg", "x"),
}


def explain_signal(session: Session, artifact: dict, symbol: str,
                   as_of: datetime | None) -> tuple[pd.DataFrame, float] | None:
    """Why the model gives `symbol` its P(up) right now.

    Returns (table, base) where table has one row per feature, biggest push
    first: feature, label, value_text, contribution. Contributions are in
    log-odds and add up exactly:  base + sum(contributions) = log(p / (1 - p)).
      * LightGBM: TreeSHAP values, built in (predict(..., pred_contrib=True)).
      * Logistic regression: coefficient x standardised feature value
        (a linear model's explanation is exact by construction).
    Baselines learn nothing, so they return None. Also None without enough history.
    """
    model = artifact["model"]
    feats = build_features(candles_frame(session, as_of))
    row = feats[feats["symbol"] == symbol].tail(1)
    if row.empty or row[artifact["features"]].isna().any(axis=None):
        return None
    X = row[artifact["features"]]

    if isinstance(model, LGBMClassifier):
        out = model.predict(X, pred_contrib=True)[0]
        contrib, base = out[:-1], float(out[-1])     # last column = the expected (base) value
    elif isinstance(model, Pipeline):
        scaler, logreg = model[0], model[-1]
        contrib = scaler.transform(X)[0] * logreg.coef_[0]
        base = float(logreg.intercept_[0])
    else:
        return None

    def show(feature: str, value: float) -> str:
        kind = FEATURE_LABELS.get(feature, (feature, "num"))[1]
        return f"{value:+.2%}" if kind == "pct" else f"{value:.2f}×" if kind == "x" else f"{value:.2f}"

    table = pd.DataFrame({
        "feature": artifact["features"],
        "label": [FEATURE_LABELS.get(f, (f, ""))[0] for f in artifact["features"]],
        "value_text": [show(f, float(X[f].iloc[0])) for f in artifact["features"]],
        "contribution": contrib,
    })
    order = table["contribution"].abs().sort_values(ascending=False).index
    return table.loc[order].reset_index(drop=True), base


def market_snapshot(session: Session, as_of: datetime | None, sectors: dict) -> pd.DataFrame:
    """Every active stock right now: symbol, sector, ltp, change_pct (vs the
    previous close) and turnover (₹ crore traded so far today), for the heatmap."""
    from src.trading.market import quotes  # local import: market imports nothing from ml

    rows = []
    for inst in session.scalars(select(Instrument).where(Instrument.is_active).order_by(Instrument.symbol)):
        q = quotes(session, inst.symbol, inst.exchange, as_of=as_of)
        if q["status"] != "success":
            continue
        q = q["data"]
        rows.append({"symbol": inst.symbol, "sector": sectors.get(inst.symbol, "Other"), "ltp": q["ltp"],
                     "change_pct": (q["ltp"] / q["prev_close"] - 1) * 100,
                     "turnover": max(q["ltp"] * q["volume"] / 1e7, 0.01)})  # ₹ crore; tiny floor keeps every tile visible
    return pd.DataFrame(rows, columns=["symbol", "sector", "ltp", "change_pct", "turnover"])
