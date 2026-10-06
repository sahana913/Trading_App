"""
train.py - Train and compare every model, pick the best, save it.

Run from the project root (PowerShell), after the data pipeline:
    .venv\\Scripts\\python.exe -m src.ml.train

Steps:
  1. features from data/processed/candles.parquet          (features.py)
  2. split by date into train / validation / test          (split.py)
  3. walk-forward CV on the training dates, every model    (split.py)
  4. fit every model on train, score on validation         (evaluate.py)
  5. pick the best by validation ROC-AUC
  6. score every model once on test + backtest each one     (backtest.py)
  7. save the best model, registry row, model card          (report.py)
"""

import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sqlalchemy.orm import Session

from src.config import CANDLES_PARQUET, PROJECT_ROOT
from src.ml.backtest import backtest, cost_per_side
from src.ml.evaluate import classification_metrics
from src.ml.features import FEATURE_COLUMNS, TARGET_COLUMN, build_features, usable_rows
from src.ml.models import SEED, describe, make_models
from src.ml.report import next_version, register, save_model, write_model_card
from src.ml.split import time_split, walk_forward_folds

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_CARD = PROJECT_ROOT / "reports" / "model_card.md"
CV_FOLDS = 5


def set_seeds(seed: int) -> None:
    """Fix every random number generator we touch."""
    random.seed(seed)
    np.random.seed(seed)


def proba_up(model, X: pd.DataFrame) -> np.ndarray:
    """P(up) from any model. Column 1 of predict_proba is the 'up' class."""
    return model.predict_proba(X)[:, 1]


def cross_validate(models: dict, train: pd.DataFrame, n_splits: int) -> dict:
    """Mean and spread of ROC-AUC / accuracy over walk-forward folds."""
    out = {}
    folds = walk_forward_folds(train, n_splits=n_splits)
    for name, model in models.items():
        aucs, accs = [], []
        for fit_rows, test_rows in folds:
            fitted = clone(model).fit(train.loc[fit_rows, FEATURE_COLUMNS], train.loc[fit_rows, TARGET_COLUMN])
            m = classification_metrics(train.loc[test_rows, TARGET_COLUMN],
                                       proba_up(fitted, train.loc[test_rows, FEATURE_COLUMNS]))
            aucs.append(m["roc_auc"])
            accs.append(m["accuracy"])
        out[name] = {"roc_auc_mean": float(np.nanmean(aucs)), "roc_auc_std": float(np.nanstd(aucs)),
                     "accuracy_mean": float(np.mean(accs)), "folds": len(folds)}
    return out


def run_training(candles: pd.DataFrame, session: Session, models_dir: Path = MODELS_DIR,
                 card_path: Path = MODEL_CARD, seed: int = SEED, cv_folds: int = CV_FOLDS) -> dict:
    """Everything from candles to saved model. Returns all results."""
    set_seeds(seed)

    # 1-2. features and time split
    rows = usable_rows(build_features(candles))
    train, val, test = time_split(rows)
    X_train, y_train = train[FEATURE_COLUMNS], train[TARGET_COLUMN]

    # 3. walk-forward CV on the training dates
    models = make_models(seed)
    cv = cross_validate(models, train, cv_folds)

    # 4. fit on train, score on validation
    fitted = {name: clone(m).fit(X_train, y_train) for name, m in models.items()}
    val_scores = {name: classification_metrics(val[TARGET_COLUMN], proba_up(m, val[FEATURE_COLUMNS]))
                  for name, m in fitted.items()}

    # 5. choose by validation ROC-AUC (NaN counts as no skill)
    best = max(val_scores, key=lambda n: np.nan_to_num(val_scores[n]["roc_auc"], nan=0.5))

    # 6. the one and only look at the test period
    test_proba = {name: proba_up(m, test[FEATURE_COLUMNS]) for name, m in fitted.items()}
    test_scores = {name: classification_metrics(test[TARGET_COLUMN], p) for name, p in test_proba.items()}
    cost = cost_per_side()
    backtests = {name: backtest(test, p, cost) for name, p in test_proba.items()}

    importance = None
    if best == "lightgbm":
        gains = fitted["lightgbm"].booster_.feature_importance(importance_type="gain")
        importance = dict(zip(FEATURE_COLUMNS, (gains / gains.sum()).tolist()))

    # 7. save, register, document
    dates = {part: (df["timestamp"].min(), df["timestamp"].max())
             for part, df in (("train", train), ("val", val), ("test", test))}
    version = next_version(session)
    path = save_model(fitted[best], best, version, models_dir, {
        "seed": seed, "threshold": 0.5,
        "train_dates": [d.isoformat() for d in dates["train"]],
    })
    params = {"seed": seed, "model": best, "cv_folds": cv_folds,
              "estimator_params": {k: repr(v) for k, v in fitted[best].get_params().items()
                                   if not hasattr(v, "get_params")}}
    metrics = {"validation": val_scores[best], "test": test_scores[best], "cv": cv[best],
               "backtest": backtests[best], "all_models": {"validation": val_scores, "test": test_scores, "cv": cv}}
    entry = register(session, version=version, algorithm=describe(fitted[best]), path=path, params=params,
                     metrics=metrics, train_start=dates["train"][0].to_pydatetime(),
                     train_end=dates["train"][1].to_pydatetime())

    result = {
        "best": best, "version": version, "seed": seed, "generated": datetime.now(),
        "cv": cv, "val": val_scores, "test": test_scores, "backtest": backtests,
        "importance": importance, "dates": dates, "cv_folds": cv_folds, "cost_per_side": cost,
        "sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "n_rows": len(rows), "n_symbols": rows["symbol"].nunique(),
        "file_path": entry.file_path, "registry_id": entry.id,
    }
    write_model_card(card_path, result)
    return result


def main() -> None:
    from src.db.session import get_engine, get_session_factory, init_db

    if not CANDLES_PARQUET.exists():
        raise SystemExit(f"{CANDLES_PARQUET} not found. Run: python -m src.data.pipeline")
    candles = pd.read_parquet(CANDLES_PARQUET)
    engine = get_engine()
    init_db(engine)
    with get_session_factory(engine)() as session:
        r = run_training(candles, session)

    print(f"\nRows: {r['n_rows']:,}  (train {r['sizes']['train']:,} / val {r['sizes']['val']:,} "
          f"/ test {r['sizes']['test']:,})\n")
    print(f"{'model':<10} {'CV AUC':>8} {'val AUC':>8} {'test AUC':>9} {'test acc':>9} "
          f"{'backtest':>9}")
    for name in r["test"]:
        print(f"{name:<10} {r['cv'][name]['roc_auc_mean']:>8.3f} {r['val'][name]['roc_auc']:>8.3f} "
              f"{r['test'][name]['roc_auc']:>9.3f} {r['test'][name]['accuracy']:>9.1%} "
              f"{r['backtest'][name]['strategy']['total_return']:>+9.2%}")
    hold = r["backtest"][r["best"]]["buy_and_hold"]["total_return"]
    print(f"{'buy&hold':<10} {'':>8} {'':>8} {'':>9} {'':>9} {hold:>+9.2%}")
    print(f"\nBest (by validation AUC): {r['best']}  ->  {r['file_path']}  (registry v{r['version']})")
    print("Model card: reports/model_card.md")


if __name__ == "__main__":
    main()
