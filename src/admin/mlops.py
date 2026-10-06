"""mlops.py - The model registry seen from the admin side: compare versions,
retrain from the database, choose which version the apps use."""

import pandas as pd
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from src.admin.audit import log_action, require_admin
from src.db.models import ModelRegistry
from src.ml.predict import candles_frame
from src.ml.report import MODEL_NAME
from src.ml.train import run_training
from src.trading.market import error, success


def registry_frame(session: Session) -> pd.DataFrame:
    """One row per model version with its key numbers."""
    rows = []
    for m in session.scalars(select(ModelRegistry).where(ModelRegistry.name == MODEL_NAME)
                             .order_by(ModelRegistry.version.desc())):
        met = m.metrics or {}
        bt = met.get("backtest", {})
        rows.append({
            "version": m.version, "active": m.is_active, "algorithm": m.algorithm, "created": m.created_at,
            "train_from": m.train_start, "train_to": m.train_end,
            "val_auc": met.get("validation", {}).get("roc_auc"),
            "test_auc": met.get("test", {}).get("roc_auc"),
            "test_accuracy": met.get("test", {}).get("accuracy"),
            "backtest_return": bt.get("strategy", {}).get("total_return"),
            "buy_and_hold": bt.get("buy_and_hold", {}).get("total_return"),
            "seed": (m.params or {}).get("seed"), "file": m.file_path,
        })
    return pd.DataFrame(rows, columns=["version", "active", "algorithm", "created", "train_from", "train_to",
                                       "val_auc", "test_auc", "test_accuracy", "backtest_return",
                                       "buy_and_hold", "seed", "file"])


def retrain(session: Session, admin_id: int, **paths) -> dict:
    """Train every model again on the REAL candles in the database (synthetic
    bars are left out: random-walk prices hold nothing to learn), register
    the best as a new active version, and log it. `paths` (models_dir,
    card_path) can redirect the output files, e.g. in tests."""
    require_admin(session, admin_id)
    candles = candles_frame(session, include_synthetic=False)
    if candles.empty:
        return error("No candles in the database to train on")
    r = run_training(candles, session, **paths)
    log_action(session, admin_id, "model_retrain",
               details={"version": r["version"], "model": r["best"],
                        "val_auc": r["val"][r["best"]]["roc_auc"], "test_auc": r["test"][r["best"]]["roc_auc"]})
    session.commit()
    return success({"version": r["version"], "best": r["best"]})


def activate_version(session: Session, admin_id: int, version: int) -> dict:
    """Make `version` the one the apps use (exactly one is active)."""
    require_admin(session, admin_id)
    target = session.scalar(select(ModelRegistry).where(ModelRegistry.name == MODEL_NAME,
                                                        ModelRegistry.version == version))
    if target is None:
        return error(f"Version {version} not found")
    session.execute(update(ModelRegistry).where(ModelRegistry.name == MODEL_NAME).values(is_active=False))
    target.is_active = True
    log_action(session, admin_id, "model_activate", details={"version": version})
    session.commit()
    return success()
