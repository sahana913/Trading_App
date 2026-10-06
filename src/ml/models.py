"""
models.py - The candidate models. All follow scikit-learn's interface
(fit / predict / predict_proba), so training and evaluation code treats them alike.

  majority    always predicts the class that was most common in training
              (a "do nothing clever" baseline; ROC-AUC is 0.5 by definition)
  yesterday   predicts tomorrow goes the same way as today (momentum baseline)
  logistic    logistic regression on standardised features
  lightgbm    gradient-boosted decision trees

A real model is only worth using if it beats BOTH baselines on unseen data.
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42  # one seed for everything random, so results are reproducible


class YesterdayDirection(BaseEstimator, ClassifierMixin):
    """Predict 'up' tomorrow if today was up (ret_1 > 0). Learns nothing."""

    def fit(self, X: pd.DataFrame, y=None):
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        up = (X["ret_1"].to_numpy() > 0).astype(float)
        return np.column_stack([1 - up, up])  # columns: P(down), P(up)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(X)[:, 1]


def make_models(seed: int = SEED) -> dict:
    """Fresh, untrained copies of every candidate (name -> model)."""
    return {
        "majority": DummyClassifier(strategy="prior"),
        "yesterday": YesterdayDirection(),
        # Scaling puts features on the same footing; C=0.1 = fairly strong
        # regularisation, which suits noisy financial data
        "logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, max_iter=2000, random_state=seed),
        ),
        # Small, shallow trees and few rounds: with ~5,000 noisy rows a big
        # model would just memorise the training years
        "lightgbm": LGBMClassifier(
            n_estimators=200, learning_rate=0.03, num_leaves=15, max_depth=4,
            min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=seed, deterministic=True, force_row_wise=True,
            n_jobs=1, verbose=-1,  # 1 thread + deterministic = identical results every run
        ),
    }


def describe(model) -> str:
    """Human-readable algorithm name for the registry."""
    if isinstance(model, DummyClassifier):
        return "Majority-class baseline"
    if isinstance(model, YesterdayDirection):
        return "Yesterday's-direction baseline"
    if isinstance(model, LGBMClassifier):
        return "LightGBM"
    return "Logistic regression"
