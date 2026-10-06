"""
evaluate.py - Classification metrics. "Positive" = the price went UP.

  accuracy   correct predictions / all predictions
  precision  of the days we said "up", the share that really went up
  recall     of the days that really went up, the share we said "up"
  F1         2 x precision x recall / (precision + recall)  (balances the two)
  ROC-AUC    chance that a random up-day gets a higher predicted probability
             than a random down-day. 0.5 = no skill (coin flip), 1.0 = perfect.
             It ignores the 0.5 threshold, so it's the fairest single number
             for comparing models.
  up_rate    share of days that actually went up (accuracy of always saying "up")
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

THRESHOLD = 0.5  # predict "up" when P(up) > 0.5


def classification_metrics(y_true, proba_up) -> dict:
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(proba_up) > THRESHOLD).astype(float)
    # ROC-AUC is undefined if only one class is present
    auc = roc_auc_score(y_true, proba_up) if len(np.unique(y_true)) == 2 else float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        # zero_division=0: if a model never says "up", its precision is 0, not an error
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(auc),
        "up_rate": float(y_true.mean()),
        "rows": int(len(y_true)),
    }
