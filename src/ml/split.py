"""
split.py - Time-based splits. NEVER shuffle time series: a model must only
ever be tested on days that come AFTER everything it was trained on.

We split by DATE (not by row): on any given day, all 10 stocks fall in the
same part. Otherwise the model could train on TCS for 5 March and be tested on
INFY for 5 March, and the market moves both on the same day.

The 1-day gap ("purge"): a row's label uses the NEXT day's close. So the last
training day's label is made from the first validation day's price. Dropping
GAP_DAYS dates between parts removes that overlap.

    |---------- train ----------| gap |-- validation --| gap |---- test ----|
    oldest                                                          newest
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

TRAIN_SHARE, VAL_SHARE = 0.60, 0.20  # the remaining 20% of dates is the test set
GAP_DAYS = 1


def time_split(rows: pd.DataFrame, train_share: float = TRAIN_SHARE, val_share: float = VAL_SHARE,
               gap: int = GAP_DAYS) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split rows into (train, validation, test) by their timestamp."""
    dates = np.sort(rows["timestamp"].unique())
    n = len(dates)
    train_end = int(n * train_share)               # dates[:train_end] are training dates
    val_start = train_end + gap
    val_end = int(n * (train_share + val_share))
    test_start = val_end + gap

    def pick(start: int, end: int) -> pd.DataFrame:
        chosen = dates[start:end]
        return rows[rows["timestamp"].isin(chosen)].reset_index(drop=True)

    return pick(0, train_end), pick(val_start, val_end), pick(test_start, n)


def walk_forward_folds(rows: pd.DataFrame, n_splits: int = 5,
                       gap: int = GAP_DAYS) -> list[tuple[np.ndarray, np.ndarray]]:
    """Walk-forward cross-validation over DATES, returned as row positions.

    Each fold trains on all dates up to a point and tests on the dates right
    after it; the training window grows each time:
        fold 1: [train]           [test]
        fold 2: [train......]           [test]
        fold 3: [train...........]            [test]  ...
    sklearn's TimeSeriesSplit does the date maths (it never shuffles).
    """
    dates = np.sort(rows["timestamp"].unique())
    folds = []
    for train_idx, test_idx in TimeSeriesSplit(n_splits=n_splits, gap=gap).split(dates):
        train_rows = np.flatnonzero(rows["timestamp"].isin(dates[train_idx]).to_numpy())
        test_rows = np.flatnonzero(rows["timestamp"].isin(dates[test_idx]).to_numpy())
        folds.append((train_rows, test_rows))
    return folds
