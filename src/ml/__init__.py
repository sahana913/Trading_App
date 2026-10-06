"""
Machine-learning module: predicts whether a stock's NEXT daily close will be
higher than today's (up = 1, down = 0). Educational signals only.

    features.py   turn candles into model inputs (past data only)
    split.py      time-based train / validation / test + walk-forward folds
    models.py     the baselines, logistic regression and LightGBM
    evaluate.py   classification metrics
    backtest.py   trade the signal vs buy-and-hold, with costs
    report.py     save the model, write the registry row and model card
    train.py      runs everything:  python -m src.ml.train
"""
