# Model card: next-day direction (lightgbm, v3)

> **Educational use only.** Paper trading on historical data. Not financial advice;
> do not use these signals to trade real money.

Generated 2026-10-07 19:12 by `python -m src.ml.train` (seed 42).

## What it predicts
For each stock and day *t*: will the next close be higher than today's?
Target = 1 if `close(t+1) / close(t) - 1 > 0`, else 0. Decided at day *t*'s close
using only information available then.

## Data
- 10 NSE stocks, daily candles from `data/processed/candles.parquet`
- Usable rows (all features + a target): 9,361

| Part | From | To | Rows |
|---|---|---|---|
| Train | 2022-03-15 | 2024-06-26 | 5,620 |
| Validation | 2024-06-28 | 2025-03-25 | 1,851 |
| Test | 2025-03-27 | 2025-12-29 | 1,870 |

Split by date, never shuffled, with a 1-day gap between parts (labels use the
next day's price). Walk-forward cross-validation: 5 expanding
folds over the training dates (`TimeSeriesSplit`).

## Features (20)
`ret_1`, `ret_2`, `ret_3`, `ret_4`, `ret_5`, `ret_5d`, `ret_20d`, `ma_ratio_5`, `ma_ratio_10`, `ma_ratio_20`, `ma_ratio_50`, `rsi_14`, `macd`, `macd_signal`, `macd_hist`, `bb_position`, `vol_10`, `vol_20`, `volume_change`, `volume_ratio_20`

Lagged returns, moving-average ratios, RSI(14), MACD, Bollinger-band position,
rolling volatility and volume change; definitions in `src/ml/features.py`.
A test recomputes every feature on data cut at each day to prove no look-ahead.

## Models compared
`majority` (always the most common class), `yesterday` (repeat today's direction),
`logistic` (logistic regression), `lightgbm` (gradient-boosted trees).
Selection rule: highest **validation** ROC-AUC. Test data is used once, for the
numbers below, and never for choosing.

### Walk-forward CV (training years)
| Model | ROC-AUC (mean ± std) | Accuracy |
|---|---|---|
| majority | 0.500 ± 0.000 | 0.519 |
| yesterday | 0.491 ± 0.004 | 0.492 |
| logistic | 0.512 ± 0.021 | 0.516 |
| lightgbm | 0.516 ± 0.019 | 0.514 |

### Validation
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| majority | 0.498 | 0.498 | 1.000 | 0.665 | 0.500 |
| yesterday | 0.494 | 0.492 | 0.493 | 0.492 | 0.494 |
| logistic | 0.484 | 0.487 | 0.738 | 0.587 | 0.477 |
| lightgbm | 0.512 | 0.508 | 0.623 | 0.559 | 0.517 |

### Test (held out)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| majority | 0.494 | 0.494 | 1.000 | 0.661 | 0.500 |
| yesterday | 0.482 | 0.475 | 0.478 | 0.476 | 0.482 |
| logistic | 0.523 | 0.512 | 0.739 | 0.605 | 0.514 |
| lightgbm | 0.497 | 0.493 | 0.720 | 0.586 | 0.491 |

Share of up-days in the test period: 49.4%.

## Backtest on the test period
Long-only: hold a stock tomorrow if P(up) > 0.5, else cash. Equal weight across
stocks. Cost 0.111% of the traded value per buy or sell
(from the app's own charges table).

| Strategy | Total return | Sharpe | Max drawdown | Trades | Time invested |
|---|---|---|---|---|---|
| majority | +8.28% | 1.029 | -5.95% | 20 | 100% |
| yesterday | -8.02% | -1.728 | -8.76% | 974 | 50% |
| logistic | +5.17% | 0.769 | -5.33% | 476 | 71% |
| lightgbm | +0.32% | 0.093 | -4.68% | 590 | 72% |
| **buy-and-hold** | +8.28% | 1.029 | -5.95% | 20 | 100% |

## Most useful features (LightGBM, share of total gain)

| Feature | Share |
|---|---|
| ret_4 | 9.7% |
| volume_change | 8.0% |
| volume_ratio_20 | 7.4% |
| ret_3 | 7.3% |
| vol_20 | 6.7% |
| ret_2 | 6.1% |
| ret_1 | 5.4% |
| ma_ratio_10 | 4.9% |

## Selected model
**lightgbm**, saved to `models/direction_v3_lightgbm.joblib` and recorded in `model_registry`
(name `direction`, version 3, active).

## Honest interpretation
On the untouched test period the selected model (lightgbm) reached ROC-AUC 0.491 and accuracy 49.7%, while simply predicting the majority class would have scored 49.4%. That is no better than a coin flip (AUC 0.5). It does not clearly beat the simple baselines. Trading the signal returned +0.32% after costs versus +8.28% for buy-and-hold. This is the expected, honest outcome: next-day direction of large, liquid stocks is very close to random, and daily trading costs eat small edges.

## Limitations
- 10 stocks, about 4 years of daily data: a small sample for any conclusion.
- One period of history; results can differ in other market conditions.
- Costs are approximate; slippage and taxes on profit are ignored.
- The 0.5 probability threshold is fixed, not tuned.

## Reproduce
```powershell
.venv\Scripts\python.exe -m src.ml.train
```
Same data + same seed (42) gives the same models and numbers.
