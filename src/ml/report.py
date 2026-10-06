"""
report.py - Persist the chosen model:
  1. the fitted model -> models/direction_v<N>_<name>.joblib
  2. a row in the model_registry table (metrics, features, dates, seed)
  3. a human-readable model card -> reports/model_card.md
"""

import math
from datetime import datetime
from pathlib import Path

import joblib
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from src.config import PROJECT_ROOT
from src.db.models import ModelRegistry
from src.ml.features import FEATURE_COLUMNS

MODEL_NAME = "direction"


def clean_json(value):
    """JSON has no NaN/Infinity; store them as null. Works through dicts/lists."""
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_model(model, model_key: str, version: int, models_dir: Path, info: dict) -> Path:
    """Write the fitted model plus what's needed to use it later."""
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{MODEL_NAME}_v{version}_{model_key}.joblib"
    joblib.dump({"model": model, "features": FEATURE_COLUMNS, "model_key": model_key,
                 "version": version, **info}, path)
    return path


def next_version(session: Session) -> int:
    latest = session.scalar(select(func.max(ModelRegistry.version)).where(ModelRegistry.name == MODEL_NAME))
    return (latest or 0) + 1


def register(session: Session, *, version: int, algorithm: str, path: Path, params: dict,
             metrics: dict, train_start: datetime, train_end: datetime) -> ModelRegistry:
    """Add the registry row and make it the active model (only one is active)."""
    session.execute(update(ModelRegistry).where(ModelRegistry.name == MODEL_NAME).values(is_active=False))
    try:
        # relative to the project, with / so it works on any OS
        stored_path = path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        stored_path = path.as_posix()
    row = ModelRegistry(
        name=MODEL_NAME, version=version, algorithm=algorithm, file_path=stored_path,
        features=FEATURE_COLUMNS, params=clean_json(params), metrics=clean_json(metrics),
        train_start=train_start, train_end=train_end, is_active=True,
    )
    session.add(row)
    session.commit()
    return row


# ---------------------------------------------------------------------------
# Model card
# ---------------------------------------------------------------------------
def _f(x, kind: str = "num") -> str:
    """Format a number for the markdown tables."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "–"
    return f"{x:+.2%}" if kind == "pct" else f"{x:.3f}"


def write_model_card(path: Path, r: dict) -> Path:
    """r is the result dict built by train.run_training."""
    best = r["best"]
    test = r["test"][best]
    names = list(r["test"])
    rows = lambda table, cols: "\n".join(  # noqa: E731  (a tiny local helper)
        f"| {n} | " + " | ".join(_f(table[n][c]) for c in cols) + " |" for n in names)

    cv_rows = "\n".join(f"| {n} | {_f(r['cv'][n]['roc_auc_mean'])} ± {_f(r['cv'][n]['roc_auc_std'])} | "
                        f"{_f(r['cv'][n]['accuracy_mean'])} |" for n in names)
    bt_rows = "\n".join(
        f"| {n} | {_f(b['strategy']['total_return'], 'pct')} | {_f(b['strategy']['sharpe'])} | "
        f"{_f(b['strategy']['max_drawdown'], 'pct')} | {b['strategy']['trades']} | "
        f"{b['strategy']['exposure']:.0%} |" for n, b in r["backtest"].items())
    hold = next(iter(r["backtest"].values()))["buy_and_hold"]

    beats_coin = test["roc_auc"] > 0.52
    beats_base = test["roc_auc"] > max(r["test"]["majority"]["roc_auc"], r["test"]["yesterday"]["roc_auc"])
    strat = r["backtest"][best]["strategy"]
    verdict = (
        f"On the untouched test period the selected model ({best}) reached ROC-AUC "
        f"{_f(test['roc_auc'])} and accuracy {test['accuracy']:.1%}, while simply predicting the "
        f"majority class would have scored {r['test']['majority']['accuracy']:.1%}. "
        + ("That is only slightly better than a coin flip (AUC 0.5). "
           if beats_coin else "That is no better than a coin flip (AUC 0.5). ")
        + ("It does beat both baselines on ROC-AUC. " if beats_base
           else "It does not clearly beat the simple baselines. ")
        + f"Trading the signal returned {strat['total_return']:+.2%} after costs versus "
        f"{hold['total_return']:+.2%} for buy-and-hold. "
        "This is the expected, honest outcome: next-day direction of large, liquid stocks is "
        "very close to random, and daily trading costs eat small edges."
    )

    importance = ""
    if r.get("importance"):
        top = sorted(r["importance"].items(), key=lambda kv: -kv[1])[:8]
        importance = ("\n## Most useful features (LightGBM, share of total gain)\n\n"
                      "| Feature | Share |\n|---|---|\n"
                      + "\n".join(f"| {k} | {v:.1%} |" for k, v in top) + "\n")

    d = r["dates"]
    text = f"""# Model card: next-day direction ({best}, v{r['version']})

> **Educational use only.** Paper trading on historical data. Not financial advice;
> do not use these signals to trade real money.

Generated {r['generated']:%Y-%m-%d %H:%M} by `python -m src.ml.train` (seed {r['seed']}).

## What it predicts
For each stock and day *t*: will the next close be higher than today's?
Target = 1 if `close(t+1) / close(t) - 1 > 0`, else 0. Decided at day *t*'s close
using only information available then.

## Data
- {r['n_symbols']} NSE stocks, daily candles from `data/processed/candles.parquet`
- Usable rows (all features + a target): {r['n_rows']:,}

| Part | From | To | Rows |
|---|---|---|---|
| Train | {d['train'][0]:%Y-%m-%d} | {d['train'][1]:%Y-%m-%d} | {r['sizes']['train']:,} |
| Validation | {d['val'][0]:%Y-%m-%d} | {d['val'][1]:%Y-%m-%d} | {r['sizes']['val']:,} |
| Test | {d['test'][0]:%Y-%m-%d} | {d['test'][1]:%Y-%m-%d} | {r['sizes']['test']:,} |

Split by date, never shuffled, with a 1-day gap between parts (labels use the
next day's price). Walk-forward cross-validation: {r['cv_folds']} expanding
folds over the training dates (`TimeSeriesSplit`).

## Features ({len(FEATURE_COLUMNS)})
`{"`, `".join(FEATURE_COLUMNS)}`

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
{cv_rows}

### Validation
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
{rows(r['val'], ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])}

### Test (held out)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
{rows(r['test'], ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])}

Share of up-days in the test period: {test['up_rate']:.1%}.

## Backtest on the test period
Long-only: hold a stock tomorrow if P(up) > 0.5, else cash. Equal weight across
stocks. Cost {r['cost_per_side']:.3%} of the traded value per buy or sell
(from the app's own charges table).

| Strategy | Total return | Sharpe | Max drawdown | Trades | Time invested |
|---|---|---|---|---|---|
{bt_rows}
| **buy-and-hold** | {_f(hold['total_return'], 'pct')} | {_f(hold['sharpe'])} | {_f(hold['max_drawdown'], 'pct')} | {hold['trades']} | 100% |
{importance}
## Selected model
**{best}**, saved to `{r['file_path']}` and recorded in `model_registry`
(name `{MODEL_NAME}`, version {r['version']}, active).

## Honest interpretation
{verdict}

## Limitations
- 10 stocks, about 4 years of daily data: a small sample for any conclusion.
- One period of history; results can differ in other market conditions.
- Costs are approximate; slippage and taxes on profit are ignored.
- The 0.5 probability threshold is fixed, not tuned.

## Reproduce
```powershell
.venv\\Scripts\\python.exe -m src.ml.train
```
Same data + same seed ({r['seed']}) gives the same models and numbers.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
