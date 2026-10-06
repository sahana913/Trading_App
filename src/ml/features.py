"""
features.py - Build model inputs from daily candles.

THE RULE: a row for day t may only use prices/volumes up to and including
day t's close. We imagine deciding at today's close what to do tomorrow.
Everything here is therefore built from:
  * .shift(k) with k >= 0   (looks back k days),
  * .rolling(n)             (the last n days, ending today),
  * .ewm(adjust=False)      (a weighted average of today and earlier days).
Never .shift(-k), centred windows or back-filling: those peek at the future.
tests/test_ml.py checks this by recomputing features on data cut at day t.

All calculations run per symbol (groupby), so one stock's history never
leaks into another's.

Features (all scale-free, so a ₹100 stock and a ₹4,000 stock look alike):
  ret_1 .. ret_5   daily return of today, yesterday, ... 4 days ago
                     ret_1 = close_t / close_(t-1) - 1
  ret_5d, ret_20d  return over the last 5 / 20 days
  ma_ratio_N       close / (N-day simple moving average) - 1, N in 5, 10, 20, 50
                     > 0 means the price is above its recent average
  rsi_14           Relative Strength Index (Wilder, 14 days), 0-100
                     avg_gain / avg_loss = RS;  RSI = 100 - 100 / (1 + RS)
                     averages are exponential with alpha = 1/14
  macd, macd_signal, macd_hist   (divided by close to make them scale-free)
                     macd   = EMA12(close) - EMA26(close)
                     signal = EMA9(macd);  hist = macd - signal
  bb_position      where the close sits inside the 20-day Bollinger band
                     (close - lower) / (upper - lower), bands = SMA20 +/- 2 std
                     0 = on the lower band, 1 = on the upper band
  vol_10, vol_20   rolling standard deviation of daily returns (volatility)
  volume_change    volume_t / volume_(t-1) - 1
  volume_ratio_20  volume_t / (20-day average volume)

Target (the answer the model learns, NOT an input):
  next_return = close_(t+1) / close_t - 1     <- uses the future on purpose
  target      = 1 if next_return > 0 else 0
"""

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "ret_1", "ret_2", "ret_3", "ret_4", "ret_5", "ret_5d", "ret_20d",
    "ma_ratio_5", "ma_ratio_10", "ma_ratio_20", "ma_ratio_50",
    "rsi_14", "macd", "macd_signal", "macd_hist", "bb_position",
    "vol_10", "vol_20", "volume_change", "volume_ratio_20",
]
TARGET_COLUMN = "target"


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI. 100 = only gains lately, 0 = only losses."""
    change = close.diff()
    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)
    # Wilder's smoothing is an EMA with alpha = 1/window; adjust=False keeps it causal
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss
    out = 100 - 100 / (1 + rs)
    # Only gains in the window: avg_loss = 0, RS = infinity, RSI = 100
    return out.where(avg_loss != 0, 100.0).where(avg_gain.notna())


def _one_symbol(bars: pd.DataFrame) -> pd.DataFrame:
    """Features and target for ONE stock, rows sorted by time."""
    close, volume = bars["close"], bars["volume"].astype(float)
    f = pd.DataFrame(index=bars.index)

    daily = close.pct_change()                 # close_t / close_(t-1) - 1
    for k in range(1, 6):
        f[f"ret_{k}"] = daily.shift(k - 1)     # ret_1 = today, ret_2 = yesterday, ...
    f["ret_5d"] = close.pct_change(5)
    f["ret_20d"] = close.pct_change(20)

    for n in (5, 10, 20, 50):
        f[f"ma_ratio_{n}"] = close / close.rolling(n).mean() - 1

    f["rsi_14"] = rsi(close, 14)

    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    f["macd"] = macd / close
    f["macd_signal"] = signal / close
    f["macd_hist"] = (macd - signal) / close

    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    lower, upper = mid - 2 * std, mid + 2 * std
    f["bb_position"] = (close - lower) / (upper - lower)

    f["vol_10"] = daily.rolling(10).std()
    f["vol_20"] = daily.rolling(20).std()

    f["volume_change"] = volume.pct_change()
    f["volume_ratio_20"] = volume / volume.rolling(20).mean()

    # --- target: the ONLY place that looks forward ---
    f["next_return"] = close.shift(-1) / close - 1
    f[TARGET_COLUMN] = (f["next_return"] > 0).astype(float).where(f["next_return"].notna())
    return f


def build_features(candles: pd.DataFrame) -> pd.DataFrame:
    """candles: symbol, timestamp, open, high, low, close, volume (any order).

    Returns one row per symbol and day with: symbol, timestamp, close, the
    FEATURE_COLUMNS, next_return and target. Rows are NOT dropped here; the
    first ~50 days of each stock have empty features (not enough history yet),
    and the last day has no target (no tomorrow yet). Use usable_rows().
    """
    df = candles.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    parts = [_one_symbol(bars) for _, bars in df.groupby("symbol", sort=True)]
    feats = pd.concat(parts).sort_index()
    out = pd.concat([df[["symbol", "timestamp", "close"]], feats], axis=1)
    # A division by zero (e.g. a zero-volume day) gives +/-inf: treat as missing
    return out.replace([np.inf, -np.inf], np.nan)


def usable_rows(feats: pd.DataFrame) -> pd.DataFrame:
    """Rows with every feature AND a target: what training can use."""
    return feats.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).reset_index(drop=True)
