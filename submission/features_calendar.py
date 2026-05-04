"""Feature builder for ``gp_calendar`` (standard calendar lags 4/7/14/28)."""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
from config import GP_CALENDAR_PARAMS, OUTCOME_LAG, TOP_K_SENSOR_COLS
from loguru import logger

_CACHE_DIR = pathlib.Path(__file__).parent / ".cache"


def _cache_path(windows: pd.DatetimeIndex) -> pathlib.Path:
    start = windows.min().date()
    end = windows.max().date()
    return _CACHE_DIR / f"features_calendar_{start}_{end}.parquet"


def build_features(
    windows: pd.DatetimeIndex,
    outcome: pd.Series,
    sensors: pd.DataFrame,
    clear_cache: bool = False,
) -> pd.DataFrame:
    """
    Build the feature matrix for gp_calendar, one row per forecast window D.

    Leakage safety: temporal features derive only from D itself. Outcome lags use
    D minus k where k >= OUTCOME_LAG (min lag = 4), so no outcome value more recent
    than D-4 is accessed. Sensor lags use total_lag = OUTCOME_LAG + k, so sensor_i_lag_4
    is the sensor value at D-4 (matching the outcome freshness cutoff). No value at
    D+1 or later is accessed.

    Feature columns (in order):

    1. Temporal (5):      dow_sin, dow_cos, month_sin, month_cos, year_norm
    2. Rolling means (3): rolling_mean_{w} for w in [7, 14, 28], anchored at D - OUTCOME_LAG
    3. Sensors (12):      sensor_{i}_lag_{OUTCOME_LAG+k} for i in [0,1,2], k in [0,3,7,28]
    4. Outcome lags (4):  outcome_lag_{k} for k in [4, 7, 14, 28]

    Total: 24 features. Max lag is OUTCOME_LAG + 28 = 32 days — within range for all
    training windows, so no NaNs from lags exceeding the data start.

    Result is cached to .cache/features_calendar_{start}_{end}.parquet keyed by window
    date range; training and assessment windows cache separately.

    :param windows: DatetimeIndex of forecast anchor days D.
    :param outcome: daily outcome Series indexed by midnight dates.
    :param sensors: daily sensor DataFrame indexed by midnight dates.
    :returns: DataFrame of shape (len(windows), 24), indexed by windows, float32 columns.
    """
    cache = _cache_path(windows)
    if clear_cache and cache.exists():
        cache.unlink()
    if cache.exists():
        logger.info(f"features_calendar: cache hit ({cache})")
        return pd.read_parquet(cache)
    logger.info(f"features_calendar: cache miss — building {len(windows)} rows")

    lag_days: list[int] = GP_CALENDAR_PARAMS["lag_days"]
    rolling_windows: list[int] = GP_CALENDAR_PARAMS["rolling_windows"]
    sensor_lag_days: list[int] = GP_CALENDAR_PARAMS["sensor_lag_days"]

    rolled = {w: outcome.rolling(w, min_periods=1).mean() for w in rolling_windows}
    dates = windows.normalize()

    dow = dates.dayofweek
    month = dates.month
    temporal = pd.DataFrame(
        {
            "dow_sin":   np.sin(2 * np.pi * dow / 7),
            "dow_cos":   np.cos(2 * np.pi * dow / 7),
            "month_sin": np.sin(2 * np.pi * (month - 1) / 12),
            "month_cos": np.cos(2 * np.pi * (month - 1) / 12),
            "year_norm": (dates.year - 2020) / 5.0,
        },
        index=windows,
    )

    anchor_dates = dates - pd.to_timedelta(OUTCOME_LAG, unit="D")
    rolling_feats = pd.DataFrame(
        {f"rolling_mean_{w}": [rolled[w].get(a, np.nan) for a in anchor_dates]
         for w in rolling_windows},
        index=windows,
    )

    sensor_frames = []
    for i, col in enumerate(TOP_K_SENSOR_COLS):
        s = sensors[col] if col in sensors.columns else pd.Series(dtype=float)
        for k in sensor_lag_days:
            total_lag = OUTCOME_LAG + k
            sensor_frames.append(
                pd.Series(
                    [s.get(d - pd.Timedelta(days=total_lag), np.nan) for d in dates],
                    index=windows,
                    name=f"sensor_{i}_lag_{total_lag}",
                )
            )
    sensor_feats = pd.concat(sensor_frames, axis=1)

    outcome_lags = pd.DataFrame(
        {f"outcome_lag_{k}": [outcome.get(d - pd.Timedelta(days=k), np.nan) for d in dates]
         for k in lag_days},
        index=windows,
    )

    result = pd.concat([temporal, rolling_feats, sensor_feats, outcome_lags], axis=1)
    result = result.astype("float32")

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    result.to_parquet(cache)
    return result
