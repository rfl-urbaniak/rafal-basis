"""Feature builder for ``gp_informed_lags`` (consecutive short lags 4-7/13-14/20-21)."""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
from config import GP_INFORMED_LAGS_PARAMS, TOP_K_SENSOR_COLS
from loguru import logger

_CACHE_DIR = pathlib.Path(__file__).parent / ".cache"


def _cache_path(windows: pd.DatetimeIndex) -> pathlib.Path:
    start = windows.min().date()
    end = windows.max().date()
    return _CACHE_DIR / f"features_informed_lags_{start}_{end}.parquet"


def build_features(
    windows: pd.DatetimeIndex,
    outcome: pd.Series,
    sensors: pd.DataFrame,
    clear_cache: bool = False,
) -> pd.DataFrame:
    """
    Build the feature matrix for gp_informed_lags, one row per forecast window D.

    Leakage safety: temporal features derive only from D itself. Outcome lags use
    D minus k where k >= 4 (min lag in lag_days), so no outcome value more recent
    than D-4 is accessed. Rolling means are anchored at D - rolling_lag (D-4).
    Sensor lags are raw offsets (no OUTCOME_LAG added): lag 0 uses the same-day
    full-day median (available by competition cutoff). No value at D+1 or later
    is accessed.

    Feature columns (in order):

    1. Temporal (5):      dow_sin, dow_cos, month_sin, month_cos, year_norm
    2. Rolling means (3): rolling_mean_{w} for w in [7, 14, 28], anchored at D - rolling_lag
    3. Sensors (27):      sensor_{i}_lag_{k} for i in [0,1,2], k in [0,1,2,6,7,13,14,20,21]
    4. Outcome lags (8):  outcome_lag_{k} for k in [4, 5, 6, 7, 13, 14, 20, 21]
    5. Outcome delta (1): outcome_delta_4_7 = outcome[D-4] - outcome[D-7]

    Total: 44 features. Max lag is 21 days — within range for all training windows.

    Result is cached to .cache/features_informed_lags_{start}_{end}.parquet keyed by
    window date range; training and assessment windows cache separately.

    :param windows: DatetimeIndex of forecast anchor days D.
    :param outcome: daily outcome Series indexed by midnight dates.
    :param sensors: daily sensor DataFrame indexed by midnight dates.
    :returns: DataFrame of shape (len(windows), 44), indexed by windows, float32 columns.
    """
    cache = _cache_path(windows)
    if clear_cache and cache.exists():
        cache.unlink()
    if cache.exists():
        logger.info(f"features_informed_lags: cache hit ({cache})")
        return pd.read_parquet(cache)
    logger.info(f"features_informed_lags: cache miss — building {len(windows)} rows")

    lag_days: list[int] = GP_INFORMED_LAGS_PARAMS["lag_days"]
    rolling_windows: list[int] = GP_INFORMED_LAGS_PARAMS["rolling_windows"]
    rolling_lag: int = GP_INFORMED_LAGS_PARAMS["rolling_lag"]
    sensor_lag_days: list[int] = GP_INFORMED_LAGS_PARAMS["sensor_lag_days"]

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

    anchor_dates = dates - pd.to_timedelta(rolling_lag, unit="D")
    rolling_feats = pd.DataFrame(
        {f"rolling_mean_{w}": [rolled[w].get(a, np.nan) for a in anchor_dates]
         for w in rolling_windows},
        index=windows,
    )

    sensor_frames = []
    for i, col in enumerate(TOP_K_SENSOR_COLS):
        s = sensors[col] if col in sensors.columns else pd.Series(dtype=float)
        for k in sensor_lag_days:
            sensor_frames.append(  # noqa: PERF401 — loop-vars used in Series name
                pd.Series(
                    [s.get(d - pd.Timedelta(days=k), np.nan) for d in dates],
                    index=windows,
                    name=f"sensor_{i}_lag_{k}",
                )
            )
    sensor_feats = pd.concat(sensor_frames, axis=1)

    lag_vals = {
        k: pd.Series(
            [outcome.get(d - pd.Timedelta(days=k), np.nan) for d in dates],
            index=windows,
            name=f"outcome_lag_{k}",
        )
        for k in lag_days
    }
    outcome_lags = pd.concat(lag_vals.values(), axis=1)
    delta = pd.Series(
        lag_vals[4].values - lag_vals[7].values,
        index=windows,
        name="outcome_delta_4_7",
    )

    result = pd.concat([temporal, rolling_feats, sensor_feats, outcome_lags, delta], axis=1)
    result = result.astype("float32")

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    result.to_parquet(cache)
    return result
