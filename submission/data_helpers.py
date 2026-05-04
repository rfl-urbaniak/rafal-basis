"""Raw-data loading, outcome/sensor extraction, and forecast-window helpers.

All public functions cache their outputs as parquet files under ``.cache/``
so subsequent runs skip the parsing/pivoting cost.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
from config import (
    ASSESSMENT_END,
    ASSESSMENT_START,
    FORECAST_DAYS,
    TRAIN_END,
    TRAIN_START,
)
from loguru import logger

DATA_PATH = pathlib.Path(__file__).parent.parent / "data" / "turingAI_forecasting_challenge_dataset.csv"
CACHE_PATH = pathlib.Path(__file__).parent / ".cache" / "dataset.parquet"
OUTCOME_CACHE_PATH = pathlib.Path(__file__).parent / ".cache" / "outcome.parquet"
SENSORS_CACHE_PATH = pathlib.Path(__file__).parent / ".cache" / "sensors.parquet"


def load_raw(data_path: pathlib.Path = DATA_PATH, clear_cache: bool = False) -> pd.DataFrame:
    """
    Read the long-format CSV dataset, caching to Parquet on first load.

    Columns: dt (datetime), metric_name, value, coverage, coverage_label, variable_type.
    One row per observation; mixed frequencies (15-min to daily).
    Shape: several million rows x 6 columns.

    :param data_path: path to the raw CSV file.
    :param clear_cache: if True, delete the existing cache file before loading.
    :returns: raw long-format DataFrame with parsed dt and numeric value columns.
    """
    if clear_cache and CACHE_PATH.exists():
        CACHE_PATH.unlink()
    if CACHE_PATH.exists():
        logger.info(f"load_raw: cache hit ({CACHE_PATH})")
        return pd.read_parquet(CACHE_PATH)
    logger.info(f"load_raw: cache miss — parsing CSV {data_path}")
    df = pd.read_csv(data_path)
    df["dt"] = pd.to_datetime(df["dt"], format="mixed")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE_PATH, index=False)
    return df


def get_outcome(df: pd.DataFrame, clear_cache: bool = False) -> pd.Series:
    """
    Extract the daily outcome series from the raw long-format DataFrame, caching to Parquet.

    Filters rows where variable_type is "outcome", normalises timestamps to midnight,
    and drops sentinel values of -9999 used as assessment-period placeholders.
    Shape: ~930 values, covering 2023-03-16 to 2025-09-30.

    :param df: raw long-format DataFrame as returned by load_raw.
    :param clear_cache: if True, delete the existing cache file before computing.
    :returns: daily Series indexed by midnight dates, named "outcome".
    """
    if clear_cache and OUTCOME_CACHE_PATH.exists():
        OUTCOME_CACHE_PATH.unlink()
    if OUTCOME_CACHE_PATH.exists():
        logger.info(f"get_outcome: cache hit ({OUTCOME_CACHE_PATH})")
        return pd.read_parquet(OUTCOME_CACHE_PATH).squeeze("columns")
    logger.info("get_outcome: cache miss — computing from raw DataFrame")
    s = df[df["variable_type"] == "outcome"].copy()
    s = s.set_index("dt")["value"].sort_index()
    s.index = s.index.normalize()
    s.index.name = "date"
    s.name = "outcome"
    s = s[s != -9999.0]
    s.to_frame().to_parquet(OUTCOME_CACHE_PATH)
    return s


def get_training_windows(outcome: pd.Series) -> pd.DatetimeIndex:
    """
    Return days D in the training period where all 10 forecast targets are observed.

    A window anchored at D requires outcome values at D+1 through D+FORECAST_DAYS.
    The anchor is hard-capped at TRAIN_END - FORECAST_DAYS so that all targets land
    inside the training period regardless of whether the assessment dataset has been
    released. This prevents post-release re-runs from silently extending the training
    set into the assessment period (which would make the assessment evaluation in-sample).

    :param outcome: daily outcome Series as returned by get_outcome.
    :returns: DatetimeIndex of valid training window anchor days.
    """
    latest_anchor = min(outcome.index.max(), pd.Timestamp(TRAIN_END)) - pd.Timedelta(days=FORECAST_DAYS)
    candidates = pd.date_range(TRAIN_START, latest_anchor, freq="D")
    return candidates[candidates.isin(outcome.index)]


def get_assessment_windows() -> pd.DatetimeIndex:
    """
    Return the 173 forecast window anchor days for the assessment period.

    The first anchor is ASSESSMENT_START minus one day (the launch day whose first
    predicted day is ASSESSMENT_START). The last anchor is ASSESSMENT_END minus
    FORECAST_DAYS days, so that D+FORECAST_DAYS lands exactly on ASSESSMENT_END.

    :returns: DatetimeIndex of 173 daily anchor days starting at ASSESSMENT_START.
    """
    first_anchor = pd.Timestamp(ASSESSMENT_START) - pd.Timedelta(days=1)
    last_anchor = pd.Timestamp(ASSESSMENT_END) - pd.Timedelta(days=FORECAST_DAYS)
    return pd.date_range(first_anchor, last_anchor, freq="D")


def get_sensors(df: pd.DataFrame, clear_cache: bool = False) -> pd.DataFrame:
    """
    Pivot feature rows to a daily wide DataFrame with midday attribution.

    A reading at or before noon on day D is attributed to D; a reading after noon on D
    is attributed to D+1, because it arrives after the competition cutoff and is only
    usable from noon on D+1 onward. Takes the median across all readings attributed
    to each day.
    Shape: ~930 rows x ~220 columns (one column per metric__coverage combination).

    :param df: raw long-format DataFrame as returned by load_raw.
    :param clear_cache: if True, delete the existing cache file before computing.
    :returns: daily DataFrame indexed by midnight dates, columns named metric__coverage.
    """
    if clear_cache and SENSORS_CACHE_PATH.exists():
        SENSORS_CACHE_PATH.unlink()
    if SENSORS_CACHE_PATH.exists():
        logger.info(f"get_sensors: cache hit ({SENSORS_CACHE_PATH})")
        return pd.read_parquet(SENSORS_CACHE_PATH)
    logger.info("get_sensors: cache miss — pivoting from raw DataFrame")
    features = df[df["variable_type"] == "feature"].copy()
    features["col"] = features["metric_name"] + "__" + features["coverage"]
    wide = features.pivot_table(index="dt", columns="col", values="value", aggfunc="mean")
    wide = wide.sort_index()
    idx = wide.index
    after_noon = (idx.hour > 12) | ((idx.hour == 12) & (idx.minute > 0))
    wide.index = np.where(
        after_noon,
        (idx.normalize() + pd.Timedelta(days=1)).asi8,
        idx.normalize().asi8,
    )
    wide.index = pd.DatetimeIndex(wide.index)
    sensors = wide.resample("D").median()
    sensors.to_parquet(SENSORS_CACHE_PATH)
    return sensors


def get_training_targets(
    outcome: pd.Series,
    windows: pd.DatetimeIndex,
    clear_cache: bool = False,
) -> pd.DataFrame:
    """
    Build the target matrix for model training, one row per forecast window D.

    For each window D, the targets are outcome[D+1] through outcome[D+FORECAST_DAYS].
    All values are guaranteed to be observed because get_training_windows only returns
    windows where the full horizon is available.

    :param outcome: daily outcome Series as returned by get_outcome.
    :param windows: DatetimeIndex of training window anchor days.
    :param clear_cache: if True, delete the existing cache file before computing.
    :returns: DataFrame of shape (len(windows), FORECAST_DAYS), columns target_day_1
              through target_day_10, indexed by windows, float32.
    """
    _cache_dir = pathlib.Path(__file__).parent / ".cache"
    cache = _cache_dir / f"training_targets_{windows.min().date()}_{windows.max().date()}.parquet"

    if clear_cache and cache.exists():
        cache.unlink()
    if cache.exists():
        logger.info(f"get_training_targets: cache hit ({cache})")
        return pd.read_parquet(cache)
    logger.info("get_training_targets: cache miss — building target matrix")

    dates = windows.normalize()
    data = {
        f"target_day_{d}": [
            outcome.get(day + pd.Timedelta(days=d), np.nan) for day in dates
        ]
        for d in range(1, FORECAST_DAYS + 1)
    }
    result = pd.DataFrame(data, index=windows).astype("float32")
    _cache_dir.mkdir(parents=True, exist_ok=True)
    result.to_parquet(cache)
    return result
