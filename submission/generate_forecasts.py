"""
NHS EAD Forecasting Competition — submission script
Rafal Urbaniak, Basis Research Institute

Model: horizon-weighted ensemble of two sparse multitask Gaussian process models —
       gp_calendar (standard calendar lags: 4, 7, 14, 28 days)
       gp_informed_lags (consecutive short lags: 4-7, 13-14, 20-21 days).
       Weights are derived from in-sample MSE on the last 30 training windows
       (separately for horizons 1-5 and 6-10) when TUNE_WEIGHTS=True; equal 0.5/0.5 otherwise.

Steps:
    1. Load config
    2. Load data (outcome, sensors)
    3. Define forecast windows (training and assessment)
    4. Build training features for each model
    5. Train both models once on the full development dataset
    5b. Derive horizon-specific ensemble weights from in-sample evaluation (TUNE_WEIGHTS flag)
    6. Build assessment features, predict with each model, combine with tuned weights
    6b. Optional in-sample sanity-eval table + plots (RUN_EVAL flag)
    7. Write pred_matrix.csv
    8. Write mse_summary.csv if assessment actuals are populated; otherwise log deferred

After Step 8, if COMPILE_REPORT, pandoc-compile report.md to report.pdf.
"""

from __future__ import annotations

import datetime
import pathlib
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from loguru import logger

_fmt = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {file}:{line} - {message}"
logger.remove()
logger.add(sys.stderr, level="INFO", format=_fmt)

_LOG_DIR = pathlib.Path(__file__).parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)
_log_file = _LOG_DIR / f"generate_forecasts_{datetime.datetime.now():%Y%m%d_%H%M%S}.log"
logger.add(_log_file, level="INFO", format=_fmt)
logger.info(f"Log file: {_log_file}")

_t_start = time.perf_counter()
_timings: dict[str, float] = {}

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

# ── Step 1: load config ───────────────────────────────────────────────────────

import eval_plots
import features_calendar
import features_informed_lags
import gp_model
from config import (
    ASSESSMENT_END,
    ASSESSMENT_START,
    FORECAST_DAYS,
    GP_CALENDAR_PARAMS,
    GP_INFORMED_LAGS_PARAMS,
    OUTCOME_LAG,
    PERSISTENCE_BLEND_WEIGHT,
    PERSISTENCE_LAG_LONG,
    PERSISTENCE_LAG_SHORT,
    PERSISTENCE_SHORT_HORIZON,
    TRAIN_END,
    TRAIN_START,
)

CLEAR_CACHE = True
USE_GPU = False
RUN_EVAL = True
TUNE_WEIGHTS = True
COMPILE_REPORT = True
DEVICE = "cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu"
logger.info(f"Device: {DEVICE}  (USE_GPU={USE_GPU}, cuda_available={torch.cuda.is_available()})")


# ── Step 2: load data ─────────────────────────────────────────────────────────

_t2 = time.perf_counter()

from data_helpers import (
    DATA_PATH,
    get_assessment_windows,
    get_outcome,
    get_sensors,
    get_training_targets,
    get_training_windows,
    load_raw,
)

df      = load_raw(DATA_PATH, clear_cache=CLEAR_CACHE)

# add horizontal line to logger
logger.info("-" * 90)
logger.info(f"Raw data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
logger.info(f"df head:\n{df.head()}")

outcome = get_outcome(df, clear_cache=CLEAR_CACHE)
logger.info("-" * 90)
logger.info(f"Outcome series extracted: {outcome.shape[0]} values from {outcome.index.min().date()} to {outcome.index.max().date()}")
logger.info(f"outcome head:\n{outcome.head()}")

sensors = get_sensors(df, clear_cache=CLEAR_CACHE)
logger.info("-" * 90)
logger.info(f"Sensors DataFrame extracted: {sensors.shape[0]} rows, {sensors.shape[1]} columns")
logger.info(f"sensors head:\n{sensors.head()}")

# check shapes and date ranges of outcome and sensors
n_assessment_days = (pd.to_datetime(ASSESSMENT_END) - pd.to_datetime(ASSESSMENT_START)).days + 1
n_windows = n_assessment_days - FORECAST_DAYS + 1

logger.info("-" * 90)
logger.info(f"Training period:   {TRAIN_START} to {TRAIN_END} ({(pd.to_datetime(TRAIN_END) - pd.to_datetime(TRAIN_START)).days + 1} days)")
logger.info(f"Assessment period: {ASSESSMENT_START} to {ASSESSMENT_END} ({n_assessment_days} days)")
logger.info(f"Forecast horizon:  {FORECAST_DAYS} days ahead per window, {n_windows} sliding windows")
logger.info(f"Outcome lag:       {OUTCOME_LAG} days (freshest outcome usable is y[D-{OUTCOME_LAG}])")

assert outcome.index.min() == pd.Timestamp(TRAIN_START), f"outcome starts {outcome.index.min().date()}, expected {TRAIN_START}"
assert outcome.index.max() >= pd.Timestamp(TRAIN_END), f"outcome ends {outcome.index.max().date()}, expected at least {TRAIN_END}"
assert outcome.isna().sum() == 0, "outcome contains NaN values"
assert outcome.index.is_monotonic_increasing, "outcome index is not sorted"
assert sensors.index.is_monotonic_increasing, "sensors index is not sorted"
logger.info(f"Outcome series check passed: {outcome.shape[0]} values, range {outcome.index.min().date()} to {outcome.index.max().date()}, no NaNs")

# ── Assessment-data readiness check (drives Step 6 assertions + Step 8 gating) ──
_assess_idx = pd.date_range(ASSESSMENT_START, ASSESSMENT_END, freq="D")
_outcome_ready = outcome.index.max() >= pd.Timestamp(ASSESSMENT_END)
_sensors_invalid = int((sensors.reindex(_assess_idx).replace(-9999, np.nan).isna()).sum().sum())
_assessment_ready = _outcome_ready and _sensors_invalid == 0
if _assessment_ready:
    logger.info(f"Assessment data ready: outcome covers through {outcome.index.max().date()}, sensors clean over assessment period")
elif _outcome_ready and not _assessment_ready:
    logger.warning(f"Assessment outcome populated but sensors have {_sensors_invalid} -9999/NaN cells in the assessment-period reindex; predictions will pull -9999 sensor lags")
else:
    _missing_days = (pd.Timestamp(ASSESSMENT_END) - outcome.index.max()).days
    logger.warning(f"Assessment data NOT ready: outcome ends at {outcome.index.max().date()} ({_missing_days} days short of {ASSESSMENT_END}). Predictions for windows beyond the first few will be unreliable until the assessment dataset is released (scheduled 6 June 2026).")

if _assessment_ready and not CLEAR_CACHE:
    logger.warning("CLEAR_CACHE=False AND assessment data is now released — feature/dataset/sensors caches may have been built from a pre-release CSV. Set CLEAR_CACHE=True for this run to ensure freshly-released data flows through.")

_timings["step_2_load_data"] = time.perf_counter() - _t2

# ── Step 3: define forecast windows ──────────────────────────────────────────

_t3 = time.perf_counter()

training_windows    = get_training_windows(outcome)
assessment_windows  = get_assessment_windows()

assert len(assessment_windows) == n_windows, f"expected {n_windows} assessment windows, got {len(assessment_windows)}"
assert training_windows.max() <= pd.Timestamp(TRAIN_END), \
    f"Training-window cap violated: max anchor {training_windows.max().date()} > TRAIN_END={TRAIN_END}. " \
    "This breaks the out-of-sample assessment property."
logger.info(f"Training-window cap check passed: max anchor {training_windows.max().date()} <= TRAIN_END={TRAIN_END}")

logger.info("-" * 90)
logger.info(f"Training windows:   {len(training_windows)} days ({training_windows.min().date()} to {training_windows.max().date()})")
logger.info(f"Assessment windows: {len(assessment_windows)} days ({assessment_windows.min().date()} to {assessment_windows.max().date()})")

_timings["step_3_define_windows"] = time.perf_counter() - _t3

# ── Step 4: build training features ──────────────────────────────────────────

_t4 = time.perf_counter()

X_cal_train_df = features_calendar.build_features(training_windows, outcome, sensors, clear_cache=CLEAR_CACHE)
X_inf_train_df = features_informed_lags.build_features(training_windows, outcome, sensors, clear_cache=CLEAR_CACHE)

assert X_cal_train_df.shape[0] == len(training_windows), \
    f"X_cal_train has {X_cal_train_df.shape[0]} rows, expected {len(training_windows)}"
assert X_inf_train_df.shape[0] == len(training_windows), \
    f"X_inf_train has {X_inf_train_df.shape[0]} rows, expected {len(training_windows)}"
assert not X_cal_train_df.isna().all(axis=0).any(), \
    "X_cal_train has at least one all-NaN feature column"
assert not X_inf_train_df.isna().all(axis=0).any(), \
    "X_inf_train has at least one all-NaN feature column"

logger.info("-" * 90)
logger.info(f"X_cal_train: {X_cal_train_df.shape}  NaN fraction: {X_cal_train_df.isna().mean().mean():.3f}")
logger.info(f"X_cal_train earliest window: {X_cal_train_df.index.min().date()}, latest window: {X_cal_train_df.index.max().date()}")
logger.info(f"X_cal_train columns: {list(X_cal_train_df.columns)}")
logger.info(f"X_cal_train sample:\n{X_cal_train_df.head()}")
logger.info(f"X_inf_train: {X_inf_train_df.shape}  NaN fraction: {X_inf_train_df.isna().mean().mean():.3f}")
logger.info(f"X_inf_train earliest window: {X_inf_train_df.index.min().date()}, latest window: {X_inf_train_df.index.max().date()}")
logger.info(f"X_inf_train columns: {list(X_inf_train_df.columns)}")
logger.info(f"X_inf_train sample:\n{X_inf_train_df.head()}")

X_cal_train = X_cal_train_df.values
X_inf_train = X_inf_train_df.values

y_train_df = get_training_targets(outcome, training_windows, clear_cache=CLEAR_CACHE)
y_train = y_train_df.values

_temporal_cols = ["dow_sin", "dow_cos", "month_sin", "month_cos", "year_norm"]
assert X_cal_train_df.index.min() >= pd.Timestamp(TRAIN_START), \
    f"X_cal earliest window {X_cal_train_df.index.min().date()} is before TRAIN_START"
assert X_cal_train_df.index.max() <= pd.Timestamp(TRAIN_END) - pd.Timedelta(days=FORECAST_DAYS), \
    f"X_cal latest window {X_cal_train_df.index.max().date()} leaves no room for targets"
assert X_inf_train_df.index.min() >= pd.Timestamp(TRAIN_START), \
    f"X_inf earliest window {X_inf_train_df.index.min().date()} is before TRAIN_START"
assert X_inf_train_df.index.max() <= pd.Timestamp(TRAIN_END) - pd.Timedelta(days=FORECAST_DAYS), \
    f"X_inf latest window {X_inf_train_df.index.max().date()} leaves no room for targets"
assert not X_cal_train_df[_temporal_cols].isna().any().any(), \
    "X_cal_train has NaNs in temporal columns — feature builder bug"
assert not X_inf_train_df[_temporal_cols].isna().any().any(), \
    "X_inf_train has NaNs in temporal columns — feature builder bug"
assert y_train_df.shape == (len(training_windows), FORECAST_DAYS), \
    f"y_train shape {y_train_df.shape} does not match expected ({len(training_windows)}, {FORECAST_DAYS})"
assert y_train_df.isna().sum().sum() == 0, \
    "y_train contains NaNs — training windows should guarantee full target availability"

logger.info(f"y_train: {y_train_df.shape}  NaN count: {y_train_df.isna().sum().sum()}")
logger.info(f"y_train sample:\n{y_train_df.head()}")

_timings["step_4_build_features"] = time.perf_counter() - _t4

# ── Step 5: train both models ─────────────────────────────────────────────────

_t5 = time.perf_counter()
_CACHE_DIR = pathlib.Path(__file__).parent / ".cache"
_FIGURES_DIR = pathlib.Path(__file__).parent / "figures" / "loss"
_win_key = f"{training_windows.min().date()}_{training_windows.max().date()}"

logger.info("-" * 90)
logger.info(f"Training gp_calendar on {X_cal_train.shape[0]} windows, {X_cal_train.shape[1]} features ...")
model_cal, lik_cal, scaler_cal, y_mean_cal, y_std_cal = gp_model.train(
    X_cal_train, y_train, GP_CALENDAR_PARAMS, device=DEVICE,
    cache_path=_CACHE_DIR / f"model_calendar_{_win_key}.pt",
    clear_cache=CLEAR_CACHE,
    name="gp_calendar",
    figures_dir=_FIGURES_DIR,
)
logger.info(f"gp_calendar done  ({time.perf_counter() - _t5:.1f}s so far)")

logger.info(f"Training gp_informed_lags on {X_inf_train.shape[0]} windows, {X_inf_train.shape[1]} features ...")
model_inf, lik_inf, scaler_inf, y_mean_inf, y_std_inf = gp_model.train(
    X_inf_train, y_train, GP_INFORMED_LAGS_PARAMS, device=DEVICE,
    cache_path=_CACHE_DIR / f"model_informed_lags_{_win_key}.pt",
    clear_cache=CLEAR_CACHE,
    name="gp_informed_lags",
    figures_dir=_FIGURES_DIR,
)
logger.info(f"gp_informed_lags done  ({time.perf_counter() - _t5:.1f}s so far)")

_timings["step_5_train_models"] = time.perf_counter() - _t5

# ── Step 5b: ensemble weight tuning (TUNE_WEIGHTS flag) ───────────────────────

def _comp_mse(yt: np.ndarray, yp: np.ndarray) -> tuple[float, float]:
    """Competition MSE for horizons 1–5 and 6–10, matching the competition formula."""
    sq = (yt - yp) ** 2
    n = yt.shape[0]
    return float(sq[:, :5].sum() / (n * 5)), float(sq[:, 5:].sum() / (n * 5))

_N_TUNE = 30
_tune_windows = training_windows[-_N_TUNE:]

if TUNE_WEIGHTS:
    _X_cal_tune = features_calendar.build_features(_tune_windows, outcome, sensors, clear_cache=CLEAR_CACHE)
    _X_inf_tune = features_informed_lags.build_features(_tune_windows, outcome, sensors, clear_cache=CLEAR_CACHE)
    _p_cal_tune = gp_model.predict(model_cal, lik_cal, scaler_cal, y_mean_cal, y_std_cal, _X_cal_tune.values, device=DEVICE)
    _p_inf_tune = gp_model.predict(model_inf, lik_inf, scaler_inf, y_mean_inf, y_std_inf, _X_inf_tune.values, device=DEVICE)
    _y_tune = y_train_df.iloc[-_N_TUNE:].values

    _mse_cal_15,  _mse_cal_610 = _comp_mse(_y_tune, _p_cal_tune)
    _mse_inf_15,  _mse_inf_610 = _comp_mse(_y_tune, _p_inf_tune)

    W_CAL_15  = _mse_inf_15  / (_mse_cal_15  + _mse_inf_15)
    W_INF_15  = 1.0 - W_CAL_15
    W_CAL_610 = _mse_inf_610 / (_mse_cal_610 + _mse_inf_610)
    W_INF_610 = 1.0 - W_CAL_610

    logger.warning("TUNE_WEIGHTS is IN-SAMPLE — weights derived from data the models were trained on")
    logger.info(f"In-sample MSE  gp_calendar:     1-5d={_mse_cal_15:.6f}  6-10d={_mse_cal_610:.6f}")
    logger.info(f"In-sample MSE  gp_informed_lags: 1-5d={_mse_inf_15:.6f}  6-10d={_mse_inf_610:.6f}")
    logger.info(f"Tuned weights  1-5d:   gp_calendar={W_CAL_15:.3f}  gp_informed_lags={W_INF_15:.3f}")
    logger.info(f"Tuned weights  6-10d:  gp_calendar={W_CAL_610:.3f}  gp_informed_lags={W_INF_610:.3f}")
else:
    W_CAL_15 = W_INF_15 = W_CAL_610 = W_INF_610 = 0.5
    logger.info("TUNE_WEIGHTS=False — using equal 0.5/0.5 weights for both horizon groups")

# ── Step 6: build assessment features & predict ──────────────────────────────

_t6 = time.perf_counter()

X_cal_assess_df = features_calendar.build_features(assessment_windows, outcome, sensors, clear_cache=CLEAR_CACHE)
X_inf_assess_df = features_informed_lags.build_features(assessment_windows, outcome, sensors, clear_cache=CLEAR_CACHE)

assert X_cal_assess_df.shape == (len(assessment_windows), X_cal_train_df.shape[1]), \
    f"X_cal_assess shape {X_cal_assess_df.shape} unexpected"
assert X_inf_assess_df.shape == (len(assessment_windows), X_inf_train_df.shape[1]), \
    f"X_inf_assess shape {X_inf_assess_df.shape} unexpected"

# Post-release feature integrity: when assessment data is supposed to be populated,
# no -9999 placeholder or NaN should leak into the assessment-feature matrices.
# Pre-release these matrices contain -9999 (sensors) and ffilled junk (outcome) by
# design; we only assert when _assessment_ready is True.
if _assessment_ready:
    for _name, _X in [("X_cal_assess_df", X_cal_assess_df), ("X_inf_assess_df", X_inf_assess_df)]:
        assert not (_X.values == -9999).any(), \
            f"{_name} contains -9999 after assessment data release — sensors or outcome partially released?"
        assert not _X.isna().any().any(), \
            f"{_name} contains NaN after assessment data release"
    logger.info("Assessment feature-integrity assertions passed (no -9999, no NaN)")

    # Lag-correctness + not-dumb spot check: the last 5 assessment anchors are the deepest
    # into the assessment period, so their outcome lags pull most heavily from newly-released
    # data. Verify each cell equals the expected outcome series value AND falls within a
    # 5-sigma band derived from the training-period outcome distribution.
    _train_outcome = outcome.loc[:TRAIN_END]
    _train_mean    = float(_train_outcome.mean())
    _train_std     = float(_train_outcome.std())
    _plausible_lo  = _train_mean - 5 * _train_std
    _plausible_hi  = _train_mean + 5 * _train_std
    _sample_anchors = assessment_windows[-5:]
    _cal_lag_days   = GP_CALENDAR_PARAMS["lag_days"]
    for _anchor in _sample_anchors:
        for _k in _cal_lag_days:
            _expected = float(outcome.loc[_anchor - pd.Timedelta(days=_k)])
            _actual   = float(X_cal_assess_df.loc[_anchor, f"outcome_lag_{_k}"])
            assert _expected == _actual, \
                f"X_cal_assess[{_anchor.date()}, outcome_lag_{_k}] = {_actual} but outcome[{(_anchor - pd.Timedelta(days=_k)).date()}] = {_expected}"
            assert np.isfinite(_actual) and _actual != -9999.0, \
                f"Lag value {_actual} is non-finite or -9999 at X_cal_assess[{_anchor.date()}, outcome_lag_{_k}]"
            assert _plausible_lo <= _actual <= _plausible_hi, \
                f"Lag value {_actual} outside training 5-sigma band [{_plausible_lo:.2f}, {_plausible_hi:.2f}] at X_cal_assess[{_anchor.date()}, outcome_lag_{_k}] — assessment data may be corrupt or rescaled"
    logger.info(f"Assessment lag-correctness + not-dumb spot-check passed (5 anchors x {len(_cal_lag_days)} lags)")

pred_cal = gp_model.predict(model_cal, lik_cal, scaler_cal, y_mean_cal, y_std_cal, X_cal_assess_df.values, device=DEVICE)
pred_inf = gp_model.predict(model_inf, lik_inf, scaler_inf, y_mean_inf, y_std_inf, X_inf_assess_df.values, device=DEVICE)

pred_ensemble = np.concatenate([
    W_CAL_15  * pred_cal[:, :5] + W_INF_15  * pred_inf[:, :5],
    W_CAL_610 * pred_cal[:, 5:] + W_INF_610 * pred_inf[:, 5:],
], axis=1)

# ── Step 6c: seasonal-naive persistence blend + non-negativity clamp ─────────
# persistence[D,h] = outcome[D + h - lag], lag=7 (h<=3) else 14. Weekday-aligned,
# all within the D-OUTCOME_LAG availability cutoff (deepest reach y[D-4] at h=10).
_persist = np.full_like(pred_ensemble, np.nan)
for _i, _D in enumerate(assessment_windows):
    for _h in range(1, FORECAST_DAYS + 1):
        _lag = PERSISTENCE_LAG_SHORT if _h <= PERSISTENCE_SHORT_HORIZON else PERSISTENCE_LAG_LONG
        _persist[_i, _h - 1] = outcome.get(_D + pd.Timedelta(days=_h - _lag), np.nan)

_persist_bad = (~np.isfinite(_persist)) | (_persist == -9999.0)
if _assessment_ready:
    assert not _persist_bad.any(), \
        f"{int(_persist_bad.sum())} persistence lookups are -9999/NaN after assessment release"
# Graceful fallback: where persistence is unavailable, fall back to GP (=> no blend there).
_persist = np.where(_persist_bad, pred_ensemble, _persist)

_w = PERSISTENCE_BLEND_WEIGHT
pred_blended = (1.0 - _w) * pred_ensemble + _w * _persist
logger.info(
    f"Persistence blend: w={_w}, seasonal_7_14 "
    f"(lag {PERSISTENCE_LAG_SHORT}d for h<={PERSISTENCE_SHORT_HORIZON}, else {PERSISTENCE_LAG_LONG}d); "
    f"{int(_persist_bad.sum())} cells fell back to GP"
)

# Non-negativity clamp — target is >= 0; clamp is no-regret under MSE.
_n_clamped = int((pred_blended < 0).sum())
if _n_clamped:
    logger.info(f"Non-negativity clamp: floored {_n_clamped} of {pred_blended.size} predictions to 0")
pred_ensemble = np.maximum(pred_blended, 0.0)

logger.info("-" * 90)
logger.info(f"pred_ensemble shape: {pred_ensemble.shape}  min={pred_ensemble.min():.2f}  max={pred_ensemble.max():.2f}")

_timings["step_6_predict"] = time.perf_counter() - _t6

# ── Step 6b: optional in-sample sanity evaluation (RUN_EVAL flag) ────────────

_mse_table_md = ""

if RUN_EVAL:
    _EVAL_DIR = pathlib.Path(__file__).parent / "figures" / "evaluation"
    _EVAL_DIR.mkdir(parents=True, exist_ok=True)
    _N_EVAL = 30
    _eval_windows = training_windows[-_N_EVAL:]

    _X_cal_ev = features_calendar.build_features(_eval_windows, outcome, sensors, clear_cache=CLEAR_CACHE)
    _X_inf_ev = features_informed_lags.build_features(_eval_windows, outcome, sensors, clear_cache=CLEAR_CACHE)
    _pred_cal_ev = gp_model.predict(model_cal, lik_cal, scaler_cal, y_mean_cal, y_std_cal, _X_cal_ev.values, device=DEVICE)
    _pred_inf_ev = gp_model.predict(model_inf, lik_inf, scaler_inf, y_mean_inf, y_std_inf, _X_inf_ev.values, device=DEVICE)
    _pred_ev = 0.5 * _pred_cal_ev + 0.5 * _pred_inf_ev
    _y_ev = y_train_df.iloc[-_N_EVAL:].values

    _mse_cal_1to5,  _mse_cal_6to10  = _comp_mse(_y_ev, _pred_cal_ev)
    _mse_inf_1to5,  _mse_inf_6to10  = _comp_mse(_y_ev, _pred_inf_ev)
    _mse_1to5,      _mse_6to10      = _comp_mse(_y_ev, _pred_ev)

    # Shipped model = persistence-blended + clamped (mirror Step 6c on eval windows).
    # Persistence here is a deterministic outcome lookup; the GP part is still in-sample.
    _persist_ev = np.full_like(_pred_ev, np.nan)
    for _i, _D in enumerate(_eval_windows):
        for _h in range(1, FORECAST_DAYS + 1):
            _lag = PERSISTENCE_LAG_SHORT if _h <= PERSISTENCE_SHORT_HORIZON else PERSISTENCE_LAG_LONG
            _persist_ev[_i, _h - 1] = outcome.get(_D + pd.Timedelta(days=_h - _lag), np.nan)
    _persist_ev = np.where(np.isfinite(_persist_ev), _persist_ev, _pred_ev)
    _pred_blend_ev = np.maximum(
        (1.0 - PERSISTENCE_BLEND_WEIGHT) * _pred_ev + PERSISTENCE_BLEND_WEIGHT * _persist_ev, 0.0
    )
    _mse_blend_1to5, _mse_blend_6to10 = _comp_mse(_y_ev, _pred_blend_ev)

    # baselines
    _last_vals = np.array([
        outcome.get(d - pd.Timedelta(days=OUTCOME_LAG), np.nan) for d in _eval_windows
    ]).reshape(-1, 1)
    _pred_last = np.repeat(_last_vals, FORECAST_DAYS, axis=1).astype("float32")
    _train_mean = float(outcome[outcome.index < _eval_windows[0]].mean())
    _pred_mean = np.full((_N_EVAL, FORECAST_DAYS), _train_mean, dtype="float32")
    _mse_last_1to5, _mse_last_6to10 = _comp_mse(_y_ev, _pred_last)
    _mse_mean_1to5, _mse_mean_6to10 = _comp_mse(_y_ev, _pred_mean)

    logger.warning(
        f"Eval is IN-SAMPLE (model trained on these {_N_EVAL} windows) — numbers optimistic vs "
        f"out-of-sample refs: Exp77 ~0.045/0.056, Exp81 ~0.037/0.044"
    )
    logger.info(f"MSE comparison (in-sample, last {_N_EVAL} windows):")
    logger.info(f"  {'Model':<22} {'1-5d':>10} {'6-10d':>10}")
    logger.info(f"  {'gp_calendar':<22} {_mse_cal_1to5:>10.6f} {_mse_cal_6to10:>10.6f}")
    logger.info(f"  {'gp_informed_lags':<22} {_mse_inf_1to5:>10.6f} {_mse_inf_6to10:>10.6f}")
    logger.info(f"  {'Ensemble (GP only)':<22} {_mse_1to5:>10.6f} {_mse_6to10:>10.6f}")
    logger.info(f"  {f'SHIPPED (blend w={PERSISTENCE_BLEND_WEIGHT})':<22} {_mse_blend_1to5:>10.6f} {_mse_blend_6to10:>10.6f}")
    logger.info(f"  {'Predict-last':<22} {_mse_last_1to5:>10.6f} {_mse_last_6to10:>10.6f}")
    logger.info(f"  {'Predict-mean':<22} {_mse_mean_1to5:>10.6f} {_mse_mean_6to10:>10.6f}")

    _mse_table_md = (
        "| Model | MSE 1–5d | MSE 6–10d |\n"
        "|:---|---:|---:|\n"
        f"| `gp_calendar` | {_mse_cal_1to5:.6f} | {_mse_cal_6to10:.6f} |\n"
        f"| `gp_informed_lags` | {_mse_inf_1to5:.6f} | {_mse_inf_6to10:.6f} |\n"
        f"| `gp_ensemble` (pre-blend) | {_mse_1to5:.6f} | {_mse_6to10:.6f} |\n"
        f"| **SHIPPED: gp_ensemble + {PERSISTENCE_BLEND_WEIGHT} persistence** | **{_mse_blend_1to5:.6f}** | **{_mse_blend_6to10:.6f}** |\n"
        f"| Predict-last | {_mse_last_1to5:.6f} | {_mse_last_6to10:.6f} |\n"
        f"| Predict-mean | {_mse_mean_1to5:.6f} | {_mse_mean_6to10:.6f} |\n"
    )

    _ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    _name = "ensemble_gp_calendar_informed_lags"
    for _fig, _stem in [
        (eval_plots.scatter_by_day(_y_ev, _pred_ev, _name),                             "scatter_by_day"),
        (eval_plots.scatter_by_period(_y_ev, _pred_ev, _mse_1to5, _mse_6to10, _name),  "scatter_by_period"),
        (eval_plots.sample_windows(_y_ev, _pred_ev, _name, _eval_windows),              "sample_windows"),
    ]:
        _fig.savefig(_EVAL_DIR / f"{_stem}_{_ts}.png", bbox_inches="tight", dpi=120)
        _fig.savefig(_EVAL_DIR / f"{_stem}.png", bbox_inches="tight", dpi=120)
        plt.close(_fig)
        logger.info(f"Eval plot saved: {_EVAL_DIR / _stem}.png  (+ timestamped copy)")

# ── Step 7: write pred_matrix.csv ────────────────────────────────────────────

_t7 = time.perf_counter()

pred_df = pd.DataFrame(
    pred_ensemble,
    columns=[f"day_{d}" for d in range(1, FORECAST_DAYS + 1)],
)
pred_df.insert(0, "forecast_id", range(1, len(assessment_windows) + 1))

_OUT_PATH = pathlib.Path(__file__).parent / "pred_matrix.csv"
pred_df.to_csv(_OUT_PATH, index=False)

logger.info("-" * 90)
logger.info(f"pred_matrix.csv written: {pred_df.shape}  →  {_OUT_PATH}")

_timings["step_7_write_csv"] = time.perf_counter() - _t7

# ── Step 8: write mse_summary.csv if assessment actuals are available ────────
# Same code path runs always; behaviour depends only on whether the assessment
# dataset has populated `outcome` for the assessment period. -9999 placeholders
# are treated as missing.

_t8 = time.perf_counter()

# day_h corresponds to outcome[anchor + h] for h in 1..FORECAST_DAYS
# (matches the get_training_targets convention).
_actuals = np.stack([
    [outcome.get(d + pd.Timedelta(days=h), np.nan) for h in range(1, FORECAST_DAYS + 1)]
    for d in assessment_windows
]).astype("float32")
_actuals = np.where(_actuals == -9999, np.nan, _actuals)

if np.isfinite(_actuals).all():
    _sq = (_actuals - pred_ensemble) ** 2
    _mse_summary_df = pd.DataFrame({
        "forecast_id": range(1, len(assessment_windows) + 1),
        "mse_1_5":     _sq[:, :5].mean(axis=1),
        "mse_6_10":    _sq[:, 5:].mean(axis=1),
    })
    _MSE_OUT = pathlib.Path(__file__).parent / "mse_summary.csv"
    _mse_summary_df.to_csv(_MSE_OUT, index=False)
    logger.info(f"mse_summary.csv written: {_mse_summary_df.shape}  →  {_MSE_OUT}")
else:
    _n_missing = int((~np.isfinite(_actuals)).sum())
    logger.info(
        f"mse_summary.csv: deferred — {_n_missing} of {_actuals.size} "
        "assessment actuals not yet available"
    )

_timings["step_8_mse_summary"] = time.perf_counter() - _t8
_timings["pipeline_total"] = time.perf_counter() - _t_start

# ── Compile report ────────────────────────────────────────────────────────────

if COMPILE_REPORT:
    _report_md   = pathlib.Path(__file__).parent / "report.md"
    _report_pdf  = pathlib.Path(__file__).parent / "report.pdf"
    _rendered_md = pathlib.Path(__file__).parent / "_report_rendered.md"
    _MSE_PLACEHOLDER = (
        "<!-- MSE_INSAMPLE_TABLE_MARKER -->\n"
        "*In-sample MSE table is populated by `generate_forecasts.py` (Step 6b). "
        "Re-run that script for the actual numbers.*"
    )
    _TIMING_PLACEHOLDER = (
        "<!-- TIMING_SUMMARY_MARKER -->\n"
        "*Runtime table is populated by `generate_forecasts.py`. "
        "Re-run that script for actual timings.*"
    )
    _timing_md = (
        "| Step | Time (s) |\n"
        "|:---|---:|\n"
        + "\n".join(f"| `{_name}` | {_secs:.1f} |" for _name, _secs in _timings.items())
        + "\n| **(report compile not included)** | |\n"
    )
    _src = _report_md.read_text()
    _filled = _src.replace(_MSE_PLACEHOLDER, _mse_table_md)
    assert _filled != _src, \
        "MSE table placeholder block not found in report.md — check the marker text matches"
    _filled2 = _filled.replace(_TIMING_PLACEHOLDER, _timing_md)
    assert _filled2 != _filled, \
        "Timing summary placeholder block not found in report.md — check the marker text matches"
    _rendered_md.write_text(_filled2)
    subprocess.run(
        ["pandoc", str(_rendered_md), "-o", str(_report_pdf),
         "--pdf-engine=pdflatex",
         "--from", "markdown+raw_tex",
         "--highlight-style=tango"],
        check=True,
        cwd=pathlib.Path(__file__).parent,
    )
    logger.info(f"Report compiled: {_report_pdf}")

logger.info("=" * 90)
logger.info("Timing summary:")
for _name, _secs in _timings.items():
    logger.info(f"  {_name:<35} {_secs:6.1f}s")
logger.info(f"  Log written to: {_log_file}")