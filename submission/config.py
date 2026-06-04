"""Frozen hyperparameters and constants for the submission pipeline.

All other submission modules import from here. Edit values in this file,
not at the call sites.
"""

FORECAST_DAYS = 10        # predict days 1-10 ahead from each origin
OUTCOME_LAG   = 4         # freshest outcome feature is y[D-4]; D-3 may be legal but we're conservative
TRAIN_START   = "2023-03-16"
TRAIN_END     = "2025-09-30"
ASSESSMENT_START = "2025-10-01"
ASSESSMENT_END   = "2026-03-31"

TOP_K_SENSOR_COLS = [
    "No. of DTAs__Bristol Royal Infirmary",
    "No. of DTAs__Weston",
    "No. of DTAs__North Bristol NHS Trust",
]

GP_CALENDAR_PARAMS = {
    "num_inducing": 100,
    "lr": 0.054,
    "batch_size": 128,
    "n_iter": 200,
    "lag_days": [4, 7, 14, 28],
    "rolling_windows": [7, 14, 28],
    "sensor_lag_days": [0, 3, 7, 28],
    "top_k_sensors": 3,
    "outputscale_init": 2.0,
}

GP_INFORMED_LAGS_PARAMS = {
    "num_inducing": 100,
    "lr": 0.1,
    "batch_size": 64,
    "n_iter": 200,
    "lag_days": [4, 5, 6, 7, 13, 14, 20, 21],
    "rolling_windows": [7, 14, 28],
    "rolling_lag": 4,
    "sensor_lag_days": [0, 1, 2, 6, 7, 13, 14, 20, 21],
    "top_k_sensors": 3,
    "outputscale_init": 2.0,
}

# Persistence blend (seasonal-naive component).
# final = (1 - W) * gp_ensemble + W * persistence, then clamped at 0.
# persistence[D, h] = outcome[D + h - lag], lag = SHORT for h <= SHORT_HORIZON else LONG.
# Weekday-aligned (7/14 = 1/2 weeks); deepest reach outcome[D-4] at h=10 (respects OUTCOME_LAG).
# W chosen by a walk-forward battery (2 winters, 53 weeks): the production-strength
# optimum is ~0.25-0.30, NOT the 0.5 of the original handicapped single-winter proxy.
# At 0.25 the improvement is significant (DM -3.06; bootstrap CI [-0.017, -0.005]),
# robust across both observed winters, sits on the flat plateau, and roughly halves
# worst-week tail risk vs 0.5. See ~/Downloads/rafal-basis-tests/.
PERSISTENCE_BLEND_WEIGHT  = 0.25
PERSISTENCE_LAG_SHORT     = 7
PERSISTENCE_LAG_LONG      = 14
PERSISTENCE_SHORT_HORIZON = 3
