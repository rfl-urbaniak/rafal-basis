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
