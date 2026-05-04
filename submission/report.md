---
title: "Forecasting Avoidable Deaths from ED Admission Delays"
author: "Rafal Urbaniak, Basis Research Institute"
date: "2026"
geometry: "margin=2.5cm"
fontsize: 11pt
colorlinks: true
header-includes:
  - \usepackage{booktabs}
  - \usepackage{float}
  - \floatplacement{figure}{H}
---

## 1. Introduction and Model Choice

This report describes a horizon-weighted ensemble of two sparse multitask Gaussian Process
models submitted to the SPHERE-PPL NHS EAD Forecasting Competition. The task is to predict
daily estimated avoidable deaths from ED admission delays across the Bristol NHS system,
ten days ahead, over 173 sliding windows.

The submitted model — **gp_ensemble** — combines two sparse RQ-iso multitask GPs: *gp_calendar*
(standard calendar lags) and *gp_informed_lags* (consecutive short lags chosen to respect
the autocorelation structure of the outcome variable). Ensemble weights are derived separately
for forecast horizons 1–5 and 6–10 by inverse-MSE weighting on a held-in training evaluation.

The architecture was selected after evaluating GP-family and tree-based candidates on a
held-out hard-week / average-difficulty-week split drawn from 2022–2024 development data;
GPs were favoured for their robustness under distribution shift on the hardest weeks.
Full architectural sweep details are documented in the development repository.

---

## 2. Data Pipeline

The development dataset spans 16 March 2023 to 30 September 2025 (930 daily observations).
The outcome variable is daily estimated avoidable deaths with no missing values. Of 220
candidate sensor variables, the three most predictive DTA columns (Bristol Royal Infirmary,
Weston, North Bristol NHS Trust) were selected based on Spearman correlation with the
outcome across all candidate sensors. Addition of further sensors 
led to performance deterioration.

Sensor readings arriving after noon on day $D$ are attributed to $D+1$ to respect the
competition's midday cutoff. A three-day outcome reporting lag means only $y_{D-4}$ and
earlier are used as input features (one day more conservative than the competition rule
allows). Rolling means and outcome lag features are anchored at $D-4$ consistently across
both models. Missing sensor values within the training set are forward-filled, backward-filled, then zero-filled.

---

## 3. Model Structure

Both component models share the `MultitaskSparseRqIsoGP` architecture: ten independent
sparse GPs (one per forecast horizon) implemented via `IndependentMultitaskVariationalStrategy`
with a `ScaleKernel(RQKernel)` covariance and Cholesky variational distribution. The mean
function is a per-task `ConstantMean`, leaving the RQ kernel to model all structure in the data.
Training maximises the ELBO via Adam with cosine-annealed learning rate and mini-batch data loading.
Models are trained once on the full development dataset; prediction over 173 assessment windows
runs in seconds per window.

The two models differ only in their feature sets:

| | **gp_calendar** | **gp_informed_lags** |
|---|---|---|
| Outcome lags | 4, 7, 14, 28 days | 4–7, 13–14, 20–21 days + $\Delta_{4,7}$ |
| Sensor lags | 0, 3, 7, 28 days (offset by outcome lag) | 0–2, 6–7, 13–14, 20–21 days (raw) |
| Rolling means | 7, 14, 28-day windows at $D-4$ | 7, 14, 28-day windows at $D-4$ |
| lr / inducing / n\_iter | 0.054 / 100 / 200 | 0.1 / 100 / 200 |

Hyperparameters and the choice of `ConstantMean` over a richer mean function were selected
on a held-out split of the development period,
documented in the development repository.

---

## 4. Evaluation

The table and trajectory plot below are in-sample over the last 30 training windows —
the model was trained on these windows, so MSE values are optimistic relative to the
out-of-sample assessment period the contest will score on. They are included as a
sanity check that the ensemble fits the training distribution and that horizon-specific
weighting yields a coherent improvement over either component model. Out-of-sample
generalisation was validated separately during development on a held-out
2025-04-01 – 2025-09-30 split, with results documented in the development repository.

<!-- MSE_INSAMPLE_TABLE_MARKER -->
*In-sample MSE table is populated by `generate_forecasts.py` (Step 6b). Re-run that script for the actual numbers.*

![Actual (blue) vs predicted (orange dashed) trajectories for 15 sampled training
windows over the 10-day forecast horizon.](figures/evaluation/sample_windows.png){ width=100% }

---

## 5. Runtime

End-to-end pipeline timing for the run that produced this report. Captured on CPU.

<!-- TIMING_SUMMARY_MARKER -->
*Runtime table is populated by `generate_forecasts.py`. Re-run that script for actual timings.*
