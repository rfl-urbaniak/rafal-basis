# Submission Instructions

**NHS EAD Forecasting Competition — Bristol Avoidable Deaths from ED Admission Delays**
Rafal Urbaniak, Basis Research Institute

A horizon-weighted ensemble of two sparse multitask Gaussian-process models
(`gp_calendar` + `gp_informed_lags`) producing 173 sliding 10-day forecasts
over Oct 2025 – Mar 2026. See `report.md` / `report.pdf` for the methodology
write-up; this file covers setup and reproduction.

---

## Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** — install via
  `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Git LFS** — required to pull the development dataset
  (`data/turingAI_forecasting_challenge_dataset.csv`)
- **Pandoc + a LaTeX engine** *(optional)* — only needed for `make report` /
  `COMPILE_REPORT=True`

No R, no `pip`. uv is the only supported install path.

---

## Setup

```bash
git clone https://github.com/<owner>/<this-fork>.git
cd <this-fork>
uv sync          # installs from the committed uv.lock
make run         # pulls data via LFS, unzips CSV, runs pipeline
```

`uv sync` reads `pyproject.toml` + `uv.lock` and produces a reproducible
environment matching the one used to generate the submitted forecasts.

---

## Run the pipeline

```bash
make run
```

This pulls the latest data via Git LFS, unpacks the CSV, and runs the pipeline.
Equivalent to:

```bash
git lfs pull
cd data && unzip -o turingAI_forecasting_challenge_dataset.csv.zip
uv run python submission/generate_forecasts.py
```

**Runtime:** ~15–25 minutes on CPU (no GPU required). The two GPs are trained
once on the full 2023-03-16 → 2025-09-30 development set; the 173 assessment
windows are then inference-only (seconds each).

**Outputs:**

- `submission/pred_matrix.csv` — 173 × 11 forecast matrix
  (`forecast_id, day_1, …, day_10`).
- `submission/mse_summary.csv` — 173 × 3 per-window MSE matrix
  (`forecast_id, mse_1_5, mse_6_10`). **Written automatically only when the
  loaded outcome series contains assessment-period actuals**; before 6 June
  2026 the script logs `mse_summary.csv: deferred — N of 1730 actuals not yet
  available` and skips the write. Re-running after the assessment dataset is
  released produces the file with no code change.
- `submission/report.pdf` — compiled report (only if pandoc + pdflatex are
  available and `COMPILE_REPORT=True`).
- `submission/figures/evaluation/` — diagnostic plots
  (in-sample scatter, sample-window trajectories).
- `submission/figures/loss/` — training loss curves.
- `submission/logs/generate_forecasts_<timestamp>.log` — full run log.

### Top-of-file flags

`generate_forecasts.py` exposes runtime flags near the top of the file. All
default to `True` and produce the submitted artefacts:

| Flag | Effect |
|------|--------|
| `CLEAR_CACHE` | Delete parquet/`.pt` caches before the run; set `False` to reuse |
| `USE_GPU` | Use CUDA if available (CPU works fine) |
| `TUNE_WEIGHTS` | Derive horizon-specific ensemble weights from in-sample MSE |
| `RUN_EVAL` | Produce in-sample MSE table + diagnostic plots |
| `COMPILE_REPORT` | pandoc-compile `report.md` → `report.pdf` |

---

## Verify against the AggregatoR contract

Before pushing the fork, replicate the [Forecast AggregatoR](https://github.com/SPHERE-PPL/Forecast-AggregatoR)'s
submission checks:

```bash
make smoke                                         # local CSVs
make smoke-remote OWNER=<gh-owner> REPO=<gh-repo>  # via GitHub Contents API (post-push)
```

`smoke` reads `submission/pred_matrix.csv` from disk; `smoke-remote` fetches
via the GitHub API the same way the AggregatoR does, which catches silent
failures the local check cannot (file gitignored, in Git LFS, or larger than
the API's 1 MB inline limit).

The script is `submission/aggregator_check.py` — a faithful port of
`Forecast-AggregatoR/FAR_app/app.R` lines 125–215 with the
NHS-EAD-forecast competition branch added. Both should pass before final
submission.

Pre-6-June (no assessment actuals): pass `--skip-mse` to silence the missing
`mse_summary.csv` notice. Once the assessment dataset is released, re-run
`generate_forecasts.py` — `mse_summary.csv` appears automatically — then drop
`--skip-mse` for the final pre-20-June verification.

---

## Hyperparameters

All hyperparameters live in `submission/config.py` as `GP_CALENDAR_PARAMS`
and `GP_INFORMED_LAGS_PARAMS` dicts. They were selected on a held-out split
of the development period; methodology is summarised in `report.md` §3.

---

## Report

The PDF is built automatically when `generate_forecasts.py` runs with
`COMPILE_REPORT=True`. To rebuild manually:

```bash
make report
```

Requires pandoc and pdflatex. The source is `submission/report.md`; the
in-sample MSE table and timing summary are injected at compile time, so a
manual rebuild after a fresh `generate_forecasts.py` run reflects the latest
evaluation numbers.

---

## Layout reference

```
submission/
├── instructions.md           # this file
├── config.py                 # frozen hyperparams, dates, sensor cols
├── data_helpers.py           # load_raw, get_outcome, get_sensors, windows, targets
├── features_calendar.py      # build_features for gp_calendar
├── features_informed_lags.py # build_features for gp_informed_lags
├── gp_model.py               # MultitaskSparseRqIsoGP train/predict
├── eval_plots.py             # diagnostic-plot helpers
├── aggregator_check.py       # pre-submission validator
├── generate_forecasts.py     # orchestrator (run this)
├── report.md                 # ≤1000-word report source
├── pred_matrix.csv           # output; regenerated each run
└── mse_summary.csv           # output; written when actuals are available (post 6 June 2026)
```
