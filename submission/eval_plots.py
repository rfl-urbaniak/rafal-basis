"""Evaluation plots matching the notebook visualisation style."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _add_diagonal(ax: plt.Axes) -> None:
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)


def scatter_by_day(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    name: str,
) -> matplotlib.figure.Figure:
    """2×5 grid of true-vs-predicted scatter plots, one panel per horizon day.

    :param y_true: observed values, shape (n_windows, 10).
    :param y_pred: predicted values, shape (n_windows, 10).
    :param name: experiment name used in the figure title.
    :returns: matplotlib Figure.
    """
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    fig.suptitle(f"{name} — True vs Predicted by horizon day", fontsize=12)
    for d in range(10):
        ax = axes[d // 5][d % 5]
        mse_d = float(np.mean((y_true[:, d] - y_pred[:, d]) ** 2))
        ax.scatter(y_true[:, d], y_pred[:, d], alpha=0.4, s=12, color="steelblue")
        _add_diagonal(ax)
        ax.set_title(f"Day {d + 1} | MSE={mse_d:.4f}", fontsize=9)
        ax.set_xlabel("Actual", fontsize=8)
        ax.set_ylabel("Predicted", fontsize=8)
    fig.tight_layout()
    return fig


def scatter_by_period(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mse_1to5: float,
    mse_6to10: float,
    name: str,
) -> matplotlib.figure.Figure:
    """Side-by-side scatter comparing the two competition evaluation periods.

    :param y_true: observed values, shape (n_windows, 10).
    :param y_pred: predicted values, shape (n_windows, 10).
    :param mse_1to5: pre-computed competition MSE for days 1–5.
    :param mse_6to10: pre-computed competition MSE for days 6–10.
    :param name: experiment name used in the figure title.
    :returns: matplotlib Figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"{name} — True vs Predicted by evaluation period", fontsize=12)
    for ax, (label, sl, mse) in zip(
        axes,
        [("Days 1–5", slice(0, 5), mse_1to5), ("Days 6–10", slice(5, 10), mse_6to10)],
    ):
        ax.scatter(y_true[:, sl].ravel(), y_pred[:, sl].ravel(), alpha=0.3, s=8, color="steelblue")
        _add_diagonal(ax)
        ax.set_title(f"{label} | MSE={mse:.4f}", fontsize=10)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
    fig.tight_layout()
    return fig


def sample_windows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    name: str,
    daily_index: pd.DatetimeIndex,
    n: int = 15,
    seed: int = 42,
) -> matplotlib.figure.Figure:
    """Line plots of actual vs predicted for n randomly sampled forecast windows.

    :param y_true: observed values, shape (n_windows, 10).
    :param y_pred: predicted values, shape (n_windows, 10).
    :param name: experiment name used in the figure title.
    :param daily_index: DatetimeIndex of forecast origin dates, length n_windows.
    :param n: number of windows to sample.
    :param seed: random seed for reproducibility.
    :returns: matplotlib Figure.
    """
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(y_true), size=min(n, len(y_true)), replace=False))
    ncols = 5
    nrows = int(np.ceil(len(idx) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3 * nrows), sharey=True)
    axes_flat = np.array(axes).ravel()
    fig.suptitle(f"{name} — Predicted vs Actual (sample forecast windows)", fontsize=11)
    days = range(1, 11)
    for plot_i, win_i in enumerate(idx):
        ax = axes_flat[plot_i]
        ax.plot(days, y_true[win_i], color="steelblue", marker="o", ms=4, label="Actual")
        ax.plot(days, y_pred[win_i], color="darkorange", linestyle="--", marker="x", ms=4, label="Predicted")
        mse_w = float(np.mean((y_true[win_i] - y_pred[win_i]) ** 2))
        ax.set_title(f"{daily_index[win_i].date()} | MSE={mse_w:.4f}", fontsize=9)
        ax.set_xlabel("Horizon (days)")
        if plot_i % ncols == 0:
            ax.set_ylabel("Avoidable deaths")
        if plot_i == 0:
            ax.legend(fontsize=7)
    for ax in axes_flat[len(idx):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig
