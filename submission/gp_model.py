"""Sparse multitask GP model used by both submission ensemble components.

Defines :class:`MultitaskSparseRqIsoGP` (RQ-iso kernel, ConstantMean,
sparse variational with learnable inducing points), plus ``train`` /
``predict`` helpers and a state-save/load cache for ``.pt`` files.
"""

from __future__ import annotations

import datetime
import pathlib

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from config import FORECAST_DAYS
from loguru import logger
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

N_TEMPORAL_COLS: int = 5


class MultitaskSparseRqIsoGP(gpytorch.models.ApproximateGP):
    """T independent sparse RQ-iso GPs via IndependentMultitaskVariationalStrategy.

    :param inducing_points: float tensor of shape (num_tasks, num_inducing, input_dim).
    :param num_tasks: number of output tasks (forecast horizon days).
    """

    def __init__(self, inducing_points: torch.Tensor, num_tasks: int = 10) -> None:
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2),
            batch_shape=torch.Size([num_tasks]),
        )
        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=True,
            ),
            num_tasks=num_tasks,
            task_dim=-1,
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_tasks]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RQKernel(batch_shape=torch.Size([num_tasks])),
            batch_shape=torch.Size([num_tasks]),
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


def _train_gp(
    model: MultitaskSparseRqIsoGP,
    likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    n_iter: int,
    lr: float,
    batch_size: int,
    device: str,
) -> list[float]:
    """
    Maximise the ELBO via Adam with CosineAnnealingLR and mini-batches.

    :param model: instantiated MultitaskSparseRqIsoGP.
    :param likelihood: MultitaskGaussianLikelihood.
    :param train_x: float tensor shape (n, input_dim).
    :param train_y: float tensor shape (n, num_tasks) — already normalised.
    :param n_iter: number of training epochs.
    :param lr: initial Adam learning rate.
    :param batch_size: mini-batch size.
    :param device: "cpu" or "cuda".
    :returns: list of per-epoch ELBO losses.
    """
    dev = torch.device(device)
    model.to(dev)
    likelihood.to(dev)
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(likelihood.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_iter, eta_min=lr * 0.01
    )
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)

    logger.info(
        f"_train_gp: n_train={len(train_x)} n_iter={n_iter} lr={lr} "
        f"batch={batch_size} device={device}"
    )
    losses: list[float] = []
    for i in range(n_iter):
        epoch_loss = 0.0
        for bx, by in loader:
            bx, by = bx.to(dev), by.to(dev)
            optimizer.zero_grad()
            loss = -mll(model(bx), by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(bx)
        scheduler.step()
        losses.append(epoch_loss / len(train_x))
        if (i + 1) % max(1, n_iter // 5) == 0:
            logger.info(
                f"  iter {i+1:4d}/{n_iter}  ELBO={-losses[-1]:.4f}"
                f"  lr={optimizer.param_groups[0]['lr']:.2e}"
            )
    logger.info(f"_train_gp: done  final_ELBO={-losses[-1]:.4f}")
    return losses


def _predict_gp(
    model: MultitaskSparseRqIsoGP,
    likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood,
    test_x: torch.Tensor,
    device: str,
) -> np.ndarray:
    """
    Return posterior mean predictions in normalised space.

    :param model: trained MultitaskSparseRqIsoGP.
    :param likelihood: trained MultitaskGaussianLikelihood.
    :param test_x: float tensor shape (n, input_dim).
    :param device: "cpu" or "cuda".
    :returns: numpy array of shape (n, num_tasks).
    """
    dev = torch.device(device)
    model.eval()
    likelihood.eval()
    means = []
    with torch.no_grad():
        for start in range(0, len(test_x), 512):
            bx = test_x[start : start + 512].to(dev)
            means.append(likelihood(model(bx)).mean.detach().cpu())
    return torch.cat(means).numpy()


def _plot_losses(losses: list[float], name: str, figures_dir: pathlib.Path) -> None:
    """
    Save a training loss curve to figures_dir with a timestamped filename.

    :param losses: per-epoch ELBO losses from _train_gp.
    :param name: model name used in title and filename.
    :param figures_dir: directory to write the PNG into.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = figures_dir / f"{name}_{ts}.png"
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(losses) + 1), losses, linewidth=1.2, color="steelblue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ELBO loss")
    ax.set_title(f"{name} — training loss")
    ax.grid(linewidth=0.4, alpha=0.5)
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info(f"Loss plot saved: {path}")


def _save_state(
    path: pathlib.Path,
    model: MultitaskSparseRqIsoGP,
    likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood,
    x_scaler: StandardScaler,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    num_inducing: int,
    input_dim: int,
) -> None:
    """
    Persist model state, scaler, and normalisation stats to a single file.

    :param path: destination .pt file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state":      model.state_dict(),
        "likelihood_state": likelihood.state_dict(),
        "x_scaler":         x_scaler,
        "y_mean":           y_mean,
        "y_std":            y_std,
        "num_inducing":     num_inducing,
        "input_dim":        input_dim,
        "num_tasks":        FORECAST_DAYS,
    }, path)
    logger.info(f"Model state saved: {path}")


def _load_state(
    path: pathlib.Path,
) -> tuple:
    """
    Reconstruct model and all inference state from a saved .pt file.

    :param path: source .pt file written by _save_state.
    :returns: (model, likelihood, x_scaler, y_mean, y_std)
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    dummy = torch.zeros(state["num_tasks"], state["num_inducing"], state["input_dim"])
    model = MultitaskSparseRqIsoGP(dummy, num_tasks=state["num_tasks"])
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=state["num_tasks"])
    model.load_state_dict(state["model_state"])
    likelihood.load_state_dict(state["likelihood_state"])
    logger.info(f"Model state loaded: {path}")
    return model, likelihood, state["x_scaler"], state["y_mean"], state["y_std"]


def _fill_nan(X: np.ndarray) -> np.ndarray:
    """
    Forward-fill, backward-fill, then zero-fill — matching Exp77/81 notebooks.

    No data leakage because:

    - At train time: rows are ordered chronologically. NaNs only appear in the
      first ~28 rows (D - k before the data start). ffill finds nothing before
      row 0 so leaves those NaNs; bfill fills them from the next valid training
      row. This is within-training-set contamination of ~3% of rows only — no
      assessment-period values are present in X at this point, so none can leak.

    - At predict time (assessment windows): all lags are ≤ OUTCOME_LAG + 28 = 32
      days. The earliest assessment window is Sep 30 2025, so the deepest lookback
      is Aug 29 2025 — well inside the training data. No NaNs are expected and
      the function is effectively a no-op for assessment features.

    :param X: float32 array shape (n, n_features), rows ordered by time.
    :returns: same shape, NaNs replaced.
    """
    return pd.DataFrame(X).ffill().bfill().fillna(0).values.astype("float32")


def _scale_X(
    X: np.ndarray,
    scaler: StandardScaler | None = None,
) -> tuple[np.ndarray, StandardScaler]:
    """
    Scale non-temporal columns with StandardScaler; leave temporal columns untouched.

    :param X: float32 array shape (n, n_features).
    :param scaler: fitted scaler to reuse at predict time; if None, fit a new one.
    :returns: scaled array and the (fitted) scaler.
    """
    temporal = X[:, :N_TEMPORAL_COLS]
    rest = X[:, N_TEMPORAL_COLS:]
    if scaler is None:
        scaler = StandardScaler()
        rest_sc = scaler.fit_transform(rest)
    else:
        rest_sc = scaler.transform(rest)
    return np.concatenate([temporal, rest_sc], axis=1).astype("float32"), scaler


def train(
    X: np.ndarray,
    y: np.ndarray,
    params: dict,
    device: str = "cpu",
    cache_path: pathlib.Path | None = None,
    clear_cache: bool = False,
    name: str = "",
    figures_dir: pathlib.Path | None = None,
    seed: int = 0,
) -> tuple:
    """
    Scale inputs, fit MultitaskSparseRqIsoGP, return model state for inference.

    If cache_path is provided and the file exists (and clear_cache is False),
    the model is loaded from cache and training is skipped — no loss plot is
    generated in that case.

    :param X: float32 array shape (n_train, n_features) — may contain NaNs.
    :param y: float32 array shape (n_train, FORECAST_DAYS).
    :param params: dict with keys num_inducing, lr, batch_size, n_iter.
    :param device: "cpu" or "cuda".
    :param cache_path: path to save/load model state (.pt file).
    :param clear_cache: if True, delete existing cache before training.
    :param name: model name for logging and plot title.
    :param figures_dir: if provided, save loss plot here after training.
    :param seed: RNG seed for inducing-point selection and mini-batch shuffling,
        making training deterministic for the pre-/post-release consistency check.
    :returns: (model, likelihood, x_scaler, y_mean, y_std)
    """
    # Seed before any stochastic step (np.random.choice for inducing points,
    # shuffle=True DataLoader). Re-seeded per call so each model is reproducible
    # independent of call order.
    np.random.seed(seed)
    torch.manual_seed(seed)

    if cache_path is not None:
        if clear_cache and cache_path.exists():
            cache_path.unlink()
        if cache_path.exists():
            logger.info(f"train ({name}): cache hit — loading from {cache_path}")
            return _load_state(cache_path)
        logger.info(f"train ({name}): cache miss — training from scratch")

    X = _fill_nan(X)
    X_sc, x_scaler = _scale_X(X)

    y_mean = y.mean(axis=0, keepdims=True)
    y_std = y.std(axis=0, keepdims=True) + 1e-8
    y_sc = ((y - y_mean) / y_std).astype("float32")

    num_inducing: int = params["num_inducing"]
    idx = np.random.choice(len(X_sc), min(num_inducing, len(X_sc)), replace=False)
    inducing_pts = (
        torch.tensor(X_sc[idx])
        .unsqueeze(0)
        .expand(FORECAST_DAYS, -1, -1)
        .clone()
    )

    model = MultitaskSparseRqIsoGP(inducing_pts, num_tasks=FORECAST_DAYS)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=FORECAST_DAYS)

    if params.get("outputscale_init") is not None:
        model.covar_module.outputscale = params["outputscale_init"]
        logger.info(f"  outputscale initialised to {params['outputscale_init']}")

    losses = _train_gp(
        model, likelihood,
        torch.tensor(X_sc), torch.tensor(y_sc),
        n_iter=params["n_iter"],
        lr=params["lr"],
        batch_size=params["batch_size"],
        device=device,
    )

    if figures_dir is not None:
        _plot_losses(losses, name or "model", figures_dir)

    if cache_path is not None:
        _save_state(cache_path, model, likelihood, x_scaler, y_mean, y_std,
                    num_inducing, X_sc.shape[1])

    return model, likelihood, x_scaler, y_mean, y_std


def predict(
    model: MultitaskSparseRqIsoGP,
    likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood,
    x_scaler: StandardScaler,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    X: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """
    Scale inputs, run GP inference, denormalise and return predictions.

    :param model: trained MultitaskSparseRqIsoGP.
    :param likelihood: trained MultitaskGaussianLikelihood.
    :param x_scaler: fitted StandardScaler from train().
    :param y_mean: per-horizon training mean, shape (1, FORECAST_DAYS).
    :param y_std: per-horizon training std + 1e-8, shape (1, FORECAST_DAYS).
    :param X: float32 array shape (n, n_features) — may contain NaNs.
    :param device: "cpu" or "cuda".
    :returns: float32 array shape (n, FORECAST_DAYS), denormalised predictions.
    """
    X = _fill_nan(X)
    X_sc, _ = _scale_X(X, scaler=x_scaler)
    mean_sc = _predict_gp(model, likelihood, torch.tensor(X_sc), device=device)
    return (mean_sc * y_std + y_mean).astype("float32")
