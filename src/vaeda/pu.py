"""PU (Positive-Unlabeled) learning implementation using PyTorch.

Replaces the TensorFlow/tf_keras-based implementation from v0.1.x.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from rich.progress import Progress
from sklearn.model_selection import RepeatedKFold
from sklearn.neighbors import NearestNeighbors

from .classifier import define_classifier
from .vae import _get_device


def _batch_slices(n: int, batch_size: int) -> list[tuple[int, int]]:
    """Return ``(start, end)`` minibatch bounds covering ``[0, n)``.

    A trailing batch of length 1 is merged into the previous batch so that
    ``BatchNorm1d`` never receives a single-sample batch during training
    (which raises "Expected more than 1 value per channel"). The only
    unavoidable singleton is the degenerate ``n == 1`` case.
    """
    slices = [(start, min(start + batch_size, n)) for start in range(0, n, batch_size)]
    if len(slices) >= 2 and slices[-1][1] - slices[-1][0] == 1:
        prev_start, _ = slices[-2]
        slices[-2] = (prev_start, slices[-1][1])
        slices.pop()
    return slices


def _normalize_pu_preds(
    preds_sum: np.ndarray,
    preds_on_p_sum: np.ndarray,
    i: int,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Average accumulated PU bagging scores.

    Each unlabeled point is scored on the held-in partition ``(k-1)`` times
    per repeat, i.e. ``i/k * (k-1)`` times total. Each positive point is
    scored on every one of the ``i`` folds, so it must divide by ``i`` (not
    ``i/k * (k-1)``) to stay on the same [0, 1] scale as ``preds``.
    """
    preds = preds_sum / ((i / k) * (k - 1))
    preds_on_p = preds_on_p_sum / i
    return preds, preds_on_p


def _train_one_epoch(
    model: torch.nn.Module,
    optimiser: torch.optim.Optimizer,
    X: torch.Tensor,
    Y: torch.Tensor,
    batch_size: int = 32,
) -> tuple[float, float]:
    """Train a single epoch with minibatches; return (loss, average precision)."""
    model.train()
    device = X.device
    n = X.shape[0]
    perm = torch.randperm(n, device=device)
    total_loss = 0.0
    n_batches = 0

    for start, end in _batch_slices(n, batch_size):
        idx = perm[start:end]
        x_batch = X[idx]
        y_batch = Y[idx]

        optimiser.zero_grad()
        preds = model(x_batch)
        loss = F.binary_cross_entropy(preds, y_batch)
        loss.backward()
        optimiser.step()
        total_loss += loss.item()
        n_batches += 1

    # Compute epoch-level average precision on full data
    model.eval()
    with torch.no_grad():
        all_preds = model(X).cpu().numpy()
        all_targets = Y.cpu().numpy()
        from sklearn.metrics import average_precision_score

        try:
            ap_val = float(average_precision_score(all_targets, all_preds))
        except ValueError:
            ap_val = 0.0

    return total_loss / n_batches, ap_val


def PU(
    U: np.ndarray,
    P: np.ndarray,
    k: int,
    N: int,
    cls_eps: int,
    seeds: np.ndarray,
    clss: str = "NN",
    _puPat: int = 5,
    puLR: float = 1e-3,
    num_layers: int = 1,
    _stop_metric: str = "ValAP",
    _verbose: int = 0,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Positive-Unlabeled bagging classifier.

    Parameters are the same as in v0.1.x for API compatibility.

    Bagging follows Mordelet & Vert: each fold fits a classifier on a small
    held-out subsample of the unlabeled set (``fit_idx``) plus all positives,
    then scores the remaining unlabeled points (``predict_idx``). Note that
    ``RepeatedKFold.split`` yields ``(train_idx, test_idx)``, so the large
    partition is used for prediction and the small one for fitting.
    """
    device = _get_device()
    random_state = seeds[0]
    rkf = RepeatedKFold(n_splits=k, n_repeats=N, random_state=random_state)

    preds = np.zeros([U.shape[0]])
    preds_on_P = np.zeros([P.shape[0]])

    hists = np.zeros([N * k, cls_eps])
    ap_hists = np.zeros([N * k, cls_eps])

    P_tensor = torch.tensor(P, dtype=torch.float32, device=device)

    i = 0
    with Progress() as progress:
        train_task = progress.add_task(description="", total=k)
        for predict_idx, fit_idx in rkf.split(U):
            i += 1
            progress.update(
                train_task,
                description=f"{i!s}/{(N * k)!s} iterations",
                refresh=True,
            )

            X = np.vstack([U[fit_idx, :], P])
            Y = np.concatenate([
                np.zeros(shape=[len(fit_idx)]),
                np.ones(shape=[P.shape[0]]),
            ])

            x = U[predict_idx, :]

            if clss == "NN":
                # Set seeds for reproducibility
                torch.manual_seed(seeds[1])

                classifier = define_classifier(ngens=X.shape[1], num_layers=num_layers)
                optimiser = torch.optim.Adam(classifier.parameters(), lr=puLR)

                # Shuffle training data
                ind = np.arange(X.shape[0])
                rng2 = np.random.Generator(np.random.PCG64(seeds[2]))
                rng2.shuffle(ind)

                X_t = torch.tensor(X[ind, :], dtype=torch.float32, device=device)
                Y_t = torch.tensor(Y[ind], dtype=torch.float32, device=device)
                x_t = torch.tensor(x, dtype=torch.float32, device=device)

                torch.manual_seed(seeds[3])
                for ep in range(cls_eps):
                    loss_val, ap_val = _train_one_epoch(classifier, optimiser, X_t, Y_t)
                    hists[i - 1, ep] = loss_val
                    ap_hists[i - 1, ep] = ap_val

                # Predictions
                classifier.eval()
                with torch.no_grad():
                    torch.manual_seed(seeds[3])
                    preds[predict_idx] = preds[predict_idx] + classifier(x_t).cpu().numpy()
                    torch.manual_seed(seeds[3])
                    preds_on_P = preds_on_P + classifier(P_tensor).cpu().numpy()

            if clss == "knn":
                neighbors = int(np.sqrt(X.shape[0]))
                knn = NearestNeighbors(n_neighbors=neighbors)
                knn.fit(X, Y)

                graph = knn.kneighbors_graph(x)
                preds[predict_idx] = preds[predict_idx] + np.squeeze(
                    np.array(np.sum(graph[:, Y == 1], axis=1) / neighbors)
                )

                graph = knn.kneighbors_graph(P)
                preds_on_P = preds_on_P + np.squeeze(
                    np.array(np.sum(graph[:, Y == 1], axis=1) / neighbors)
                )

    preds, preds_on_P = _normalize_pu_preds(preds, preds_on_P, i, k)

    return preds, preds_on_P, hists, ap_hists


def epoch_PU(
    U: np.ndarray,
    P: np.ndarray,
    k: int,
    N: int,
    cls_eps: int,
    seeds: np.ndarray,
    _puPat: int = 5,
    puLR: float = 1e-3,
    num_layers: int = 1,
    _stop_metric: str = "ValAP",
    _verbose: int = 0,
) -> _EpochHistory:
    """Train a single PU fold to determine optimal epoch count.

    Returns a history-like object with a ``.history`` dict containing
    ``"loss"`` and ``"ap"`` (average precision) lists, matching the
    tf_keras History API used by the caller.
    """
    device = _get_device()
    random_state = seeds[0]
    rkf = RepeatedKFold(n_splits=k, n_repeats=N, random_state=random_state)

    i = 0
    with Progress() as progress:
        train_task = progress.add_task(description="", total=k)
        for _, fit_idx in rkf.split(U):
            i += 1
            progress.update(
                train_task,
                description=f"{i!s}/{(N * k)!s} iterations",
                refresh=True,
            )
            X = np.vstack([U[fit_idx, :], P])
            Y = np.concatenate([
                np.zeros([len(fit_idx)]),
                np.ones([P.shape[0]]),
            ])

            torch.manual_seed(seeds[1])
            classifier = define_classifier(X.shape[1], num_layers=num_layers)
            optimiser = torch.optim.Adam(classifier.parameters(), lr=puLR)

            # Shuffle training data
            ind = np.arange(X.shape[0])
            rng2 = np.random.Generator(np.random.PCG64(seeds[2]))
            rng2.shuffle(ind)

            X_t = torch.tensor(X[ind, :], dtype=torch.float32, device=device)
            Y_t = torch.tensor(Y[ind], dtype=torch.float32, device=device)

            torch.manual_seed(seeds[3])
            loss_history: list[float] = []
            ap_history: list[float] = []
            for _ in range(cls_eps):
                loss_val, ap_val = _train_one_epoch(classifier, optimiser, X_t, Y_t)
                loss_history.append(loss_val)
                ap_history.append(ap_val)

            break  # Only first fold, matching v0.1.x behaviour

    return _EpochHistory({"loss": loss_history, "ap": ap_history})


class _EpochHistory:
    """Minimal history object mimicking the tf_keras History API."""

    def __init__(self, history: dict[str, list[float]]) -> None:
        self.history = history
