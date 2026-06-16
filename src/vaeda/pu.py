"""PU (Positive-Unlabeled) learning orchestration.

Backend-agnostic bagging: this module handles the numpy-level fold splitting
and score averaging, delegating each fold's neural-network training to the
compute backend (see :func:`vaeda.backends._torch.train.train_pu_fold`).
The ``knn`` classifier branch is pure scikit-learn and stays here.
"""

from __future__ import annotations

import numpy as np
from rich.progress import Progress
from sklearn.model_selection import RepeatedKFold
from sklearn.neighbors import NearestNeighbors

from .backends import get_backend


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
    random_state = seeds[0]
    rkf = RepeatedKFold(n_splits=k, n_repeats=N, random_state=random_state)

    preds = np.zeros([U.shape[0]])
    preds_on_P = np.zeros([P.shape[0]])

    hists = np.zeros([N * k, cls_eps])
    ap_hists = np.zeros([N * k, cls_eps])

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
                fold = get_backend().train_pu_fold(
                    X,
                    Y,
                    x,
                    P,
                    cls_eps=cls_eps,
                    num_layers=num_layers,
                    pu_lr=puLR,
                    seeds=seeds,
                )
                # x and P were supplied, so the fold always carries predictions
                assert fold.pred_x is not None and fold.pred_P is not None
                hists[i - 1, :] = fold.loss_hist
                ap_hists[i - 1, :] = fold.ap_hist
                preds[predict_idx] = preds[predict_idx] + fold.pred_x
                preds_on_P = preds_on_P + fold.pred_P

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
    random_state = seeds[0]
    rkf = RepeatedKFold(n_splits=k, n_repeats=N, random_state=random_state)

    i = 0
    loss_history: list[float] = []
    ap_history: list[float] = []
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

            fold = get_backend().train_pu_fold(
                X,
                Y,
                None,
                None,
                cls_eps=cls_eps,
                num_layers=num_layers,
                pu_lr=puLR,
                seeds=seeds,
            )
            loss_history = fold.loss_hist
            ap_history = fold.ap_hist

            break  # Only first fold, matching v0.1.x behaviour

    return _EpochHistory({"loss": loss_history, "ap": ap_history})


class _EpochHistory:
    """Minimal history object mimicking the tf_keras History API."""

    def __init__(self, history: dict[str, list[float]]) -> None:
        self.history = history
