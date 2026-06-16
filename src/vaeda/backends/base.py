"""The compute-backend seam.

A backend takes numpy arrays in and returns numpy arrays out, hiding the
framework-specific training loop. Both the torch backend and the (Phase 4)
TensorFlow backend implement this protocol so the orchestration code in
``vaeda.vaeda`` and ``vaeda.pu`` never imports a framework directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Protocol

if TYPE_CHECKING:
    import numpy as np


class PuFoldResult(NamedTuple):
    """Result of training a single PU bagging fold.

    ``pred_x`` / ``pred_P`` are the classifier scores on the held-out unlabeled
    points and the positive set (``None`` when that input was not supplied);
    ``loss_hist`` / ``ap_hist`` are the per-epoch training metrics.
    """

    pred_x: np.ndarray | None
    pred_P: np.ndarray | None
    loss_hist: list[float]
    ap_hist: list[float]


class Backend(Protocol):
    """A pluggable VAE / PU-classifier compute backend."""

    name: str

    def train_clust_vae(
        self,
        x_mat: np.ndarray,
        X_train: np.ndarray,
        X_test: np.ndarray,
        clust_train_oh: np.ndarray,
        clust_test_oh: np.ndarray,
        *,
        enc_sze: int,
        num_clust: int,
        lr: float,
        clust_weight: float,
        rate: float,
        patience: int,
        max_epochs: int,
        seeds: np.ndarray,
        verbose: int = 0,
    ) -> np.ndarray:
        """Train the cluster-supervised VAE; return the latent encoding of ``x_mat``."""
        ...

    def train_pu_fold(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        x_predict: np.ndarray | None,
        P: np.ndarray | None,
        *,
        cls_eps: int,
        num_layers: int,
        pu_lr: float,
        seeds: np.ndarray,
    ) -> PuFoldResult:
        """Train one PU bagging fold's classifier and score the held-out points."""
        ...
