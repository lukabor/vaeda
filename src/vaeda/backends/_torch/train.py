"""PyTorch training loops for the VAE and PU classifier.

These functions take numpy arrays in and return numpy arrays out, so the
orchestration code in ``vaeda.vaeda`` and ``vaeda.pu`` stays free of torch.
They were lifted verbatim (same statement order, same seeding) from the
previous inline loops in those modules — see the characterization tests in
``tests/test_torch_backend.py``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from sklearn.metrics import average_precision_score

from ..base import PuFoldResult
from .classifier import define_classifier
from .vae import _EarlyStopper, _get_device, define_clust_vae


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
        try:
            ap_val = float(average_precision_score(all_targets, all_preds))
        except ValueError:
            ap_val = 0.0

    return total_loss / n_batches, ap_val


def train_clust_vae(
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
    """Train the cluster-supervised VAE and return the latent encoding of ``x_mat``.

    Mirrors the v0.1.x flow: minibatch training (Keras default batch size 32),
    an exponential LR decay after epoch 3, and early stopping that restores the
    best-validation-loss weights before the final encode.
    """
    torch.manual_seed(seeds[6])
    vae, optimiser = define_clust_vae(
        enc_sze,
        x_mat.shape[1],
        num_clust,
        LR=lr,
        clust_weight=clust_weight,
    )
    device = _get_device()

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    clust_train_t = torch.tensor(clust_train_oh, dtype=torch.float32, device=device)
    clust_test_t = torch.tensor(clust_test_oh, dtype=torch.float32, device=device)

    # Learning-rate scheduler (exponential decay after epoch 3)
    def lr_lambda(epoch: int) -> float:
        if epoch < 3:
            return 1.0
        return float(np.exp(rate))

    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimiser, lr_lambda=lr_lambda)

    # Early stopping (snapshots the best-validation-loss weights)
    stopper = _EarlyStopper(patience=patience)
    batch_size = 32  # Keras default

    for epoch in range(max_epochs):
        # Train step (minibatch, matching Keras default batch_size=32)
        vae.train()
        n_train = X_train_t.shape[0]
        # Shuffle training data each epoch
        perm = torch.randperm(n_train, device=device)

        for start, end in _batch_slices(n_train, batch_size):
            idx = perm[start:end]
            x_batch = X_train_t[idx]
            c_batch = clust_train_t[idx]

            optimiser.zero_grad()
            recon_mu, recon_logvar, clust_pred, _, enc_mu, enc_logvar = vae(x_batch)
            batch_loss, _, _ = vae.loss(
                x_batch,
                recon_mu,
                recon_logvar,
                enc_mu,
                enc_logvar,
                clust_pred,
                c_batch,
            )
            batch_loss.backward()
            optimiser.step()

        scheduler.step()

        # Validation step
        vae.eval()
        with torch.no_grad():
            (
                v_recon_mu,
                v_recon_logvar,
                v_clust_pred,
                _,
                v_enc_mu,
                v_enc_logvar,
            ) = vae(X_test_t)
            val_loss, _, _ = vae.loss(
                X_test_t,
                v_recon_mu,
                v_recon_logvar,
                v_enc_mu,
                v_enc_logvar,
                v_clust_pred,
                clust_test_t,
            )
            val_loss_val = val_loss.item()

        # Early stopping check (snapshots best weights internally)
        if stopper.step(val_loss_val, vae):
            if verbose != 0:
                logger.info(f"VAE early stopping at epoch {epoch}")
            break

    # Restore the best-validation-loss weights before encoding
    stopper.restore(vae)

    # Extract encodings
    vae.eval()
    x_mat_t = torch.tensor(x_mat, dtype=torch.float32, device=device)
    with torch.no_grad():
        torch.manual_seed(seeds[7])
        z, _, _ = vae.encoder(x_mat_t)
        encoding = z.detach().cpu().numpy()

    return encoding


def train_pu_fold(
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
    """Train one PU bagging fold's NN classifier and score the held-out points."""
    device = _get_device()

    torch.manual_seed(seeds[1])
    classifier = define_classifier(ngens=X.shape[1], num_layers=num_layers)
    optimiser = torch.optim.Adam(classifier.parameters(), lr=pu_lr)

    # Shuffle training data
    ind = np.arange(X.shape[0])
    rng2 = np.random.Generator(np.random.PCG64(seeds[2]))
    rng2.shuffle(ind)

    X_t = torch.tensor(X[ind, :], dtype=torch.float32, device=device)
    Y_t = torch.tensor(Y[ind], dtype=torch.float32, device=device)

    torch.manual_seed(seeds[3])
    loss_hist: list[float] = []
    ap_hist: list[float] = []
    for _ in range(cls_eps):
        loss_val, ap_val = _train_one_epoch(classifier, optimiser, X_t, Y_t)
        loss_hist.append(loss_val)
        ap_hist.append(ap_val)

    classifier.eval()
    pred_x: np.ndarray | None = None
    pred_P: np.ndarray | None = None
    with torch.no_grad():
        if x_predict is not None:
            x_t = torch.tensor(x_predict, dtype=torch.float32, device=device)
            torch.manual_seed(seeds[3])
            pred_x = classifier(x_t).cpu().numpy()
        if P is not None:
            P_t = torch.tensor(P, dtype=torch.float32, device=device)
            torch.manual_seed(seeds[3])
            pred_P = classifier(P_t).cpu().numpy()

    return PuFoldResult(
        pred_x=pred_x,
        pred_P=pred_P,
        loss_hist=loss_hist,
        ap_hist=ap_hist,
    )
