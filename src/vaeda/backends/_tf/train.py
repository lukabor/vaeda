"""TensorFlow training loops for the VAE and PU classifier.

Same numpy-in/numpy-out seam as the torch backend (see
``vaeda.backends.base.Backend``), implemented with Keras ``fit`` and the
TFP/Keras models. Ported from the upstream/kostkalab v0.1.x training flow so
the doublet scores reproduce the original vaeda results.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
import tf_keras as tfk

from ..base import PuFoldResult
from .classifier import define_classifier
from .vae import define_clust_vae


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

    Matches the v0.1.x flow: ``model.fit`` with a val-loss EarlyStopping
    callback (``restore_best_weights=False``, as upstream) and an exponential
    LR decay after epoch 3, then a single forward pass of the encoder.
    """
    tf.random.set_seed(seeds[6])
    vae = define_clust_vae(
        enc_sze, x_mat.shape[1], num_clust, LR=lr, clust_weight=clust_weight
    )

    early_stopping = tfk.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        min_delta=0,
        patience=patience,
        verbose=0,
        restore_best_weights=False,
    )

    def scheduler(epoch: int, current_lr: float) -> float:
        if epoch < 3:
            return current_lr
        return current_lr * tf.math.exp(rate)

    lr_schedule = tfk.callbacks.LearningRateScheduler(scheduler)

    vae.fit(
        x=[X_train],
        y=[X_train, clust_train_oh],
        validation_data=([X_test], [X_test, clust_test_oh]),
        epochs=max_epochs,
        callbacks=[early_stopping, lr_schedule],
        verbose=verbose,
    )

    encoder = vae.get_layer("encoder")
    tf.random.set_seed(seeds[7])
    return np.array(tf.convert_to_tensor(encoder(x_mat)))


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
    """Train one PU bagging fold's classifier and score the held-out points."""
    np.random.seed(1)
    tf.random.set_seed(seeds[1])
    classifier = define_classifier(X.shape[1], num_layers=num_layers)

    # Shuffle indices (consumes the seeded RNG even in the unshuffled branch,
    # matching v0.1.x)
    ind = np.arange(X.shape[0])
    np.random.seed(seeds[2])
    np.random.shuffle(ind)

    auc = tfk.metrics.AUC(curve="PR", name="auc")
    classifier.compile(
        optimizer=tfk.optimizers.Adam(learning_rate=pu_lr),
        loss="binary_crossentropy",
        metrics=[auc],
    )

    if (X.shape[0] * 0.1) >= 50:
        tf.random.set_seed(seeds[3])
        hist = classifier.fit(x=X, y=Y, epochs=cls_eps, verbose=0)
    else:
        ind = np.arange(X.shape[0])
        np.random.seed(seeds[2])
        np.random.shuffle(ind)
        tf.random.set_seed(seeds[3])
        hist = classifier.fit(x=X[ind, :], y=Y[ind], epochs=cls_eps, verbose=0)

    loss_hist = list(hist.history["loss"])
    ap_hist = list(hist.history.get("auc", [0.0] * cls_eps))

    pred_x: np.ndarray | None = None
    pred_P: np.ndarray | None = None
    if x_predict is not None:
        tf.random.set_seed(seeds[3])
        pred_x = np.array(classifier(x_predict)).flatten()
    if P is not None:
        tf.random.set_seed(seeds[3])
        pred_P = np.array(classifier(P)).flatten()

    return PuFoldResult(
        pred_x=pred_x,
        pred_P=pred_P,
        loss_hist=loss_hist,
        ap_hist=ap_hist,
    )
