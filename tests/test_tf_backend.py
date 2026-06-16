"""Functional contract tests for the TensorFlow backend (Phase 4 of Roadmap).

These run only where tensorflow is installed (the ``vaeda[tensorflow]`` extra),
so they skip in the default torch-only environment. They assert the seam works
end to end; exact parity with upstream is covered separately (Phase 5).
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

pytest.importorskip("tensorflow", reason="tensorflow not installed (vaeda[tensorflow])")


@pytest.fixture(scope="module")
def tf_backend():
    """The TensorFlow backend, loaded via the resolver."""
    os.environ["VAEDA_BACKEND"] = "tensorflow"
    from vaeda.backends import get_backend

    get_backend.cache_clear()
    backend = get_backend()
    assert backend.name == "tensorflow"
    return backend


class TestTfTrainClustVae:
    """_tf.train.train_clust_vae: VAE encoding via TFP/Keras fit."""

    def test_returns_finite_encoding_of_expected_shape(self, tf_backend):
        """
        Given small fixed cluster-labelled inputs
        When the TF VAE is trained and the full matrix is encoded
        Then a finite encoding of shape (n_cells, enc_sze) is returned
        """
        rng = np.random.default_rng(0)
        n, ngens, n_clust = 60, 12, 3
        x_mat = rng.normal(size=(n, ngens)).astype(np.float32)
        clust = rng.integers(0, n_clust, size=n)
        X_train, X_test, c_train, c_test = train_test_split(
            x_mat, clust, test_size=0.1, random_state=12345
        )
        c_train_oh = np.eye(n_clust)[c_train.astype(int)].astype(np.float32)
        c_test_oh = np.eye(n_clust)[c_test.astype(int)].astype(np.float32)
        seeds = np.arange(13) + 100

        encoding = tf_backend.train_clust_vae(
            x_mat,
            X_train,
            X_test,
            c_train_oh,
            c_test_oh,
            enc_sze=5,
            num_clust=n_clust,
            lr=1e-3,
            clust_weight=20000,
            rate=-0.75,
            patience=5,
            max_epochs=8,
            seeds=seeds,
        )

        assert encoding.shape == (n, 5)
        assert np.isfinite(encoding).all()


class TestTfTrainPuFold:
    """_tf.train.train_pu_fold: one PU fold via Keras fit."""

    def test_predictions_are_probabilities_with_full_history(self, tf_backend):
        """
        Given a single PU fold's fit/predict/positive matrices
        When the TF classifier is trained and scores the held-out points
        Then predictions are probabilities in [0, 1] and the loss history has
             one entry per epoch
        """
        rng = np.random.default_rng(1)
        ngens, cls_eps = 12, 6
        U = rng.normal(size=(30, ngens)).astype(np.float32)
        P = rng.normal(size=(15, ngens)).astype(np.float32)
        X = np.vstack([U[:10], P])
        Y = np.concatenate([np.zeros(10), np.ones(15)])
        seeds = np.arange(13) + 100

        result = tf_backend.train_pu_fold(
            X, Y, U[10:20], P, cls_eps=cls_eps, num_layers=1, pu_lr=1e-3, seeds=seeds
        )

        assert result.pred_x.shape == (10,)
        assert result.pred_P.shape == (15,)
        assert (result.pred_x >= 0).all() and (result.pred_x <= 1).all()
        assert len(result.loss_hist) == cls_eps
