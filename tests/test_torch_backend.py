"""Characterization tests pinning the extracted torch backend functions.

Phase 1 of docs/Roadmap.md lifts the VAE training loop out of ``vaeda.vaeda``
and the PU per-fold training out of ``vaeda.pu`` into
``vaeda.backends._torch.train``. These goldens were captured from the previous
inline implementations, so they fail if the extraction changes the numerics.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
from sklearn.model_selection import train_test_split


def _vae_inputs():
    """Build the small fixed VAE training inputs used to capture the golden."""
    rng = np.random.default_rng(0)
    n, ngens, n_clust = 40, 12, 3
    x_mat = rng.normal(size=(n, ngens)).astype(np.float64)
    clust = rng.integers(0, n_clust, size=n)
    X_train, X_test, clust_train, clust_test = train_test_split(
        x_mat, clust, test_size=0.1, random_state=12345
    )
    clust_train_oh = np.eye(n_clust)[clust_train.astype(int)]
    clust_test_oh = np.eye(n_clust)[clust_test.astype(int)]
    return x_mat, X_train, X_test, clust_train_oh, clust_test_oh, n_clust


class TestTrainClustVae:
    """backends._torch.train.train_clust_vae: extracted VAE training loop."""

    def test_reproduces_inline_encoding(self):
        """
        Given the small fixed inputs and seeds used by the old inline VAE loop
        When the extracted train_clust_vae is run
        Then it returns an encoding identical (to float tolerance) to the golden
             captured from the pre-refactor inline loop
        """
        from vaeda.backends._torch.train import train_clust_vae

        x_mat, X_train, X_test, ctrain_oh, ctest_oh, n_clust = _vae_inputs()
        seeds = np.arange(10) + 100

        encoding = train_clust_vae(
            x_mat,
            X_train,
            X_test,
            ctrain_oh,
            ctest_oh,
            enc_sze=5,
            num_clust=n_clust,
            lr=1e-3,
            clust_weight=20000,
            rate=-0.75,
            patience=5,
            max_epochs=25,
            seeds=seeds,
        )

        assert encoding.shape == (40, 5)
        assert float(encoding.sum()) == pytest.approx(-16.39426377415657, rel=1e-5)


class TestTrainPuFold:
    """backends._torch.train.train_pu_fold: extracted PU per-fold training."""

    def test_reproduces_inline_fold_predictions(self):
        """
        Given a single PU fold's fit/predict/positive matrices and the seeds
             used by the old inline PU loop
        When the extracted train_pu_fold is run
        Then its predictions on the unlabeled and positive points match the
             goldens captured from the pre-refactor inline loop
        """
        from vaeda.backends._torch.train import train_pu_fold

        rng = np.random.default_rng(0)
        ngens = 12
        # advance rng the same way the capture script did (x_mat then clust)
        rng.normal(size=(40, ngens))
        rng.integers(0, 3, size=40)
        U = rng.normal(size=(30, ngens)).astype(np.float64)
        P = rng.normal(size=(15, ngens)).astype(np.float64)

        fit_idx = np.arange(10)
        X = np.vstack([U[fit_idx, :], P])
        Y = np.concatenate([np.zeros(len(fit_idx)), np.ones(P.shape[0])])
        x_predict = U[10:20]
        seeds = np.arange(10) + 100

        result = train_pu_fold(
            X,
            Y,
            x_predict,
            P,
            cls_eps=8,
            num_layers=1,
            pu_lr=1e-3,
            seeds=seeds,
        )

        assert result.pred_x.shape == (10,)
        assert result.pred_P.shape == (15,)
        assert float(result.pred_x.sum()) == pytest.approx(3.99595744907856, rel=1e-5)
        assert float(result.pred_P.sum()) == pytest.approx(7.703938618302345, rel=1e-5)
        assert len(result.loss_hist) == 8
        assert len(result.ap_hist) == 8
