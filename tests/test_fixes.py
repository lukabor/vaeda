"""Regression unit tests for correctness fixes (no network required).

Covers three bugs found in review:
1. match-knee clobbering the detected knee to 250 in the common case.
2. preds_on_P normalised by the wrong divisor (scale mismatch vs preds).
3. BatchNorm1d crashing on a trailing size-1 minibatch.
"""

import numpy as np
import pytest
import torch


class TestClampKnee:
    """vaeda._clamp_knee: adjust the detected elbow epoch count."""

    def test_normal_range_is_preserved(self):
        """
        Given a detected knee inside [20, 250] and a large sample (num >= 500)
        When the knee is clamped
        Then the detected value is returned unchanged (not forced to 250)
        """
        from vaeda.vaeda import _clamp_knee

        assert _clamp_knee(120, num=600) == 120

    def test_small_sample_adds_one(self):
        """
        Given a small sample (num < 500)
        When the knee is clamped
        Then one epoch is added before bounding
        """
        from vaeda.vaeda import _clamp_knee

        assert _clamp_knee(120, num=300) == 121

    def test_below_floor_clamps_up(self):
        """
        Given a detected knee below 20
        When the knee is clamped
        Then it is raised to the floor of 20
        """
        from vaeda.vaeda import _clamp_knee

        assert _clamp_knee(5, num=600) == 20

    def test_above_ceiling_clamps_down(self):
        """
        Given a detected knee above 250
        When the knee is clamped
        Then it is lowered to the ceiling of 250
        """
        from vaeda.vaeda import _clamp_knee

        assert _clamp_knee(300, num=600) == 250


class TestNormalizePuPreds:
    """pu._normalize_pu_preds: average accumulated PU bagging scores."""

    def test_p_scores_divided_by_fold_count(self):
        """
        Given P scored on every fold (i additions) and U scored once per repeat
        When the accumulated sums are normalised
        Then P divides by the fold count i and U by N*(k-1), giving equal means
        """
        from vaeda.pu import _normalize_pu_preds

        k, n_repeats = 2, 2
        i = n_repeats * k  # 4 folds total
        # Every P point predicted 0.5 on all i folds -> sum 2.0, mean 0.5
        preds_on_p_sum = np.array([0.5 * i])
        # Every U point predicted 0.5 on (k-1) folds per repeat -> sum 1.0
        preds_sum = np.array([0.5 * n_repeats * (k - 1)])

        preds, preds_on_p = _normalize_pu_preds(preds_sum, preds_on_p_sum, i, k)

        assert preds_on_p[0] == pytest.approx(0.5)
        assert preds[0] == pytest.approx(0.5)


class TestBatchSlices:
    """pu._batch_slices: minibatch bounds safe for BatchNorm."""

    def test_clean_split_has_no_singleton(self):
        """
        Given n evenly divisible by the batch size
        When slices are computed
        Then each batch is full and none has length 1
        """
        from vaeda.pu import _batch_slices

        assert _batch_slices(64, 32) == [(0, 32), (32, 64)]

    def test_trailing_singleton_is_merged(self):
        """
        Given n leaving a trailing batch of length 1
        When slices are computed
        Then the singleton is merged into the previous batch
        """
        from vaeda.pu import _batch_slices

        slices = _batch_slices(33, 32)
        assert all(end - start > 1 for start, end in slices)
        # full coverage of [0, n)
        assert slices[0][0] == 0
        assert slices[-1][1] == 33

    def test_train_one_epoch_survives_singleton_tail(self):
        """
        Given training data whose size leaves a size-1 final batch (n % 32 == 1)
        When a single epoch is trained
        Then BatchNorm does not raise on the singleton batch
        """
        from vaeda.classifier import define_classifier
        from vaeda.pu import _train_one_epoch

        torch.manual_seed(0)
        n = 33  # 33 % 32 == 1
        X = torch.randn(n, 4)
        Y = (torch.rand(n) > 0.5).float()
        model = define_classifier(ngens=4)
        optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

        loss, auc = _train_one_epoch(model, optimiser, X, Y)

        assert np.isfinite(loss)
