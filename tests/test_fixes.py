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


class TestClusterHeadIsCategorical:
    """vae.ClustVAE: cluster head uses softmax categorical cross-entropy."""

    def test_classifier_emits_logits_not_independent_probabilities(self):
        """
        Given the cluster classifier head
        When it scores a batch of latent vectors
        Then it emits raw logits (softmax sums to 1, values may be negative),
             not independent per-class sigmoid probabilities in (0, 1)
        """
        from vaeda.vae import ClustClassifier

        torch.manual_seed(0)
        head = ClustClassifier(n_latent=4, n_clusters=3)
        out = head(torch.randn(64, 4))

        softmax_sums = torch.softmax(out, dim=1).sum(dim=1)
        assert torch.allclose(softmax_sums, torch.ones(64), atol=1e-5)
        # sigmoid output would be strictly in (0, 1); logits go negative
        assert (out < 0).any()

    def test_loss_rewards_correct_cluster_over_wrong_one(self):
        """
        Given one-hot cluster targets and the VAE loss
        When the classifier confidently predicts the correct vs wrong cluster
        Then the correct prediction yields a strictly lower loss
             (categorical CE accepts logits that BCE-on-sigmoid could not)
        """
        from vaeda.vae import ClustVAE

        vae = ClustVAE(n_input=4, n_latent=2, n_clusters=3)
        x = torch.randn(5, 4)
        recon_mu = x.clone()
        recon_logvar = torch.zeros_like(x)
        mu = torch.zeros(5, 2)
        logvar = torch.zeros(5, 2)
        target = torch.eye(3)[torch.tensor([0, 1, 2, 0, 1])]

        good = vae.loss(x, recon_mu, recon_logvar, mu, logvar, target * 10.0, target)[0]
        bad = vae.loss(x, recon_mu, recon_logvar, mu, logvar, (1 - target) * 10.0, target)[0]

        assert good.item() < bad.item()


class TestTopVariableGenes:
    """vaeda._top_variable_genes: rank genes by log-scaled variance."""

    def test_high_fold_change_gene_beats_high_count_gene(self):
        """
        Given a high-count gene with tiny fold change (high raw variance) and a
        low-count gene with large fold change (high log variance)
        When the single most variable gene is selected
        Then the high-fold-change gene wins (log-scaled, not raw, variance)
        """
        from vaeda.vaeda import _top_variable_genes

        # gene 0: huge counts, ~flat -> high raw var, ~zero log var
        # gene 1: low counts, big fold change -> high log var
        x = np.array(
            [[10000.0, 1.0], [10100.0, 4.0], [10200.0, 16.0]],
            dtype=np.float64,
        )
        idx = _top_variable_genes(x, num_hvgs=1)
        assert list(idx) == [1]


class TestAvoidSelfPairs:
    """mk_doublets._avoid_self_pairs: no cell is paired with itself."""

    def test_self_pair_is_broken(self):
        """
        Given parent index arrays where some positions point a cell at itself
        When self-pairs are repaired
        Then no position has ind1 == ind2
        """
        from vaeda.mk_doublets import _avoid_self_pairs

        ind1 = np.array([0, 1, 2, 3])
        ind2 = np.array([0, 2, 2, 3])  # positions 0 and 3 are self-pairs
        out2 = _avoid_self_pairs(ind1, ind2, n=4)

        assert not np.any(ind1 == out2)
