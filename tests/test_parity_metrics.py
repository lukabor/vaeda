"""Tests for the parity comparison metrics (Phase 5 of docs/Roadmap.md).

Pure functions over score / call vectors, so they are exercised here with
small hand-checkable inputs. The slow cross-backend parity test
(tests/test_parity.py) builds on these.
"""

import numpy as np
import pytest


class TestCallsFromScores:
    """parity_metrics.calls_from_scores: flag the top-rate fraction as doublets."""

    def test_top_fraction_is_flagged(self):
        """
        Given ten scores and a doublet rate of 0.2
        When calls are derived
        Then the two highest-scoring points are flagged and the rest are not
        """
        from parity_metrics import calls_from_scores

        scores = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.4, 0.5, 0.6, 0.7, 0.05])
        calls = calls_from_scores(scores, rate=0.2)

        assert calls.dtype == bool
        assert calls.sum() == 2
        assert calls[1] and calls[3]  # the 0.9 and 0.8 points


class TestJaccard:
    """parity_metrics.jaccard: overlap of the doublet-positive sets."""

    def test_identical_calls_score_one(self):
        """
        Given two identical boolean call vectors with at least one doublet
        When the Jaccard index is computed
        Then it is 1.0
        """
        from parity_metrics import jaccard

        a = np.array([True, False, True, False])
        assert jaccard(a, a) == pytest.approx(1.0)

    def test_disjoint_calls_score_zero(self):
        """
        Given two call vectors whose doublet sets do not overlap
        When the Jaccard index is computed
        Then it is 0.0
        """
        from parity_metrics import jaccard

        a = np.array([True, True, False, False])
        b = np.array([False, False, True, True])
        assert jaccard(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        """
        Given call sets {0,1} and {1,2}
        When the Jaccard index is computed
        Then it is intersection 1 over union 3
        """
        from parity_metrics import jaccard

        a = np.array([True, True, False])
        b = np.array([False, True, True])
        assert jaccard(a, b) == pytest.approx(1 / 3)


class TestScoreAgreement:
    """parity_metrics.pearson/spearman/rmse: score-level agreement."""

    def test_perfect_linear_correlation(self):
        """
        Given y = 2x + 1
        When Pearson and Spearman are computed
        Then both are 1.0
        """
        from parity_metrics import pearson, spearman

        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        y = 2 * x + 1
        assert pearson(x, y) == pytest.approx(1.0)
        assert spearman(x, y) == pytest.approx(1.0)

    def test_monotonic_nonlinear_is_spearman_one_pearson_less(self):
        """
        Given a monotonic but non-linear relation
        When correlations are computed
        Then Spearman is 1.0 while Pearson is strictly below 1.0
        """
        from parity_metrics import pearson, spearman

        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = x**3
        assert spearman(x, y) == pytest.approx(1.0)
        assert pearson(x, y) < 1.0

    def test_rmse_zero_for_identical_and_positive_otherwise(self):
        """
        Given identical vectors, then vectors differing by a constant
        When RMSE is computed
        Then it is 0.0 for identical inputs and equals the constant offset
        """
        from parity_metrics import rmse

        a = np.array([0.1, 0.5, 0.9])
        assert rmse(a, a) == pytest.approx(0.0)
        assert rmse(a, a + 0.2) == pytest.approx(0.2)


class TestAri:
    """parity_metrics.ari: adjusted Rand index between two call labelings."""

    def test_identical_labelings_score_one(self):
        """
        Given two identical call labelings
        When the ARI is computed
        Then it is 1.0
        """
        from parity_metrics import ari

        a = np.array([True, False, True, False, True])
        assert ari(a, a) == pytest.approx(1.0)


class TestCompare:
    """parity_metrics.compare: bundle every metric for two score vectors."""

    def test_returns_all_expected_metrics(self):
        """
        Given two score vectors and a doublet rate
        When compare() is called
        Then the result holds every parity metric plus per-vector min/max
        """
        from parity_metrics import compare

        rng = np.random.default_rng(0)
        a = rng.random(50)
        b = a + rng.normal(0, 0.01, 50)

        m = compare(a, b, rate=0.1)

        for key in (
            "jaccard",
            "ari",
            "pearson",
            "spearman",
            "rmse",
            "min_a",
            "max_a",
            "min_b",
            "max_b",
        ):
            assert key in m
        assert m["pearson"] > 0.9  # a and b are nearly identical
