"""Tests for the seed-stability metrics (Phase 7 of docs/Roadmap.md).

Pure functions over (n_runs, n_cells) call / score matrices, exercised here
with small hand-checkable inputs. The slow per-env stability runs build the
real 50-seed matrices and feed them through these same helpers.
"""

import numpy as np
import pytest


class TestProducerStringCalls:
    """Call metrics must read vaeda's native "doublet"/"singlet" string calls.

    vaeda.py writes adata.obs["vaeda_calls"] as a string array, not bool. A raw
    astype(bool) would flag every non-empty string as a doublet, so the call
    metrics must map strings through the doublet label.
    """

    def test_call_frequency_reads_doublet_label(self):
        """
        Given a calls matrix of "doublet"/"singlet" strings (as vaeda emits)
        When call_frequency is computed
        Then only the "doublet" entries count toward the per-cell frequency
        """
        from stability_metrics import call_frequency

        calls = np.array(
            [
                ["doublet", "singlet", "doublet"],
                ["singlet", "singlet", "doublet"],
            ]
        )
        np.testing.assert_allclose(call_frequency(calls), [0.5, 0.0, 1.0])

    def test_derive_threshold_reads_doublet_label(self):
        """
        Given string calls alongside scores
        When derive_threshold is computed
        Then the "doublet"/"singlet" split brackets the cutoff correctly
        """
        from stability_metrics import derive_threshold

        scores = np.array([0.10, 0.20, 0.80, 0.90])
        calls = np.array(["singlet", "singlet", "doublet", "doublet"])
        assert derive_threshold(scores, calls) == pytest.approx(0.50)

    def test_unexpected_string_labels_raise(self):
        """
        Given a string calls matrix with an unexpected label
        When a call metric is computed
        Then it raises rather than silently miscounting
        """
        from stability_metrics import call_frequency

        calls = np.array([["doublet", "maybe"], ["singlet", "singlet"]])
        with pytest.raises(ValueError, match="unexpected call labels"):
            call_frequency(calls)


class TestCallFrequency:
    """stability_metrics.call_frequency: per-cell fraction of runs calling doublet."""

    def test_fraction_of_runs_per_cell(self):
        """
        Given a 4-run by 3-cell boolean calls matrix
        When call_frequency is computed
        Then each cell carries the fraction of runs that flagged it a doublet
        """
        from stability_metrics import call_frequency

        calls = np.array(
            [
                [True, False, True],
                [True, False, False],
                [False, False, True],
                [True, False, False],
            ]
        )
        freq = call_frequency(calls)

        assert freq.shape == (3,)
        np.testing.assert_allclose(freq, [3 / 4, 0.0, 2 / 4])


class TestFlipRate:
    """stability_metrics.flip_rate: fraction of cells that ever disagree across runs."""

    def test_only_unstable_cells_counted(self):
        """
        Given a calls matrix where some cells are pinned (all 0 or all 1)
            and others flip across runs
        When flip_rate is computed
        Then it returns the fraction of cells with 0 < frequency < 1
        """
        from stability_metrics import flip_rate

        calls = np.array(
            [
                [True, False, True, False],
                [True, False, False, True],
                [True, False, True, False],
            ]
        )
        # cell0 always doublet, cell1 never -> stable; cell2, cell3 flip.
        assert flip_rate(calls) == pytest.approx(2 / 4)


class TestFleissKappa:
    """stability_metrics.fleiss_kappa: chance-corrected agreement over runs."""

    def test_unanimous_runs_score_one(self):
        """
        Given runs that are identical to each other (perfect agreement)
            but with cells split across both categories
        When fleiss_kappa is computed
        Then it is 1.0
        """
        from stability_metrics import fleiss_kappa

        row = [True, False, True, False]
        calls = np.array([row, row, row])
        assert fleiss_kappa(calls) == pytest.approx(1.0)

    def test_partial_agreement_below_one(self):
        """
        Given runs that disagree on some cells
        When fleiss_kappa is computed
        Then it matches the hand-computed value (below 1.0)
        """
        from stability_metrics import fleiss_kappa

        calls = np.array(
            [
                [True, False, True, False],
                [True, True, False, False],
                [False, False, True, True],
            ]
        )
        # P_bar = 1/3, P_e = 1/2 -> (1/3 - 1/2)/(1 - 1/2) = -1/3.
        assert fleiss_kappa(calls) == pytest.approx(-1 / 3)


class TestMeanPairwise:
    """stability_metrics.mean_pairwise_*: agreement averaged over run pairs."""

    def test_identical_runs_give_mean_one_sd_zero(self):
        """
        Given three identical runs
        When mean_pairwise_jaccard / mean_pairwise_ari are computed
        Then mean is 1.0 and sd is 0.0 for both
        """
        from stability_metrics import mean_pairwise_ari, mean_pairwise_jaccard

        row = [True, False, True, False, True]
        calls = np.array([row, row, row])

        for metric in (mean_pairwise_jaccard, mean_pairwise_ari):
            mean, sd = metric(calls)
            assert mean == pytest.approx(1.0)
            assert sd == pytest.approx(0.0)

    def test_jaccard_averages_over_all_pairs(self):
        """
        Given three runs with pairwise Jaccard 1/2, 1/2 and 1/3
        When mean_pairwise_jaccard is computed
        Then mean is the average of the three unordered-pair values
        """
        from stability_metrics import mean_pairwise_jaccard

        calls = np.array(
            [
                [True, True, False],   # doublets {0,1}
                [True, False, True],   # doublets {0,2}
                [True, True, True],    # doublets {0,1,2}
            ]
        )
        # J(0,1)=|{0}|/|{0,1,2}|=1/3; J(0,2)={0,1}/{0,1,2}=2/3; J(1,2)={0,2}/{0,1,2}=2/3.
        mean, sd = mean_pairwise_jaccard(calls)
        assert mean == pytest.approx((1 / 3 + 2 / 3 + 2 / 3) / 3)
        assert sd == pytest.approx(np.std([1 / 3, 2 / 3, 2 / 3]))


class TestDoubletCountCv:
    """stability_metrics.doublet_count_cv: spread of per-run doublet counts."""

    def test_equal_counts_give_zero_cv(self):
        """
        Given runs that each call the same number of doublets
        When doublet_count_cv is computed
        Then the coefficient of variation is 0.0
        """
        from stability_metrics import doublet_count_cv

        calls = np.array(
            [
                [True, True, False, False],
                [True, False, True, False],
                [False, True, False, True],
            ]
        )  # 2 doublets per run
        assert doublet_count_cv(calls) == pytest.approx(0.0)

    def test_cv_is_sd_over_mean(self):
        """
        Given per-run doublet counts of 1, 2 and 3
        When doublet_count_cv is computed
        Then it is sd / mean of those counts
        """
        from stability_metrics import doublet_count_cv

        calls = np.array(
            [
                [True, False, False],   # 1
                [True, True, False],    # 2
                [True, True, True],     # 3
            ]
        )
        counts = [1, 2, 3]
        assert doublet_count_cv(calls) == pytest.approx(np.std(counts) / np.mean(counts))


class TestPerCellScoreSd:
    """stability_metrics.per_cell_score_sd: per-cell score spread across runs."""

    def test_sd_down_the_runs_axis(self):
        """
        Given a 3-run by 2-cell scores matrix
        When per_cell_score_sd is computed
        Then each cell gets the sd of its scores across the three runs
        """
        from stability_metrics import per_cell_score_sd

        scores = np.array(
            [
                [0.1, 0.5],
                [0.2, 0.5],
                [0.3, 0.5],
            ]
        )
        sd = per_cell_score_sd(scores)

        assert sd.shape == (2,)
        np.testing.assert_allclose(sd, [np.std([0.1, 0.2, 0.3]), 0.0])


class TestIcc:
    """stability_metrics.icc: one-way ICC of scores, cells as targets."""

    def test_constant_per_cell_scores_give_icc_one(self):
        """
        Given scores that are identical across runs but differ between cells
            (all variance is between cells, none within)
        When icc is computed
        Then it is 1.0
        """
        from stability_metrics import icc

        scores = np.array([[0.1, 0.9], [0.1, 0.9]])
        assert icc(scores) == pytest.approx(1.0)

    def test_indistinguishable_cells_give_nonpositive_icc(self):
        """
        Given cells with equal means but all variance within-cell across runs
        When icc is computed
        Then it collapses to its negative floor (-1.0 here), well below 1.0
        """
        from stability_metrics import icc

        scores = np.array([[0.1, 0.9], [0.9, 0.1]])
        # MSB = 0, MSW = 0.32, k = 2 -> (0 - MSW)/(MSW) = -1.0.
        assert icc(scores) == pytest.approx(-1.0)


class TestDeriveThreshold:
    """stability_metrics.derive_threshold: recover a seed's cutoff from outputs."""

    def test_midpoint_of_the_separating_bracket(self):
        """
        Given per-cell scores and the boolean calls a backend made
            (calls == score > t for some hidden t)
        When derive_threshold is computed
        Then it returns the midpoint of (max singlet score, min doublet score)
        """
        from stability_metrics import derive_threshold

        scores = np.array([0.10, 0.20, 0.80, 0.90])
        calls = np.array([False, False, True, True])
        # max singlet = 0.20, min doublet = 0.80 -> midpoint 0.50.
        assert derive_threshold(scores, calls) == pytest.approx(0.50)

    def test_returns_nan_when_a_class_is_empty(self):
        """
        Given calls that flag every cell the same way (no separating bracket)
        When derive_threshold is computed
        Then it returns nan
        """
        from stability_metrics import derive_threshold

        scores = np.array([0.1, 0.2, 0.3])
        assert np.isnan(derive_threshold(scores, np.array([False, False, False])))
        assert np.isnan(derive_threshold(scores, np.array([True, True, True])))
