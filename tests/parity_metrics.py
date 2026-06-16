"""Parity comparison metrics for vaeda doublet scores and calls.

Pure numpy/scipy/sklearn helpers used by the cross-backend parity test
(tests/test_parity.py) to compare the torch backend, the TensorFlow backend,
and the frozen legacy fixture. Kept under tests/ because they are validation
tooling, not part of the installed package.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import adjusted_rand_score


def calls_from_scores(scores: np.ndarray, rate: float) -> np.ndarray:
    """Flag the top ``rate`` fraction of ``scores`` as doublets (boolean array).

    A common rule applied identically to every workflow's scores so the call
    sets are comparable, independent of each backend's internal thresholding.
    The cutoff is the ``1 - rate`` quantile; ties at the cutoff are included.
    """
    scores = np.asarray(scores, dtype=float)
    cutoff = np.quantile(scores, 1.0 - rate)
    return scores >= cutoff


def jaccard(calls_a: np.ndarray, calls_b: np.ndarray) -> float:
    """Jaccard index of two doublet-positive sets (1.0 if both are empty)."""
    a = np.asarray(calls_a, dtype=bool)
    b = np.asarray(calls_b, dtype=bool)
    union = np.count_nonzero(a | b)
    if union == 0:
        return 1.0
    return float(np.count_nonzero(a & b) / union)


def ari(calls_a: np.ndarray, calls_b: np.ndarray) -> float:
    """Adjusted Rand index between two binary call labelings."""
    return float(
        adjusted_rand_score(
            np.asarray(calls_a, dtype=int), np.asarray(calls_b, dtype=int)
        )
    )


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation coefficient between two score vectors."""
    return float(pearsonr(np.asarray(a, dtype=float), np.asarray(b, dtype=float))[0])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation coefficient between two score vectors."""
    return float(spearmanr(np.asarray(a, dtype=float), np.asarray(b, dtype=float))[0])


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Root-mean-square error between two score vectors."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def compare(scores_a: np.ndarray, scores_b: np.ndarray, *, rate: float) -> dict[str, float]:
    """Compute every parity metric between two score vectors.

    Doublet calls are derived from each vector via :func:`calls_from_scores`
    with the shared ``rate``; the returned dict also carries each vector's
    min/max for sanity checks.
    """
    calls_a = calls_from_scores(scores_a, rate)
    calls_b = calls_from_scores(scores_b, rate)
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    return {
        "jaccard": jaccard(calls_a, calls_b),
        "ari": ari(calls_a, calls_b),
        "pearson": pearson(a, b),
        "spearman": spearman(a, b),
        "rmse": rmse(a, b),
        "min_a": float(a.min()),
        "max_a": float(a.max()),
        "min_b": float(b.min()),
        "max_b": float(b.max()),
    }
