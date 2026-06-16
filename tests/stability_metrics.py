"""Seed-stability metrics for vaeda doublet scores and calls (Phase 7).

Pure numpy/scipy helpers that quantify how much a single backend's doublet
calls / scores move when only the RNG seed changes (same data, same backend)
across N runs. The companion to tests/parity_metrics.py (cross-backend), kept
under tests/ because it is validation tooling, not part of the package.

Every function takes a matrix shaped ``(n_runs, n_cells)`` — one row per seed.
"""

from __future__ import annotations

import itertools

import numpy as np
from parity_metrics import ari, jaccard


def _as_doublet_bool(calls: np.ndarray) -> np.ndarray:
    """Normalise a calls array to a boolean doublet mask.

    vaeda emits ``adata.obs["vaeda_calls"]`` as the strings ``"doublet"`` /
    ``"singlet"``; a bare ``astype(bool)`` on those would flag every non-empty
    string as a doublet. This maps the string form by label and passes through
    numeric / boolean arrays unchanged.

    Args:
        calls: Array of ``"doublet"``/``"singlet"`` strings, or any numeric /
            boolean array already encoding doublet as truthy.

    Returns:
        Boolean array, ``True`` where the cell is a doublet.

    Raises:
        ValueError: If a string array carries labels other than the two
            expected ones.
    """
    calls = np.asarray(calls)
    if calls.dtype.kind in ("U", "S", "O"):
        as_str = calls.astype(str)
        allowed = {"doublet", "singlet"}
        seen = set(np.unique(as_str).tolist())
        if not seen <= allowed:
            raise ValueError(f"unexpected call labels {sorted(seen - allowed)}; expected {allowed}")
        return as_str == "doublet"
    return calls.astype(bool)


def _mean_pairwise(matrix: np.ndarray, metric) -> tuple[float, float]:
    """Apply ``metric`` to every unordered pair of rows; return (mean, sd).

    Args:
        matrix: Matrix shaped ``(n_runs, n_cells)``; rows are compared pairwise.
        metric: A two-row comparison returning a float (e.g. ``jaccard``).

    Returns:
        Population mean and standard deviation of the pairwise values. A single
        row (no pairs) yields ``(nan, nan)``.
    """
    rows = np.asarray(matrix)
    values = [metric(rows[i], rows[j]) for i, j in itertools.combinations(range(len(rows)), 2)]
    if not values:
        return float("nan"), float("nan")
    return float(np.mean(values)), float(np.std(values))


def call_frequency(calls: np.ndarray) -> np.ndarray:
    """Per-cell fraction of runs that flagged the cell a doublet.

    Args:
        calls: Boolean matrix shaped ``(n_runs, n_cells)``.

    Returns:
        Float array shaped ``(n_cells,)`` with each cell's doublet frequency
        ``f_i`` in ``[0, 1]``.
    """
    return _as_doublet_bool(calls).mean(axis=0)


def flip_rate(calls: np.ndarray) -> float:
    """Fraction of cells whose call is not unanimous across runs.

    A cell is *stable* when every run agrees (frequency 0 or 1) and *flipping*
    otherwise. The flip rate is the share of flipping cells — a direct read of
    how many calls are seed-sensitive.

    Args:
        calls: Boolean matrix shaped ``(n_runs, n_cells)``.

    Returns:
        Fraction of cells with ``0 < f_i < 1``.
    """
    freq = call_frequency(calls)
    return float(np.mean((freq > 0.0) & (freq < 1.0)))


def fleiss_kappa(calls: np.ndarray) -> float:
    """Fleiss' kappa over the binary doublet verdicts, treating runs as raters.

    Chance-corrected agreement across ``n_runs`` raters who each assign every
    cell to one of two categories (doublet / singlet). ``1.0`` is perfect
    agreement; ``0.0`` is chance-level.

    Args:
        calls: Boolean matrix shaped ``(n_runs, n_cells)``.

    Returns:
        Fleiss' kappa. When agreement is perfect and the expected agreement is
        also degenerate (all cells in one category), returns ``1.0``.
    """
    calls = _as_doublet_bool(calls)
    n_runs = calls.shape[0]
    # Per-cell counts in each category: doublet then singlet, shape (n_cells, 2).
    n_doublet = calls.sum(axis=0)
    counts = np.column_stack([n_doublet, n_runs - n_doublet])

    # P_i: observed agreement per cell, averaged over cells.
    p_i = (np.square(counts).sum(axis=1) - n_runs) / (n_runs * (n_runs - 1))
    p_bar = float(p_i.mean())

    # P_e: agreement expected by chance from the category marginals.
    p_j = counts.mean(axis=0) / n_runs
    p_e = float(np.square(p_j).sum())

    if p_e >= 1.0:
        return 1.0
    return (p_bar - p_e) / (1.0 - p_e)


def mean_pairwise_jaccard(calls: np.ndarray) -> tuple[float, float]:
    """Mean ± sd of the Jaccard index over every pair of runs.

    Args:
        calls: Boolean matrix shaped ``(n_runs, n_cells)``.

    Returns:
        ``(mean, sd)`` of the doublet-set Jaccard index across run pairs.
    """
    return _mean_pairwise(_as_doublet_bool(calls), jaccard)


def mean_pairwise_ari(calls: np.ndarray) -> tuple[float, float]:
    """Mean ± sd of the adjusted Rand index over every pair of runs.

    Args:
        calls: Boolean matrix shaped ``(n_runs, n_cells)``.

    Returns:
        ``(mean, sd)`` of the ARI between run pairs' call labelings.
    """
    return _mean_pairwise(_as_doublet_bool(calls), ari)


def doublet_count_cv(calls: np.ndarray) -> float:
    """Coefficient of variation of the per-run doublet count.

    Each run calls some number of doublets; this is the spread of that count
    relative to its mean (``sd / mean``), a scale-free read of how much the
    total doublet load wobbles with the seed.

    Args:
        calls: Boolean matrix shaped ``(n_runs, n_cells)``.

    Returns:
        ``sd / mean`` of the per-run doublet counts, or ``0.0`` when the mean
        count is zero (no doublets in any run).
    """
    counts = _as_doublet_bool(calls).sum(axis=1)
    mean = float(counts.mean())
    if mean == 0.0:
        return 0.0
    return float(counts.std() / mean)


def per_cell_score_sd(scores: np.ndarray) -> np.ndarray:
    """Per-cell standard deviation of the doublet score across runs.

    Args:
        scores: Float matrix shaped ``(n_runs, n_cells)``.

    Returns:
        Float array shaped ``(n_cells,)`` — each cell's score sd over the runs.
    """
    return np.asarray(scores, dtype=float).std(axis=0)


def icc(scores: np.ndarray) -> float:
    """One-way random-effects ICC(1,1) of the scores, cells as targets.

    Treats each cell as a target measured once per run; ICC is the share of
    total score variance attributable to between-cell differences rather than
    run-to-run noise. ``1.0`` means runs reproduce each cell's score exactly;
    values toward / below ``0`` mean the seed dominates the signal.

    Args:
        scores: Float matrix shaped ``(n_runs, n_cells)``.

    Returns:
        ICC(1,1). Returns ``1.0`` in the degenerate all-constant case (no
        variance anywhere).
    """
    x = np.asarray(scores, dtype=float)
    n_runs, n_cells = x.shape  # k measurements per target, n targets
    grand = x.mean()
    cell_means = x.mean(axis=0)

    ss_between = n_runs * np.sum((cell_means - grand) ** 2)
    ss_within = np.sum((x - cell_means) ** 2)
    ms_between = ss_between / (n_cells - 1)
    ms_within = ss_within / (n_cells * (n_runs - 1))

    denom = ms_between + (n_runs - 1) * ms_within
    if denom == 0.0:
        return 1.0
    return float((ms_between - ms_within) / denom)


def derive_threshold(scores: np.ndarray, calls: np.ndarray) -> float:
    """Recover a run's doublet cutoff from its scores and boolean calls.

    A run calls cell ``i`` a doublet iff ``score_i > t`` for some hidden ``t``,
    so ``max(score | singlet) < t <= min(score | doublet)``. This returns the
    midpoint of that bracket — a single representative cutoff per run whose
    spread across seeds shows how much the threshold itself moves.

    Args:
        scores: Float per-cell scores shaped ``(n_cells,)``.
        calls: Boolean per-cell calls shaped ``(n_cells,)``.

    Returns:
        Bracket midpoint, or ``nan`` when either class is empty (no bracket).
    """
    scores = np.asarray(scores, dtype=float)
    calls = _as_doublet_bool(calls)
    if not calls.any() or calls.all():
        return float("nan")
    max_singlet = scores[~calls].max()
    min_doublet = scores[calls].min()
    return float((max_singlet + min_doublet) / 2.0)
