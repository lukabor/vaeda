"""Cross-backend parity test (Phase 5 of docs/Roadmap.md).

Compares the doublet scores of the three vaeda lineages on the same pbmc3k
cells (same vaeda seed): the legacy upstream TensorFlow implementation, this
repo's TensorFlow backend, and this repo's torch backend. Each pairwise
comparison must meet thresholds on doublet-call overlap (Jaccard), call
agreement (ARI), score agreement (Pearson/Spearman) and score error (RMSE).

The scores are read from committed fixtures rather than computed here, because
torch and TensorFlow cannot both run in one process (native-lib conflict) and
the legacy lineage needs python 3.8. Regenerate the fixtures with:

    # legacy  -> data/legacy_pbmc3k_scores.csv   (see docker/legacy/README.md)
    VAEDA_BACKEND=torch      python tools/score_pbmc3k.py tests/fixtures/torch_pbmc3k_scores.csv
    VAEDA_BACKEND=tensorflow python tools/score_pbmc3k.py tests/fixtures/tf_pbmc3k_scores.csv

The thresholds were set from the measured 3-way run (see docs/Roadmap.md,
Phase 5) with margin; they are floors (RMSE is a ceiling), not exact values.
"""

import csv
from pathlib import Path

import numpy as np
import pytest
from parity_metrics import compare

# Fraction of cells flagged as doublets when deriving calls for Jaccard/ARI.
RATE = 0.08

_ROOT = Path(__file__).parent.parent
_FIXTURES = Path(__file__).parent / "fixtures"
SCORE_CSVS = {
    "legacy": _ROOT / "data" / "legacy_pbmc3k_scores.csv",
    "torch": _FIXTURES / "torch_pbmc3k_scores.csv",
    "tensorflow": _FIXTURES / "tf_pbmc3k_scores.csv",
}

# Lower bounds (RMSE is an upper bound), set from the measured 3-way run with
# margin. Worst observed across the three pairs: Jaccard 0.65, ARI 0.74,
# Pearson 0.92, Spearman 0.87, RMSE 0.066. These floors leave room for the
# small run-to-run variance of the (stochastic) TF/torch pipelines while still
# catching a real regression (e.g. the pre-fix TF backend scored Pearson 0.61).
THRESHOLDS = {
    "jaccard": 0.55,
    "ari": 0.65,
    "pearson": 0.85,
    "spearman": 0.80,
    "rmse_max": 0.10,
}


def _load_scores(path: Path) -> dict[str, float]:
    with path.open() as fh:
        return {row["obs_id"]: float(row["doublet_score"]) for row in csv.DictReader(fh)}


def _aligned(a: dict[str, float], b: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    keys = [k for k in a if k in b]
    return np.array([a[k] for k in keys]), np.array([b[k] for k in keys])


@pytest.mark.parametrize(
    ("name_a", "name_b"),
    [("torch", "legacy"), ("tensorflow", "legacy"), ("torch", "tensorflow")],
)
def test_backend_pair_meets_parity_thresholds(name_a, name_b):
    """
    Given two lineages' doublet scores over the same pbmc3k cells
    When the parity metrics are computed on the aligned scores
    Then doublet-call overlap, agreement and error meet the thresholds
    """
    for name in (name_a, name_b):
        if not SCORE_CSVS[name].exists():
            pytest.skip(f"missing fixture {SCORE_CSVS[name]} (see this module's docstring)")

    a, b = _aligned(_load_scores(SCORE_CSVS[name_a]), _load_scores(SCORE_CSVS[name_b]))
    assert a.size > 0, "no overlapping cells between the two fixtures"

    metrics = compare(a, b, rate=RATE)
    print(f"\n[parity] {name_a} vs {name_b}: {metrics}")

    assert metrics["jaccard"] >= THRESHOLDS["jaccard"]
    assert metrics["ari"] >= THRESHOLDS["ari"]
    assert metrics["pearson"] >= THRESHOLDS["pearson"]
    assert metrics["spearman"] >= THRESHOLDS["spearman"]
    assert metrics["rmse"] <= THRESHOLDS["rmse_max"]
