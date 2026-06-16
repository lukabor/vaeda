"""Build the Phase 7 cross-backend seed-stability table.

Loads each available backend's 50-seed score + call matrices and prints one
stability row per backend side by side (which lineage's doublet calls move
least when only the seed changes). Backends whose matrices are absent are
skipped, so this can run incrementally as each lineage finishes generating.

Sources (override with --backend LABEL SCORES_CSV CALLS_CSV, repeatable):
    torch    tests/fixtures/stability/torch/torch_scores.csv      + _calls.csv
    tf       tests/fixtures/stability/tf/tensorflow_scores.csv    + _calls.csv
    legacy   data/legacy_stability_scores.csv                     + _calls.csv

Run from the repo root in the torch env (pure numpy/scipy, no backend compute):
    uv run --no-sync python tools/stability_table.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from stability_metrics import summarize  # noqa: E402

DEFAULT_SOURCES = [
    ("torch", "tests/fixtures/stability/torch/torch_scores.csv", "tests/fixtures/stability/torch/torch_calls.csv"),
    ("tf", "tests/fixtures/stability/tf/tensorflow_scores.csv", "tests/fixtures/stability/tf/tensorflow_calls.csv"),
    ("legacy", "data/legacy_stability_scores.csv.gz", "data/legacy_stability_calls.csv.gz"),
]

# (key, header, format) for the printed table, in column order.
COLUMNS = [
    ("mean_doublet_count", "n_doub", "{:.1f}"),
    ("sd_doublet_count", "±sd", "{:.2f}"),
    ("doublet_count_cv", "count_cv", "{:.4f}"),
    ("flip_rate", "flip_rate", "{:.4f}"),
    ("fleiss_kappa", "fleiss_k", "{:.4f}"),
    ("mean_jaccard", "jaccard", "{:.4f}"),
    ("mean_ari", "ari", "{:.4f}"),
    ("mean_score_sd", "score_sd", "{:.4f}"),
    ("icc", "icc", "{:.4f}"),
    ("mean_threshold", "thresh", "{:.4f}"),
    ("threshold_sd", "thr_sd", "{:.4f}"),
]


def _load_matrix(path: str):
    """Read a cells×seeds CSV into a (n_runs, n_cells) numpy matrix."""
    df = pd.read_csv(path, index_col="obs_id")
    return df.to_numpy().T  # CSV is cells x seeds; metrics want seeds x cells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        nargs=3,
        action="append",
        metavar=("LABEL", "SCORES_CSV", "CALLS_CSV"),
        help="add/override a backend source (repeatable)",
    )
    args = parser.parse_args()
    sources = args.backend if args.backend else DEFAULT_SOURCES

    rows = {}
    for label, scores_csv, calls_csv in sources:
        if not (Path(scores_csv).exists() and Path(calls_csv).exists()):
            print(f"skip {label}: matrices not found ({scores_csv})", file=sys.stderr)
            continue
        rows[label] = summarize(_load_matrix(scores_csv), _load_matrix(calls_csv))

    if not rows:
        print("no backend matrices found; generate them first", file=sys.stderr)
        raise SystemExit(1)

    width = max(len(label) for label in rows)
    header = " ".join(f"{h:>9}" for _, h, _ in COLUMNS)
    print(f"{'backend':<{width}} {header}")
    for label, s in rows.items():
        cells = " ".join(f"{fmt.format(s[key]):>9}" for key, _, fmt in COLUMNS)
        print(f"{label:<{width}} {cells}")


if __name__ == "__main__":
    main()
