"""Score pbmc3k across many seeds for the Phase 7 seed-stability analysis.

Runs the current vaeda once per seed on the full pbmc3k dataset and collects
both the per-cell doublet scores and the per-cell calls into two matrices
(one row per seed), written as CSVs the stability metrics consume.

Run once per backend in an environment where that backend's compute is healthy
(torch in the default .venv; tensorflow in the vaeda[tensorflow] env) — torch
and TensorFlow cannot both *run* in one process. The legacy lineage is produced
by docker/legacy/ instead, in py3.8.

Usage:
    VAEDA_BACKEND=torch \\
        python tools/stability_pbmc3k.py tests/fixtures/stability/torch [--seeds 0 50]

Writes <out_dir>/<backend>_scores.csv and <out_dir>/<backend>_calls.csv, each
indexed by obs_id with one column per seed ("seed_0", "seed_1", ...). Progress
is printed per seed so a long background run is observable.
"""

import argparse
import os
import tempfile

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

import vaeda
from vaeda.backends import get_backend


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", help="directory for the <backend>_{scores,calls}.csv matrices")
    parser.add_argument(
        "--seeds",
        nargs=2,
        type=int,
        default=(0, 50),
        metavar=("START", "STOP"),
        help="half-open seed range [START, STOP); default 0 50 (seeds 0-49)",
    )
    args = parser.parse_args()

    backend = get_backend().name
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seeds[0], args.seeds[1]))

    sc.settings.datasetdir = Path(tempfile.gettempdir()) / "vaeda_cache_stability"
    base = sc.datasets.pbmc3k()
    base.var_names_make_unique()
    print(f"backend={backend} loaded pbmc3k: {base.shape}; seeds {seeds[0]}..{seeds[-1]}", flush=True)

    scores: dict[str, np.ndarray] = {}
    calls: dict[str, np.ndarray] = {}
    obs_ids = None
    for i, seed in enumerate(seeds):
        result = vaeda.vaeda(base.copy(), seed=seed)
        col = f"seed_{seed}"
        scores[col] = np.asarray(result.obs["vaeda_scores"], dtype=float)
        calls[col] = np.asarray(result.obs["vaeda_calls"]).astype(str)
        if obs_ids is None:
            obs_ids = result.obs_names
        n_doub = int((calls[col] == "doublet").sum())
        print(
            f"[{i + 1}/{len(seeds)}] seed={seed} doublets={n_doub} "
            f"score_range=[{scores[col].min():.4f}, {scores[col].max():.4f}]",
            flush=True,
        )

    scores_path = out_dir / f"{backend}_scores.csv"
    calls_path = out_dir / f"{backend}_calls.csv"
    pd.DataFrame(scores, index=obs_ids).to_csv(scores_path, index_label="obs_id")
    pd.DataFrame(calls, index=obs_ids).to_csv(calls_path, index_label="obs_id")
    print(f"wrote {scores_path} and {calls_path}: {len(obs_ids)} cells x {len(seeds)} seeds", flush=True)


if __name__ == "__main__":
    main()
