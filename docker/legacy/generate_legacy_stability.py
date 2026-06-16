"""Generate the legacy seed-stability fixture (Phase 7).

Runs the original kostkalab/vaeda (TensorFlow/TFP, python 3.8) on the full
pbmc3k dataset once per seed and freezes per-cell scores and calls as two
matrices (one column per seed) — the legacy lineage of the within-backend
seed-stability analysis.

Mounted into docker/legacy/'s image at runtime (no rebuild needed):

    docker run --rm \\
        -v "$PWD/data:/out" \\
        -v "$PWD/docker/legacy/generate_legacy_stability.py:/gen.py" \\
        vaeda-legacy python /gen.py

Writes /out/legacy_stability_scores.csv and /out/legacy_stability_calls.csv,
each indexed by obs_id with one "seed_<n>" column per seed. Frozen and
committed (via a .gitignore negation) like the single-seed parity fixture.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
import scanpy as sc

import vaeda

SEEDS = range(0, 50)
# Gzipped: the score matrix is high-entropy float text and exceeds jj's file-size
# guard uncompressed. pandas reads/writes .csv.gz transparently by extension.
SCORES_PATH = "/out/legacy_stability_scores.csv.gz"
CALLS_PATH = "/out/legacy_stability_calls.csv.gz"


def main() -> None:
    base = sc.datasets.pbmc3k()  # ~2700 cells x 32738 genes, raw counts
    base.var_names_make_unique()
    print(f"loaded pbmc3k: {base.shape}; seeds {SEEDS.start}..{SEEDS.stop - 1}", flush=True)

    scores = {}
    calls = {}
    obs_ids = None
    seeds = list(SEEDS)
    for i, seed in enumerate(seeds):
        result = vaeda.vaeda(base.copy(), seed=seed)
        col = f"seed_{seed}"
        scores[col] = np.asarray(result.obs["vaeda_scores"]).astype(float)
        calls[col] = np.asarray(result.obs["vaeda_calls"]).astype(str)
        if obs_ids is None:
            obs_ids = result.obs_names
        n_doub = int((calls[col] == "doublet").sum())
        print(f"[{i + 1}/{len(seeds)}] seed={seed} doublets={n_doub}", flush=True)

    pd.DataFrame(scores, index=obs_ids).to_csv(SCORES_PATH, index_label="obs_id")
    pd.DataFrame(calls, index=obs_ids).to_csv(CALLS_PATH, index_label="obs_id")
    print(f"wrote {SCORES_PATH} and {CALLS_PATH}: {len(obs_ids)} cells x {len(seeds)} seeds", flush=True)


if __name__ == "__main__":
    main()
