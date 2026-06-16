"""Generate the legacy (upstream) vaeda doublet-score fixture.

Runs the original kostkalab/vaeda (TensorFlow/TFP, the version this repo's
torch + tf backends are validated against) on the full pbmc3k dataset with a
fixed seed, and writes a two-column CSV (cell barcode, doublet score) that the
parity tests in tests/test_parity.py compare the current backends against.

Intended to run inside docker/legacy/Dockerfile (python 3.8 + pinned old TF).
See docker/legacy/README.md for provenance and how to regenerate.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
import scanpy as sc

import vaeda

# Fixed so the legacy fixture is reproducible (within the legacy TF RNG).
SEED = 12345
OUT_PATH = "/out/legacy_pbmc3k_scores.csv"


def main() -> None:
    adata = sc.datasets.pbmc3k()  # ~2700 cells x 32738 genes, raw counts
    adata.var_names_make_unique()
    print(f"loaded pbmc3k: {adata.shape}")

    result = vaeda.vaeda(adata, seed=SEED)

    scores = np.asarray(result.obs["vaeda_scores"]).astype(float)
    df = pd.DataFrame({"obs_id": result.obs_names, "doublet_score": scores})
    df.to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH}: {df.shape}, score range [{scores.min()}, {scores.max()}]")


if __name__ == "__main__":
    main()
