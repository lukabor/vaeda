"""Score pbmc3k with the current vaeda and write a parity fixture CSV.

Run once per backend in an environment where that backend's compute is healthy
(torch in the default .venv; tensorflow in the vaeda[tensorflow] env). torch and
TensorFlow cannot both *run* in one process (native-lib/OpenMP conflict), which
is why each fixture is generated separately. The legacy fixture comes from
docker/legacy/ instead.

Usage:
    VAEDA_BACKEND=torch       python tools/score_pbmc3k.py tests/fixtures/torch_pbmc3k_scores.csv
    VAEDA_BACKEND=tensorflow  python tools/score_pbmc3k.py tests/fixtures/tf_pbmc3k_scores.csv

The SEED matches docker/legacy/generate_legacy_scores.py so all fixtures score
the same pbmc3k cells under the same vaeda seed.
"""

import os
import sys
import tempfile

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

import vaeda
from vaeda.backends import get_backend

SEED = 12345


def main() -> None:
    out = sys.argv[1]
    sc.settings.datasetdir = Path(tempfile.gettempdir()) / f"vaeda_cache_{Path(out).stem}"
    adata = sc.datasets.pbmc3k()
    adata.var_names_make_unique()

    result = vaeda.vaeda(adata, seed=SEED)
    scores = np.asarray(result.obs["vaeda_scores"], dtype=float)
    pd.DataFrame({"obs_id": result.obs_names, "doublet_score": scores}).to_csv(
        out, index=False
    )
    print(
        f"backend={get_backend().name} wrote {out}: n={len(scores)} "
        f"range=[{scores.min():.4f}, {scores.max():.4f}]"
    )


if __name__ == "__main__":
    main()
