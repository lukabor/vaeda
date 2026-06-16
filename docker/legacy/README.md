# Legacy parity fixture

This directory builds an isolated environment to score pbmc3k with the
**original published** [kostkalab/vaeda](https://github.com/kostkalab/vaeda)
(TensorFlow / TFP, python 3.8), and freeze the result as the parity baseline
the current torch and TensorFlow backends are validated against
(`tests/test_parity.py`).

It is **not** part of the vaeda package or the normal `docker-compose` stack.

## What it produces

`data/legacy_pbmc3k_scores.csv` — two columns, one row per pbmc3k cell:

```
obs_id,doublet_score
AAACATACAACCAC-1,0.04231934497753779
...
```

## Provenance

- Upstream commit: `7b4979c12d92ce9b01256e106f4bb0f2b814b0b2` (kostkalab/vaeda `main`).
- Dataset: full `scanpy.datasets.pbmc3k()` (~2700 cells × 32738 genes, raw counts).
- Seed: `SEED = 12345` (see `generate_legacy_scores.py`), matching
  `tools/score_pbmc3k.py` so every lineage scores the same cells under the same
  vaeda seed.
- Pinned stack (python 3.8): tensorflow 2.13.1, tensorflow-probability 0.21.0,
  numpy 1.24.3 — the last TF whose `tf.keras` is Keras 2, which upstream's
  TFP/Keras code requires. See `Dockerfile`.

## Regenerate

```bash
docker build -t vaeda-legacy -f docker/legacy/Dockerfile docker/legacy
docker run --rm -v "$PWD/data:/out" vaeda-legacy
```

The CSV is committed (via a `.gitignore` negation), so this only needs to run if
the baseline is intentionally refreshed. Doublet scoring has stochastic steps;
expect small run-to-run variation, which the parity thresholds tolerate.

## The seed-stability fixture (Phase 7)

`data/legacy_stability_{scores,calls}.csv.gz` — the same upstream lineage scored
across **seeds 0–49** (one column per seed), per-cell scores *and* calls, frozen
for the within-backend seed-stability analysis. Gzipped because the score matrix
is high-entropy float text. Regenerate by mounting the loop generator into the
same image (no rebuild):

```bash
docker run --rm \
    -v "$PWD/data:/out" \
    -v "$PWD/docker/legacy/generate_legacy_stability.py:/gen.py" \
    vaeda-legacy python /gen.py
```

The torch and TF stability matrices are recomputed live (gitignored under
`tests/fixtures/stability/`) via `tools/stability_pbmc3k.py`; the cross-backend
table is built by `tools/stability_table.py`. See `docs/Roadmap.md` Phase 7.

## The other two fixtures

`tests/fixtures/{torch,tf}_pbmc3k_scores.csv` are scored by the *current* repo,
one backend each (torch and TensorFlow cannot run in one process). Regenerate
with `tools/score_pbmc3k.py` — see the docstring in `tests/test_parity.py`.
