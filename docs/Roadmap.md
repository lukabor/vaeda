# vaeda multi-framework backend — Roadmap

Goal: ship vaeda with **selectable VAE/classifier backends** so users install
`vaeda[torch]` (the default) or `vaeda[tensorflow]`, while the orchestration code
stays framework-agnostic.

## Locked decisions

- **Default = torch.** `pip install vaeda` is equivalent to `pip install vaeda[torch]`.
  Only `pip install vaeda[tensorflow]` adds the TensorFlow stack. torch therefore
  stays in core dependencies; `[torch]` is an explicit alias.
- **Backend resolution** (`get_backend()` order): `VAEDA_BACKEND` env var →
  autodetect the installed framework → both installed & env unset ⇒ torch + warn →
  none installed ⇒ raise with a `pip install vaeda[tensorflow]` hint.
- **Seam altitude = numpy in, numpy out.** The only altitude that hides torch's
  hand-rolled epoch loop behind the same interface as TF's `model.fit()`. The
  backend owns the *whole* training loop, not just the model definition.
- **TF parity target = upstream `kostkalab/vaeda` main**, tolerance-based (not
  bit-exact — RNG differs across frameworks).

## Constraints / context

- jujutsu repo; commit per phase with `/jj-atomic`. Tip `295fc8c9` is unbookmarked
  (set a bookmark before any push). See `memory/vaeda-open-items.md`.
- `.venv` can be root-owned/broken — see `memory/vaeda-dev-env.md`. Fast tests
  (`tests/test_fixes.py`) run in seconds; the full pbmc3k pipeline
  (`tests/test_vaeda.py`) takes ~65 min on CPU.
- torch touches 4 files today: `vae.py` (models), `classifier.py` (PU net),
  `pu.py` (PU train loop), `vaeda.py` (VAE train loop + scheduler + inference).
  `cluster.py`, `mk_doublets.py`, `logger.py` are already framework-agnostic.

## Target layout

```
src/vaeda/
  vaeda.py          # orchestrator — numpy/anndata only, calls get_backend()
  cluster.py        # agnostic, unchanged
  mk_doublets.py    # agnostic, unchanged
  logger.py         # agnostic, unchanged
  backends/
    __init__.py     # get_backend(): env -> autodetect -> error
    base.py         # Backend Protocol (the seam)
    _torch/{vae,classifier,train}.py
    _tf/{vae,classifier,train}.py
```

The seam (`backends/base.py`, as realized in Phases 1–2):

```python
class Backend(Protocol):
    name: str
    def train_clust_vae(self, x_mat, X_train, X_test, clust_train_oh, clust_test_oh,
                        *, enc_sze, num_clust, lr, clust_weight, rate, patience,
                        max_epochs, seeds, verbose=0) -> np.ndarray: ...
    def train_pu_fold(self, X, Y, x_predict, P, *, cls_eps, num_layers, pu_lr,
                      seeds) -> PuFoldResult: ...
```

Note: the PU seam is *per fold* (`train_pu_fold`), not whole-classifier — the
`RepeatedKFold` bagging and score averaging stay in `pu.py` as backend-agnostic
numpy orchestration. Seeding is internal to each call (via the `seeds` array),
so no separate `set_seed` is needed. `PuFoldResult` is a `NamedTuple` defined in
`backends/base.py`.

## Phases

### Phase 1 — Extract torch into a backend (no behavior change) ⚠️ riskiest
Move `vae.py`, `classifier.py` into `backends/_torch/`. Lift the VAE training loop
out of `vaeda.py` and the PU loop out of `pu.py` into `backends/_torch/train.py`
(carrying `_EarlyStopper`, the LR scheduler, `_get_device`, seeding). `vaeda.py`
and `pu.py` keep only numpy-level orchestration.
- **Verify**: capture a golden output (doublet scores on a small fixture) *before*
  the move; assert identical after. `tests/test_fixes.py` stays green.
- **Status**: ✅ done & verified (2026-06-16). Goldens in
  `tests/test_torch_backend.py` reproduce the pre-refactor VAE encoding and PU
  fold scores exactly; full pbmc3k pipeline (`tests/test_vaeda.py`, 15 tests)
  green on the relocated backend. `vaeda.py` and `pu.py` are now torch-free.

### Phase 2 — Seam + resolver
Add `backends/base.py` (Protocol) and `backends/__init__.py` (`get_backend()` with
env → autodetect → error). `vaeda.py` calls `get_backend()` instead of importing
`_torch` directly. Still torch-only.
- **Verify**: resolver unit tests (env set/unset, missing-backend error message);
  fast suite green.
- **Status**: ✅ done (2026-06-16). `backends/base.py` holds the `Backend`
  protocol + `PuFoldResult`; `backends/__init__.py` has `get_backend()` (cached)
  and the pure, unit-tested `_resolve_backend_name()`. Framework imports are
  deferred so importing vaeda forces neither torch nor tensorflow.
  `tests/test_backend_resolver.py` covers all branches; env smoke-tested.

### Phase 3 — pyproject extras
Add `[project.optional-dependencies]` `torch` and `tensorflow`; keep torch in core
so bare install stays torch. Document the `VAEDA_BACKEND=tensorflow` selector and
the tradeoff that a `[tensorflow]` env also carries torch.
- **Verify**: `uv sync` resolves; `pip install -e .[tensorflow]` resolves the TF
  stack in a throwaway env.
- **Status**: pending.

### Phase 4 — TensorFlow backend
Build `backends/_tf/` from upstream `kostkalab/vaeda` (tfp `IndependentNormal` +
`KLDivergenceRegularizer` + Keras `fit`). Map upstream `define_clust_vae` / `PU` /
`epoch_PU` onto `train_clust_vae` / `train_pu_classifier`.
- **Verify**: with `VAEDA_BACKEND=tensorflow`, the fast suite passes against the TF
  backend.
- **Risk**: `tensorflow_probability` + `tf_keras` is thinning-support — pin hard.
- **Status**: pending.

### Phase 5 — Parity validation
Generate golden doublet scores by running upstream main in an isolated legacy env
(py3.9 + old TF) on the pbmc3k tutorial dataset
(`doc/vaeda_scanpy-pbmc3k-tutorial.ipynb`). Assert the `_tf` backend matches within
tolerance (score correlation ≥ 0.95 + identical hard calls on clear doublets).
Cross-check torch vs TF agreement on the same dataset.
- **Verify**: tolerance assertions pass; document per-backend reproducibility
  baselines (do not cross-assert exact values — RNG differs).
- **Status**: pending.

### Phase 6 — CI matrix
One job per extra (`[torch]`, `[tensorflow]`); each runs the suite plus its parity
check. Upstream goldens are generated out-of-band, not in the main interpreter.
- **Verify**: both matrix legs green.
- **Status**: pending.

## Risks / gotchas

- Phase 1 is the bulk of the work and the only step that can silently change
  numerics — the pre-move golden is the safety net.
- `tensorflow_probability` / `tf_keras` rot is the most likely future breakage.
- Upstream main may not run on modern Python — hence goldens come from an isolated
  legacy env, never CI's main interpreter.
- torch and TF RNG never produce identical results — two reproducibility baselines,
  not one.
