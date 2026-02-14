# vaeda

A computational tool for annotating doublets in scRNAseq data using a variational autoencoder and PU (Positive-Unlabeled) learning.

**v0.2.0** — Now powered by **PyTorch** (replacing TensorFlow).

## Installation

### Recommended: uv (fastest)

```bash
# Install the latest release
uv pip install git+https://github.com/lukabor/vaeda.git@v0.2.0

# Or install from a local clone
git clone https://github.com/lukabor/vaeda.git
cd vaeda
uv sync
```

### pip

```bash
pip install git+https://github.com/lukabor/vaeda.git@v0.2.0
```

### GPU Support

vaeda v0.2.0 uses PyTorch and automatically selects the best available device (CUDA > MPS > CPU).

For NVIDIA GPU support, install PyTorch with CUDA before installing vaeda:

```bash
# CUDA 12.4
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Then install vaeda
pip install git+https://github.com/lukabor/vaeda.git@v0.2.0
```

For Apple Silicon (M1/M2/M3), MPS acceleration is used automatically with the standard PyTorch install.

### Migrating from v0.1.x (TensorFlow)

v0.2.0 replaces TensorFlow with PyTorch. The public API is unchanged — only the backend has changed:

```bash
# Remove old TensorFlow dependencies (optional)
pip uninstall tensorflow tensorflow-probability tf_keras

# Install v0.2.0
pip install git+https://github.com/lukabor/vaeda.git@v0.2.0
```

> **Note on conda:** Mixing conda and pip can lead to conflicts, especially with PyTorch CUDA builds. If you must use conda, create a minimal environment and use pip for the actual installation:

```bash
conda create -n vaeda_env python=3.13
conda activate vaeda_env
pip install git+https://github.com/lukabor/vaeda.git@v0.2.0
```

## Quick Start

```python
import vaeda

# adata is an AnnData object with raw counts in adata.X
result = vaeda.vaeda(adata)

# Results are stored in the AnnData object:
# - adata.obsm['vaeda_embedding']: VAE encoding
# - adata.obs['vaeda_scores']: doublet scores
# - adata.obs['vaeda_calls']: doublet/singlet calls
```

For a detailed example, see the [tutorial notebook](https://github.com/lukabor/vaeda/blob/main/doc/vaeda_scanpy-pbmc3k-tutorial.ipynb), which adapts the [scanpy PBMC3k tutorial](https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html) to demonstrate doublet annotation with vaeda.

## Development

### Setup

Clone the repository and install with development dependencies:

```bash
git clone https://github.com/lukabor/vaeda.git
cd vaeda

# Install all dependencies including dev group (uses uv.lock for reproducibility)
uv sync

# Or install with specific dependency group
uv sync --group dev
```

### Makefile Commands

Common development tasks are available via Makefile:

```bash
make test     # Run tests
make lint     # Run linting
make format   # Format code
```

### Running Tests

```bash
# Using Make (recommended)
make test

# Or manually with pytest
uv run pytest

# Run with verbose output
uv run pytest -v

# Run a specific test file
uv run pytest tests/test_vaeda.py

# Run tests with coverage
uv run pytest --cov=vaeda
```

### Code Quality

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting. Configuration is in `pyproject.toml` with a line length of 90 characters.

```bash
# Using Make (recommended)
make lint     # Check for issues
make format   # Format code

# Or manually
uv run ruff check src/
uv run ruff check --fix src/
uv run ruff format src/
```

### Type Checking

Type checking is done with [ty](https://github.com/astral-sh/ty):

```bash
uv run ty check src/
```

### Docker Development Environment

A Docker-based development environment is available for consistent, reproducible development across different machines. The Dockerfile supports both `dev` and `prod` targets.

#### Building the Container

```bash
docker compose build --no-cache dev
docker compose build --no-cache prod
```

#### Running the Container

```bash
docker compose up dev
docker compose up -d dev   # detached
docker compose exec dev bash
```

## Project Structure

```
vaeda/
├── src/vaeda/
│   ├── __init__.py      # Package exports
│   ├── vaeda.py         # Main vaeda function
│   ├── vae.py           # VAE model definition (PyTorch)
│   ├── pu.py            # PU learning implementation (PyTorch)
│   ├── classifier.py    # Neural network classifier (PyTorch)
│   ├── cluster.py       # Clustering utilities
│   ├── mk_doublets.py   # Synthetic doublet generation
│   └── logger.py        # Logging configuration
├── tests/               # Test suite
├── doc/                 # Documentation and tutorials
├── pyproject.toml       # Project configuration (includes ruff config)
├── Dockerfile           # Container definition (dev and prod targets)
├── docker-compose.yml   # Container orchestration
├── Makefile             # Development shortcuts
└── CHANGELOG.md         # Version history
```

## API Reference

### `vaeda.vaeda()`

Main function for doublet annotation.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adata` | `AnnData` | required | Annotated data matrix with raw counts in `adata.X` |
| `layer` | `str \| None` | `None` | Use `adata.layers[layer]` instead of `adata.X` |
| `filter_genes` | `bool` | `True` | Select 2000 most variable genes |
| `verbose` | `int` | `0` | Verbosity level (0, 1, 2, or 3) |
| `gene_thresh` | `float` | `0.01` | Filter genes expressed in ≤ threshold × cells |
| `num_hvgs` | `int` | `2000` | Number of highly variable genes to use |
| `pca_comp` | `int` | `30` | Number of principal components |
| `enc_sze` | `int` | `5` | Size of VAE encoding |
| `seed` | `int \| None` | `None` | Random seed for reproducibility |
| `optimized` | `bool` | `False` | Use vectorized doublet generation O(n log n) vs legacy O(n²) |

**Returns:** `AnnData` with added fields:
- `adata.obsm['vaeda_embedding']`: VAE encoding for cells
- `adata.obs['vaeda_scores']`: Doublet probability scores
- `adata.obs['vaeda_calls']`: Binary doublet/singlet calls

## Reproducibility

vaeda uses stochastic algorithms (VAE training, PU learning) that introduce natural run-to-run variation. Even with the same seed, consecutive runs may produce slightly different results due to non-determinism in PyTorch operations.

**Typical run-to-run variation (same seed, same code):**

| Metric | Expected Range |
|--------|----------------|
| Classification agreement | ~99% |
| Score correlation (Pearson r) | > 0.9 |
| Score correlation (Spearman r) | > 0.9 |

**For maximum reproducibility:**

```python
import torch
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)

# For CUDA
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import vaeda
result = vaeda.vaeda(adata, seed=42)
```

Note: Enabling deterministic operations may significantly slow down computation and some operations may not have deterministic implementations.
