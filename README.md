# vaeda

vaeda (variational auto-encoder for doublet annotation) is a Python package for doublet annotation in single-cell RNA sequencing data. For method details and comparisons to alternative doublet annotation tools, see the [vaeda publication](https://academic.oup.com/bioinformatics/article/39/1/btac720/6808614).

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
  - [Using uv (Recommended)](#using-uv-recommended)
  - [Using pip](#using-pip)
  - [Using conda](#using-conda-not-recommended)
- [Quick Start](#quick-start)
- [Development](#development)
  - [Setup](#setup)
  - [Makefile Commands](#makefile-commands)
  - [Running Tests](#running-tests)
  - [Code Quality](#code-quality)
  - [Type Checking](#type-checking)
  - [Docker Development Environment](#docker-development-environment)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [Other Doublet Detection Tools](#other-doublet-detection-tools)
- [Citation](#citation)
- [License](#license)

## Requirements

- Python 3.12+
- TensorFlow 2.20+
- TensorFlow Probability 0.25+

## Installation

### Using uv (Recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager that handles vaeda's complex dependencies (TensorFlow, TensorFlow Probability, tf_keras) automatically with full reproducibility via lockfiles.

```bash
# Install uv if you haven't already
# See: https://docs.astral.sh/uv/getting-started/installation/

# Install from GitHub (specific release)
uv pip install git+https://github.com/lukabor/vaeda.git@v0.1.1

# Or install from a cloned repository
git clone https://github.com/lukabor/vaeda.git
cd vaeda
uv pip install .
```

### Using pip

```bash
# Install from GitHub (specific release)
pip install git+https://github.com/lukabor/vaeda.git@v0.1.1

# Or from a cloned repository
git clone https://github.com/lukabor/vaeda.git
cd vaeda
pip install .
```

### Using conda (Not Recommended)

> ⚠️ **Warning:** Using conda for vaeda installation is not recommended due to several issues:
>
> - **Reproducibility:** Conda's dependency resolution can produce different environments on different machines or at different times, making it difficult to reproduce results.
> - **Nested environments:** Mixing conda and pip can lead to conflicts, especially with complex dependencies like TensorFlow. Conda may install its own Python, creating nested or conflicting virtual environments.
> - **Dependency conflicts:** TensorFlow and TensorFlow Probability versions must be carefully coordinated. Conda's solver may not respect these constraints correctly.
> - **Slower resolution:** Conda's dependency resolution is significantly slower than uv or pip.
>
> If you must use conda, create a minimal environment and use pip for the actual installation:

```bash
# Create a minimal conda environment with just Python
conda create -n vaeda_env python=3.13
conda activate vaeda_env

# Use pip to install vaeda (not conda)
pip install git+https://github.com/lukabor/vaeda.git@v0.1.1
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
make build    # Build the Docker container
make test     # Run tests
make lint     # Run linting
make format   # Format code
make shell    # Start development shell in container
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
# Build the development image (recommended: use --no-cache for clean builds)
docker compose build --no-cache dev

# Or build the production image
docker compose build --no-cache prod
```

#### Running the Container

```bash
# Start the development container
docker compose up dev

# Or start the production container
docker compose up prod

# Run in detached mode
docker compose up -d dev
```

#### Accessing the Container

```bash
# Execute a shell in the running container
docker compose exec dev bash
```

#### Remote Development with distant.nvim

For neovim users, the container supports [distant.nvim](https://github.com/chipsenkbeil/distant.nvim) for seamless remote editing:

```bash
# Start the container with distant server
docker compose up -d dev

# Connect from neovim using distant.nvim on port 8080
```

This allows you to use your local neovim configuration while executing commands inside the container.

## Project Structure

```
vaeda/
├── src/vaeda/
│   ├── __init__.py      # Package exports
│   ├── vaeda.py         # Main vaeda function
│   ├── vae.py           # VAE model definition
│   ├── pu.py            # PU learning implementation
│   ├── classifier.py    # Neural network classifier
│   ├── cluster.py       # Clustering utilities
│   ├── mk_doublets.py   # Synthetic doublet generation
│   └── logger.py        # Logging configuration
├── tests/               # Test suite
├── doc/                 # Documentation and tutorials
├── pyproject.toml       # Project configuration (includes ruff config)
├── Dockerfile           # Container definition (dev and prod targets)
├── docker-compose.yml   # Container orchestration
├── Makefile             # Development shortcuts
└── uv.lock              # Locked dependencies
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
| `verbose` | `int` | `0` | Verbosity level (0, 1, or 2) |
| `gene_thresh` | `float` | `0.01` | Filter genes expressed in ≤ threshold × cells |
| `num_hvgs` | `int` | `2000` | Number of highly variable genes to use |
| `pca_comp` | `int` | `30` | Number of principal components |
| `enc_sze` | `int` | `5` | Size of VAE encoding |
| `seed` | `int \| None` | `None` | Random seed for reproducibility |
| `optimized` | `bool` | `False` | Use vectorized doublet generation O(n log n) vs legacy O(n²). Faster on large datasets, ~98% agreement with legacy. |

**Returns:** `AnnData` with added fields:
- `adata.obsm['vaeda_embedding']`: VAE encoding for cells
- `adata.obs['vaeda_scores']`: Doublet probability scores
- `adata.obs['vaeda_calls']`: Binary doublet/singlet calls

## Reproducibility

vaeda uses stochastic algorithms (VAE training, PU learning) that introduce natural run-to-run variation. Even with the same seed, consecutive runs may produce slightly different results due to non-determinism in TensorFlow/Keras operations.

**Typical run-to-run variation (same seed, same code):**

| Metric | Expected Range |
|--------|----------------|
| Classification agreement | ~99% |
| Score correlation (Pearson r) | > 0.9 |
| Score correlation (Spearman r) | > 0.9 |

This variation is inherent to the deep learning components and does not affect the scientific validity of the results. The doublet/singlet classifications are highly stable across runs.

**For maximum reproducibility:**

```python
import os
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import tensorflow as tf
tf.random.set_seed(42)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

import vaeda
result = vaeda.vaeda(adata, seed=42)
```

Note: Enabling deterministic operations may significantly slow down computation.

## Other Doublet Detection Tools

- [scds](https://github.com/kostkalab/scds) - Computational doublet detection for scRNA-seq (R)
- [scDblFinder](https://github.com/plger/scDblFinder) - Fast doublet detection in R
- [DoubletFinder](https://github.com/chris-mcginnis-ucsf/DoubletFinder) - R package for doublet detection
- [Scrublet](https://github.com/AllonKleinLab/scrublet) - Python tool for detecting doublets
- [Solo](https://github.com/calico/Solo) - Deep learning approach to doublet detection

## Citation

If you use vaeda in your research, please cite:

```bibtex
@article{schriever2022vaeda,
  title={vaeda: a computational framework for detection of doublets in single-cell RNA sequencing data using variational autoencoders},
  author={Schriever, Hannah and Kostka, Dennis},
  journal={Bioinformatics},
  volume={39},
  number={1},
  pages={btac720},
  year={2023},
  publisher={Oxford University Press}
}
```

## License

MIT
