# Legacy fixture generator: original kostkalab/vaeda on python 3.8 + old TF/TFP.
#
# Builds an isolated environment matching the upstream lineage so we can score
# pbmc3k with the *published* vaeda and freeze the result as a parity fixture
# (data/legacy_pbmc3k_scores.csv). This image is NOT part of the vaeda package
# or its normal docker-compose stack — it exists only to regenerate that CSV.
#
# Build & run (writes into ./data on the host):
#   docker build -t vaeda-legacy -f docker/legacy/Dockerfile docker/legacy
#   docker run --rm -v "$PWD/data:/out" vaeda-legacy
FROM python:3.8-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Pinned scientific stack compatible with python 3.8 + TF 2.13 (the last TF
# whose tf.keras is Keras 2, which upstream vaeda's tf.keras/TFP code needs).
RUN pip install --no-cache-dir \
    "numpy==1.24.3" \
    "scipy==1.10.1" \
    "scikit-learn==1.3.2" \
    "pandas==2.0.3" \
    "tensorflow==2.13.1" \
    "tensorflow-probability==0.21.0" \
    "anndata==0.9.2" \
    "scanpy==1.9.8" \
    "kneed==0.8.5" \
    "leidenalg==0.10.2" \
    "igraph==0.11.3" \
    "numba==0.58.1"

# Upstream vaeda has unpinned deps, so install it without dependencies to keep
# the pinned stack above intact. Pinned to the kostkalab/vaeda main commit the
# torch/tf backends were ported from.
RUN pip install --no-cache-dir --no-deps \
    "git+https://github.com/kostkalab/vaeda@7b4979c12d92ce9b01256e106f4bb0f2b814b0b2"

COPY generate_legacy_scores.py /generate_legacy_scores.py

ENV CUDA_VISIBLE_DEVICES=-1 \
    TF_CPP_MIN_LOG_LEVEL=3

CMD ["python", "/generate_legacy_scores.py"]
