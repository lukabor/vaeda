"""Unit tests for vaeda doublet detection.

Tests follow Given-When-Then structure:
- Given: Setup and preconditions
- When: Action being tested
- Then: Expected outcomes
"""

import os
import tarfile
import tempfile
from pathlib import Path

import pytest
import requests
import scanpy as sc


@pytest.fixture(scope="module")
def pbmc3k_adata():
    """Download and prepare pbmc3k test dataset. This fixture follows documentation provided by vaeda."""
    original_dir = Path.cwd()
    temp_dir = tempfile.TemporaryDirectory()

    try:
        os.chdir(temp_dir.name)

        url = "http://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz"
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        with tarfile.open(fileobj=response.raw, mode="r|gz") as file:
            file.extractall(path=".")

        adata = sc.read_10x_mtx(
            "./filtered_gene_bc_matrices/hg19/",
            var_names="gene_symbols",
            cache=False,
        )
        adata.var_names_make_unique()

        os.chdir(original_dir)
        yield adata

    finally:
        os.chdir(original_dir)
        temp_dir.cleanup()


class TestVaedaImport:
    """Test vaeda module imports."""

    def test_vaeda_module_exposes_main_function(self):
        """
        Given the vaeda package is installed
        When I import the vaeda module
        Then the main vaeda function should be available
        """
        # Given
        import vaeda

        # When
        has_vaeda_function = hasattr(vaeda, "vaeda")

        # Then
        assert has_vaeda_function

    def test_vaeda_module_exposes_cluster_function(self):
        """
        Given the vaeda package is installed
        When I import the vaeda module
        Then the cluster function should be available
        """
        # Given
        import vaeda

        # When
        has_cluster_function = hasattr(vaeda, "cluster")

        # Then
        assert has_cluster_function

    def test_vaeda_module_exposes_sim_inflate_function(self):
        """
        Given the vaeda package is installed
        When I import the vaeda module
        Then the sim_inflate function should be available
        """
        # Given
        import vaeda

        # When
        has_sim_inflate_function = hasattr(vaeda, "sim_inflate")

        # Then
        assert has_sim_inflate_function


class TestVaedaOutput:
    """Test vaeda function output structure."""

    def test_vaeda_returns_anndata_object(self, pbmc3k_adata):
        """
        Given an AnnData object with raw counts
        When I run vaeda on the data
        Then it should return an AnnData object
        """
        # Given
        import anndata as ad

        import vaeda

        adata = pbmc3k_adata[:500, :].copy()

        # When
        result = vaeda.vaeda(adata, verbose=0)

        # Then
        assert isinstance(result, ad.AnnData)

    def test_vaeda_adds_doublet_scores_to_obs(self, pbmc3k_adata):
        """
        Given an AnnData object with raw counts
        When I run vaeda on the data
        Then vaeda_scores should be added to adata.obs
        """
        # Given
        import vaeda

        adata = pbmc3k_adata[:500, :].copy()

        # When
        result = vaeda.vaeda(adata, verbose=0)

        # Then
        assert "vaeda_scores" in result.obs.columns

    def test_vaeda_adds_doublet_calls_to_obs(self, pbmc3k_adata):
        """
        Given an AnnData object with raw counts
        When I run vaeda on the data
        Then vaeda_calls should be added to adata.obs
        """
        # Given
        import vaeda

        adata = pbmc3k_adata[:500, :].copy()

        # When
        result = vaeda.vaeda(adata, verbose=0)

        # Then
        assert "vaeda_calls" in result.obs.columns

    def test_vaeda_adds_embedding_to_obsm(self, pbmc3k_adata):
        """
        Given an AnnData object with raw counts
        When I run vaeda on the data
        Then vaeda_embedding should be added to adata.obsm
        """
        # Given
        import vaeda

        adata = pbmc3k_adata[:500, :].copy()

        # When
        result = vaeda.vaeda(adata, verbose=0)

        # Then
        assert "vaeda_embedding" in result.obsm


class TestVaedaScoreValidation:
    """Test vaeda score validity."""

    def test_doublet_scores_are_non_negative(self, pbmc3k_adata):
        """
        Given an AnnData object with raw counts
        When I run vaeda on the data
        Then all doublet scores should be >= 0
        """
        # Given
        import vaeda

        adata = pbmc3k_adata[:500, :].copy()

        # When
        result = vaeda.vaeda(adata, verbose=0)
        scores = result.obs["vaeda_scores"]

        # Then
        assert scores.min() >= 0.0

    def test_doublet_scores_do_not_exceed_one(self, pbmc3k_adata):
        """
        Given an AnnData object with raw counts
        When I run vaeda on the data
        Then all doublet scores should be <= 1
        """
        # Given
        import vaeda

        adata = pbmc3k_adata[:500, :].copy()

        # When
        result = vaeda.vaeda(adata, verbose=0)
        scores = result.obs["vaeda_scores"]

        # Then
        assert scores.max() <= 1.0

    def test_doublet_calls_are_binary(self, pbmc3k_adata):
        """
        Given an AnnData object with raw counts
        When I run vaeda on the data
        Then doublet calls should be binary values
        """
        # Given
        import vaeda

        adata = pbmc3k_adata[:500, :].copy()
        valid_values = {0, 1, "singlet", "doublet"}

        # When
        result = vaeda.vaeda(adata, verbose=0)
        calls = result.obs["vaeda_calls"]
        unique_values = set(calls.unique())

        # Then
        assert unique_values.issubset(valid_values)


class TestVaedaCluster:
    """Test vaeda clustering component."""

    def test_cluster_returns_labels_for_each_cell(self, pbmc3k_adata):
        """
        Given preprocessed gene expression data
        When I run the cluster function
        Then it should return a label for each cell
        """
        # Given
        from vaeda import cluster

        adata = pbmc3k_adata[:500, :].copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=2000)
        adata_hvg = adata[:, adata.var.highly_variable].copy()

        x_mat = adata_hvg.X
        if hasattr(x_mat, "toarray"):
            x_mat = x_mat.toarray()

        n_cells = adata.n_obs

        # When
        labels = cluster(x_mat)

        # Then
        assert len(labels) == n_cells

    def test_cluster_returns_integer_labels(self, pbmc3k_adata):
        """
        Given preprocessed gene expression data
        When I run the cluster function
        Then labels should be integers
        """
        # Given
        import numpy as np

        from vaeda import cluster

        adata = pbmc3k_adata[:500, :].copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=2000)
        adata_hvg = adata[:, adata.var.highly_variable].copy()

        x_mat = adata_hvg.X
        if hasattr(x_mat, "toarray"):
            x_mat = x_mat.toarray()

        # When
        labels = cluster(x_mat)

        # Then
        assert np.issubdtype(labels.dtype, np.integer)


class TestVaedaSimInflate:
    """Test vaeda doublet simulation component."""

    def test_sim_inflate_returns_augmented_data(self, pbmc3k_adata):
        """
        Given a gene expression matrix
        When I run sim_inflate to generate synthetic doublets
        Then it should return augmented data
        """
        # Given
        from vaeda import sim_inflate

        adata = pbmc3k_adata[:500, :].copy()
        x_mat = adata.X
        if hasattr(x_mat, "toarray"):
            x_mat = x_mat.toarray()

        # When
        result = sim_inflate(x_mat)

        # Then
        assert result is not None


@pytest.mark.slow
class TestVaedaFullDataset:
    """Full dataset tests (marked as slow)."""

    def test_vaeda_processes_full_pbmc3k_dataset(self, pbmc3k_adata):
        """
        Given the full pbmc3k dataset
        When I run vaeda with default parameters
        Then it should complete successfully with all expected outputs
        """
        # Given
        import vaeda

        adata = pbmc3k_adata.copy()

        # When
        result = vaeda.vaeda(adata, verbose=1)

        # Then
        assert "vaeda_scores" in result.obs.columns
        assert "vaeda_calls" in result.obs.columns
        assert "vaeda_embedding" in result.obsm

    def test_vaeda_embedding_has_correct_dimensions(self, pbmc3k_adata):
        """
        Given the full pbmc3k dataset
        When I run vaeda with default encoding size
        Then the embedding should have shape (n_cells, enc_sze)
        """
        # Given
        import vaeda

        adata = pbmc3k_adata.copy()
        enc_sze = 5  # default encoding size

        # When
        result = vaeda.vaeda(adata, verbose=0, enc_sze=enc_sze)
        embedding = result.obsm["vaeda_embedding"]

        # Then
        assert embedding.shape[0] == adata.n_obs
        assert embedding.shape[1] == enc_sze + 1  # encoding + knn_feature
