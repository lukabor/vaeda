"""Unit tests for vaeda doublet detection.

Tests follow Given-When-Then structure:
- Given: Setup and preconditions
- When: Action being tested
- Then: Expected outcomes
"""

import os
import shutil
import tarfile
import tempfile

import pytest
import requests
import scanpy as sc


@pytest.fixture(scope="module")
def pbmc3k_adata():
    """Download and prepare pbmc3k test dataset. This fixture follows documentation provided by vaeda."""
    original_dir = os.getcwd()
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
