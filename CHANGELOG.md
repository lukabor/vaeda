# Changelog

All notable changes to the `vaeda` package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2025-10-01

### Added
- Added modern Python packaging configuration with `pyproject.toml` following PEP 621 and PEP 660 best practices
- Added version-specific dependencies for better compatibility
- Added dependency lock file `uv.lock` for reproducible builds and better dependency resolution

### Changed
- **BREAKING**: Migrated from `tf.keras` to `tf_keras` imports to resolve compatibility issues with TensorFlow Probability and Keras 3.x. This was necessary due to the rapid evolution of the TensorFlow ecosystem - TensorFlow Probability layers were designed for Keras 2.x, but newer TensorFlow versions ship with Keras 3.x by default. The `tf_keras` package provides backward compatibility by maintaining the Keras 2.x API. See [TensorFlow Probability v0.25.0 release notes](https://github.com/tensorflow/probability/releases/tag/v0.25.0) and GitHub issues [#2006](https://github.com/tensorflow/probability/issues/2006), [#1795](https://github.com/tensorflow/probability/issues/1795), [#1774](https://github.com/tensorflow/probability/issues/1774).
- Updated dependency specifications with minimum versions for better stability:
  - `tensorflow>=2.13.1` (was unversioned)
  - `tensorflow-probability[tf]>=0.21.0` (was unversioned)
  - `scikit-learn>=1.3.2` (was unversioned)
- Enhanced sparse matrix handling to support both sparse and dense AnnData objects

### Fixed
- **Critical**: Fixed `ValueError: Only instances of keras.Layer can be added to a Sequential model` error when using TensorFlow Probability layers
- Fixed optimizer compatibility issues between TensorFlow and tf_keras
- Fixed callback compatibility issues in VAE training
- Fixed metrics compatibility in PU learning classifier
- Improved error handling for different matrix formats in AnnData objects

### Technical Details
The main compatibility issue was caused by version conflicts between TensorFlow Probability (which uses the older Keras 2.x API) and newer TensorFlow versions (which ship with Keras 3.x). The solution involved:

1. **vaeda/vae.py**: Replaced `tf.keras` imports with `tf_keras`, fixed optimizer references
2. **vaeda/vaeda.py**: Added `tf_keras` import, updated callbacks, improved matrix handling  
3. **vaeda/classifier.py**: Migrated to `tf_keras` for model definition
4. **vaeda/PU.py**: Updated metrics and optimizers to use `tf_keras` API

This ensures compatibility with:
- TensorFlow 2.13.1+ 
- TensorFlow Probability 0.21.0+
- Python 3.8+

## [0.0.30] - 2022-04-10

### Added
- Initial release by Hannah Schriever (hcs31@pitt.edu) using `setup.py` packaging (standard approach at the time)
- Core vaeda algorithm for doublet detection in single-cell RNA sequencing data
- Variational autoencoder-based approach for learning cell representations
- PU (Positive-Unlabeled) learning framework for doublet classification
- KNN-based feature extraction for doublet estimation
- Clustering-based simulated doublet filtering
- Comprehensive doublet scoring and calling functionality

### Features
- Support for AnnData objects (scanpy ecosystem compatibility)
- Configurable gene filtering and highly variable gene selection
- PCA-based dimensionality reduction
- Early stopping and learning rate scheduling for model training
- Flexible doublet rate estimation with user-defined or heuristic approaches
- Comprehensive logging and intermediate result saving capabilities

### Dependencies (Original)
- numpy
- tensorflow (unversioned)
- scipy  
- scikit-learn (unversioned)
- kneed
- anndata
- tensorflow_probability (unversioned)
- scanpy>1.3.3

### Technical Implementation
- Built using TensorFlow/Keras for deep learning components
- Scikit-learn for traditional ML tasks (PCA, clustering, cross-validation)
- Scanpy integration for single-cell analysis workflows
- Custom VAE architecture with clustering-aware loss function
- Bagging-based PU learning with configurable fold structures

