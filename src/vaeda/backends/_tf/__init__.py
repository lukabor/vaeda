"""TensorFlow backend: TFP/Keras VAE + classifier and their training loops.

``tensorflow_probability`` 0.25 is built on Keras 2, so ``TF_USE_LEGACY_KERAS``
must be set to point ``tf.keras`` at ``tf_keras`` *before* tensorflow is first
imported. We set it here, at the top of the package, before importing any
submodule that pulls in tensorflow.
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

from .train import train_clust_vae, train_pu_fold  # noqa: E402


class TfBackend:
    """TensorFlow implementation of the :class:`vaeda.backends.base.Backend` seam."""

    name = "tensorflow"
    train_clust_vae = staticmethod(train_clust_vae)
    train_pu_fold = staticmethod(train_pu_fold)


BACKEND = TfBackend()
