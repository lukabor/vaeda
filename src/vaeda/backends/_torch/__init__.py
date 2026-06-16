"""PyTorch backend: VAE/classifier model definitions and training loops."""

from __future__ import annotations

from .train import train_clust_vae, train_pu_fold


class TorchBackend:
    """PyTorch implementation of the :class:`vaeda.backends.base.Backend` seam."""

    name = "torch"
    train_clust_vae = staticmethod(train_clust_vae)
    train_pu_fold = staticmethod(train_pu_fold)


BACKEND = TorchBackend()
