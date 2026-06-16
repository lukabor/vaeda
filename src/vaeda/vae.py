"""VAE model definitions using PyTorch.

Replaces the TensorFlow/TensorFlow Probability implementation from v0.1.x
with native PyTorch modules and torch.distributions.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent, Normal


class Encoder(nn.Module):
    """Probabilistic encoder that maps input to a latent distribution."""

    def __init__(self, n_input: int, n_latent: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_input, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)
        # Output mean and log-variance for the latent distribution
        self.fc_mu = nn.Linear(256, n_latent)
        self.fc_logvar = nn.Linear(256, n_latent)
        self._init_weights()

    def _init_weights(self) -> None:
        """Use Glorot uniform (Xavier) to match Keras Dense defaults."""
        for m in [self.fc1, self.fc_mu, self.fc_logvar]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (z_sample, mu, logvar)."""
        h = F.relu(self.fc1(x))
        h = self.bn1(h)
        h = self.dropout1(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        # Reparameterisation trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar


class Decoder(nn.Module):
    """Probabilistic decoder that maps latent samples to a
    reconstruction distribution."""

    def __init__(self, n_latent: int, n_output: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_latent, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)
        # Output mean and log-variance for the reconstruction
        self.fc_mu = nn.Linear(256, n_output)
        self.fc_logvar = nn.Linear(256, n_output)
        self._init_weights()

    def _init_weights(self) -> None:
        """Use Glorot uniform (Xavier) to match Keras Dense defaults."""
        for m in [self.fc1, self.fc_mu, self.fc_logvar]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (recon_mu, recon_logvar)."""
        h = F.relu(self.fc1(z))
        h = self.bn1(h)
        h = self.dropout1(h)
        recon_mu = self.fc_mu(h)
        recon_logvar = self.fc_logvar(h)
        return recon_mu, recon_logvar


class ClustClassifier(nn.Module):
    """Cluster classifier head on the latent space."""

    def __init__(self, n_latent: int, n_clusters: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(n_latent)
        self.fc = nn.Linear(n_latent, n_clusters)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return raw cluster logits (softmax is applied inside the loss)."""
        h = self.bn(z)
        return self.fc(h)


class ClustVAE(nn.Module):
    """VAE with an auxiliary cluster-classification head.

    This is the PyTorch equivalent of ``define_clust_vae`` from v0.1.x.
    The loss consists of:
      1. Negative log-likelihood of the reconstruction (Gaussian).
      2. KL divergence from the latent posterior to a unit Gaussian prior.
      3. Categorical cross-entropy on cluster labels (weighted by
         ``clust_weight``).
    """

    def __init__(
        self,
        n_input: int,
        n_latent: int,
        n_clusters: int,
        clust_weight: float = 10000.0,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(n_input, n_latent)
        self.decoder = Decoder(n_latent, n_input)
        self.clust_classifier = ClustClassifier(n_latent, n_clusters)
        self.clust_weight = clust_weight

    def forward(
        self, x: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return (recon_mu, recon_logvar, clust_pred, z, mu, logvar)."""
        z, mu, logvar = self.encoder(x)
        recon_mu, recon_logvar = self.decoder(z)
        clust_pred = self.clust_classifier(z)
        return recon_mu, recon_logvar, clust_pred, z, mu, logvar

    def loss(
        self,
        x: torch.Tensor,
        recon_mu: torch.Tensor,
        recon_logvar: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        clust_pred: torch.Tensor,
        clust_target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute total loss.

        Matches the TFP/Keras loss semantics:
        - NLL: sum over features, mean over batch (Keras default)
        - KL: sum over latent dims, mean over batch
          (TFP KLDivergenceRegularizer behaviour)
        - Cluster CE: mean over batch

        Parameters
        ----------
        x : input data
        recon_mu, recon_logvar : decoder outputs
        mu, logvar : encoder outputs (passed through, not recomputed)
        clust_pred : cluster classifier logits (pre-softmax)
        clust_target : one-hot cluster labels

        Returns (total_loss, recon_loss, kl_loss) averaged over batch.
        """
        # Clamp logvar to prevent numerical instability
        recon_logvar = recon_logvar.clamp(-20, 20)

        # Reconstruction NLL (Gaussian): sum over features, mean over
        # batch — matches TFP's nll = -reduce_sum(log_prob, axis=-1)
        # averaged by Keras over the batch.
        recon_dist = Independent(Normal(recon_mu, torch.exp(0.5 * recon_logvar)), 1)
        # log_prob already sums over features (Independent, dim=1)
        nll = -recon_dist.log_prob(x).mean()

        # KL divergence: sum over latent dims, mean over batch
        # This matches TFP KLDivergenceRegularizer which computes
        # per-sample KL and adds it to the layer's activity
        # regularisation losses (then Keras averages over batch).
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()

        # Cluster classification loss: softmax categorical cross-entropy.
        # Clusters are mutually exclusive, so this uses cross_entropy over
        # logits with soft (one-hot) targets, not independent per-class BCE.
        clust_loss = F.cross_entropy(clust_pred, clust_target, reduction="mean")

        total = nll + kl + self.clust_weight * clust_loss
        return total, nll, kl


# ---------------------------------------------------------------------------
# Factory functions (preserve the v0.1.x public API names)
# ---------------------------------------------------------------------------


def define_clust_vae(
    enc_sze: int,
    ngens: int,
    num_clust: int,
    LR: float = 1e-3,
    clust_weight: float = 10000.0,
) -> tuple[ClustVAE, torch.optim.Optimizer]:
    """Build a ClustVAE model and its Adamax optimiser.

    Returns
    -------
    model : ClustVAE
    optimiser : torch.optim.Adamax
    """
    device = _get_device()
    model = ClustVAE(ngens, enc_sze, num_clust, clust_weight).to(device)
    optimiser = torch.optim.Adamax(model.parameters(), lr=LR)
    return model, optimiser


class _EarlyStopper:
    """Track validation loss, snapshot the best weights, and signal stopping.

    On each :meth:`step` the model is snapshotted (a deep copy of its
    ``state_dict``) whenever the validation loss improves. After training,
    :meth:`restore` loads the best snapshot back, so the encoding is taken
    from the best epoch rather than the last one.
    """

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0
        self._best_state: dict | None = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Record ``val_loss``; snapshot ``model`` if it improved.

        Returns True once ``patience`` consecutive non-improving epochs have
        elapsed (the caller should stop training).
        """
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self._best_state = copy.deepcopy(model.state_dict())
            return False
        self.counter += 1
        return self.counter >= self.patience

    def restore(self, model: nn.Module) -> None:
        """Load the best snapshot back into ``model`` (no-op if never set)."""
        if self._best_state is not None:
            model.load_state_dict(self._best_state)


def _get_device() -> torch.device:
    """Return the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
