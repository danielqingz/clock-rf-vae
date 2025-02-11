"""ops.py"""

import torch
import torch.nn.functional as F


def recon_loss(x, x_recon):
    n = x.size(0)
    # BCE with logits, sum over the batch, then / n
    loss = F.binary_cross_entropy_with_logits(x_recon, x, reduction="sum") / n
    return loss


def kl_divergence(mu, logvar, r):
    # standard KL
    kld = -0.5 * (1 + logvar - mu**2 - logvar.exp()).sum(1)
    lam = -9.9 * r + 10.0
    weighted_kld = lam * kld
    return weighted_kld.mean()


def entropy(r):
    # note that r.log() is only valid for r>0
    H = (r * r.log() + (1 - r) * (1 - r).log()).sum()
    return H


def permute_dims(z):
    """Permute each dimension of z across the batch."""
    assert z.dim() == 2

    B, _ = z.size()
    perm_z = []
    for z_j in z.split(1, 1):
        perm = torch.randperm(B, device=z.device)
        perm_z_j = z_j[perm]
        perm_z.append(perm_z_j)

    return torch.cat(perm_z, 1)
