"""Feature encoders for ALIGNN - atom, bond, and angle features."""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional
import numpy as np
import dgl


class AtomEmbedding(nn.Module):
    """Learnable atom embedding from atomic number."""

    def __init__(
        self,
        num_elements: int = 100,
        embedding_dim: int = 64,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_elements, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, atomic_numbers: torch.Tensor) -> torch.Tensor:
        z = atomic_numbers.clamp(min=0, max=self.embedding.num_embeddings - 1)
        return self.embedding(z)


class RBFExpansion(nn.Module):
    """Expand interatomic distances with Gaussian radial basis functions.

    Uses SchNet-style RBF expansion:
        exp(-gamma * (d - center)^2)

    where gamma is 1 / lengthscale^2 and centers are evenly spaced.
    """

    def __init__(
        self,
        vmin: float = 0.0,
        vmax: float = 8.0,
        bins: int = 80,
        lengthscale: Optional[float] = None,
    ):
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins

        centers = torch.linspace(vmin, vmax, bins)
        self.register_buffer("centers", centers)

        if lengthscale is None:
            lengthscale = np.diff(centers).mean()
        self.lengthscale = lengthscale
        self.gamma = 1.0 / (lengthscale**2)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        distance = distance.unsqueeze(1)
        return torch.exp(-self.gamma * (distance - self.centers) ** 2)


class AngleRBFExpansion(nn.Module):
    """Expand bond angle cosines with Gaussian RBF.

    Range is [-1, 1] for cosine values.
    """

    def __init__(
        self,
        vmin: float = -1.0,
        vmax: float = 1.0,
        bins: int = 40,
        lengthscale: Optional[float] = None,
    ):
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins

        centers = torch.linspace(vmin, vmax, bins)
        self.register_buffer("centers", centers)

        if lengthscale is None:
            lengthscale = np.diff(centers).mean()
        self.lengthscale = lengthscale
        self.gamma = 1.0 / (lengthscale**2)

    def forward(self, cosine: torch.Tensor) -> torch.Tensor:
        cosine = cosine.unsqueeze(1)
        return torch.exp(-self.gamma * (cosine - self.centers) ** 2)


class MLPLayer(nn.Module):
    """MLP layer: Linear -> BatchNorm -> SiLU."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class AtomFeatureEncoder(nn.Module):
    """Encode atom features from atomic numbers.

    Uses element properties from pymatgen or learned embeddings.
    For simplicity, uses learnable embeddings.
    """

    def __init__(
        self,
        num_elements: int = 100,
        atom_features: int = 92,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.atom_embedding = nn.Linear(atom_features, hidden_dim)

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        element_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if element_features is not None:
            return self.atom_embedding(element_features)
        z = atomic_numbers.clamp(min=0, max=99)
        return self.atom_embedding(z.unsqueeze(-1).float().expand(-1, 92))


class BondEncoder(nn.Module):
    """Encode bond features from distances."""

    def __init__(
        self,
        rbf_bins: int = 80,
        embedding_dim: int = 64,
    ):
        super().__init__()
        self.rbf = RBFExpansion(vmin=0.0, vmax=8.0, bins=rbf_bins)
        self.encoder = nn.Sequential(
            nn.Linear(rbf_bins, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.SiLU(),
        )

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        rbf_features = self.rbf(distances)
        return self.encoder(rbf_features)


class AngleEncoder(nn.Module):
    """Encode angle features from bond angle cosines."""

    def __init__(
        self,
        rbf_bins: int = 40,
        embedding_dim: int = 64,
    ):
        super().__init__()
        self.rbf = AngleRBFExpansion(vmin=-1.0, vmax=1.0, bins=rbf_bins)
        self.encoder = nn.Sequential(
            nn.Linear(rbf_bins, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.SiLU(),
        )

    def forward(self, cosines: torch.Tensor) -> torch.Tensor:
        rbf_features = self.rbf(cosines)
        return self.encoder(rbf_features)


class CrystalFeatureEncoder(nn.Module):
    """Combined encoder for crystal graph features.

    Encodes:
    - Atom features (atomic numbers -> embeddings)
    - Bond features (distances -> RBF -> embeddings)
    - Angle features (cosines -> RBF -> embeddings)
    """

    def __init__(
        self,
        num_elements: int = 100,
        atom_features: int = 92,
        rbf_bins: int = 80,
        angle_bins: int = 40,
        embedding_dim: int = 64,
    ):
        super().__init__()
        self.atom_encoder = AtomFeatureEncoder(
            num_elements=num_elements,
            atom_features=atom_features,
            hidden_dim=embedding_dim,
        )
        self.bond_encoder = BondEncoder(
            rbf_bins=rbf_bins,
            embedding_dim=embedding_dim,
        )
        self.angle_encoder = AngleEncoder(
            rbf_bins=angle_bins,
            embedding_dim=embedding_dim,
        )

    def encode_atoms(self, g: dgl.DGLGraph) -> torch.Tensor:
        z = g.ndata["atomic_number"]
        return self.atom_encoder(z)

    def encode_bonds(self, g: dgl.DGLGraph) -> torch.Tensor:
        d = g.edata["d"]
        return self.bond_encoder(d)

    def encode_angles(self, lg: dgl.DGLGraph) -> torch.Tensor:
        h = lg.edata["h"]
        return self.angle_encoder(h)

    def forward(
        self,
        g: dgl.DGLGraph,
        lg: Optional[dgl.DGLGraph] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        atom_feats = self.encode_atoms(g)
        bond_feats = self.encode_bonds(g)

        angle_feats = None
        if lg is not None:
            angle_feats = self.encode_angles(lg)

        return atom_feats, bond_feats, angle_feats