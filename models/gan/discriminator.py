"""
WGAN-GP Discriminator / Critic network.

No sigmoid output — uses Wasserstein loss.
Also serves as an anomaly scorer: lower scores indicate more anomalous input.
Spectral normalization is applied for training stability.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class Discriminator(nn.Module):
    """
    WGAN-GP Critic (Discriminator).

    Architecture: Linear → LayerNorm → LeakyReLU → Dropout → ... → Linear

    No sigmoid output: raw real-valued score.
    Higher scores mean "more real"; lower scores indicate anomalies / fraud.

    Args:
        input_dim: Dimensionality of the input feature vector.
        hidden_dims: List of hidden layer sizes.
        dropout: Dropout probability.
        use_spectral_norm: Apply spectral normalization to linear layers.
    """

    def __init__(
        self,
        input_dim: int = 30,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.3,
        use_spectral_norm: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim

        hidden_dims = hidden_dims or [256, 512, 256]

        def linear(in_f: int, out_f: int) -> nn.Linear:
            layer = nn.Linear(in_f, out_f)
            return spectral_norm(layer) if use_spectral_norm else layer

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim

        layers.append(linear(prev_dim, 1))  # Raw score — no activation

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute critic scores.

        Args:
            x: Input feature tensor of shape (batch_size, input_dim).

        Returns:
            Score tensor of shape (batch_size, 1).
        """
        return self.model(x)

    def get_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute anomaly scores.  Low scores = anomalous / potentially fraudulent.

        Scores are negated so that higher values indicate more anomalous inputs,
        consistent with typical anomaly detection conventions.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Anomaly score tensor of shape (batch_size,).
        """
        with torch.no_grad():
            scores = self.forward(x).squeeze(-1)
        return -scores  # Negate: higher = more anomalous
