"""
WGAN-GP Generator network.

Generates realistic synthetic transaction feature vectors from a noise vector.
Supports conditional generation conditioned on fraud type/category.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    WGAN-GP Generator.

    Architecture: Linear → BatchNorm → LeakyReLU → ... → Tanh

    Args:
        noise_dim: Dimensionality of the input noise vector (default 128).
        output_dim: Dimensionality of the output (= feature vector length).
        hidden_dims: List of hidden layer sizes.
        condition_dim: If > 0, condition the generator on a class embedding.
    """

    def __init__(
        self,
        noise_dim: int = 128,
        output_dim: int = 30,
        hidden_dims: Optional[List[int]] = None,
        condition_dim: int = 0,
    ) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        self.output_dim = output_dim
        self.condition_dim = condition_dim

        hidden_dims = hidden_dims or [256, 512, 256]
        input_dim = noise_dim + condition_dim

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Tanh())

        self.model = nn.Sequential(*layers)

    def forward(
        self,
        z: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate synthetic samples.

        Args:
            z: Noise vector of shape (batch_size, noise_dim).
            condition: Optional condition tensor (batch_size, condition_dim).

        Returns:
            Synthetic feature tensor of shape (batch_size, output_dim).
        """
        if condition is not None and self.condition_dim > 0:
            z = torch.cat([z, condition], dim=1)
        return self.model(z)

    def sample_noise(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample noise vectors from N(0, 1)."""
        return torch.randn(batch_size, self.noise_dim, device=device)
