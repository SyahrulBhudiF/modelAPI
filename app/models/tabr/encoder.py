"""
TabR Encoder Module

Implements the encoder component for the TabR model architecture.
"""

import torch
import torch.nn as nn


class TabREncoder(nn.Module):
    """
    TabR Encoder for transforming input features into embeddings.

    Uses a linear projection followed by optional residual blocks
    for deep feature extraction.
    """

    def __init__(self, in_dim: int, d: int, n_blocks: int = 0, dropout: float = 0.0):
        """
        Initialize the TabR encoder.

        Args:
            in_dim: Input feature dimension
            d: Output embedding dimension
            n_blocks: Number of residual blocks (default: 0)
            dropout: Dropout probability (default: 0.0)
        """
        super().__init__()
        self.linear = nn.Linear(in_dim, d)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(d),
                    nn.Linear(d, d),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d, d),
                )
                for _ in range(n_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.

        Args:
            x: Input tensor of shape (batch_size, in_dim)

        Returns:
            Encoded tensor of shape (batch_size, d)
        """
        x = self.linear(x)
        for blk in self.blocks:
            x = x + blk(x)
        return x
