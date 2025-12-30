"""
Value Correction Module for TabR.

Implements the T transformation for correcting retrieved values
based on the difference between query and context keys.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueCorrection(nn.Module):
    """
    Value Correction transformation T.

    Transforms the delta (difference between query key and context keys)
    into a correction term that adjusts the retrieved values.
    """

    def __init__(self, d: int, dropout: float = 0.1):
        """
        Initialize ValueCorrection module.

        Args:
            d: Dimension of the embedding space.
            dropout: Dropout probability.
        """
        super().__init__()
        self.fc1 = nn.Linear(d, d, bias=False)
        self.fc2 = nn.Linear(d, d, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            delta: Difference tensor between query key and context keys.
                   Shape: (batch_size, m, d) where m is number of retrieved neighbors.

        Returns:
            Correction term of the same shape as delta.
        """
        x = self.fc1(delta)
        x = F.relu(x)
        x = self.dropout(x)
        return self.fc2(x)
