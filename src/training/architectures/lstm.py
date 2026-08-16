"""
StockLSTM — multi-layer LSTM for next-day return regression.

Architecture (matches docs/phases/phase_04_deep_learning_forecasting.md):
    Input  : (batch, sequence_length, n_features)
    LSTM   : n_layers stacked, hidden_dim=128, dropout between layers
    Head   : Linear(hidden_dim, 64) -> ReLU -> Dropout -> Linear(64, 1)
    Output : (batch, 1) — predicted next-day return

Hyperparameters come from src.utils.config.DL_* constants.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.utils.config import DL_HIDDEN_DIM, DL_NUM_LAYERS, DL_DROPOUT


class StockLSTM(nn.Module):
    """Stacked LSTM with a small MLP regression head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = DL_HIDDEN_DIM,
        num_layers: int = DL_NUM_LAYERS,
        output_dim: int = 1,
        dropout: float = DL_DROPOUT,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout = dropout

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, n_features) -> (batch, 1)"""
        # lstm_out: (batch, seq_len, hidden_dim)
        lstm_out, _ = self.lstm(x)
        # Use the final time-step's hidden state (most standard convention).
        last_out = lstm_out[:, -1, :]
        return self.head(last_out)

    def config_dict(self) -> dict:
        """Return architecture hyperparameters (for checkpoint sidecars)."""
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "output_dim": self.output_dim,
            "dropout": self.dropout,
            "arch": "StockLSTM",
        }


__all__ = ["StockLSTM"]
