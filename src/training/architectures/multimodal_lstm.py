"""
Multimodal LSTM architectures for Phase 7 (price + sentiment).

Two fusion strategies, both inheriting a shared `MultimodalBase` for sidecar
compatibility (`config_dict()` round-trip):

1.  MultimodalLSTMEarly — concatenate price and sentiment features at every
    timestep, single LSTM on the concatenated stream. Mirrors Phase 4 hyper-
    parameters (3 layers, hidden=128, dropout=0.2).

2.  MultimodalLSTMLate  — two-stream: separate LSTM for price (128 hidden,
    2 layers) and a small LSTM for sentiment (32 hidden, 1 layer). Concatenate
    final hidden states, MLP head.

Both output a single next-day return prediction: forward(price, sentiment) -> (B, 1).
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class MultimodalBase(nn.Module):
    """Shared base exposing config_dict() for sidecar round-tripping."""

    arch_name: str = "MultimodalBase"
    fusion: str = "unknown"

    def config_dict(self) -> dict:
        """Serialize hyperparameters for the checkpoint sidecar.

        Includes all constructor args needed by build_multimodal() to
        reconstruct the model from a sidecar pickle.
        """
        raise NotImplementedError

    def forward(self, price: torch.Tensor, sentiment: torch.Tensor) -> torch.Tensor:  # noqa: D401
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Early fusion: concat at every timestep, single LSTM
# ---------------------------------------------------------------------------

class MultimodalLSTMEarly(MultimodalBase):
    """Concat price + sentiment at every timestep, single LSTM.

    Input:
        price     (B, seq_len, P)
        sentiment (B, seq_len, S)
    Output:
        (B, 1) — predicted next-day return
    """

    arch_name = "MultimodalLSTMEarly"
    fusion = "early"

    def __init__(
        self,
        input_dim_price: int,
        input_dim_sentiment: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        output_dim: int = 1,
    ):
        super().__init__()
        self.input_dim_price = input_dim_price
        self.input_dim_sentiment = input_dim_sentiment
        self.concat_dim = input_dim_price + input_dim_sentiment
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_dim = output_dim

        self.lstm = nn.LSTM(
            input_size=self.concat_dim,
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

    def forward(self, price: torch.Tensor, sentiment: torch.Tensor) -> torch.Tensor:
        # Concat along feature axis: (B, seq_len, P) + (B, seq_len, S) -> (B, seq_len, P+S)
        x = torch.cat([price, sentiment], dim=-1)
        out, _ = self.lstm(x)
        last = out[:, -1, :]            # last timestep hidden
        return self.head(last)

    def config_dict(self) -> dict:
        return {
            "arch": self.arch_name,
            "fusion": self.fusion,
            "input_dim_price": self.input_dim_price,
            "input_dim_sentiment": self.input_dim_sentiment,
            "concat_dim": self.concat_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "output_dim": self.output_dim,
        }


# ---------------------------------------------------------------------------
# Late fusion: two-stream, concat final hidden states
# ---------------------------------------------------------------------------

class MultimodalLSTMLate(MultimodalBase):
    """Two-stream: separate LSTMs for price and sentiment, concat final hidden.

    Sentiment branch uses smaller capacity because forward-filled sentiment is
    low-information (sticky).

    Input:
        price     (B, seq_len, P)
        sentiment (B, seq_len, S)
    Output:
        (B, 1)
    """

    arch_name = "MultimodalLSTMLate"
    fusion = "late"

    def __init__(
        self,
        input_dim_price: int,
        input_dim_sentiment: int,
        hidden_dim_price: int = 128,
        hidden_dim_sent: int = 32,
        num_layers_price: int = 2,
        num_layers_sent: int = 1,
        dropout: float = 0.2,
        dropout_sent: float = 0.1,
        output_dim: int = 1,
    ):
        super().__init__()
        self.input_dim_price = input_dim_price
        self.input_dim_sentiment = input_dim_sentiment
        self.hidden_dim_price = hidden_dim_price
        self.hidden_dim_sent = hidden_dim_sent
        self.num_layers_price = num_layers_price
        self.num_layers_sent = num_layers_sent
        self.dropout = dropout
        self.dropout_sent = dropout_sent
        self.output_dim = output_dim
        self.fusion_dim = hidden_dim_price + hidden_dim_sent

        self.price_lstm = nn.LSTM(
            input_size=input_dim_price,
            hidden_size=hidden_dim_price,
            num_layers=num_layers_price,
            batch_first=True,
            dropout=dropout if num_layers_price > 1 else 0.0,
        )
        self.sent_lstm = nn.LSTM(
            input_size=input_dim_sentiment,
            hidden_size=hidden_dim_sent,
            num_layers=num_layers_sent,
            batch_first=True,
            dropout=dropout_sent if num_layers_sent > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(self.fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim),
        )

    def forward(self, price: torch.Tensor, sentiment: torch.Tensor) -> torch.Tensor:
        price_out, _ = self.price_lstm(price)
        sent_out, _ = self.sent_lstm(sentiment)
        h = torch.cat([price_out[:, -1, :], sent_out[:, -1, :]], dim=-1)
        return self.head(h)

    def config_dict(self) -> dict:
        return {
            "arch": self.arch_name,
            "fusion": self.fusion,
            "input_dim_price": self.input_dim_price,
            "input_dim_sentiment": self.input_dim_sentiment,
            "hidden_dim_price": self.hidden_dim_price,
            "hidden_dim_sent": self.hidden_dim_sent,
            "num_layers_price": self.num_layers_price,
            "num_layers_sent": self.num_layers_sent,
            "dropout": self.dropout,
            "dropout_sent": self.dropout_sent,
            "output_dim": self.output_dim,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_multimodal(config: dict) -> MultimodalBase:
    """Construct the right multimodal model from a sidecar config_dict().

    Branches on `config["arch"]` (string). Used by inference and resume paths.
    """
    arch = config.get("arch", "")
    if arch == MultimodalLSTMEarly.arch_name:
        return MultimodalLSTMEarly(
            input_dim_price=config["input_dim_price"],
            input_dim_sentiment=config["input_dim_sentiment"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            output_dim=config["output_dim"],
        )
    if arch == MultimodalLSTMLate.arch_name:
        return MultimodalLSTMLate(
            input_dim_price=config["input_dim_price"],
            input_dim_sentiment=config["input_dim_sentiment"],
            hidden_dim_price=config["hidden_dim_price"],
            hidden_dim_sent=config["hidden_dim_sent"],
            num_layers_price=config["num_layers_price"],
            num_layers_sent=config["num_layers_sent"],
            dropout=config["dropout"],
            dropout_sent=config["dropout_sent"],
            output_dim=config["output_dim"],
        )
    raise ValueError(f"Unknown multimodal arch: {arch!r}")


__all__ = [
    "MultimodalBase",
    "MultimodalLSTMEarly",
    "MultimodalLSTMLate",
    "build_multimodal",
]
