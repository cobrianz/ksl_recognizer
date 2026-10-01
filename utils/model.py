"""
LSTM model for KSL sequence classification.
"""

import torch
import torch.nn as nn
from typing import Optional


class KSL_LSTM(nn.Module):
    """
    Bidirectional LSTM classifier for landmark sequences.
    Input shape: (batch, seq_len, input_dim)
    """

    def __init__(
        self,
        input_dim: int = 1662,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 10,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * self.num_directions, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use the last hidden state from both directions
        if self.bidirectional:
            # h_n shape: (num_layers * 2, batch, hidden_dim)
            # Take the last layer's forward and backward states
            h_forward = h_n[-2, :, :]
            h_backward = h_n[-1, :, :]
            h = torch.cat([h_forward, h_backward], dim=1)
        else:
            h = h_n[-1, :, :]

        h = self.dropout(h)
        logits = self.fc(h)
        return logits


class KSL_Transformer(nn.Module):
    """
    Simple Transformer encoder for landmark sequences (optional alternative).
    """

    def __init__(
        self,
        input_dim: int = 1662,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        num_classes: int = 10,
        dropout: float = 0.3,
        max_seq_len: int = 60,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.transformer(x)
        # Global average pooling over time
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.fc(x)


def create_model(config: dict) -> nn.Module:
    """Factory to create model from config."""
    model_cfg = config["model"]
    model_type = model_cfg.get("type", "lstm").lower()

    if model_type == "lstm":
        return KSL_LSTM(
            input_dim=model_cfg["input_dim"],
            hidden_dim=model_cfg["hidden_dim"],
            num_layers=model_cfg["num_layers"],
            num_classes=model_cfg["num_classes"],
            dropout=model_cfg["dropout"],
        )
    elif model_type == "transformer":
        return KSL_Transformer(
            input_dim=model_cfg["input_dim"],
            d_model=model_cfg["hidden_dim"],
            num_layers=model_cfg["num_layers"],
            num_classes=model_cfg["num_classes"],
            dropout=model_cfg["dropout"],
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
