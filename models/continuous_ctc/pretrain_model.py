"""
Pretraining model for CISLR: same BiLSTM encoder architecture as the
CTC model, but with mean-pooling over time + a linear classifier, since
CISLR pretraining is standard single-label classification (one video ->
one word), not sequence-to-sequence alignment.

After pretraining, only the LSTM encoder's weights get transferred into
the CTC model -- the classification head here is CISLR-vocab-specific
(4,764 words) and has no meaning for ISL-CSLTR's much smaller gloss
vocabulary, so it's intentionally left behind.
"""

import torch
import torch.nn as nn


class PretrainEncoder(nn.Module):
    def __init__(self, input_dim: int = 228, hidden_dim: int = 128,
                 num_layers: int = 2, num_classes: int = None, dropout: float = 0.3):
        super().__init__()
        if num_classes is None:
            raise ValueError("num_classes must be provided")

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)  # (batch, T, hidden*2)

        # Mean-pool over the REAL (unpadded) timesteps only -- averaging in
        # the zero-padding would bias short sequences' representations
        # toward zero, proportionally more so the shorter they are.
        mask = torch.arange(out.shape[1], device=out.device).unsqueeze(0) < input_lengths.unsqueeze(1).to(out.device)
        mask = mask.unsqueeze(-1).float()  # (batch, T, 1)
        summed = (out * mask).sum(dim=1)
        pooled = summed / input_lengths.unsqueeze(1).to(out.device).float()

        return self.classifier(pooled)
