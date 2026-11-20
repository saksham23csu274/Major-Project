"""
BiLSTM + CTC model for continuous ISL sign recognition.

Architecture: keypoint sequence (T, 228) -> BiLSTM encoder -> per-timestep
linear projection to vocab size -> log-softmax -> CTC loss/decoding.

Why BiLSTM over Transformer: see docs/architecture-decisions.md. Short
version: ~100 sentence-level training examples is too little data for a
Transformer encoder to reliably learn attention patterns without
overfitting; BiLSTM is lighter, faster to train on an M4 (no CUDA), and
well-precedented for skeleton-based continuous sign recognition.
"""

import torch
import torch.nn as nn


class BiLSTMCTCModel(nn.Module):
    def __init__(self, input_dim: int = 228, hidden_dim: int = 128,
                 num_layers: int = 2, vocab_size: int = None, dropout: float = 0.3):
        super().__init__()
        if vocab_size is None:
            raise ValueError("vocab_size must be provided (including the blank token)")

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # bidirectional doubles the effective hidden size
        self.classifier = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, T, input_dim), padded
        input_lengths: (batch,) real (unpadded) sequence lengths

        Returns log-probabilities shaped (T, batch, vocab_size) -- CTCLoss
        expects time-major input by default, not batch-major.
        """
        # Packing tells the LSTM to ignore padded timesteps rather than
        # treating zero-padding as real (mostly-zero) input, which would
        # otherwise subtly bias the learned representation.
        packed = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        logits = self.classifier(out)  # (batch, T, vocab_size)
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs.transpose(0, 1)  # -> (T, batch, vocab_size) for CTCLoss


def greedy_ctc_decode(log_probs: torch.Tensor, input_lengths: torch.Tensor, blank_idx: int = 0) -> list:
    """
    Simple greedy CTC decoding: take the argmax class at each timestep,
    then collapse repeated consecutive symbols and remove blanks. This is
    the standard baseline decoder -- not as accurate as beam search, but
    simple, fast, and a fine starting point before optimizing.

    log_probs: (T, batch, vocab_size)
    Returns: list of lists of token indices, one per batch item.
    """
    T, batch_size, _ = log_probs.shape
    predictions = log_probs.argmax(dim=-1)  # (T, batch)

    decoded = []
    for b in range(batch_size):
        length = int(input_lengths[b])
        seq = predictions[:length, b].tolist()

        collapsed = []
        prev = None
        for token in seq:
            if token != prev and token != blank_idx:
                collapsed.append(token)
            prev = token
        decoded.append(collapsed)
    return decoded
