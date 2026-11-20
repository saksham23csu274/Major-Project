"""
Loads the trained BiLSTM+CTC checkpoint for use by the backend's
inference pipeline. Kept separate from the training code so the backend
doesn't need to import anything training-specific (optimizers, loss
functions, etc.) -- just the model architecture and the saved weights.
"""

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "models" / "continuous_ctc"))

from model import BiLSTMCTCModel  # noqa: E402 -- import after sys.path setup, intentional

DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "models" / "checkpoints" / "ctc_finetuned_best.pt"


def load_model(checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH):
    """
    Returns (model, token_to_idx, idx_to_token, device). Raises
    FileNotFoundError with a clear message if no checkpoint exists yet --
    this is expected before the model has been trained, and the caller
    should handle it gracefully (e.g. the API can still start up and
    report "model not available yet" rather than crashing).
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No trained checkpoint found at {checkpoint_path}. "
            f"Run models/continuous_ctc/train.py first."
        )

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    token_to_idx = checkpoint["vocab"]
    idx_to_token = {v: k for k, v in token_to_idx.items()}

    model = BiLSTMCTCModel(
        input_dim=228,
        hidden_dim=checkpoint["hidden_dim"],
        num_layers=checkpoint["num_layers"],
        vocab_size=len(token_to_idx),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, token_to_idx, idx_to_token, device
