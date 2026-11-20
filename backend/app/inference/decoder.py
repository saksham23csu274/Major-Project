"""
Ties together the CTC model's greedy decoding with the rule-based
translation layer, so the backend has one simple function to call:
raw keypoint sequence in, (gloss string, English sentence) out.
"""

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "models" / "continuous_ctc"))
sys.path.insert(0, str(PROJECT_ROOT))

from model import greedy_ctc_decode  # noqa: E402
from translation_layer.rules.gloss_reorder import gloss_to_sentence  # noqa: E402


def decode_keypoint_sequence(model, keypoints: np.ndarray, idx_to_token: dict, device) -> tuple:
    """
    keypoints: (T, 228) numpy array, a single sequence (no batching --
    this is for one live stream at a time, matching the real-time
    single-user inference use case).

    Returns (gloss_string, english_sentence).
    """
    if keypoints.shape[0] == 0:
        return "", ""

    x = torch.from_numpy(keypoints).float().unsqueeze(0).to(device)  # (1, T, 228)
    input_lengths = torch.tensor([keypoints.shape[0]], dtype=torch.long)

    with torch.no_grad():
        log_probs = model(x, input_lengths)  # (T, 1, vocab_size)
        decoded_indices = greedy_ctc_decode(log_probs.cpu(), input_lengths)[0]

    gloss_tokens = [idx_to_token[i] for i in decoded_indices if idx_to_token.get(i) != "<blank>"]
    gloss_string = " ".join(gloss_tokens)

    sentence = gloss_to_sentence(gloss_string) if gloss_string else ""
    return gloss_string, sentence
