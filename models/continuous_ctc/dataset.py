"""
PyTorch Dataset + collate function for the BiLSTM+CTC model.

CTC training needs, per batch:
  - padded input sequences (batch, T_max, feature_dim)
  - padded target sequences (batch, L_max) — the gloss token indices
  - input_lengths: the REAL (unpadded) length of each input sequence
  - target_lengths: the REAL (unpadded) length of each target sequence

The lengths matter because CTCLoss uses them to know which parts of the
padded tensors are real data vs. padding — getting this wrong is the most
common source of silent CTC bugs (the loss still computes a number, it's
just meaningless).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

VOCAB_PATH = Path(__file__).resolve().parent / "vocab.json"


def load_vocab() -> dict:
    with open(VOCAB_PATH) as f:
        return json.load(f)


def gloss_to_indices(gloss: str, token_to_idx: dict) -> list:
    tokens = str(gloss).upper().replace(",", " ").split()
    return [token_to_idx[t] for t in tokens]  # KeyError on unseen token is intentional -- fail loudly


class ISLKeypointDataset(Dataset):
    def __init__(self, split_csv_path: str, token_to_idx: dict):
        self.df = pd.read_csv(split_csv_path)
        self.token_to_idx = token_to_idx

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        keypoints = np.load(row["keypoints_path"])  # (T, 228)
        target = gloss_to_indices(row["gloss"], self.token_to_idx)  # (L,)
        return (
            torch.from_numpy(keypoints).float(),
            torch.tensor(target, dtype=torch.long),
            row["clip_id"],
        )


def ctc_collate_fn(batch):
    """
    Pads a list of (keypoints, target, clip_id) tuples into batched,
    padded tensors plus the length vectors CTCLoss needs.
    """
    keypoints_list, targets_list, clip_ids = zip(*batch)

    input_lengths = torch.tensor([kp.shape[0] for kp in keypoints_list], dtype=torch.long)
    target_lengths = torch.tensor([len(t) for t in targets_list], dtype=torch.long)

    max_t = int(input_lengths.max())
    feature_dim = keypoints_list[0].shape[1]
    padded_inputs = torch.zeros(len(batch), max_t, feature_dim, dtype=torch.float32)
    for i, kp in enumerate(keypoints_list):
        padded_inputs[i, :kp.shape[0], :] = kp

    # CTCLoss expects targets as a single concatenated 1D tensor (not
    # padded per-sample) when target_lengths is provided -- padding the
    # targets would silently corrupt the loss since CTCLoss doesn't know
    # where the padding starts without target_lengths matching exactly.
    concatenated_targets = torch.cat(targets_list)

    return padded_inputs, concatenated_targets, input_lengths, target_lengths, clip_ids
