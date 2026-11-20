"""
Dataset loader for CISLR pretraining. Unlike the tiny word-level baseline
dataset (which fit entirely in memory as static single-frame vectors),
CISLR has 7,050 variable-length video sequences -- loading everything
eagerly into one big tensor risks exceeding the 16GB RAM budget on
worst-case combinations of many long sequences. This loads each
sequence's .npy file lazily in __getitem__, same pattern as the
sentence-level ISLKeypointDataset.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CISLRPretrainDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_to_idx: dict):
        self.df = df.reset_index(drop=True)
        self.label_to_idx = label_to_idx

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        keypoints = np.load(row["keypoints_path"])  # (T, 228)
        label = self.label_to_idx[row["gloss"]]
        return torch.from_numpy(keypoints).float(), label


def pretrain_collate_fn(batch):
    """Pads variable-length sequences; returns input_lengths for
    pack_padded_sequence, plus a plain label tensor (no CTC-style
    concatenated targets needed here -- one label per whole sequence)."""
    keypoints_list, labels = zip(*batch)

    input_lengths = torch.tensor([kp.shape[0] for kp in keypoints_list], dtype=torch.long)
    max_t = int(input_lengths.max())
    feature_dim = keypoints_list[0].shape[1]

    padded = torch.zeros(len(batch), max_t, feature_dim, dtype=torch.float32)
    for i, kp in enumerate(keypoints_list):
        padded[i, :kp.shape[0], :] = kp

    labels_t = torch.tensor(labels, dtype=torch.long)
    return padded, labels_t, input_lengths
