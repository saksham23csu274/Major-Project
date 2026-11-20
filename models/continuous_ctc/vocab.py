"""
Builds the gloss token vocabulary from the training split.

CTC needs a fixed, closed vocabulary of output classes plus one reserved
"blank" token (index 0, by convention) used internally by the CTC
algorithm to represent "no new symbol at this timestep" — this is what
lets CTC align a long keypoint sequence to a much shorter gloss sequence
without requiring pre-segmented frame-to-word boundaries.

Usage:
    python models/continuous_ctc/vocab.py
Output:
    models/continuous_ctc/vocab.json
"""

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TRAIN_SPLIT = PROJECT_ROOT / "data" / "splits" / "train.csv"
VOCAB_PATH = Path(__file__).resolve().parent / "vocab.json"

BLANK_TOKEN = "<blank>"


def build_vocab(gloss_series) -> dict:
    tokens = set()
    for gloss in gloss_series.dropna():
        tokens.update(str(gloss).upper().replace(",", " ").split())

    sorted_tokens = sorted(tokens)
    token_to_idx = {BLANK_TOKEN: 0}
    for i, tok in enumerate(sorted_tokens, start=1):
        token_to_idx[tok] = i
    return token_to_idx


def main() -> None:
    if not TRAIN_SPLIT.exists():
        print(f"[ERROR] {TRAIN_SPLIT} not found. Run 05_create_splits.py first.")
        return

    df = pd.read_csv(TRAIN_SPLIT)
    token_to_idx = build_vocab(df["gloss"])

    with open(VOCAB_PATH, "w") as f:
        json.dump(token_to_idx, f, indent=2)

    print(f"Vocabulary size (including blank): {len(token_to_idx)}")
    print(f"Saved to: {VOCAB_PATH}")


if __name__ == "__main__":
    main()
