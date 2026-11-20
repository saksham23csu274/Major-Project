"""
Phase 2 — Pretrain the BiLSTM encoder on CISLR (isolated-word classification).

DESIGN NOTE — why a simple random split, not stratified/rare-class-filtered
like the baseline: CISLR has ~4,764 gloss classes across 7,050 videos --
roughly 1.5 instances per class on average, far sparser than the
word-level baseline data (~9 instances/class). Stratified splitting would
fail or be meaningless for the vast majority of classes here. Since this
pretraining stage's real purpose is to shape the LSTM encoder's learned
representations into something generally useful for sign recognition
(not to produce a benchmark-quality CISLR classifier -- that's not a
project goal), we use a plain random train/val split and treat val
accuracy as a rough sanity signal that training is working, not a
rigorous result to report on its own. The real evaluation of whether
this pretraining helped happens after fine-tuning on ISL-CSLTR, by
comparing WER with vs. without this step.

Usage:
    python models/continuous_ctc/pretrain_cislr.py
    python models/continuous_ctc/pretrain_cislr.py --epochs 15 --batch-size 16
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cislr_dataset import CISLRPretrainDataset, pretrain_collate_fn
from pretrain_model import PretrainEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CISLR_INDEX = PROJECT_ROOT / "data" / "processed" / "labels" / "cislr_index.csv"
CISLR_MANIFEST = PROJECT_ROOT / "data" / "processed" / "labels" / "cislr_keypoints_manifest.csv"
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"
ENCODER_CHECKPOINT_PATH = CHECKPOINT_DIR / "ctc_pretrained_encoder.pt"


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    args = parser.parse_args()

    if not CISLR_INDEX.exists() or not CISLR_MANIFEST.exists():
        print("[ERROR] cislr_index.csv or cislr_keypoints_manifest.csv not found. "
              "Run 07_build_cislr_index.py and 08_extract_cislr_keypoints.py first.")
        return

    index_df = pd.read_csv(CISLR_INDEX)
    manifest_df = pd.read_csv(CISLR_MANIFEST)
    manifest_df = manifest_df[manifest_df["status"] == "ok"]

    df = index_df.merge(manifest_df[["uid", "keypoints_path"]], on="uid", how="inner")
    dropped = len(index_df) - len(df)
    if dropped:
        print(f"Dropping {dropped} video(s) without successfully extracted keypoints.")

    print(f"Usable videos: {len(df)}")
    print(f"Distinct gloss classes: {df['gloss'].nunique()}")
    class_counts = df["gloss"].value_counts()
    print(f"Instances per class: min={class_counts.min()}, max={class_counts.max()}, "
          f"median={class_counts.median():.0f}, mean={class_counts.mean():.2f}")

    label_to_idx = {label: i for i, label in enumerate(sorted(df["gloss"].unique()))}
    num_classes = len(label_to_idx)

    full_dataset = CISLRPretrainDataset(df, label_to_idx)

    # Plain random split (see module docstring for why not stratified).
    val_size = max(1, int(0.1 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=pretrain_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=pretrain_collate_fn)

    device = get_device()
    print(f"Using device: {device}")

    model = PretrainEncoder(
        input_dim=228, hidden_dim=args.hidden_dim, num_layers=args.num_layers, num_classes=num_classes
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for padded_inputs, labels, input_lengths in train_loader:
            padded_inputs, labels = padded_inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(padded_inputs, input_lengths)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for padded_inputs, labels, input_lengths in val_loader:
                padded_inputs, labels = padded_inputs.to(device), labels.to(device)
                logits = model(padded_inputs, input_lengths)
                correct += (logits.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total if total else 0.0

        print(f"Epoch {epoch:3d}/{args.epochs} | train_loss={avg_loss:.4f} | val_acc={val_acc:.3f} "
              f"(chance level: {1/num_classes:.4f})")

        # Using >= rather than > guarantees at least epoch 1 always saves a
        # checkpoint, even if val_acc is 0.0 throughout (e.g. very early in
        # training, or in a pathological case) -- with a strict >, a
        # checkpoint might never be saved at all if val_acc never exceeds
        # its own starting value of 0.0, while still printing a false
        # "success" message at the end.
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            # Save the FULL model (useful for resuming/inspection) AND
            # separately, just the LSTM encoder's state_dict, which is
            # what actually gets transferred into the CTC model -- the
            # classification head here is CISLR-vocab-specific and
            # meaningless for ISL-CSLTR's different gloss vocabulary.
            torch.save({
                "encoder_state_dict": model.lstm.state_dict(),
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "val_acc": val_acc,
                "epoch": epoch,
                "num_cislr_classes": num_classes,
            }, ENCODER_CHECKPOINT_PATH)

    print(f"\nPretraining complete. Best val accuracy: {best_val_acc:.4f} "
          f"(chance level: {1/num_classes:.4f})")
    if ENCODER_CHECKPOINT_PATH.exists():
        print(f"Encoder checkpoint saved to: {ENCODER_CHECKPOINT_PATH}")
    else:
        print(f"[WARNING] No checkpoint was saved at {ENCODER_CHECKPOINT_PATH} -- "
              f"this shouldn't happen; please report this if you see it.")


if __name__ == "__main__":
    main()
