"""
Training script for the BiLSTM+CTC continuous sign recognition model.

Trains on data/splits/train.csv, evaluates Word Error Rate (WER) on
data/splits/val.csv each epoch, and saves the best checkpoint (by val WER)
to models/checkpoints/.

Usage:
    python models/continuous_ctc/train.py
    python models/continuous_ctc/train.py --epochs 50 --batch-size 8 --lr 0.001
    python models/continuous_ctc/train.py --epochs 30 --pretrained-encoder ../checkpoints/ctc_pretrained_encoder.pt
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import ISLKeypointDataset, ctc_collate_fn
from model import BiLSTMCTCModel, greedy_ctc_decode

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TRAIN_SPLIT = PROJECT_ROOT / "data" / "splits" / "train.csv"
VAL_SPLIT = PROJECT_ROOT / "data" / "splits" / "val.csv"
VOCAB_PATH = Path(__file__).resolve().parent / "vocab.json"
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"


def edit_distance(a: list, b: list) -> int:
    """Standard Levenshtein distance between two token sequences (used for WER)."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


def compute_wer(predictions: list, targets: list) -> float:
    """
    predictions, targets: lists of token-index sequences (one per sample).
    WER = total edit distance / total reference length, across the whole set
    (this is the standard "corpus-level WER" definition, not averaged
    per-sentence, since per-sentence averaging over-weights short sentences).
    """
    total_edits = 0
    total_ref_len = 0
    for pred, target in zip(predictions, targets):
        total_edits += edit_distance(pred, target)
        total_ref_len += len(target)
    return total_edits / total_ref_len if total_ref_len > 0 else 0.0


def unbatch_targets(concatenated_targets: torch.Tensor, target_lengths: torch.Tensor) -> list:
    """Splits CTC's concatenated target tensor back into per-sample lists."""
    result = []
    offset = 0
    for length in target_lengths.tolist():
        result.append(concatenated_targets[offset:offset + length].tolist())
        offset += length
    return result


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(model, loader, device) -> float:
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for padded_inputs, targets, input_lengths, target_lengths, _ in loader:
            padded_inputs = padded_inputs.to(device)
            log_probs = model(padded_inputs, input_lengths)
            preds = greedy_ctc_decode(log_probs.cpu(), input_lengths)
            all_preds.extend(preds)
            all_targets.extend(unbatch_targets(targets, target_lengths))
    return compute_wer(all_preds, all_targets)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--pretrained-encoder", type=str, default=None,
                         help="Path to a CISLR-pretrained encoder checkpoint "
                              "(from pretrain_cislr.py) to initialize the LSTM "
                              "weights from before fine-tuning.")
    args = parser.parse_args()

    if not TRAIN_SPLIT.exists() or not VAL_SPLIT.exists():
        print("[ERROR] train.csv/val.csv not found. Run 05_create_splits.py first.")
        return
    if not VOCAB_PATH.exists():
        print("[ERROR] vocab.json not found. Run models/continuous_ctc/vocab.py first.")
        return

    with open(VOCAB_PATH) as f:
        token_to_idx = json.load(f)
    vocab_size = len(token_to_idx)
    print(f"Vocabulary size (incl. blank): {vocab_size}")

    device = get_device()
    print(f"Using device: {device}")

    train_dataset = ISLKeypointDataset(TRAIN_SPLIT, token_to_idx)
    val_dataset = ISLKeypointDataset(VAL_SPLIT, token_to_idx)
    print(f"Train clips: {len(train_dataset)}, Val clips: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=ctc_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=ctc_collate_fn)

    model = BiLSTMCTCModel(
        input_dim=228, hidden_dim=args.hidden_dim, num_layers=args.num_layers, vocab_size=vocab_size
    ).to(device)

    if args.pretrained_encoder:
        pretrained_path = Path(args.pretrained_encoder)
        if not pretrained_path.exists():
            print(f"[ERROR] Pretrained encoder checkpoint not found: {pretrained_path}")
            return
        checkpoint = torch.load(pretrained_path, map_location=device, weights_only=False)

        if checkpoint["hidden_dim"] != args.hidden_dim or checkpoint["num_layers"] != args.num_layers:
            print(f"[ERROR] Pretrained encoder was trained with hidden_dim={checkpoint['hidden_dim']}, "
                  f"num_layers={checkpoint['num_layers']}, but this run is configured with "
                  f"hidden_dim={args.hidden_dim}, num_layers={args.num_layers}. These must match "
                  f"exactly for the weights to transfer -- adjust --hidden-dim/--num-layers to match.")
            return

        model.lstm.load_state_dict(checkpoint["encoder_state_dict"])
        print(f"Loaded pretrained encoder from {pretrained_path} "
              f"(CISLR val_acc was {checkpoint['val_acc']:.4f} at epoch {checkpoint['epoch']})")

    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_wer = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for padded_inputs, targets, input_lengths, target_lengths, _ in train_loader:
            padded_inputs = padded_inputs.to(device)

            optimizer.zero_grad()
            log_probs = model(padded_inputs, input_lengths)
            loss = ctc_loss_fn(log_probs.cpu(), targets, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_wer = evaluate(model, val_loader, device)

        print(f"Epoch {epoch:3d}/{args.epochs} | train_loss={avg_loss:.4f} | val_WER={val_wer:.3f}")

        if val_wer < best_val_wer:
            best_val_wer = val_wer
            checkpoint_path = CHECKPOINT_DIR / "ctc_finetuned_best.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "vocab": token_to_idx,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "val_wer": val_wer,
                "epoch": epoch,
            }, checkpoint_path)
            print(f"  -> New best val WER ({val_wer:.3f}), saved checkpoint to {checkpoint_path}")

    print(f"\nTraining complete. Best val WER: {best_val_wer:.3f}")


if __name__ == "__main__":
    main()
