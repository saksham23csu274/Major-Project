"""
Phase 2 — Isolated-sign recognition baseline model.

This is the RQ1/RQ3 comparison point: a simple classifier over single-frame
keypoints, with NO temporal modeling and NO CTC alignment. It's expected
to underperform the continuous BiLSTM+CTC model specifically because it
can't use motion information at all -- that gap (not the absolute
accuracy number) is what RQ1 is asking about.

Splitting note: unlike the sentence-level data, ISL-CSLTR's word-level
images have no signer information (no per-signer subfolder), so we can't
do a signer-disjoint split here the way we did for the sequence model.
We use a random stratified split by class instead -- a documented
limitation, not an oversight.

Usage:
    python models/baseline/train_baseline.py
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "keypoints_word" / "word_features.npz"
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "checkpoints" / "baseline_isolated_sign.pt"


class IsolatedSignClassifier(nn.Module):
    def __init__(self, input_dim: int = 228, hidden_dim: int = 128, num_classes: int = None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def main() -> None:
    if not FEATURES_PATH.exists():
        print(f"[ERROR] {FEATURES_PATH} not found. Run 10_extract_word_keypoints.py first.")
        return

    data = np.load(FEATURES_PATH)
    X, y_raw = data["X"], data["y"]
    print(f"Loaded {X.shape[0]} instances, {X.shape[1]}-dim features")

    # Drop classes with fewer than 3 instances -- can't meaningfully split
    # them across train/val/test, and a class the model can only ever see
    # once or twice isn't a fair evaluation of anything.
    unique, counts = np.unique(y_raw, return_counts=True)
    valid_classes = set(unique[counts >= 3])
    mask = np.array([label in valid_classes for label in y_raw])
    dropped = (~mask).sum()
    if dropped:
        print(f"Dropping {dropped} instance(s) from classes with <3 total examples "
              f"({len(unique) - len(valid_classes)} classes affected).")
    X, y_raw = X[mask], y_raw[mask]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    num_classes = len(label_encoder.classes_)
    print(f"Training on {num_classes} classes, {len(X)} total instances after filtering.")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    # The second split can fail if a class ended up with only 1 instance in
    # X_temp (e.g. a class with exactly 3 total instances: ~2 train, ~1
    # temp -- that 1 instance can't be stratified further into val/test).
    # Fall back to a non-stratified split for val/test in that case rather
    # than crashing -- a small amount of class imbalance in val/test is a
    # far smaller problem than the whole script failing.
    try:
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
    except ValueError:
        print("[NOTE] Some classes have too few instances to stratify the val/test "
              "split perfectly; falling back to a non-stratified split for this step.")
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    X_train_t = torch.from_numpy(X_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).long().to(device)
    X_val_t = torch.from_numpy(X_val).float().to(device)
    y_val_t = torch.from_numpy(y_val).long().to(device)
    X_test_t = torch.from_numpy(X_test).float().to(device)
    y_test_t = torch.from_numpy(y_test).long().to(device)

    model = IsolatedSignClassifier(input_dim=X.shape[1], num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    epochs = 100
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = loss_fn(logits, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_acc = (val_logits.argmax(dim=1) == y_val_t).float().mean().item()

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} | train_loss={loss.item():.4f} | val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "label_classes": label_encoder.classes_,
                "input_dim": X.shape[1],
                "num_classes": num_classes,
                "val_acc": val_acc,
            }, CHECKPOINT_PATH)

    # Final test evaluation using the best checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_t)
        test_acc = (test_logits.argmax(dim=1) == y_test_t).float().mean().item()

    print(f"\nTraining complete.")
    print(f"Best val accuracy: {best_val_acc:.3f}")
    print(f"Test accuracy (best checkpoint): {test_acc:.3f}")
    print(f"Chance-level accuracy for reference: {1/num_classes:.3f} ({num_classes} classes)")
    print(f"Checkpoint saved to: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
