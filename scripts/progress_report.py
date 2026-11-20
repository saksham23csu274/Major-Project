"""
Project progress tracker.

Re-scans the actual project folder every time it runs — file existence,
folder contents, row counts in key CSVs — so it always reflects current
reality instead of a number someone forgot to update by hand.

Each phase has a weight (rough effort share across the whole 9-10 month
timeline) and a list of checks. Each check is a (description, detector)
pair where detector is a function returning True/False.

TO EXTEND: as new milestones get built (e.g. once we start Phase 2), just
add new check tuples to the relevant phase's list below. Nothing else
needs to change — percentages recompute automatically.

Usage:
    python scripts/progress_report.py
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_SPLITS = PROJECT_ROOT / "data" / "splits"
MODELS = PROJECT_ROOT / "models"
TRANSLATION = PROJECT_ROOT / "translation_layer"
BACKEND = PROJECT_ROOT / "backend"
FRONTEND = PROJECT_ROOT / "frontend"
DOCS = PROJECT_ROOT / "docs"


# ---------- generic detector helpers ----------

def _real_files(folder: Path, exts=None):
    """Files in folder (recursive), excluding .gitkeep, optionally filtered by extension."""
    if not folder.exists():
        return []
    files = [p for p in folder.rglob("*") if p.is_file() and p.name != ".gitkeep"]
    if exts:
        files = [p for p in files if p.suffix.lower() in exts]
    return files


def dir_has_files(folder: Path, min_count: int = 1, exts=None):
    return lambda: len(_real_files(folder, exts)) >= min_count


def file_exists(path: Path):
    return lambda: path.exists() and path.stat().st_size > 0


def csv_has_min_rows(path: Path, min_rows: int):
    def check():
        if not path.exists():
            return False
        try:
            return len(pd.read_csv(path)) >= min_rows
        except Exception:
            return False
    return check


def any_file_matching(folder: Path, pattern: str):
    return lambda: folder.exists() and len(list(folder.rglob(pattern))) > 0


# ---------- phase definitions ----------
# weight = rough share of total project effort (should sum to 100)

PHASES = [
    {
        "name": "Phase 1 — Data Pipeline",
        "weight": 20,
        "checks": [
            ("ISL-CSLTR downloaded", dir_has_files(DATA_RAW / "isl_cslrt", 100)),
            ("CISLR downloaded", dir_has_files(DATA_RAW / "cislr", 10)),
            ("Fingerspelling dataset downloaded (supplementary)", dir_has_files(DATA_RAW / "fingerspelling", 10)),
            ("Clip index built (ISL-CSLTR sentence-level)", csv_has_min_rows(DATA_PROCESSED / "labels" / "clip_index.csv", 100)),
            ("Keypoints extracted from clips", dir_has_files(DATA_PROCESSED / "keypoints", 300, exts={".npy", ".npz"})),
            ("Train/val/test signer-based splits created", dir_has_files(DATA_SPLITS, 1)),
        ],
    },
    {
        "name": "Phase 2 — Baseline + Continuous Sequence Model",
        "weight": 25,
        "checks": [
            ("Baseline (isolated-sign) model code written", file_exists(MODELS / "baseline" / "train_baseline.py")),
            ("Baseline model trained (checkpoint saved)", any_file_matching(MODELS / "checkpoints", "*baseline*")),
            ("BiLSTM+CTC model code written", dir_has_files(MODELS / "continuous_ctc", 1, exts={".py"})),
            ("CTC model pretrained on CISLR", any_file_matching(MODELS / "checkpoints", "*pretrain*")),
            ("CTC model fine-tuned on ISL-CSLTR", any_file_matching(MODELS / "checkpoints", "*finetune*")),
        ],
    },
    {
        "name": "Phase 3 — Translation Layer",
        "weight": 15,
        "checks": [
            ("Rule-based gloss reordering implemented", dir_has_files(TRANSLATION / "rules", 1, exts={".py"})),
            ("Translation layer unit tests written", dir_has_files(TRANSLATION / "tests", 1, exts={".py"})),
        ],
    },
    {
        "name": "Phase 4 — Backend + Frontend Integration",
        "weight": 20,
        "checks": [
            ("FastAPI app entrypoint exists", any_file_matching(BACKEND, "main.py")),
            ("WebSocket streaming endpoint implemented", dir_has_files(BACKEND / "app" / "websocket", 1, exts={".py"})),
            ("Inference pipeline (sliding window / incremental decode) implemented", dir_has_files(BACKEND / "app" / "inference", 1, exts={".py"})),
            ("Frontend project scaffolded (Next.js)", file_exists(FRONTEND / "package.json")),
            ("Frontend webcam + client-side MediaPipe integration", dir_has_files(FRONTEND / "src", 3)),
        ],
    },
    {
        "name": "Phase 5 — Evaluation",
        "weight": 10,
        "checks": [
            ("WER / accuracy results recorded", any_file_matching(DOCS, "*evaluation*")),
            ("Latency measurements recorded", any_file_matching(DOCS, "*latency*")),
            ("Unseen-signer generalization results recorded", any_file_matching(DOCS, "*generalization*")),
        ],
    },
    {
        "name": "Phase 6 — Polish + Documentation/Report",
        "weight": 10,
        "checks": [
            ("Architecture decisions documented", file_exists(DOCS / "architecture-decisions.md")),
            ("Final report drafted", any_file_matching(DOCS, "*report*")),
        ],
    },
]


def main() -> None:
    print("=" * 70)
    print("ISL Translation System — Project Progress")
    print("=" * 70)

    overall_earned = 0.0
    overall_total = 0.0

    for phase in PHASES:
        checks = phase["checks"]
        results = [(desc, bool(fn())) for desc, fn in checks]
        done_count = sum(1 for _, ok in results if ok)
        phase_pct = 100 * done_count / len(results) if results else 0

        weight = phase["weight"]
        overall_earned += weight * (phase_pct / 100)
        overall_total += weight

        print(f"\n{phase['name']}  —  {phase_pct:.0f}% complete ({done_count}/{len(results)})")
        for desc, ok in results:
            mark = "[x]" if ok else "[ ]"
            print(f"  {mark} {desc}")

    overall_pct = 100 * overall_earned / overall_total if overall_total else 0

    print("\n" + "=" * 70)
    print(f"OVERALL PROJECT COMPLETION: {overall_pct:.1f}%")
    print("=" * 70)

    print("\nWhat's left (unchecked items, in phase order):")
    any_left = False
    for phase in PHASES:
        pending = [desc for desc, fn in phase["checks"] if not fn()]
        if pending:
            any_left = True
            print(f"\n  {phase['name']}:")
            for desc in pending:
                print(f"    - {desc}")
    if not any_left:
        print("  Nothing — every tracked milestone is complete!")

    print("\n(This scan reflects the current state of your project folder.")
    print(" Add new checks to scripts/progress_report.py as new milestones")
    print(" come up — nothing else needs to change.)")


if __name__ == "__main__":
    main()
