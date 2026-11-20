"""
Phase 1a verification step.

Walks data/raw/<dataset>/ for each expected dataset and prints back:
  - folder depth / structure sample
  - file extension counts
  - a preview of any CSV/text annotation files found

Run this BEFORE writing any keypoint-extraction or label-parsing code.
The goal is to confirm ground truth about each dataset's actual on-disk
layout (which varies by download/version) rather than assuming it.

Usage:
    python scripts/01_inspect_dataset.py
"""

import os
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

DATASETS = {
    "isl_cslrt": "ISL-CSLTR (sentence-level)",
    "cislr": "CISLR (word-level)",
    "fingerspelling": "Continuous Fingerspelling Dataset",
}

MAX_TREE_LINES = 40
ANNOTATION_EXTENSIONS = {".csv", ".txt", ".json", ".tsv", ".xml"}
PREVIEW_LINES = 8


def print_header(text: str) -> None:
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def walk_and_report(dataset_dir: Path, label: str) -> None:
    print_header(f"{label} -> {dataset_dir}")

    if not dataset_dir.exists():
        print("  [MISSING] Folder does not exist yet.")
        return

    all_files = [p for p in dataset_dir.rglob("*") if p.is_file()]
    if not all_files:
        print("  [EMPTY] Folder exists but contains no files.")
        print("  -> Did you complete the manual download step in")
        print("     docs/phase1-data-acquisition.md ?")
        return

    ext_counts = Counter(p.suffix.lower() for p in all_files)
    print(f"  Total files: {len(all_files)}")
    print("  File extension breakdown:")
    for ext, count in ext_counts.most_common():
        ext_label = ext if ext else "(no extension)"
        print(f"    {ext_label:15s} {count}")

    # Directory tree sample (depth-limited, line-limited)
    print("\n  Folder structure (sample, depth-limited):")
    lines_printed = 0
    for root, dirs, files in os.walk(dataset_dir):
        depth = len(Path(root).relative_to(dataset_dir).parts)
        if depth > 2:
            dirs[:] = []  # don't descend further
            continue
        indent = "    " + "  " * depth
        rel = Path(root).relative_to(dataset_dir)
        print(f"{indent}{rel if str(rel) != '.' else dataset_dir.name}/")
        lines_printed += 1
        for f in sorted(files)[:5]:
            print(f"{indent}  {f}")
            lines_printed += 1
            if lines_printed >= MAX_TREE_LINES:
                print("    ... (truncated)")
                break
        if lines_printed >= MAX_TREE_LINES:
            break

    # Preview any annotation-like files
    annotation_files = [p for p in all_files if p.suffix.lower() in ANNOTATION_EXTENSIONS]
    if annotation_files:
        print(f"\n  Found {len(annotation_files)} potential annotation file(s). Previewing first one:")
        sample = annotation_files[0]
        print(f"  --- {sample.relative_to(dataset_dir)} ---")
        try:
            with open(sample, "r", errors="replace") as fh:
                for i, line in enumerate(fh):
                    if i >= PREVIEW_LINES:
                        print("    ...")
                        break
                    print(f"    {line.rstrip()}")
        except Exception as e:
            print(f"    [could not read file: {e}]")
    else:
        print("\n  No obvious annotation files (.csv/.txt/.json/.tsv/.xml) found at any depth.")
        print("  -> This dataset may store labels differently (e.g. filename encodes label,")
        print("     or labels are in a nested subfolder we didn't print above).")


def main() -> None:
    print("ISL Translation System — Dataset Inventory")
    print(f"Scanning: {RAW_DIR}")

    if not RAW_DIR.exists():
        print(f"[ERROR] {RAW_DIR} does not exist. Are you running this from the project root?")
        return

    for folder_name, label in DATASETS.items():
        walk_and_report(RAW_DIR / folder_name, label)

    print_header("Next step")
    print("Share this script's output back so we can write the real keypoint")
    print("extraction + label alignment logic against the actual file layout,")
    print("instead of guessing.")


if __name__ == "__main__":
    main()
