"""
CISLR dataset inspection — same approach as 01/02 for ISL-CSLTR: report
the real on-disk structure before writing any preprocessing code against
assumptions. CISLR's Hugging Face dataset viewer showed columns
(uid, gloss, duration, category) in what looked like a single "test"
split of ~2,285 rows, but that's the auto-converted Parquet view — the
raw downloaded files (via `hf download`) may be organized differently
(separate CSVs + video files, or something else entirely).

Usage:
    python scripts/06_inspect_cislr.py
"""

import os
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CISLR_DIR = PROJECT_ROOT / "data" / "raw" / "cislr"

MAX_TREE_LINES = 60
ANNOTATION_EXTENSIONS = {".csv", ".txt", ".json", ".tsv", ".xml", ".jsonl"}
PREVIEW_LINES = 10


def print_header(text: str) -> None:
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def main() -> None:
    print_header(f"CISLR inventory -> {CISLR_DIR}")

    if not CISLR_DIR.exists():
        print("[MISSING] Folder does not exist.")
        return

    all_files = [p for p in CISLR_DIR.rglob("*") if p.is_file()]
    if not all_files:
        print("[EMPTY] Folder exists but contains no files.")
        return

    ext_counts = Counter(p.suffix.lower() for p in all_files)
    print(f"Total files: {len(all_files)}")
    print("File extension breakdown:")
    for ext, count in ext_counts.most_common(20):
        ext_label = ext if ext else "(no extension)"
        print(f"  {ext_label:15s} {count}")

    print("\nFolder structure (depth-limited sample):")
    lines_printed = 0
    for root, dirs, files in os.walk(CISLR_DIR):
        # skip hidden/cache dirs like .cache created by huggingface_hub
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        depth = len(Path(root).relative_to(CISLR_DIR).parts)
        if depth > 2:
            dirs[:] = []
            continue
        indent = "  " + "  " * depth
        rel = Path(root).relative_to(CISLR_DIR)
        print(f"{indent}{rel if str(rel) != '.' else CISLR_DIR.name}/")
        lines_printed += 1
        for f in sorted(files)[:8]:
            print(f"{indent}  {f}")
            lines_printed += 1
            if lines_printed >= MAX_TREE_LINES:
                print("  ... (truncated)")
                break
        if lines_printed >= MAX_TREE_LINES:
            break

    annotation_files = [p for p in all_files if p.suffix.lower() in ANNOTATION_EXTENSIONS]
    if annotation_files:
        print(f"\nFound {len(annotation_files)} potential annotation file(s). Previewing up to 3:")
        for sample in annotation_files[:3]:
            print(f"\n--- {sample.relative_to(CISLR_DIR)} ---")
            try:
                with open(sample, "r", errors="replace") as fh:
                    for i, line in enumerate(fh):
                        if i >= PREVIEW_LINES:
                            print("  ...")
                            break
                        print(f"  {line.rstrip()}")
            except Exception as e:
                print(f"  [could not read: {e}]")
    else:
        print("\nNo obvious annotation files (.csv/.txt/.json/.tsv/.xml/.jsonl) found.")

    # Video/media files specifically, since that's what we'll need to run
    # MediaPipe extraction against
    media_exts = {".mp4", ".avi", ".mov", ".webm", ".mkv"}
    media_files = [p for p in all_files if p.suffix.lower() in media_exts]
    print(f"\nVideo/media files found: {len(media_files)}")
    if media_files:
        print("Sample paths:")
        for f in media_files[:5]:
            print(f"  {f.relative_to(CISLR_DIR)}")


if __name__ == "__main__":
    main()
