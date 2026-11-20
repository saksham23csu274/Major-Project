"""
Phase 2 — Build the isolated-sign (word-level) index for the baseline model.

Unlike the sentence-level data, ISL-CSLTR's word-level images have no
signer subfolder structure -- word_details.xlsx just maps each image
directly to its word label (e.g. "HEAR/HEAR (2).jpg" -> "HEAR"). Each
image is treated as one independent training INSTANCE (not a temporal
sequence), which is a deliberate design choice for the baseline: real
sign recognition needs the temporal/motion information a single frame
can't capture, so a single-frame classifier is expected to perform much
worse than the continuous sequence model. That gap is exactly the point
of having a baseline -- it's what RQ1 and RQ3 are measuring against.

Usage:
    python scripts/09_build_word_index.py
"""

import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = PROJECT_ROOT / "data" / "raw" / "isl_cslrt" / "ISL_CSLRT_Corpus"
WORD_DETAILS_XLSX = CORPUS_DIR / "corpus_csv_files" / "ISL_CSLRT_Corpus_word_details.xlsx"
FRAMES_WORD_ROOT = CORPUS_DIR / "Frames_Word_Level"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "labels"
OUTPUT_CSV = OUTPUT_DIR / "word_index.csv"


def resolve_path(raw_path: str) -> Path:
    normalized = str(raw_path).replace("\\", "/")
    parts = [p for p in normalized.split("/") if p]
    try:
        idx = parts.index("Frames_Word_Level")
    except ValueError:
        return None
    relative_parts = parts[idx + 1:]
    return FRAMES_WORD_ROOT.joinpath(*relative_parts)


def main() -> None:
    if not WORD_DETAILS_XLSX.exists():
        print(f"[ERROR] Not found: {WORD_DETAILS_XLSX}")
        return

    df = pd.read_excel(WORD_DETAILS_XLSX)
    print(f"Loaded {len(df)} rows. Distinct words: {df['Word'].nunique()}")

    records = []
    missing = 0
    for i, row in df.iterrows():
        resolved = resolve_path(row["Frames path"])
        if resolved is None or not resolved.exists():
            missing += 1
            continue
        records.append({
            "instance_id": f"{str(row['Word']).strip().replace(' ', '_')}__{i}",
            "word": str(row["Word"]).strip().upper(),
            "frame_path": str(resolved),
        })

    index_df = pd.DataFrame(records)
    print(f"\nMatched {len(index_df)} / {len(df)} rows to real image files.")
    if missing:
        print(f"[WARNING] {missing} rows had no matching file on disk.")

    word_counts = index_df["word"].value_counts()
    print(f"\nDistinct word classes: {index_df['word'].nunique()}")
    print(f"Instances per class: min={word_counts.min()}, max={word_counts.max()}, "
          f"median={word_counts.median():.0f}")

    # Flag classes with very few instances -- these will be hard/impossible
    # to reliably train and evaluate on with a random split.
    rare_classes = word_counts[word_counts < 2]
    if len(rare_classes) > 0:
        print(f"\n[NOTE] {len(rare_classes)} word class(es) have only 1 instance "
              f"(can't be split across train/val/test). These will need to be "
              f"either excluded or handled specially in the next step.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    index_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
