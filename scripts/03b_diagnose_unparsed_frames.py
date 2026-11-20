"""
Diagnostic follow-up to 03_build_clip_index.py.

That script skipped 6,808 / 18,863 frame rows as "unexpected path format."
This script shows us the RAW, unmodified path strings for a sample of the
skipped rows, categorized by *why* they were skipped, so we can fix the
parser against reality instead of guessing again.

Usage:
    python scripts/03b_diagnose_unparsed_frames.py
"""

import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = PROJECT_ROOT / "data" / "raw" / "isl_cslrt" / "ISL_CSLRT_Corpus"
CSV_DIR = CORPUS_DIR / "corpus_csv_files"
FRAME_DETAILS_XLSX = CSV_DIR / "ISL_CSLRT_Corpus_frame_details.xlsx"

FRAME_NUMBER_RE = re.compile(r"(\d+)\)?\.jpg$", re.IGNORECASE)
SAMPLES_PER_CATEGORY = 8


def classify(raw_path: str) -> str:
    normalized = str(raw_path).replace("\\", "/")
    parts = [p for p in normalized.split("/") if p]

    if len(parts) < 3:
        return "too_few_path_segments"

    filename = parts[-1]
    if not FRAME_NUMBER_RE.search(filename):
        return "filename_no_trailing_number_jpg"

    if "Frames_Sentence_Level" not in parts:
        return "missing_frames_sentence_level_token"

    return "PARSED_OK"


def main() -> None:
    frame_df = pd.read_excel(FRAME_DETAILS_XLSX)
    print(f"Loaded {len(frame_df)} frame rows.\n")

    frame_df["_raw_path"] = frame_df["Frames path"].astype(str)
    frame_df["_category"] = frame_df["_raw_path"].apply(classify)

    counts = frame_df["_category"].value_counts()
    print("Breakdown by category:")
    print(counts.to_string())
    print()

    for category, count in counts.items():
        if category == "PARSED_OK":
            continue
        print("=" * 70)
        print(f"Category: {category}  ({count} rows)")
        print("=" * 70)
        sample = frame_df[frame_df["_category"] == category].head(SAMPLES_PER_CATEGORY)
        for _, row in sample.iterrows():
            print(f"  Sentence: {row['Sentence']!r}")
            print(f"  Raw path: {row['_raw_path']!r}")
            print()


if __name__ == "__main__":
    main()
