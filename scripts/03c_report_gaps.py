"""
Quick gap report on clip_index.csv — identifies:
  1. Which sentences from the gloss CSV never showed up in the frame data
     (100 gloss sentences vs 97 recovered -> which 3 are missing)
  2. Which specific clip has missing frame files on disk (662/663 clean)

Usage:
    python scripts/03c_report_gaps.py
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = PROJECT_ROOT / "data" / "raw" / "isl_cslrt" / "ISL_CSLRT_Corpus"
GLOSS_CSV = CORPUS_DIR / "corpus_csv_files" / "ISL Corpus sign glosses.csv"
CLIP_INDEX = PROJECT_ROOT / "data" / "processed" / "labels" / "clip_index.csv"


def normalize(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


def main() -> None:
    gloss_df = pd.read_csv(GLOSS_CSV).drop_duplicates()
    clip_df = pd.read_csv(CLIP_INDEX)

    gloss_sentences = set(gloss_df["Sentence"].apply(normalize))
    clip_sentences = set(clip_df["sentence"].apply(normalize))

    missing_from_clips = gloss_sentences - clip_sentences
    print(f"Sentences in gloss CSV: {len(gloss_sentences)}")
    print(f"Sentences recovered in clip index: {len(clip_sentences)}")
    print(f"\nSentences in gloss CSV but NOT found in any clip ({len(missing_from_clips)}):")
    for s in sorted(missing_from_clips):
        print(f"  - {s!r}")

    print("\n" + "=" * 70)
    incomplete = clip_df[clip_df["num_frames"] != clip_df["num_frames_found_on_disk"]]
    print(f"Clips with missing frame files on disk: {len(incomplete)}")
    for _, row in incomplete.iterrows():
        missing_n = row["num_frames"] - row["num_frames_found_on_disk"]
        print(f"  clip_id={row['clip_id']}  expected={row['num_frames']}  "
              f"found={row['num_frames_found_on_disk']}  missing={missing_n}")


if __name__ == "__main__":
    main()
