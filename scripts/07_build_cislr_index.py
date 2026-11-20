"""
Phase 2 (pretraining prep) — Build the CISLR clip index.

Unlike ISL-CSLTR (sentence-level, needed grouping by signer+sentence),
CISLR is already a flat, isolated-word dataset: one row per video, one
gloss label per video. This script verifies that every video file
referenced in dataset.csv actually exists on disk (filename = uid +
".mp4" -- confirmed by testing against the real file listing, which
matched 7050/7050 exactly; an initial guess that a "_" prefix was always
added was tested and disproved), and produces a clean index for the
keypoint extraction step.

SCOPING NOTE: CISLR ships its own prototype.csv/test.csv split for the
paper's one-shot prototype-learning benchmark -- a different task from
ours (we're using CISLR for standard supervised pretraining of our
BiLSTM encoder, not one-shot classification). We intentionally use the
full dataset.csv and create our own train/val split later, rather than
CISLR's official split, since it doesn't match our use case.

Usage:
    python scripts/07_build_cislr_index.py
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CISLR_DIR = PROJECT_ROOT / "data" / "raw" / "cislr"
DATASET_CSV = CISLR_DIR / "dataset.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "labels"
OUTPUT_CSV = OUTPUT_DIR / "cislr_index.csv"


def build_video_filename_index() -> dict:
    """
    Scans the CISLR video folder ONCE and builds a filename -> path lookup,
    rather than re-walking the directory tree for every one of 7,050 rows
    (which would be extremely slow). We search recursively once here for
    the same reason noted in the module docstring: zip extraction nesting
    can vary.
    """
    index = {}
    for path in CISLR_DIR.rglob("*.mp4"):
        index[path.name] = path
    return index


def main() -> None:
    if not DATASET_CSV.exists():
        print(f"[ERROR] {DATASET_CSV} not found.")
        return

    df = pd.read_csv(DATASET_CSV)
    print(f"Loaded {len(df)} rows from dataset.csv")
    print(f"Distinct gloss words: {df['gloss'].nunique()}")

    print("\nScanning video folder once to build a filename index...")
    filename_index = build_video_filename_index()
    print(f"Found {len(filename_index)} .mp4 files on disk.")

    # Naming convention confirmed against real data: filename is simply
    # uid + ".mp4", with NO prefix. (An earlier guess that a "_" was always
    # prepended was wrong -- checked and disproved against the real file
    # listing: some filenames appear to start with "_" only because that
    # particular uid itself begins with "_", not because of a safety
    # prefix. Direct uid+".mp4" matched 7050/7050 real files exactly.)
    sample = df.head(5)
    print("\nVerifying naming convention on a sample of 5:")
    sample_ok = True
    for _, row in sample.iterrows():
        expected_filename = f"{row['uid']}.mp4"
        found = filename_index.get(expected_filename)
        status = f"FOUND: {found}" if found else "NOT FOUND"
        print(f"  uid={row['uid']!r} -> expected {expected_filename!r} -> {status}")
        if found is None:
            sample_ok = False

    if not sample_ok:
        print("\n[WARNING] At least one sample file was not found. Stopping before "
              "processing all rows -- check the naming convention before proceeding.")
        return

    print("\nSample verified. Matching all rows against the file index...")

    records = []
    not_found_count = 0
    for _, row in df.iterrows():
        expected_filename = f"{row['uid']}.mp4"
        video_path = filename_index.get(expected_filename)
        if video_path is None:
            not_found_count += 1
            continue
        records.append({
            "uid": row["uid"],
            "gloss": str(row["gloss"]).upper().strip(),
            "duration": row["duration"],
            "category": row["category"],
            "video_path": str(video_path),
        })

    index_df = pd.DataFrame(records)
    print(f"\nMatched {len(index_df)} / {len(df)} rows to real video files.")
    if not_found_count:
        print(f"[WARNING] {not_found_count} rows had no matching video file.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    index_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved CISLR index to: {OUTPUT_CSV}")
    print(f"Distinct gloss vocabulary in matched rows: {index_df['gloss'].nunique()}")


if __name__ == "__main__":
    main()
