"""
Phase 1a verification step, round 2 — ISL-CSLTR label files.

Now that we know ISL-CSLTR's folder layout (from 01_inspect_dataset.py),
this script looks specifically at:
  - the CSV/XLSX label files in corpus_csv_files/ (columns + sample rows)
  - a sample of actual filenames inside Videos_Sentence_Level/,
    Frames_Word_Level/, and Frames_Sentence_Level/

so we can confirm exactly how video files map to sentence/gloss labels
before writing the real preprocessing pipeline.

Usage:
    python scripts/02_inspect_isl_cslrt_labels.py
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = PROJECT_ROOT / "data" / "raw" / "isl_cslrt" / "ISL_CSLRT_Corpus"
CSV_DIR = CORPUS_DIR / "corpus_csv_files"

TABLE_FILES = [
    "ISL Corpus sign glosses.csv",
    "ISL_CSLRT_Corpus details.xlsx",
    "ISL_CSLRT_Corpus_frame_details.xlsx",
    "ISL_CSLRT_Corpus_word_details.xlsx",
]

MEDIA_FOLDERS = [
    "Videos_Sentence_Level",
    "Frames_Word_Level",
    "Frames_Sentence_Level",
]

SAMPLE_ROWS = 5
SAMPLE_FILENAMES = 10


def print_header(text: str) -> None:
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def load_table(path: Path) -> pd.DataFrame | None:
    try:
        if path.suffix.lower() == ".csv":
            # try a couple common encodings, ISL-CSLTR CSVs have been reported
            # to sometimes use latin-1
            for enc in ("utf-8", "latin-1"):
                try:
                    return pd.read_csv(path, encoding=enc)
                except UnicodeDecodeError:
                    continue
            return pd.read_csv(path, encoding="utf-8", errors="replace")
        elif path.suffix.lower() == ".xlsx":
            return pd.read_excel(path)
    except Exception as e:
        print(f"  [ERROR loading {path.name}]: {e}")
        return None
    return None


def inspect_table_files() -> None:
    print_header("Label / metadata files in corpus_csv_files/")
    if not CSV_DIR.exists():
        print(f"  [MISSING] {CSV_DIR} does not exist.")
        return

    for fname in TABLE_FILES:
        path = CSV_DIR / fname
        print(f"\n--- {fname} ---")
        if not path.exists():
            print("  [MISSING]")
            continue

        df = load_table(path)
        if df is None:
            continue

        print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns")
        print(f"  Columns: {list(df.columns)}")
        print(f"  First {SAMPLE_ROWS} rows:")
        with pd.option_context("display.max_columns", None, "display.width", 160):
            print(df.head(SAMPLE_ROWS).to_string(index=False))

        # flag any obviously null-heavy columns, useful to know early
        null_frac = df.isnull().mean()
        heavy_nulls = null_frac[null_frac > 0.3]
        if not heavy_nulls.empty:
            print(f"  Columns with >30% missing values: {dict(heavy_nulls.round(2))}")


def inspect_media_folders() -> None:
    print_header("Media folder filename samples")
    for folder_name in MEDIA_FOLDERS:
        folder = CORPUS_DIR / folder_name
        print(f"\n--- {folder_name} ---")
        if not folder.exists():
            print("  [MISSING]")
            continue

        all_files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() != ".db"]
        print(f"  Non-.db file count: {len(all_files)}")

        # show whether files sit directly in this folder or in subfolders
        subdirs = [p for p in folder.iterdir() if p.is_dir()]
        print(f"  Immediate subdirectories: {len(subdirs)}")
        if subdirs:
            print(f"  Sample subdirectory names: {[d.name for d in subdirs[:SAMPLE_FILENAMES]]}")

        print(f"  Sample filenames:")
        for f in all_files[:SAMPLE_FILENAMES]:
            print(f"    {f.relative_to(folder)}")


def main() -> None:
    print("ISL-CSLTR — Deep Label Inspection")
    inspect_table_files()
    inspect_media_folders()

    print_header("Next step")
    print("Paste this output back. It tells us the exact schema linking")
    print("video/frame filenames to sentence text and gloss sequences,")
    print("which is what the keypoint-extraction + label-alignment script")
    print("needs to be built against.")


if __name__ == "__main__":
    main()
