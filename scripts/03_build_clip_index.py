"""
Phase 1b — Build the ISL-CSLTR sentence-level clip index.

Turns the raw ISL_CSLRT_Corpus_frame_details.xlsx (one row per frame image)
into a clean, structured index: one row per CLIP, where a clip is a single
signer's performance of a single sentence, with its ordered frame sequence
and its ISL gloss label attached.

Frame path format (as shipped by the dataset, Windows-style separators):
    ISL_CSLRT_Corpus\\Frames_Sentence_Level\\<sentence>\\<signer_id>\\<sentence> <NN>.jpg

We treat (sentence, signer_id) as the unique clip key: all frames sharing
that folder pair belong to one continuous signing performance.

Output:
    data/processed/labels/clip_index.csv
    Columns: clip_id, sentence, gloss, signer_id, num_frames, frame_paths
    (frame_paths is a ';'-separated ordered list of absolute paths)

Usage:
    python scripts/03_build_clip_index.py
"""

import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = PROJECT_ROOT / "data" / "raw" / "isl_cslrt" / "ISL_CSLRT_Corpus"
CSV_DIR = CORPUS_DIR / "corpus_csv_files"
FRAMES_ROOT = CORPUS_DIR / "Frames_Sentence_Level"

FRAME_DETAILS_XLSX = CSV_DIR / "ISL_CSLRT_Corpus_frame_details.xlsx"
GLOSS_CSV = CSV_DIR / "ISL Corpus sign glosses.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "labels"
OUTPUT_CSV = OUTPUT_DIR / "clip_index.csv"

# Frame numbers appear in two conventions in this dataset:
#   "are you free today 01.jpg"          -> bare number before .jpg
#   "are you hiding something (3).jpg"   -> number wrapped in parentheses
# The optional `\)?` handles both.
FRAME_NUMBER_RE = re.compile(r"(\d+)\)?\.jpg$", re.IGNORECASE)


def normalize_sentence(s: str) -> str:
    """Lowercase + collapse whitespace, so minor formatting differences
    between the two label files don't break the join."""
    return " ".join(str(s).strip().lower().split())


def parse_frame_path(raw_path: str):
    """
    Given a path like:
        ISL_CSLRT_Corpus\\Frames_Sentence_Level\\are you free today\\1\\are you free today 01.jpg
    Return (signer_id, frame_number, resolved_absolute_path) or None if it
    doesn't match the expected shape.
    """
    normalized = raw_path.replace("\\", "/")
    parts = [p for p in normalized.split("/") if p]

    if len(parts) < 3:
        return None

    filename = parts[-1]
    signer_id = parts[-2]
    # sentence folder is parts[-3], but we trust the 'Sentence' column instead
    # of re-deriving it from the path, since that's the authoritative label

    match = FRAME_NUMBER_RE.search(filename)
    if not match:
        return None
    frame_number = int(match.group(1))

    # Resolve against actual dataset location, taking only the part of the
    # path from "Frames_Sentence_Level" onward, since the xlsx stores paths
    # rooted at ISL_CSLRT_Corpus which we already know the location of.
    try:
        idx = parts.index("Frames_Sentence_Level")
    except ValueError:
        return None
    relative_parts = parts[idx + 1:]  # sentence / signer / filename
    resolved = FRAMES_ROOT.joinpath(*relative_parts)

    return signer_id, frame_number, resolved


def main() -> None:
    print("Building ISL-CSLTR clip index...")

    if not FRAME_DETAILS_XLSX.exists():
        print(f"[ERROR] Not found: {FRAME_DETAILS_XLSX}")
        return
    if not GLOSS_CSV.exists():
        print(f"[ERROR] Not found: {GLOSS_CSV}")
        return

    frame_df = pd.read_excel(FRAME_DETAILS_XLSX)
    gloss_df = pd.read_csv(GLOSS_CSV)

    print(f"Loaded {len(frame_df)} frame rows, {len(gloss_df)} gloss rows.")

    # Check for exact duplicate rows in gloss csv (101 rows for 100 sentences
    # was flagged as worth checking)
    dup_count = gloss_df.duplicated().sum()
    if dup_count:
        print(f"[NOTE] {dup_count} fully duplicate row(s) in gloss CSV — dropping.")
        gloss_df = gloss_df.drop_duplicates()

    # Build normalized sentence -> gloss lookup
    gloss_df["_norm_sentence"] = gloss_df["Sentence"].apply(normalize_sentence)
    gloss_lookup = dict(zip(gloss_df["_norm_sentence"], gloss_df["SIGN GLOSSES"]))

    dup_sentences = gloss_df["_norm_sentence"].duplicated().sum()
    if dup_sentences:
        print(f"[WARNING] {dup_sentences} sentence(s) appear more than once in gloss CSV "
              f"with possibly different glosses — lookup keeps the last occurrence.")

    # Parse every frame row
    records = []
    unparsed = 0
    for _, row in frame_df.iterrows():
        parsed = parse_frame_path(str(row["Frames path"]))
        if parsed is None:
            unparsed += 1
            continue
        signer_id, frame_number, resolved_path = parsed
        records.append({
            "sentence_raw": row["Sentence"],
            "sentence_norm": normalize_sentence(row["Sentence"]),
            "signer_id": signer_id,
            "frame_number": frame_number,
            "frame_path": resolved_path,
        })

    if unparsed:
        print(f"[WARNING] {unparsed} frame row(s) had an unexpected path format and were skipped.")

    frames_clean = pd.DataFrame(records)
    print(f"Successfully parsed {len(frames_clean)} frame rows.")

    # Group into clips: one row per (sentence, signer_id)
    clips = []
    missing_gloss = set()
    missing_files = 0

    grouped = frames_clean.groupby(["sentence_norm", "signer_id"])
    for (sentence_norm, signer_id), group in grouped:
        group_sorted = group.sort_values("frame_number")
        frame_paths = list(group_sorted["frame_path"])

        gloss = gloss_lookup.get(sentence_norm)
        if gloss is None:
            missing_gloss.add(sentence_norm)

        existing_paths = [p for p in frame_paths if p.exists()]
        missing_files += (len(frame_paths) - len(existing_paths))

        clips.append({
            "clip_id": f"{sentence_norm.replace(' ', '_')}__signer{signer_id}",
            "sentence": group_sorted["sentence_raw"].iloc[0],
            "gloss": gloss,
            "signer_id": signer_id,
            "num_frames": len(frame_paths),
            "num_frames_found_on_disk": len(existing_paths),
            "frame_paths": ";".join(str(p) for p in frame_paths),
        })

    clip_df = pd.DataFrame(clips)

    print(f"\nBuilt {len(clip_df)} clips across "
          f"{clip_df['sentence'].nunique()} distinct sentences and "
          f"{clip_df['signer_id'].nunique()} distinct signer IDs.")
    print(f"Signer IDs found: {sorted(clip_df['signer_id'].unique(), key=lambda x: (len(x), x))}")

    if missing_gloss:
        print(f"\n[WARNING] {len(missing_gloss)} sentence(s) had no matching gloss entry:")
        for s in list(missing_gloss)[:10]:
            print(f"    '{s}'")
        if len(missing_gloss) > 10:
            print(f"    ... and {len(missing_gloss) - 10} more")

    if missing_files:
        print(f"\n[WARNING] {missing_files} frame file(s) listed in the index do not exist on disk "
              f"(path resolution mismatch, or files genuinely missing).")

    clips_with_full_frames = (clip_df["num_frames"] == clip_df["num_frames_found_on_disk"]).sum()
    print(f"\nClips with 100% of frames found on disk: {clips_with_full_frames} / {len(clip_df)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clip_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved clip index to: {OUTPUT_CSV}")

    print("\nSample rows:")
    with pd.option_context("display.max_colwidth", 60):
        print(clip_df[["clip_id", "sentence", "gloss", "signer_id", "num_frames"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
