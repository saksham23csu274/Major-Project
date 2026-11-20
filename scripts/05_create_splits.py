"""
Phase 1d — Train/val/test splits, held out by SIGNER (not by clip).

Why signer-disjoint, not a random clip-level split:
  RQ4 asks whether the model generalizes to signers it has never seen.
  A random split over clips would let the same signer's other sentences
  leak into training, so the model could partly learn to recognize that
  signer's individual signing style rather than the sign language itself.
  Holding out entire signers for val/test is the only way to honestly
  measure generalization to new people.

Default split (7 signers in ISL-CSLTR):
    Train: signers 1, 2, 3, 4, 5
    Val:   signer  6
    Test:  signer  7
  This is a fixed, documented choice rather than a random draw, so the
  split is reproducible and identical every time this script runs.
  Override with --val-signers / --test-signers if you want to experiment
  with a different holdout (e.g. to check whether results are sensitive
  to *which* signer gets held out).

Only clips that were successfully processed by 04_extract_keypoints.py
(status == "ok" or "already_done" in the manifest, with a real .npy file
on disk) are included — this automatically drops the one known-bad clip
(i_am_hungry__signer3) without needing a manual exclusion list.

Usage:
    python scripts/05_create_splits.py
    python scripts/05_create_splits.py --val-signers 6 --test-signers 7
    python scripts/05_create_splits.py --val-signers 4 --test-signers 2,7
"""

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLIP_INDEX = PROJECT_ROOT / "data" / "processed" / "labels" / "clip_index.csv"
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "labels" / "keypoints_manifest.csv"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

DEFAULT_VAL_SIGNERS = {6}
DEFAULT_TEST_SIGNERS = {7}


def parse_signer_list(s: str) -> set:
    return {int(x.strip()) for x in s.split(",") if x.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-signers", type=str, default=None,
                         help="Comma-separated signer IDs for validation, e.g. '6'")
    parser.add_argument("--test-signers", type=str, default=None,
                         help="Comma-separated signer IDs for test, e.g. '7'")
    args = parser.parse_args()

    val_signers = parse_signer_list(args.val_signers) if args.val_signers else DEFAULT_VAL_SIGNERS
    test_signers = parse_signer_list(args.test_signers) if args.test_signers else DEFAULT_TEST_SIGNERS

    overlap = val_signers & test_signers
    if overlap:
        print(f"[ERROR] Signers {overlap} appear in both val and test — a signer must be in exactly one split.")
        return

    if not CLIP_INDEX.exists() or not MANIFEST_PATH.exists():
        print("[ERROR] clip_index.csv or keypoints_manifest.csv not found. "
              "Run 03_build_clip_index.py and 04_extract_keypoints.py first.")
        return

    clip_df = pd.read_csv(CLIP_INDEX)
    manifest_df = pd.read_csv(MANIFEST_PATH)

    # Drop clips with a missing/NaN gloss (e.g. "which college school are
    # you from", which had no matching entry in the gloss CSV due to a
    # source-data text mismatch -- see 03c_report_gaps.py). These clips
    # have real keypoints but no usable training label, so they can't be
    # included in any split.
    missing_gloss_mask = clip_df["gloss"].isna()
    if missing_gloss_mask.any():
        dropped_sentences = clip_df.loc[missing_gloss_mask, "sentence"].unique()
        print(f"Dropping {missing_gloss_mask.sum()} clip(s) with no gloss label "
              f"(sentences: {list(dropped_sentences)}).")
        clip_df = clip_df[~missing_gloss_mask]

    # Keep only clips that actually have usable keypoints on disk.
    usable_statuses = {"ok", "already_done"}
    manifest_df = manifest_df[manifest_df["status"].isin(usable_statuses)]
    manifest_df = manifest_df[manifest_df["keypoints_path"].notna()]

    merged = clip_df.merge(
        manifest_df[["clip_id", "keypoints_path"]],
        on="clip_id",
        how="inner",  # drops any clip without a successful keypoints file
    )

    dropped = len(clip_df) - len(merged)
    if dropped:
        print(f"Dropped {dropped} clip(s) with no usable keypoints file "
              f"(e.g. the known i_am_hungry__signer3 extraction failure).")

    all_signers = set(merged["signer_id"].unique())
    unexpected = (val_signers | test_signers) - all_signers
    if unexpected:
        print(f"[WARNING] Requested val/test signers {unexpected} don't exist in the data. Check --val-signers/--test-signers.")

    train_signers = all_signers - val_signers - test_signers

    def assign_split(signer_id):
        if signer_id in val_signers:
            return "val"
        elif signer_id in test_signers:
            return "test"
        else:
            return "train"

    merged["split"] = merged["signer_id"].apply(assign_split)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for split_name in ["train", "val", "test"]:
        split_df = merged[merged["split"] == split_name].drop(columns=["split"])
        out_path = SPLITS_DIR / f"{split_name}.csv"
        split_df.to_csv(out_path, index=False)

    print("\nSplit summary:")
    print(f"  Train signers: {sorted(int(s) for s in train_signers)}")
    print(f"  Val signers:   {sorted(int(s) for s in val_signers)}")
    print(f"  Test signers:  {sorted(int(s) for s in test_signers)}")
    print()
    for split_name in ["train", "val", "test"]:
        split_df = merged[merged["split"] == split_name]
        n_clips = len(split_df)
        n_sentences = split_df["sentence"].nunique()
        n_frames = split_df["num_frames"].sum()
        print(f"  {split_name:5s}: {n_clips:4d} clips, {n_sentences:3d} distinct sentences, {n_frames:6d} total frames")

    # Sanity check: how many sentences appear in ALL THREE splits (fully
    # covered) vs. only in some — informative for interpreting later
    # results, not an error either way.
    sentence_splits = merged.groupby("sentence")["split"].apply(set)
    all_three = (sentence_splits.apply(lambda s: s == {"train", "val", "test"})).sum()
    print(f"\n  Sentences appearing in all 3 splits (different signers): {all_three} / {merged['sentence'].nunique()}")

    print(f"\nSaved to: {SPLITS_DIR}/{{train,val,test}}.csv")


if __name__ == "__main__":
    main()
