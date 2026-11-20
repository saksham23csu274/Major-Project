"""
Phase 2 (pretraining prep) — MediaPipe keypoint extraction for CISLR.

Unlike ISL-CSLTR (pre-extracted frame images), CISLR gives raw .mp4 video
files, so this script decodes frames itself via OpenCV before running the
same PoseLandmarker + HandLandmarker pipeline used in 04_extract_keypoints.py
(same 228-dim feature vector, same normalization, same reasons for using
separate landmarkers instead of the buggy combined Holistic bundle on
macOS -- see that script's docstring for the full explanation).

IMPORTANT — LONG RUNTIME: with 7,050 videos, this can realistically take
many hours (ISL-CSLTR's 18,863 frames took ~13 minutes at ~24 frames/sec;
CISLR likely has well over 1 million total frames). This script is
designed to survive being interrupted:
  - Progress is written to the manifest CSV incrementally (flushed after
    every video), not just once at the end -- an interruption never loses
    more than the single video in progress.
  - Already-completed videos (present in the manifest with status "ok")
    are skipped on re-run, so you can stop and resume across sessions.

RECOMMENDED: run this with `caffeinate` on macOS so the laptop doesn't
sleep partway through:
    caffeinate -i python scripts/08_extract_cislr_keypoints.py

Usage:
    python scripts/08_extract_cislr_keypoints.py [--limit N] [--overwrite]
"""

import argparse
import csv
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CISLR_INDEX = PROJECT_ROOT / "data" / "processed" / "labels" / "cislr_index.csv"
KEYPOINTS_DIR = PROJECT_ROOT / "data" / "processed" / "keypoints_cislr"
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "labels" / "cislr_keypoints_manifest.csv"

MODEL_DIR = PROJECT_ROOT / "models" / "mediapipe"
POSE_MODEL_PATH = MODEL_DIR / "pose_landmarker_full.task"
HAND_MODEL_PATH = MODEL_DIR / "hand_landmarker.task"

POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

NUM_POSE = 33
NUM_HAND = 21
FEATURE_DIM = (NUM_POSE + 2 * NUM_HAND) * 3 + 3  # 228, same as 04_extract_keypoints.py

LEFT_SHOULDER_IDX = 11
RIGHT_SHOULDER_IDX = 12

MAX_FRAMES_PER_VIDEO = 300  # safety cap -- a handful of CISLR videos may be
                             # unexpectedly long; this bounds worst-case
                             # per-video runtime without affecting the vast
                             # majority of normal isolated-sign clips


def _download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading model bundle...\n  From: {url}\n  To:   {dest}")
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as e:
        print(f"[ERROR] Could not download model: {e}")
        sys.exit(1)
    print(f"Downloaded ({dest.stat().st_size / 1e6:.1f} MB).")


def ensure_models_downloaded() -> tuple:
    _download(POSE_MODEL_URL, POSE_MODEL_PATH)
    _download(HAND_MODEL_URL, HAND_MODEL_PATH)
    return POSE_MODEL_PATH, HAND_MODEL_PATH


# ---------- Feature vector construction (identical logic to 04_extract_keypoints.py) ----------

def landmarks_to_array(landmark_list, expected_count: int):
    if not landmark_list:
        return np.zeros((expected_count, 3), dtype=np.float32), False
    arr = np.array([[lm.x, lm.y, lm.z] for lm in landmark_list], dtype=np.float32)
    if arr.shape[0] != expected_count:
        return np.zeros((expected_count, 3), dtype=np.float32), False
    return arr, True


def normalize_frame(pose_arr, left_arr, right_arr, pose_present, left_present, right_present):
    if not pose_present:
        return pose_arr, left_arr, right_arr
    left_shoulder = pose_arr[LEFT_SHOULDER_IDX]
    right_shoulder = pose_arr[RIGHT_SHOULDER_IDX]
    center = (left_shoulder + right_shoulder) / 2.0
    scale = np.linalg.norm(left_shoulder - right_shoulder)
    scale = max(scale, 1e-6)
    pose_norm = (pose_arr - center) / scale
    left_norm = (left_arr - center) / scale if left_present else left_arr
    right_norm = (right_arr - center) / scale if right_present else right_arr
    return pose_norm, left_norm, right_norm


def build_feature_vector(pose_landmarks, left_hand_landmarks, right_hand_landmarks) -> np.ndarray:
    pose_arr, pose_present = landmarks_to_array(pose_landmarks, NUM_POSE)
    left_arr, left_present = landmarks_to_array(left_hand_landmarks, NUM_HAND)
    right_arr, right_present = landmarks_to_array(right_hand_landmarks, NUM_HAND)
    pose_arr, left_arr, right_arr = normalize_frame(
        pose_arr, left_arr, right_arr, pose_present, left_present, right_present
    )
    presence = np.array([float(pose_present), float(left_present), float(right_present)], dtype=np.float32)
    return np.concatenate([pose_arr.flatten(), left_arr.flatten(), right_arr.flatten(), presence])


def process_video(pose_landmarker, hand_landmarker, video_path: str) -> np.ndarray:
    import mediapipe as mp

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0  # sane fallback -- some codecs report 0 for FPS metadata
    ms_per_frame = 1000.0 / fps

    frame_vectors = []
    frame_idx = 0
    try:
        while frame_idx < MAX_FRAMES_PER_VIDEO:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # Using frame_idx * ms_per_frame rather than trusting
            # CAP_PROP_POS_MSEC directly -- some codecs report unreliable
            # (sometimes non-increasing) timestamps there, and MediaPipe's
            # VIDEO mode requires strictly increasing timestamps or it
            # raises an error.
            timestamp_ms = int(frame_idx * ms_per_frame)

            pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
            hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

            pose_lms = pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None

            left_lms, right_lms = None, None
            for hand_lms, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
                label = handedness[0].category_name
                if label == "Left":
                    left_lms = hand_lms
                elif label == "Right":
                    right_lms = hand_lms

            frame_vectors.append(build_feature_vector(pose_lms, left_lms, right_lms))
            frame_idx += 1
    finally:
        cap.release()

    if not frame_vectors:
        raise RuntimeError(f"No frames decoded from video: {video_path}")

    return np.stack(frame_vectors, axis=0)


def load_completed_uids() -> set:
    """Reads the manifest (if it exists from a previous run) to know which
    uids are already done, so we can skip them and resume cleanly."""
    if not MANIFEST_PATH.exists():
        return set()
    try:
        df = pd.read_csv(MANIFEST_PATH)
        return set(df.loc[df["status"] == "ok", "uid"])
    except Exception:
        return set()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not CISLR_INDEX.exists():
        print(f"[ERROR] {CISLR_INDEX} not found. Run 07_build_cislr_index.py first.")
        return

    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    pose_model_path, hand_model_path = ensure_models_downloaded()

    pose_options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=str(pose_model_path),
            delegate=mp_python.BaseOptions.Delegate.CPU,
        ),
        running_mode=mp_vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
    )
    hand_options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=str(hand_model_path),
            delegate=mp_python.BaseOptions.Delegate.CPU,
        ),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
    )

    df = pd.read_csv(CISLR_INDEX)
    if args.limit:
        df = df.head(args.limit)

    KEYPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    already_done = set() if args.overwrite else load_completed_uids()
    if already_done:
        print(f"Resuming: {len(already_done)} video(s) already completed in a previous run, skipping those.")

    # Open manifest in append mode and write incrementally -- critical for
    # a multi-hour run, so an interruption never loses more than the one
    # video currently in progress.
    manifest_exists = MANIFEST_PATH.exists() and not args.overwrite
    manifest_file = open(MANIFEST_PATH, "a" if manifest_exists else "w", newline="")
    manifest_writer = csv.DictWriter(manifest_file, fieldnames=["uid", "keypoints_path", "num_frames", "status"])
    if not manifest_exists:
        manifest_writer.writeheader()

    processed, skipped, failed = 0, 0, 0

    try:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting CISLR keypoints"):
            uid = row["uid"]
            if uid in already_done:
                skipped += 1
                continue

            out_path = KEYPOINTS_DIR / f"{uid}.npy"

            try:
                with mp_vision.PoseLandmarker.create_from_options(pose_options) as pose_landmarker, \
                     mp_vision.HandLandmarker.create_from_options(hand_options) as hand_landmarker:
                    keypoints = process_video(pose_landmarker, hand_landmarker, row["video_path"])
                np.save(out_path, keypoints)
                processed += 1
                manifest_writer.writerow({
                    "uid": uid, "keypoints_path": str(out_path),
                    "num_frames": keypoints.shape[0], "status": "ok",
                })
            except Exception as e:
                failed += 1
                manifest_writer.writerow({
                    "uid": uid, "keypoints_path": "", "num_frames": 0, "status": f"FAILED: {e}",
                })

            manifest_file.flush()  # ensure progress survives an interruption
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Progress so far has been saved to the manifest.")
        print("Re-run this script to resume from where you left off.")
    finally:
        manifest_file.close()

    print(f"\nDone (this run). Processed: {processed}, skipped (already done): {skipped}, failed: {failed}")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
