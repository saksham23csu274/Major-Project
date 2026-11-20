"""
Phase 2 — Extract MediaPipe keypoints from isolated word-level images.

Unlike the sentence-level and CISLR extraction (VIDEO mode, since those
are temporal sequences), this uses IMAGE mode -- each word-level picture
is an independent still with no preceding/following frame to relate it
to. Same 225-dim landmark portion (pose + both hands) and normalization
as the sequence pipelines, minus the "VIDEO mode" timestamp machinery
that doesn't apply here.

Since there are only ~1000 images total (vs. 18,863 sentence-level frames
or 7,050 CISLR videos), this consolidates everything into ONE output file
(features.npz: X, y, label names) rather than one .npy per instance --
simpler for the baseline classifier to load, and there's no real
resumability concern at this small scale.

Usage:
    python scripts/10_extract_word_keypoints.py
"""

import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORD_INDEX = PROJECT_ROOT / "data" / "processed" / "labels" / "word_index.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "keypoints_word" / "word_features.npz"

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
FEATURE_DIM = (NUM_POSE + 2 * NUM_HAND) * 3 + 3
LEFT_SHOULDER_IDX = 11
RIGHT_SHOULDER_IDX = 12


def _download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading model bundle...\n  From: {url}\n  To:   {dest}")
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


def ensure_models_downloaded():
    _download(POSE_MODEL_URL, POSE_MODEL_PATH)
    _download(HAND_MODEL_URL, HAND_MODEL_PATH)
    return POSE_MODEL_PATH, HAND_MODEL_PATH


def landmarks_to_array(landmark_list, expected_count):
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
    scale = max(np.linalg.norm(left_shoulder - right_shoulder), 1e-6)
    pose_norm = (pose_arr - center) / scale
    left_norm = (left_arr - center) / scale if left_present else left_arr
    right_norm = (right_arr - center) / scale if right_present else right_arr
    return pose_norm, left_norm, right_norm


def build_feature_vector(pose_landmarks, left_hand_landmarks, right_hand_landmarks):
    pose_arr, pose_present = landmarks_to_array(pose_landmarks, NUM_POSE)
    left_arr, left_present = landmarks_to_array(left_hand_landmarks, NUM_HAND)
    right_arr, right_present = landmarks_to_array(right_hand_landmarks, NUM_HAND)
    pose_arr, left_arr, right_arr = normalize_frame(
        pose_arr, left_arr, right_arr, pose_present, left_present, right_present
    )
    presence = np.array([float(pose_present), float(left_present), float(right_present)], dtype=np.float32)
    return np.concatenate([pose_arr.flatten(), left_arr.flatten(), right_arr.flatten(), presence])


def extract_single_image(pose_landmarker, hand_landmarker, image_path: str) -> np.ndarray:
    import mediapipe as mp

    mp_image = mp.Image.create_from_file(str(image_path))
    pose_result = pose_landmarker.detect(mp_image)
    hand_result = hand_landmarker.detect(mp_image)

    pose_lms = pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None
    left_lms, right_lms = None, None
    for hand_lms, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
        label = handedness[0].category_name
        if label == "Left":
            left_lms = hand_lms
        elif label == "Right":
            right_lms = hand_lms

    return build_feature_vector(pose_lms, left_lms, right_lms)


def main() -> None:
    if not WORD_INDEX.exists():
        print(f"[ERROR] {WORD_INDEX} not found. Run 09_build_word_index.py first.")
        return

    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    pose_model_path, hand_model_path = ensure_models_downloaded()

    # IMAGE mode (not VIDEO) -- each instance is an independent still image.
    pose_options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=str(pose_model_path), delegate=mp_python.BaseOptions.Delegate.CPU
        ),
        running_mode=mp_vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.5,
    )
    hand_options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=str(hand_model_path), delegate=mp_python.BaseOptions.Delegate.CPU
        ),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
    )

    df = pd.read_csv(WORD_INDEX)
    print(f"Processing {len(df)} word-level images...")

    features, labels, failed = [], [], []

    with mp_vision.PoseLandmarker.create_from_options(pose_options) as pose_landmarker, \
         mp_vision.HandLandmarker.create_from_options(hand_options) as hand_landmarker:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting word keypoints"):
            try:
                vec = extract_single_image(pose_landmarker, hand_landmarker, row["frame_path"])
                features.append(vec)
                labels.append(row["word"])
            except Exception as e:
                failed.append((row["instance_id"], str(e)))

    X = np.stack(features, axis=0)
    y = np.array(labels)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUTPUT_PATH, X=X, y=y)

    print(f"\nExtracted {len(features)} / {len(df)} instances. Failed: {len(failed)}")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Distinct classes: {len(set(labels))}")
    print(f"Saved to: {OUTPUT_PATH}")

    if failed:
        print("\nFailed instances (first 10):")
        for instance_id, err in failed[:10]:
            print(f"  {instance_id}: {err}")


if __name__ == "__main__":
    main()
