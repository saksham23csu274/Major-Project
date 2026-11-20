"""
Phase 1c — MediaPipe keypoint extraction.

For every clip in data/processed/labels/clip_index.csv, runs MediaPipe's
PoseLandmarker and HandLandmarker (Tasks API) over its ordered frame images,
extracts pose + both-hand landmarks, normalizes them (translation/scale
invariant), and saves the resulting (T, 228) keypoint sequence as a .npy file.

Why two separate landmarkers instead of the combined HolisticLandmarker:
  The combined Holistic bundle wires pose+face+hand sub-models together
  internally with cropped-region handoffs between them. On macOS this
  triggers a hard native crash (SIGABRT, "Check failed: service_ Service
  is unavailable", inside a Metal/GPU helper) regardless of an explicit
  CPU delegate setting — a known class of issue with that bundle on this
  platform. PoseLandmarker and HandLandmarker run as independent, much
  simpler graphs and don't hit this. Since we already decided against
  using face landmarks at all, running these two separately costs us
  nothing we needed anyway.

Why pose + hands only (no face mesh):
  Face mesh adds 478 landmarks per frame — a huge increase in input
  dimensionality for a BiLSTM trained on ~100 sentences of data, with little
  benefit for the core lexical recognition task. This matches our
  architecture decision to favor simplicity. Facial expression / non-manual
  markers are a documented stretch-goal extension, not a core-path blocker.

Feature vector layout (228 dims per frame), all counts as landmarks x (x,y,z):
  [0:99]    pose landmarks (33 x 3), normalized
  [99:162]  left hand landmarks (21 x 3), normalized
  [162:225] right hand landmarks (21 x 3), normalized
  [225]     pose_present (1.0 / 0.0)
  [226]     left_hand_present (1.0 / 0.0)
  [227]     right_hand_present (1.0 / 0.0)

Normalization: every frame's landmarks are re-centered on the midpoint of
the two shoulders (pose landmarks 11, 12) and scaled by shoulder width, so
the representation is invariant to the signer's distance from the camera
and position in frame. If pose isn't detected in a frame, that frame's
values are zero-filled and presence flags reflect it (the sequence model
will need to tolerate occasional dropped frames — normal for real video).

Usage:
    python scripts/04_extract_keypoints.py [--limit N] [--overwrite]

    --limit N     only process the first N clips (useful for a quick test
                  run before committing to the full ~1-2 hour extraction)
    --overwrite   redo clips that already have a saved .npy file
                  (by default, already-processed clips are skipped, so you
                  can safely re-run this after an interruption)
"""

import argparse
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLIP_INDEX = PROJECT_ROOT / "data" / "processed" / "labels" / "clip_index.csv"
KEYPOINTS_DIR = PROJECT_ROOT / "data" / "processed" / "keypoints"
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "labels" / "keypoints_manifest.csv"

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
FEATURE_DIM = (NUM_POSE + 2 * NUM_HAND) * 3 + 3  # +3 presence flags = 228

LEFT_SHOULDER_IDX = 11
RIGHT_SHOULDER_IDX = 12


def _download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading model bundle...")
    print(f"  From: {url}")
    print(f"  To:   {dest}")
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as e:
        print(f"[ERROR] Could not download model: {e}")
        print("If this is a network/firewall issue, download the file manually from")
        print(f"  {url}")
        print(f"and save it to: {dest}")
        sys.exit(1)
    print(f"Downloaded ({dest.stat().st_size / 1e6:.1f} MB).")


def ensure_models_downloaded() -> tuple[Path, Path]:
    _download(POSE_MODEL_URL, POSE_MODEL_PATH)
    _download(HAND_MODEL_URL, HAND_MODEL_PATH)
    return POSE_MODEL_PATH, HAND_MODEL_PATH


def landmarks_to_array(landmark_list, expected_count: int) -> tuple[np.ndarray, bool]:
    """
    Convert a MediaPipe landmark list to a flat (expected_count, 3) array.
    Returns (array, present) where present=False and array is all-zero if
    the landmark list was empty/missing (e.g. hand not visible this frame).
    """
    if not landmark_list:
        return np.zeros((expected_count, 3), dtype=np.float32), False
    arr = np.array([[lm.x, lm.y, lm.z] for lm in landmark_list], dtype=np.float32)
    if arr.shape[0] != expected_count:
        # Defensive: shouldn't happen with a well-formed model output, but
        # don't silently misalign features if it does.
        return np.zeros((expected_count, 3), dtype=np.float32), False
    return arr, True


def normalize_frame(pose_arr, left_arr, right_arr, pose_present: bool,
                     left_present: bool, right_present: bool):
    """
    Re-center on shoulder midpoint and scale by shoulder width. If pose
    wasn't detected, we can't normalize meaningfully — return arrays as-is.
    Critically: a hand that wasn't detected stays exactly zero (not shifted
    by the center subtraction) so "all zero" reliably means "missing" and
    doesn't get corrupted into some other non-zero value that looks like
    real (but wrong) data to the model.
    """
    if not pose_present:
        return pose_arr, left_arr, right_arr

    left_shoulder = pose_arr[LEFT_SHOULDER_IDX]
    right_shoulder = pose_arr[RIGHT_SHOULDER_IDX]
    center = (left_shoulder + right_shoulder) / 2.0
    scale = np.linalg.norm(left_shoulder - right_shoulder)
    scale = max(scale, 1e-6)  # avoid divide-by-zero on degenerate detections

    pose_norm = (pose_arr - center) / scale
    left_norm = (left_arr - center) / scale if left_present else left_arr
    right_norm = (right_arr - center) / scale if right_present else right_arr
    return pose_norm, left_norm, right_norm


def build_feature_vector(pose_landmarks, left_hand_landmarks, right_hand_landmarks) -> np.ndarray:
    """
    Pure function (no MediaPipe types required beyond duck-typed .x/.y/.z
    attributes) so this can be unit tested without a live model.
    """
    pose_arr, pose_present = landmarks_to_array(pose_landmarks, NUM_POSE)
    left_arr, left_present = landmarks_to_array(left_hand_landmarks, NUM_HAND)
    right_arr, right_present = landmarks_to_array(right_hand_landmarks, NUM_HAND)

    pose_arr, left_arr, right_arr = normalize_frame(
        pose_arr, left_arr, right_arr, pose_present, left_present, right_present
    )

    presence = np.array(
        [float(pose_present), float(left_present), float(right_present)],
        dtype=np.float32,
    )
    return np.concatenate([pose_arr.flatten(), left_arr.flatten(), right_arr.flatten(), presence])


def process_clip(pose_landmarker, hand_landmarker, frame_paths: list[str]) -> np.ndarray:
    import mediapipe as mp  # local import so build_feature_vector is testable without mediapipe installed

    frame_vectors = []
    for i, frame_path in enumerate(frame_paths):
        mp_image = mp.Image.create_from_file(str(frame_path))
        timestamp_ms = i * 33  # arbitrary monotonic timestamp (~30fps assumption)

        pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
        hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

        pose_lms = pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None

        # HandLandmarker doesn't separate left/right the way Holistic did —
        # it returns a list of detected hands, each with its own
        # 'handedness' classification we have to check ourselves.
        left_lms, right_lms = None, None
        for hand_lms, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
            label = handedness[0].category_name  # "Left" or "Right"
            if label == "Left":
                left_lms = hand_lms
            elif label == "Right":
                right_lms = hand_lms

        frame_vectors.append(build_feature_vector(pose_lms, left_lms, right_lms))

    return np.stack(frame_vectors, axis=0)  # (T, 228)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not CLIP_INDEX.exists():
        print(f"[ERROR] {CLIP_INDEX} not found. Run 03_build_clip_index.py first.")
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

    clip_df = pd.read_csv(CLIP_INDEX)
    if args.limit:
        clip_df = clip_df.head(args.limit)

    KEYPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    skipped, processed, failed = 0, 0, 0

    for _, row in tqdm(clip_df.iterrows(), total=len(clip_df), desc="Extracting keypoints"):
        clip_id = row["clip_id"]
        out_path = KEYPOINTS_DIR / f"{clip_id}.npy"

        if out_path.exists() and not args.overwrite:
            skipped += 1
            manifest_rows.append({"clip_id": clip_id, "keypoints_path": str(out_path), "status": "already_done"})
            continue

        frame_paths = str(row["frame_paths"]).split(";")

        # Fresh landmarker instances per clip: avoids any tracking-state
        # leakage between different signers/sentences.
        try:
            with mp_vision.PoseLandmarker.create_from_options(pose_options) as pose_landmarker, \
                 mp_vision.HandLandmarker.create_from_options(hand_options) as hand_landmarker:
                keypoints = process_clip(pose_landmarker, hand_landmarker, frame_paths)
            np.save(out_path, keypoints)
            processed += 1
            manifest_rows.append({
                "clip_id": clip_id,
                "keypoints_path": str(out_path),
                "num_frames": keypoints.shape[0],
                "feature_dim": keypoints.shape[1],
                "status": "ok",
            })
        except Exception as e:
            failed += 1
            manifest_rows.append({"clip_id": clip_id, "keypoints_path": None, "status": f"FAILED: {e}"})

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(MANIFEST_PATH, index=False)

    print(f"\nDone. Processed: {processed}, already done (skipped): {skipped}, failed: {failed}")
    print(f"Manifest saved to: {MANIFEST_PATH}")
    if failed:
        print("\nFailed clips:")
        print(manifest_df[manifest_df["status"].str.startswith("FAILED", na=False)][["clip_id", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
