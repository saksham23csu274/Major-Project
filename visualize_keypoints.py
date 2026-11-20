"""
Produces one image: a real dataset frame with MediaPipe's detected
pose + hand keypoints drawn on top, for use as a figure in the Synopsis
Report. Uses the same PoseLandmarker/HandLandmarker setup as the main
extraction pipeline, in IMAGE mode (single still frame).

Usage:
    python visualize_keypoints.py
    python visualize_keypoints.py --image /path/to/any/frame.jpg
"""

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
CLIP_INDEX = PROJECT_ROOT / "data" / "processed" / "labels" / "clip_index.csv"
MODEL_DIR = PROJECT_ROOT / "models" / "mediapipe"
POSE_MODEL_PATH = MODEL_DIR / "pose_landmarker_full.task"
HAND_MODEL_PATH = MODEL_DIR / "hand_landmarker.task"
OUTPUT_PATH = PROJECT_ROOT / "keypoint_overlay_example.png"


def get_sample_frame_path(explicit_path: str = None) -> str:
    if explicit_path:
        return explicit_path
    df = pd.read_csv(CLIP_INDEX)
    first_row = df.iloc[0]
    frame_paths = str(first_row["frame_paths"]).split(";")
    # pick a frame from the middle of the clip -- more likely to show an
    # actual sign in progress than the very first/last (often neutral) frame
    return frame_paths[len(frame_paths) // 2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None)
    args = parser.parse_args()

    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    import mediapipe as mp

    frame_path = get_sample_frame_path(args.image)
    print(f"Using frame: {frame_path}")

    image_bgr = cv2.imread(frame_path)
    if image_bgr is None:
        print(f"[ERROR] Could not read image at {frame_path}")
        sys.exit(1)
    h, w = image_bgr.shape[:2]

    pose_options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=str(POSE_MODEL_PATH), delegate=mp_python.BaseOptions.Delegate.CPU
        ),
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    hand_options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=str(HAND_MODEL_PATH), delegate=mp_python.BaseOptions.Delegate.CPU
        ),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=2,
    )

    mp_image = mp.Image.create_from_file(str(frame_path))

    with mp_vision.PoseLandmarker.create_from_options(pose_options) as pose_landmarker, \
         mp_vision.HandLandmarker.create_from_options(hand_options) as hand_landmarker:
        pose_result = pose_landmarker.detect(mp_image)
        hand_result = hand_landmarker.detect(mp_image)

    output = image_bgr.copy()

    # Draw pose landmarks (cyan)
    if pose_result.pose_landmarks:
        for lm in pose_result.pose_landmarks[0]:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(output, (x, y), 4, (255, 255, 0), -1)

    # Draw hand landmarks (magenta for left, yellow for right)
    for hand_lms, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
        color = (255, 0, 255) if handedness[0].category_name == "Left" else (0, 255, 255)
        for lm in hand_lms:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(output, (x, y), 4, color, -1)

    cv2.imwrite(str(OUTPUT_PATH), output)
    print(f"Saved: {OUTPUT_PATH}")
    print("Open this file and insert it directly into the Synopsis Report,")
    print("replacing the bracketed note near the Background section.")


if __name__ == "__main__":
    main()
