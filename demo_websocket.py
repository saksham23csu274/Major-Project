"""
Live demo script for the WebSocket streaming endpoint. Connects to the
running backend, streams a REAL clip's keypoint frames one at a time
(exactly as a browser would send them), then prints every translation
update the server sends back -- showing it update as more frames arrive,
the same way it would during actual live use.

Usage:
    python demo_websocket.py
    python demo_websocket.py --clip-id are_you_free_today__signer1
"""

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
import pandas as pd
import websockets

PROJECT_ROOT = Path(__file__).resolve().parent
CLIP_INDEX = PROJECT_ROOT / "data" / "processed" / "labels" / "clip_index.csv"
WS_URL = "ws://127.0.0.1:8000/ws/translate"


def pick_clip(explicit_clip_id: str = None) -> tuple:
    df = pd.read_csv(CLIP_INDEX)
    if explicit_clip_id:
        row = df[df["clip_id"] == explicit_clip_id].iloc[0]
    else:
        row = df.iloc[0]
    return row["clip_id"], row["sentence"], row["gloss"]


async def demo(clip_id_arg: str = None):
    clip_id, sentence, gold_gloss = pick_clip(clip_id_arg)
    keypoints_path = PROJECT_ROOT / "data" / "processed" / "keypoints" / f"{clip_id}.npy"
    keypoints = np.load(keypoints_path)

    print(f"Streaming real clip: {clip_id}")
    print(f"  Original sentence: {sentence!r}")
    print(f"  Real gold gloss:   {gold_gloss!r}")
    print(f"  Frames to stream:  {keypoints.shape[0]}")
    print()

    async with websockets.connect(WS_URL) as ws:
        # Send every frame, with a small delay between each to mimic real
        # camera frame timing (~30fps) rather than sending everything
        # instantly -- purely cosmetic for a live demo's pacing.
        for i, frame in enumerate(keypoints):
            await ws.send(json.dumps({"keypoints": frame.tolist()}))
            await asyncio.sleep(0.03)

        print("All frames sent -- waiting for translation updates...\n")

        # Drain every response the server sends back. Using a generous
        # timeout and looping until it's exhausted is more reliable than
        # trying to time a receive after each individual frame.
        update_num = 0
        try:
            while True:
                response = await asyncio.wait_for(ws.recv(), timeout=2.0)
                data = json.loads(response)
                update_num += 1
                print(f"  Update {update_num}: gloss={data.get('gloss', ''):25s} "
                      f"-> {data.get('translation', '')}")
        except asyncio.TimeoutError:
            pass

    if update_num == 0:
        print("No updates received -- check that the backend is running and a trained")
        print("checkpoint exists at models/checkpoints/ctc_finetuned_best.pt")
    else:
        print(f"\nDone. Compare the final update above to the real gold gloss/sentence printed at the top.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-id", type=str, default=None)
    args = parser.parse_args()
    asyncio.run(demo(args.clip_id))
