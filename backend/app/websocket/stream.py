"""
WebSocket endpoint for real-time ISL translation.

Protocol (JSON messages, one per WebSocket message):
  Client -> Server:
    {"keypoints": [228 floats]}   -- one frame of keypoints from the browser
    {"reset": true}               -- clear the buffer (start of a new sentence)

  Server -> Client:
    {"gloss": "...", "translation": "..."}  -- sent periodically as new
                                                frames accumulate

DESIGN NOTE — sliding window: we don't wait for the client to say "the
sentence is finished" before responding; we re-run decoding every
DECODE_EVERY_N_FRAMES new frames against the buffer accumulated so far,
and send an updated translation each time. This is what gives the
"live, continuously updating" experience described in the project brief,
at the cost of some wasted recomputation (each decode reprocesses the
whole buffer from scratch, since our current model has no incremental/
streaming inference mode -- a documented simplification, not an
oversight, and a natural target for optimization once the core pipeline
is proven out).

CURRENT LIMITATION (expected, worth being upfront about): the model was
trained on only ~470 sentence-level clips with no CISLR pretraining yet,
so translation quality from a live camera feed will likely be poor. This
endpoint's job is to prove the STREAMING INFRASTRUCTURE works correctly
end-to-end -- decoding accuracy is a separate, already-tracked concern
(Phase 2/5).
"""

import json
from pathlib import Path

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect, APIRouter

from app.inference.model_loader import load_model, DEFAULT_CHECKPOINT_PATH
from app.inference.decoder import decode_keypoint_sequence

router = APIRouter()

DECODE_EVERY_N_FRAMES = 10  # re-run decoding after this many new frames arrive
MAX_BUFFER_FRAMES = 300     # cap buffer length to bound per-decode compute cost

# Loaded once at import time (module-level), shared across connections --
# the model itself is stateless at inference time (no gradient tracking,
# no per-request mutation), so this is safe to share.
_model_state = {"loaded": False}


def _try_load_model():
    if _model_state["loaded"]:
        return
    try:
        model, token_to_idx, idx_to_token, device = load_model()
        _model_state.update({
            "model": model, "token_to_idx": token_to_idx,
            "idx_to_token": idx_to_token, "device": device, "loaded": True,
        })
    except FileNotFoundError as e:
        _model_state["error"] = str(e)


@router.websocket("/ws/translate")
async def translate_stream(websocket: WebSocket):
    await websocket.accept()
    _try_load_model()

    if not _model_state["loaded"]:
        await websocket.send_json({
            "error": f"Model not available: {_model_state.get('error', 'unknown error')}"
        })
        await websocket.close()
        return

    buffer = []
    frames_since_last_decode = 0

    try:
        while True:
            raw_message = await websocket.receive_text()
            message = json.loads(raw_message)

            if message.get("reset"):
                buffer = []
                frames_since_last_decode = 0
                await websocket.send_json({"gloss": "", "translation": ""})
                continue

            keypoints_frame = message.get("keypoints")
            if keypoints_frame is None or len(keypoints_frame) != 228:
                await websocket.send_json({"error": "Expected 'keypoints' as a 228-length array"})
                continue

            buffer.append(keypoints_frame)
            if len(buffer) > MAX_BUFFER_FRAMES:
                buffer = buffer[-MAX_BUFFER_FRAMES:]  # drop oldest frames, keep most recent window

            frames_since_last_decode += 1
            if frames_since_last_decode >= DECODE_EVERY_N_FRAMES:
                frames_since_last_decode = 0
                keypoints_array = np.array(buffer, dtype=np.float32)
                gloss, sentence = decode_keypoint_sequence(
                    _model_state["model"], keypoints_array,
                    _model_state["idx_to_token"], _model_state["device"],
                )
                await websocket.send_json({"gloss": gloss, "translation": sentence})

    except WebSocketDisconnect:
        pass  # normal disconnect, nothing to clean up (buffer is per-connection, local variable)
