# Real-Time Sentence-Level Indian Sign Language Translation System

Final-year major project (CS, AI/ML) — translates continuous Indian Sign Language (ISL)
video into readable sentence-level text (English, with Hindi/Hinglish output in progress),
aimed at real-time use on consumer hardware.

**Author:** Saksham Mathur (Roll No. 23csu274)

---

## Problem

India has a severe shortage of sign-language interpreters. Most existing ISL recognition
work focuses on isolated signs or fingerspelling; this project targets the harder,
more practically useful problem: continuous, sentence-level signing, translated to text
in real time, running on ordinary hardware rather than requiring GPUs or cloud inference.

## How it works

1. **Keypoint extraction** — MediaPipe PoseLandmarker + HandLandmarker extract pose and
   hand keypoints from video frames, instead of feeding raw video into the model. This
   keeps the input lightweight enough for real-time, CPU-only inference.
2. **Sign recognition** — A BiLSTM + CTC (Connectionist Temporal Classification) model
   takes the keypoint sequence and predicts a gloss sequence (the sequence of signs),
   without needing frame-by-frame sign boundaries labelled in advance.
   - Trained on **ISL-CSLTR** (100 sentences, 7 signers, 663 clips), with a
     signer-disjoint train/val/test split (signers 1–5 train, 6 validation, 7 held out
     for test) so results reflect generalization to an unseen signer, not memorization.
   - Optionally pretrained on **CISLR** (~4,765 word classes, 7,050 videos) before
     fine-tuning on ISL-CSLTR, which improves results (see below).
3. **Translation layer** — A rule-based (non-ML) layer reorders the predicted gloss
   sequence into a grammatical sentence, since there isn't enough parallel gloss↔sentence
   data to train a translation model from scratch. Currently supports English output;
   a Hindi/Hinglish extension is prototyped (see `translation_layer/rules/` /
   `hindi_output.py`) but not yet wired into the live backend.
4. **Serving** — A FastAPI backend streams keypoints over a WebSocket connection and
   returns live gloss/translation updates using a sliding-window buffer for incremental
   decoding.

## Results so far

| Model | Word Error Rate (WER) |
|---|---|
| BiLSTM+CTC, no pretraining | 0.829 |
| BiLSTM+CTC, CISLR-pretrained encoder | 0.820 |

| Component | Result |
|---|---|
| Isolated-sign baseline (113 classes) | 28.8% test accuracy (vs. 0.9% chance) |
| CISLR pretraining encoder (4,765 classes) | 5.11% accuracy (vs. 0.02% chance) |
| Rule-based gloss→English translation | 65% exact match, 93.8% word-level recall |

All evaluation numbers are measured on the signer-disjoint held-out test signer, not seen
during training.

## Project structure

```
.
├── scripts/              # dataset inspection, keypoint extraction, index building, training helpers
├── models/               # model definitions (isolated-sign baseline, continuous BiLSTM+CTC)
├── translation_layer/    # rule-based gloss → sentence translation (English, Hindi/Hinglish prototype)
├── backend/ (or app/)    # FastAPI app, WebSocket streaming endpoint, inference pipeline
├── docs/ or reports/     # Synopsis Report, Progress Report, presentation
└── README.md
```

> Update this tree to match your actual folder names before committing — this is a
> template based on the project's design, not a scan of your real filesystem.

## Setup

Datasets (ISL-CSLTR, CISLR) and trained model checkpoints are **not** included in this
repo (see `.gitignore`) — they're too large for git. Download them separately and place
them under `data/` (or wherever your scripts expect them), then run the pipeline scripts
in `scripts/` in order (dataset inspection → keypoint extraction → index/splits →
training).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the backend:
```bash
# fill in your actual entrypoint/command here
uvicorn backend.main:app --reload
```

## Status

~74% complete. Core recognition pipeline (data → keypoints → BiLSTM+CTC → rule-based
English translation → FastAPI/WebSocket backend) is working end-to-end. Remaining work:
frontend (Next.js + webcam/MediaPipe client), formal latency measurement, Hindi/Hinglish
integration into the live backend, and the final written report.

## Known limitations

- Small training data (663 sentence-level clips) limits recognition accuracy.
- Translation layer is rule-based, not learned — covers the vocabulary seen in training
  data, not open-vocabulary ISL.
- No evaluated reverse-communication channel yet (how a hearing person's reply reaches
  the Deaf user) — currently a design proposal only.
- Hindi/Hinglish output is a tested prototype, not yet integrated into the live pipeline.

## License

See `LICENSE`.
