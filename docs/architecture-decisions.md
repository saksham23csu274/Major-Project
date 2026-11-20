# Architecture Decisions — CONFIRMED

_Confirmed on: 2026-09-08_

| Area | Decision | Justification |
|---|---|---|
| Pose extraction | MediaPipe Holistic, **client-side** (browser WASM build) for live inference; Python MediaPipe for offline dataset preprocessing | Keeps only keypoints (not video) crossing the network in production, minimizing latency/bandwidth. Same landmark model in both runtimes keeps keypoint format consistent between training and inference. |
| Sequence model | **BiLSTM + CTC** as the primary continuous-recognition architecture. Transformer encoder + CTC is a stretch-goal/ablation only | With ~100 sentence-level ISL-CSLTR examples and ~4,700-word CISLR for pretraining, a Transformer's data appetite risks overfitting and slower/heavier training on an M4 with no CUDA. BiLSTM is lighter, well-precedented for skeleton-based continuous sign recognition, and trains comfortably on CPU/MPS. |
| Baseline model | Isolated-sign classifier (per-segment classification, no CTC, no translation layer) | Required comparison point for RQ1 (does continuous+CTC beat isolated recognition) and RQ3 (does explicit translation beat naive concatenation). |
| Translation layer | Rule-based ISL gloss → English/Hindi reordering (core deliverable). Learned seq2seq gloss→sentence model is a stretch goal only | Matches project brief; safer, more controllable, and more explainable for a solo 9-10 month timeline and for evaluators who want to see linguistic reasoning (topic-comment structure, etc). |
| Backend | FastAPI + WebSocket | Good fit for continuous streaming inference; minimal boilerplate; easy to run and debug locally. |
| Frontend | Next.js/React, plain React state (no Redux) for the live transcript buffer | Keeps state management simple for a solo project; React state is sufficient for a single live transcript panel. |
| Compute strategy | PyTorch with `mps` backend (Apple Silicon), CPU fallback. Batch sizes kept modest (exact numbers set after inspecting real keypoint sequence lengths in Phase 1) | Fits the 16GB RAM / no-CUDA-GPU constraint. |

## Confirmed datasets and access method

| Dataset | Purpose | Access | Notes |
|---|---|---|---|
| **ISL-CSLTR** | Sentence-level fine-tuning + evaluation | Mendeley Data, DOI `10.17632/kcmpdxky7p.1` — publicly downloadable, no login | 700 annotated videos, 18,863 sentence-level frames, 1,036 word-level images, 100 spoken-language sentences, 7 signers. This is our RQ4 (unseen-signer generalization) test bed since it has multiple signers. |
| **CISLR** | Word-level pretraining | Hugging Face `Exploration-Lab/CISLR` — publicly accessible but requires a free HF account + accepting terms before download | ~4,700-word vocabulary, ~7,050 videos, sourced from the web (varied backgrounds/quality — expect to filter/clean). |
| **Continuous Fingerspelling Dataset for ISL (Kirandevraj et al., WSLP 2025)** | Handling fingerspelled segments | https://kirandevraj.github.io/ISL-Fingerspelling/ | 1,308 segments, 499 videos, 70.85 minutes, 14,814 characters, extracted from ISH News YouTube channel with synchronized on-screen text. Very recent (Dec 2025) — treat as supplementary, not a load-bearing dependency, in case format/availability shifts. |

## Open items to revisit once Phase 1 data pipeline is running

- Exact keypoint feature dimensionality (which MediaPipe Holistic landmark subsets we keep — likely hands + upper-body pose + reduced face mesh, to control input size).
- Real batch size / sequence length numbers for BiLSTM+CTC training on the M4.
- Whether CISLR's web-sourced video quality needs a filtering pass before pretraining.

## Known issues and fixes (Phase 1 data pipeline)

- **MediaPipe version pin**: `mediapipe==1.0.x` crashes with a hard native abort (`SIGABRT`, "Check failed: service_ Service is unavailable") inside a Metal/GPU helper during pose/hand landmark detection on macOS Apple Silicon — occurs even with the CPU delegate explicitly forced, across both the combined `HolisticLandmarker` bundle and the separate `PoseLandmarker`/`HandLandmarker` tasks. Pinned to `mediapipe==0.10.21`, a well-established release, which resolved it. `requirements.txt` is pinned accordingly to prevent regression on a future fresh install.
- **Feature choice**: extracting pose (33 landmarks) + both hands (21 each) = 228-dim feature vector per frame (including 3 presence flags for missing detections), explicitly excluding the 478-point face mesh to keep input dimensionality manageable given ~100 sentence-level training examples.
- **ISL-CSLTR data quality**: 662 of 663 (sentence, signer) clips extracted successfully. One clip (`i_am_hungry__signer3`) excluded due to a stray-space filename inconsistency between the dataset's metadata spreadsheet and the actual file on disk — a source-data quirk, not a pipeline bug.
