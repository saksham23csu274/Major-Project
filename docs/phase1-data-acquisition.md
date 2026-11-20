# Phase 1a — Data Acquisition (manual step, do this first)

Mendeley and Hugging Face both require a browser click-through (login/terms
acceptance), so this step can't be scripted — do it once, by hand, on your
MacBook. Everything after this is scripted.

## 1. ISL-CSLTR (sentence-level, primary fine-tuning + eval set)

1. Go to https://data.mendeley.com/datasets/kcmpdxky7p/1
2. Click "Download all files" (no login required).
3. Unzip the download.
4. Move/copy the extracted contents into:
   `isl-translation-system/data/raw/isl_cslrt/`

You should end up with (roughly) subfolders of videos organized by sentence
and/or signer, plus a labels/annotation file — the exact internal layout
varies by dataset version, which is why the next script (`scripts/01_inspect_dataset.py`)
first *reports* the structure back to you before we write any parsing logic
against it. Do not assume the layout — we verify it.

## 2. CISLR (word-level, pretraining set)

1. Go to https://huggingface.co/datasets/Exploration-Lab/CISLR
2. Log in (free account) and accept the access conditions.
3. Either use the "Files and versions" tab to download directly, or (recommended,
   once you accept terms) use the `huggingface_hub` Python library:
   ```bash
   pip install huggingface_hub
   huggingface-cli login   # paste your HF access token
   huggingface-cli download Exploration-Lab/CISLR --repo-type dataset --local-dir data/raw/cislr
   ```
4. Confirm `data/raw/cislr/` contains `dataset.csv`, `prototype.csv`, `test.csv`
   and the video files/folders referenced by them.

## 3. Continuous Fingerspelling Dataset for ISL (supplementary, fingerspelled segments)

1. Go to https://kirandevraj.github.io/ISL-Fingerspelling/
2. Follow the download link/instructions on that page (this is a very recently
   published dataset — Dec 2025 — so check the page directly for the current
   access method rather than trusting anything cached).
3. Place contents into `data/raw/fingerspelling/`.

**Note:** this dataset is source video from a YouTube news channel with
on-screen text cues, not clean studio ISL — it's a stretch/supplementary
resource for fingerspelling handling, not a load-bearing dependency. If it
turns out to be awkward to obtain or license-restricted, we can drop it
without hurting the core deliverable (RQ1–RQ3 don't depend on it).

## Once you've done the above

Run:
```bash
cd isl-translation-system
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/01_inspect_dataset.py
```
This won't train anything — it just walks `data/raw/` and prints back an
inventory (file counts, extensions, folder structure, a sample annotation
file if found) so we can confirm what we're actually working with before
writing the keypoint-extraction and label-alignment code together.
