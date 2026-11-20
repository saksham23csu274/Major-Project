"""
Validates the rule-based translation layer against the real ISL-CSLTR
gloss corpus in both directions:

  1. sentence_to_gloss: how well do our deletion rules reproduce the
     real annotated gloss for each of the 100 sentences? (quantifies
     RQ3's "how gloss-like is a simple rule-based model" baseline)

  2. gloss_to_sentence: given the REAL gloss (what our recognition model
     will eventually output), how reasonable is the reconstructed
     English sentence? This is the direction that matters for the
     deployed system, so it's checked qualitatively (printed for
     inspection) rather than exact-match (perfect reconstruction of the
     original wording usually isn't expected, e.g. we can't always know
     if "the" was in the original).

Usage:
    python translation_layer/tests/validate_against_corpus.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from translation_layer.rules.gloss_reorder import sentence_to_gloss, gloss_to_sentence

GLOSS_CSV = (
    PROJECT_ROOT / "data" / "raw" / "isl_cslrt" / "ISL_CSLRT_Corpus"
    / "corpus_csv_files" / "ISL Corpus sign glosses.csv"
)


def word_level_accuracy(predicted: str, gold: str) -> float:
    """Fraction of gold gloss tokens that appear in the predicted gloss,
    in a simple order-agnostic sense (good enough for a quick per-sentence
    signal; exact match is checked separately)."""
    pred_tokens = predicted.lower().replace(",", "").split()
    gold_tokens = gold.lower().replace(",", "").split()
    if not gold_tokens:
        return 1.0
    matched = sum(1 for t in gold_tokens if t in pred_tokens)
    return matched / len(gold_tokens)


def main() -> None:
    if not GLOSS_CSV.exists():
        print(f"[ERROR] Not found: {GLOSS_CSV}")
        return

    df = pd.read_csv(GLOSS_CSV).drop_duplicates()
    df = df[df["Sentence"].str.strip().str.lower() != "i am (age)"]  # template placeholder, not a real sentence

    print("=" * 78)
    print("PART 1: sentence_to_gloss forward-direction accuracy vs. real corpus")
    print("=" * 78)

    exact_matches = 0
    word_accs = []
    mismatches = []

    for _, row in df.iterrows():
        sentence, gold_gloss = row["Sentence"], row["SIGN GLOSSES"]
        predicted_gloss = sentence_to_gloss(sentence)

        is_exact = predicted_gloss.strip().upper() == str(gold_gloss).strip().upper()
        if is_exact:
            exact_matches += 1
        else:
            mismatches.append((sentence, gold_gloss, predicted_gloss))

        word_accs.append(word_level_accuracy(predicted_gloss, str(gold_gloss)))

    n = len(df)
    print(f"\nExact match: {exact_matches}/{n} ({100*exact_matches/n:.1f}%)")
    print(f"Average word-level recall: {100*sum(word_accs)/len(word_accs):.1f}%")

    print(f"\nSample mismatches (first 10 of {len(mismatches)}):")
    for sentence, gold, pred in mismatches[:10]:
        print(f"  sentence: {sentence!r}")
        print(f"    gold:      {gold!r}")
        print(f"    predicted: {pred!r}")
        print()

    print("=" * 78)
    print("PART 2: gloss_to_sentence reverse-direction — qualitative spot-check")
    print("=" * 78)
    print("(This is the direction the deployed system actually uses. Perfect")
    print(" reconstruction of exact original wording isn't the goal — readable,")
    print(" reasonably correct English is.)\n")

    sample_indices = list(range(0, n, max(1, n // 15)))  # ~15 spread-out examples
    for idx in sample_indices:
        row = df.iloc[idx]
        gold_gloss = row["SIGN GLOSSES"]
        original_sentence = row["Sentence"]
        reconstructed = gloss_to_sentence(gold_gloss)
        print(f"  gloss:     {gold_gloss!r}")
        print(f"  original:  {original_sentence!r}")
        print(f"  rebuilt:   {reconstructed!r}")
        print()


if __name__ == "__main__":
    main()
