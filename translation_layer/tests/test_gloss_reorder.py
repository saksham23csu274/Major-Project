"""
Unit tests for the rule-based translation layer.

These lock in the high-confidence rules (copula/modal deletion, negation
preservation, WH-word position) derived from real corpus analysis — see
gloss_reorder.py's module docstring for the full quantitative breakdown.
Run with: python -m pytest translation_layer/tests/test_gloss_reorder.py -v
(or just: python translation_layer/tests/test_gloss_reorder.py)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from translation_layer.rules.gloss_reorder import sentence_to_gloss, gloss_to_sentence


def test_copula_dropped():
    assert sentence_to_gloss("i am hungry") == "I HUNGRY"
    assert sentence_to_gloss("you are good") == "YOU GOOD"
    assert sentence_to_gloss("this place is beautiful") == "THIS PLACE BEAUTIFUL"


def test_modal_dropped():
    assert sentence_to_gloss("can i help you") == "I HELP YOU"
    assert sentence_to_gloss("shall we go outside") == "WE GO OUTSIDE"


def test_be_kept_after_negation_as_main_verb():
    # "be" here is the imperative main verb, not an auxiliary -- must be kept
    assert sentence_to_gloss("do not be stubborn") == "DO NOT BE STUBBORN"


def test_be_dropped_as_auxiliary():
    # "be" here is part of the "would be coming" aux chain -- must be dropped
    assert sentence_to_gloss("he would be coming today") == "HE COMING TODAY"


def test_negation_preserved_not_reordered():
    assert sentence_to_gloss("i do not agree") == "I DO NOT AGREE"
    assert sentence_to_gloss("do not worry") == "DO NOT WORRY"


def test_wh_word_stays_sentence_initial():
    # Confirms our key empirical finding: WH-words are NOT moved to the
    # end, contrary to some generic claims about ISL/ASL structure.
    result = sentence_to_gloss("why are you angry")
    assert result.split()[0] == "WHY"


def test_gloss_to_sentence_basic_reconstruction():
    assert gloss_to_sentence("I HUNGRY") == "I am hungry."
    assert gloss_to_sentence("YOU GOOD") == "You are good."


def test_gloss_to_sentence_question_punctuation():
    result = gloss_to_sentence("WHY YOU ANGRY")
    assert result.endswith("?")


def test_gloss_to_sentence_negation_expansion():
    # "DONOT"/"DONT" (merged tokens seen in the raw corpus) should expand
    # back into readable "do not"
    result = gloss_to_sentence("I DONOT AGREE")
    assert "do not" in result.lower()


def _run_all():
    """Fallback runner if pytest isn't installed."""
    import inspect
    tests = [obj for name, obj in globals().items() if name.startswith("test_") and callable(obj)]
    passed, failed = 0, 0
    for test in tests:
        try:
            test()
            print(f"  PASS: {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {test.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")


if __name__ == "__main__":
    _run_all()
