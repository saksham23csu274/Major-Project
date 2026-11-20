"""
translation_layer/rules/hindi_output.py

Prototype extension of the existing rule-based translation layer
(translation_layer/rules/gloss_reorder.py) to produce Hindi (Devanagari)
and Hinglish (Latin-script, code-mixed) output alongside the existing
English output — in response to mentor feedback item 3 (multi-language
output for real deployment: banks, hospitals, public spaces).

DESIGN NOTE (read this before extending):
This follows the SAME philosophy as gloss_reorder.py: a small, hand-written
dictionary + rules, not a trained model — because there is no Hindi-annotated
training data for ISL-CSLTR, and the project's whole translation layer is
already deliberately rule-based for exactly that reason (see Synopsis
Report, Section 3, papers [24]-[26] on the data cost of learned
gloss<->text approaches).

Two different strategies are used for the two outputs, because ISL and
Hindi share word order (both lean SOV / topic-comment: see Zeshan, 2000,
2003, *Indo-Pakistani Sign Language Grammar*), while English does not:

  * HINDI  : ISL gloss order needs little to no reordering to become valid
             Hindi word order. This function mainly does word-for-word
             lexical substitution (gloss token -> Hindi token) plus
             copula insertion/placement, which is simpler than the
             English path's reordering rules.
  * HINGLISH: real spoken Hinglish is not "Hindi grammar in Roman script".
             It is closer to "English sentence structure with Hindi words
             swapped in for common function/content words" -- so this
             reuses the EXISTING English gloss_to_sentence() output and
             substitutes selected words, rather than building a second
             grammar engine from scratch.

COVERAGE: the lexicons below only cover the vocabulary already documented
in this project's own gloss data (PRONOUN_COPULA / KNOWN_PREDICATES from
gloss_reorder.py, plus a handful of common ISL-CSLTR sentence words such
as question words and greetings). Once you have run
`models/continuous_ctc/vocab.py` against the real ISL-CSLTR gloss column,
extend HINDI_LEXICON / HINGLISH_LEXICON with every remaining vocabulary
token the real dataset uses -- this file cannot guess words it has never
seen, and shipping an incomplete lexicon silently is worse than flagging
what's missing, so untranslated tokens are returned unchanged and logged
(see `translate_tokens()`), never silently dropped.
"""

from typing import List, Tuple

# ---------------------------------------------------------------------------
# Re-declared here for a standalone prototype; in the real project these
# already exist in gloss_reorder.py and should be imported from there
# instead of duplicated:
#   from translation_layer.rules.gloss_reorder import (
#       PRONOUN_COPULA, KNOWN_PREDICATES, ARTICLES, MODAL_DROP,
#       gloss_to_sentence,
#   )
# ---------------------------------------------------------------------------
PRONOUN_COPULA = {"i": "am", "you": "are", "we": "are", "they": "are", "he": "is", "she": "is", "it": "is"}
KNOWN_PREDICATES = {
    "free", "hungry", "tired", "bad", "good", "welcome", "afraid", "grateful",
    "sure", "happy", "bored", "cold", "beautiful", "fine", "kind", "nice", "friend",
}

# ---------------------------------------------------------------------------
# 1. Devanagari Hindi copula, keyed the same way as PRONOUN_COPULA.
#    Hindi conjugates the copula by person AND (for "you") by formality --
#    this project defaults "you" -> aap (आप) / hain (हैं), the respectful
#    form, since the deployment context is public-facing (banks, hospitals).
# ---------------------------------------------------------------------------
PRONOUN_HINDI = {
    "i": "मैं", "you": "आप", "we": "हम", "they": "वे",
    "he": "वह", "she": "वह", "it": "यह",
}
COPULA_HINDI = {
    "i": "हूँ", "you": "हैं", "we": "हैं", "they": "हैं",
    "he": "है", "she": "है", "it": "है",
}

# Known predicate adjectives/nouns -> Hindi. NOTE: Hindi adjectives agree in
# gender with the subject (e.g. "अच्छा" masc. vs "अच्छी" fem.) -- this
# prototype ships the masculine/neutral form only and documents gender
# agreement as a known gap, the same way gloss_reorder.py documents its own
# WH-question and possessive-interruption gaps rather than hiding them.
PREDICATE_HINDI = {
    "free": "फ़ुर्सत में", "hungry": "भूखा", "tired": "थका हुआ", "bad": "बुरा",
    "good": "अच्छा", "welcome": "स्वागत है", "afraid": "डरा हुआ",
    "grateful": "आभारी", "sure": "निश्चित", "happy": "खुश", "bored": "बोर",
    "cold": "ठंडा", "beautiful": "सुंदर", "fine": "ठीक", "kind": "दयालु",
    "nice": "अच्छा", "friend": "दोस्त",
}

# A handful of common ISL-CSLTR-style function/question words. Extend this
# once the real vocab list is available.
FUNCTION_WORD_HINDI = {
    "today": "आज", "tomorrow": "कल", "yesterday": "कल", "not": "नहीं",
    "who": "कौन", "what": "क्या", "where": "कहाँ", "when": "कब",
    "why": "क्यों", "how": "कैसे", "yes": "हाँ", "no": "नहीं",
    "my": "मेरा", "your": "आपका", "his": "उसका", "her": "उसकी",
    "please": "कृपया", "thank": "धन्यवाद", "sorry": "माफ़ कीजिए",
    "help": "मदद", "name": "नाम", "hospital": "अस्पताल", "bank": "बैंक",
    "money": "पैसे", "doctor": "डॉक्टर", "account": "खाता", "form": "फॉर्म",
}

# ---------------------------------------------------------------------------
# 2. Hinglish (Latin-script, code-mixed): reuses English sentence STRUCTURE,
#    swaps in Roman-Hindi words for common terms. This is deliberately a
#    lighter-touch transform than the Hindi path above.
# ---------------------------------------------------------------------------
HINGLISH_SUBSTITUTIONS = {
    "i": "main", "you": "aap", "we": "hum", "they": "wo log",
    "he": "wo", "she": "wo", "it": "ye",
    "today": "aaj", "tomorrow": "kal", "yesterday": "kal", "not": "nahi",
    "who": "kaun", "what": "kya", "where": "kahan", "when": "kab", "why": "kyun",
    "how": "kaise", "yes": "haan", "no": "nahi",
    "my": "mera", "your": "aapka", "his": "uska", "her": "uski",
    "free": "free", "hungry": "bhookha", "tired": "thaka hua", "bad": "bura",
    "good": "accha", "friend": "dost", "happy": "khush", "fine": "theek",
    "thank": "dhanyavaad", "please": "please", "sorry": "sorry",
    "help": "help", "name": "naam", "hospital": "hospital", "bank": "bank",
    "money": "paise", "doctor": "doctor", "account": "account", "form": "form",
}


def translate_tokens(tokens: List[str], lexicon: dict) -> Tuple[List[str], List[str]]:
    """Word-for-word substitution against a lexicon. Returns (translated,
    unmatched) so callers can see exactly what wasn't covered -- silent
    partial coverage is how a translation layer quietly ships wrong output,
    so this project logs the gap instead of hiding it."""
    out, unmatched = [], []
    for tok in tokens:
        key = tok.lower()
        if key in lexicon:
            out.append(lexicon[key])
        else:
            out.append(tok)  # leave English word in place rather than dropping it
            unmatched.append(tok)
    return out, unmatched


def gloss_to_hindi(gloss_tokens: List[str]) -> Tuple[str, List[str]]:
    """
    ISL gloss (e.g. ["YOU", "FREE", "TODAY"]) -> Hindi sentence.
    Because ISL and Hindi both lean SOV/topic-comment (Zeshan, 2000), this
    does far LESS reordering than the English path: mostly substitute
    words in place, then attach the copula at the END (Hindi's default
    clause-final copula position) when the gloss looks like a
    pronoun+predicate sentence, mirroring how gloss_to_sentence() detects
    the same pattern for English.
    """
    tokens = [t.lower() for t in gloss_tokens]
    unmatched: List[str] = []
    words: List[str] = []

    pronoun = tokens[0] if tokens and tokens[0] in PRONOUN_HINDI else None
    predicate = next((t for t in tokens if t in PREDICATE_HINDI), None)

    for tok in tokens:
        if tok in PRONOUN_HINDI:  # translate ANY pronoun occurrence, not just the subject
            words.append(PRONOUN_HINDI[tok])
        elif tok in PREDICATE_HINDI:
            words.append(PREDICATE_HINDI[tok])
        elif tok in FUNCTION_WORD_HINDI:
            words.append(FUNCTION_WORD_HINDI[tok])
        else:
            words.append(tok.capitalize())  # unknown token, kept visible
            unmatched.append(tok)

    if pronoun and predicate:
        words.append(COPULA_HINDI[pronoun])  # copula goes last, Hindi-style

    sentence = " ".join(words) + "।"  # Devanagari full stop
    return sentence, unmatched


ROMAN_COPULA_BY_PRONOUN = {
    "i": "hoon", "you": "hain", "we": "hain", "they": "hain", "he": "hai", "she": "hai", "it": "hai",
}


def gloss_to_hinglish(gloss_tokens: List[str], english_sentence: str) -> Tuple[str, List[str]]:
    """
    Takes the ALREADY-BUILT English sentence from the existing
    gloss_to_sentence() (English grammar/word order kept as-is) and
    substitutes common words with their Roman-Hindi equivalents --
    matching how Hinglish is actually spoken (English sentence frame,
    Hindi content words), rather than trying to apply Hindi grammar in
    Latin script. The copula (am/is/are) is substituted CONTEXTUALLY based
    on the most recent pronoun seen, since "hoon/ho/hai/hain" depend on
    person -- a single flat word->word lookup can't do this correctly.
    """
    words = english_sentence.rstrip("?.").split(" ")
    unmatched: List[str] = []
    out_words = []
    last_pronoun = None
    for w in words:
        key = w.lower().strip(",")
        if key in PRONOUN_HINDI:  # tracks i/you/we/they/he/she/it
            last_pronoun = key
            out_words.append(HINGLISH_SUBSTITUTIONS.get(key, w))
        elif key in ("am", "is", "are") and last_pronoun:
            out_words.append(ROMAN_COPULA_BY_PRONOUN[last_pronoun])
        elif key in HINGLISH_SUBSTITUTIONS:
            out_words.append(HINGLISH_SUBSTITUTIONS[key])
        else:
            out_words.append(w)
            if key.isalpha() and key not in ("is", "are", "am", "a", "an", "the"):
                unmatched.append(w)
    suffix = "?" if english_sentence.strip().endswith("?") else "."
    return " ".join(out_words) + suffix, unmatched


if __name__ == "__main__":
    # Smoke test against the same example used elsewhere in this project's
    # demos: gold gloss for "are_you_free_today"
    demo_gloss = ["YOU", "FREE", "TODAY"]
    demo_english = "You are free today."  # what gloss_to_sentence() already produces

    hi, hi_missing = gloss_to_hindi(demo_gloss)
    hg, hg_missing = gloss_to_hinglish(demo_gloss, demo_english)

    print("Gloss   :", " ".join(demo_gloss))
    print("English :", demo_english)
    print("Hindi   :", hi, " (untranslated tokens:", hi_missing, ")")
    print("Hinglish:", hg, " (untranslated tokens:", hg_missing, ")")
