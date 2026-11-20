"""
Phase 3 — Rule-based ISL <-> English translation layer.

RULES DERIVED FROM REAL DATA, NOT GENERIC ISL GRAMMAR ASSUMPTIONS:
  Every rule below was checked against all 100 real (English sentence,
  ISL gloss) pairs in ISL_CSLTR's "ISL Corpus sign glosses.csv" before
  being encoded — see docs/translation-layer-analysis.md for the full
  quantitative breakdown. This matters because generic claims about ISL
  grammar (e.g. "ISL is SOV" or "question words move to the end") did NOT
  hold in this corpus — what we found instead:

  KEY EMPIRICAL FINDING: content word order in this corpus's glosses is
  IDENTICAL to English word order in every single example. There is no
  observed subject/object/verb reordering, and WH-question words stay
  sentence-initial (not moved to the end as sometimes claimed for ISL).
  The gloss transformation is almost entirely about DELETING certain
  function-word categories, not reordering content words. This may be a
  property of how this specific corpus was glossed (possible influence
  from the English source sentence during annotation) rather than a
  universal fact about ISL — worth discussing as a limitation in the
  final report, not presenting as a general ISL grammar fact.

  Rule confidence, measured against all 100 real pairs:
    - Copula (am/is/are/was/were) dropped: 38/39 (97%) -- very reliable
    - Modal (will/would/shall/can/could) dropped: 11/11 (100%) -- fully reliable
    - "be" dropped only when part of an aux chain (would/will/can + be),
      kept when it's the imperative main verb after negation: n=2, both
      correct with this rule, though small sample
    - Negation (not/dont/don't) kept as-is, never reordered: consistent
      in every example -- fully reliable
    - Articles (a/an/the) dropped: 8/12 (67%) -- weak majority, kept as a
      rule anyway since "mostly dropped" beats "never handled", but this
      is our least reliable rule
    - do/does/did (question auxiliary): kept in 15/21 (71%), dropped in
      6/21 (29%) -- too inconsistent to rule either way; we KEEP by
      default (the majority behavior, and the safer/more information-
      preserving choice)
    - Prepositions: kept in 16/26 (62%), dropped in 10/26 (38%) --
      likewise too inconsistent; we KEEP by default

FORWARD direction (sentence_to_gloss): mainly useful for (a) generating
CTC training targets automatically from new sentence data, and (b) as a
benchmark to quantify how well a simple deletion-only model explains the
real corpus (this itself is a data point for RQ3 in the report).

REVERSE direction (gloss_to_sentence): this is the actual deployed
translation layer for the live system -- turning a recognized sign/gloss
sequence back into a readable English sentence. This direction is lossy
to invert perfectly (we can't always know if "the" was in the original,
or whether "am/is/are" should come back), so it uses a small, extensible
lexicon of pronouns/adjectives seen in the corpus to make a reasonable
reconstruction, and intentionally keeps output simple/telegraphic rather
than guessing wrong.
"""

# ---------- Word categories, derived from the analysis above ----------

COPULA = {"am", "is", "are", "was", "were"}
MODAL_DROP = {"will", "would", "shall", "can", "could"}
ARTICLES = {"a", "an", "the"}

# Pronoun -> appropriate copula, for gloss_to_sentence reconstruction
PRONOUN_COPULA = {
    "i": "am",
    "you": "are",
    "we": "are",
    "they": "are",
    "he": "is",
    "she": "is",
    "it": "is",
}

# Small, extensible lexicon of adjectives/state-nouns seen directly after a
# subject pronoun in the corpus -- used to decide whether to insert a
# copula in gloss_to_sentence. Extend this list as more vocabulary is
# added (e.g. once CISLR is incorporated).
KNOWN_PREDICATES = {
    "free", "hungry", "tired", "bad", "good", "welcome", "afraid",
    "grateful", "sure", "happy", "bored", "cold", "beautiful", "fine",
    "kind", "nice", "friend",
}

WH_WORDS = {"what", "who", "why", "when", "where", "which", "how"}


def _tokenize(text: str) -> list:
    # Keep commas as separate tokens (a couple of gold sentences use them
    # meaningfully, e.g. "you do anything, i do not care"), drop periods.
    text = text.replace(".", "")
    text = text.replace(",", " , ")
    return [t for t in text.lower().split() if t]


def sentence_to_gloss(sentence: str) -> str:
    """
    Forward direction: approximate the ISL gloss for an English sentence
    using the deletion rules above. Returns an uppercase, space-joined
    gloss string (commas preserved where present in input).
    """
    tokens = _tokenize(sentence)
    output = []

    for i, tok in enumerate(tokens):
        prev_tok = tokens[i - 1] if i > 0 else None

        if tok in COPULA:
            continue  # drop: 97% confidence rule
        if tok in MODAL_DROP:
            continue  # drop: 100% confidence rule
        if tok == "be":
            # Drop only when part of an aux chain (preceded by a modal
            # that's also being dropped); keep as imperative main verb
            # otherwise (e.g. "do not be stubborn").
            if prev_tok in MODAL_DROP:
                continue
            output.append(tok)
            continue
        if tok in ARTICLES:
            continue  # drop: 67% confidence rule (our weakest rule)

        output.append(tok)

    result = " ".join(output).upper().replace(" , ", ", ")
    return result


def gloss_to_sentence(gloss: str) -> str:
    """
    Reverse direction: the actual deployed translation layer. Takes a
    recognized gloss sequence and reconstructs a readable English
    sentence. Intentionally conservative -- if we're not confident about
    an insertion (e.g. which article), we leave it out rather than guess
    wrong, since a slightly telegraphic sentence is more honest than a
    fluent but incorrect one.
    """
    gloss = gloss.replace(",", " , ")
    tokens = gloss.lower().split()
    if not tokens:
        return ""

    # Expand merged negation tokens seen in the raw corpus (e.g. "donot")
    expanded = []
    for tok in tokens:
        if tok in ("donot", "dont"):
            expanded.extend(["do", "not"])
        else:
            expanded.append(tok)
    tokens = expanded

    is_question = tokens[0] in WH_WORDS

    # Copula insertion: pronoun immediately followed by a known
    # adjective/state-noun predicate.
    result = []
    for i, tok in enumerate(tokens):
        result.append(tok)
        next_tok = tokens[i + 1] if i + 1 < len(tokens) else None
        if tok in PRONOUN_COPULA and next_tok in KNOWN_PREDICATES:
            result.append(PRONOUN_COPULA[tok])

    sentence = " ".join(result)
    sentence = sentence.replace(" , ", ", ")
    if sentence:
        sentence = sentence[0].upper() + sentence[1:]
    sentence += "?" if is_question else "."
    return sentence
