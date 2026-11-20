# Translation Layer — Data-Driven Analysis (Phase 3)

Before writing any reordering rules, we analyzed all 100 real (English
sentence, ISL gloss) pairs in ISL-CSLTR's `ISL Corpus sign glosses.csv`
to find the actual transformation patterns, rather than assuming generic
claims about ISL grammar from general linguistic literature.

## Key empirical finding

**Content word order in this corpus's glosses is identical to English
word order in every single one of the 100 examples.** No subject/object/
verb reordering was observed, and WH-question words (what/who/why/when/
where/which/how) always stay sentence-initial — they are never moved to
the end, contrary to some generic claims made about ISL/ASL topic-comment
structure.

The gloss transformation in this corpus is almost entirely about
**deleting specific categories of function words**, not reordering
content. This may reflect how this particular corpus was glossed
(possible influence from the English source sentence during annotation)
rather than being a universal property of ISL — worth discussing as a
limitation/consideration in the final report rather than presenting as a
general fact about the language.

## Rule confidence, measured against all 100 real pairs

| Rule | Confidence | Evidence |
|---|---|---|
| Copula (am/is/are/was/were) dropped | 97% (38/39) | 1 exception: "i am so sorry to hear that" keeps "am" |
| Modal (will/would/shall/can/could) dropped | 100% (11/11) | No exceptions found |
| "be" dropped only in aux chains, kept as imperative main verb after negation | Both correct (n=2) | "would be coming"→dropped; "not be stubborn"→kept |
| Negation (not/dont/don't) preserved, never reordered | 100% | Consistent in every example |
| Articles (a/an/the) dropped | 67% (8/12) | Weakest rule; kept anyway as majority behavior |
| do/does/did (question auxiliary) | Inconsistent: 71% kept, 29% dropped | Rule: keep by default (majority + information-preserving) |
| Prepositions | Inconsistent: 62% kept, 38% dropped | Rule: keep by default (majority + information-preserving) |

## Forward-direction validation (sentence → gloss)

Running our deletion rules against all 100 real sentences and comparing
to the real annotated gloss:

- **Exact match: 65/100 (65%)**
- Remaining mismatches cluster into specific, identifiable categories:
  - Inconsistent do-support handling (~6 cases)
  - Inconsistent preposition handling (~10 cases)
  - Inconsistent `-ing`/base-form lexical choice, e.g. "hiding"→"HIDE" vs "feeling"→"FEELING" both occur (~8 cases)
  - A few outright annotation quirks/typos (e.g. "congratulations"→"CONGRATULATIIONS", a typo in the source data)

This 65% is itself a useful RQ3 data point: **a purely rule-based,
word-list-driven model explains about two-thirds of the real
transformations exactly**, with the remaining third concentrated in
identifiable soft/inconsistent patterns — which is exactly the kind of
gap a learned sequence-to-sequence model (the documented stretch goal)
would be expected to close, since it can learn probabilistic tendencies
rather than requiring hard rules.

## Reverse-direction (gloss → sentence) — the actual deployed layer

This is the direction the live system uses: turning a recognized gloss
sequence back into readable English. Spot-checked against real gold
glosses:

**Working well:**
- `I HUNGRY` → "I am hungry."
- `YOU GOOD` → "You are good."
- `DO NOT BE STUBBORN` → "Do not be stubborn."
- `I DONOT AGREE` → "I do not agree." (merged negation token correctly expanded)

**Known limitation (documented, not yet fixed):** copula insertion only
triggers for the `pronoun → predicate` pattern (as in statements like "I
[am] hungry"). WH-questions where the pronoun comes right after the
WH-word don't get their copula inserted:
- `WHO YOU` → "Who you?" (should ideally be "Who are you?")
- `HOW THINGS` → "How things?" (should ideally be "How are things?")

This needs a second, separate insertion rule (`WH-word → pronoun` pattern)
that wasn't built yet — a clear, scoped next-iteration item rather than a
rushed patch.

## Design choices and why

- **Conservative insertion**: when the reverse direction isn't confident
  about a word choice (e.g. which article to reinsert), it leaves it out
  rather than guessing — a slightly telegraphic sentence is more honest
  than a fluent but wrong one.
- **Small, extensible predicate lexicon**: `KNOWN_PREDICATES` in
  `gloss_reorder.py` is currently built from words seen directly in this
  100-sentence corpus. It's designed to be extended as vocabulary grows
  (e.g. once CISLR's larger vocabulary is incorporated).
