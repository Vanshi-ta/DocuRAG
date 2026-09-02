"""
Lightweight query understanding for DocuRAG.

Deliberately NOT a real NER system — no spaCy, no LLM call, no external
model. It is a single, cheap heuristic used for exactly one purpose:
deciding whether a question mentions two or more distinct named entities
(e.g. "Vanshita" and "Manas"), so the retriever can fan out one
sub-retrieval per entity instead of relying on a single query embedding to
somehow retrieve evidence for all of them at once (see retriever.py
`retrieve_for_question` for why that matters).

HEURISTIC: capitalized word runs, excluding the sentence's first word
(capitalized purely by position) and a small stoplist of capitalized
question/functional words that show up mid-sentence ("Compare", "What").
Consecutive capitalized words are merged into one entity ("Vanshita
Suryavanshi" -> one entity, not two).

KNOWN LIMITATIONS (stated explicitly, not hidden):
  - Single-word ALL-CAPS acronyms (SQL, API, CEO...) are deliberately
    excluded from entity detection so they don't get treated as named
    entities or merged onto an adjacent real name. This means a genuinely
    named all-caps entity (rare) would be missed — an accepted trade-off
    given how much more common resume/document acronyms are.
  - Pronouns ("her", "their", "his") are NOT resolved to a prior entity —
    there is no conversational memory here. "What are her skills?" will
    not fan out and will retrieve based on the literal query text alone.
    This is a deliberate scope decision, not an oversight — see
    docs/RETRIEVAL.md "Ambiguous questions".
"""

from __future__ import annotations

import re
from typing import List

_QUESTION_OR_FUNCTIONAL_WORDS = {
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "is", "are", "does", "do", "did", "can", "could", "would", "should",
    "compare", "list", "show", "tell", "give", "summarize", "explain",
    "the", "a", "an", "and", "or", "of", "in", "on", "for", "to", "that",
    "than", "does", "not", "between", "with", "has", "have", "had",
}

_WORD_RE = re.compile(r"[A-Za-z]+")


def _is_entity_candidate(word: str) -> bool:
    """
    A capitalized word counts as a possible entity token UNLESS it's an
    all-uppercase acronym (CGPA, SQL, CEO, API, ...) — those are common in
    resumes/technical documents and are not named entities. Restricting to
    Title Case ("Vanshita", "Google") rather than any-capitalized also
    prevents an acronym immediately after a real name from being merged
    into it (e.g. "Vanshita's CGPA" must extract only "Vanshita", not
    "Vanshita CGPA").
    """
    if len(word) < 2:
        return False
    if word.isupper():
        return False
    return word[0].isupper()


def extract_entities(query: str) -> List[str]:
    """
    Return a list of likely named entities mentioned in `query`, in the
    order they first appear, deduplicated case-insensitively. Returns an
    empty list if none are found (e.g. no capitalized words, or a
    lowercase/pronoun-only question).
    """
    words = _WORD_RE.findall(query)
    entities: List[str] = []

    i = 0
    n = len(words)
    while i < n:
        word = words[i]
        is_sentence_start = i == 0
        is_stopword = word.lower() in _QUESTION_OR_FUNCTIONAL_WORDS

        if _is_entity_candidate(word) and not is_sentence_start and not is_stopword:
            span = [word]
            j = i
            while (
                j + 1 < n
                and _is_entity_candidate(words[j + 1])
                and words[j + 1].lower() not in _QUESTION_OR_FUNCTIONAL_WORDS
            ):
                j += 1
                span.append(words[j])
            entities.append(" ".join(span))
            i = j + 1
        else:
            i += 1

    seen = set()
    unique_entities = []
    for entity in entities:
        key = entity.lower()
        if key not in seen:
            seen.add(key)
            unique_entities.append(entity)
    return unique_entities
