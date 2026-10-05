

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
    
    if len(word) < 2:
        return False
    if word.isupper():
        return False
    return word[0].isupper()


def extract_entities(query: str) -> List[str]:
    
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
