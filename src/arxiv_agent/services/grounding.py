"""Conservative sentence-level checks for source-linked generated claims.

These checks are a guardrail, not a general natural-language entailment model.
They reject claims that need facts scattered across unrelated source sentences.
"""

import re
import unicodedata

_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "in", "is", "it", "of", "on", "or", "our", "the", "their", "this", "to", "we",
    "with", "was", "were", "that", "which", "while",
}
_ACTIONS = {
    "freeze": re.compile(r"\b(?:freez\w*|frozen)\b", re.I),
    "train": re.compile(r"\btrain\w*\b", re.I),
    "reduce": re.compile(r"\breduc\w*\b", re.I),
    "increase": re.compile(r"\bincreas\w*\b", re.I),
    "improve": re.compile(r"\bimprov\w*\b", re.I),
    "outperform": re.compile(r"\boutperform\w*\b", re.I),
    "replace": re.compile(r"\b(?:replac\w*|dispens\w* with)\b", re.I),
    "evaluate": re.compile(r"\bevaluat\w*\b", re.I),
    "batch": re.compile(r"\bbatch\w*\b", re.I),
    "update": re.compile(r"\bupdat\w*\b", re.I),
    "achieve": re.compile(r"\bachiev\w*\b", re.I),
}
_NEGATION = re.compile(r"\b(?:not|never|no|cannot|can't|without)\b", re.I)
_HYPOTHETICAL = re.compile(r"\b(?:if|might|could|would|may)\b", re.I)
_ABSOLUTE = re.compile(r"\b(?:all|every|always|never|only)\b", re.I)
_NUMBER = re.compile(r"(?<!\d)\d[\d,]*(?:\.\d+)?")
_CAUSAL = re.compile(r"\b([A-Za-z0-9]+)\s+(?:causes?|leads? to)\s+([A-Za-z0-9]+)\b", re.I)
_COMPARISON = {
    "positive": re.compile(r"\b(?:higher|better|greater|more than|outperform\w*)\b", re.I),
    "negative": re.compile(r"\b(?:lower|worse|less than|underperform\w*)\b", re.I),
}


def _words(value: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    words = re.findall(r"[a-z0-9]+", normalized)
    result = set()
    for word in words:
        if word in _STOP or len(word) < 2:
            continue
        if word.endswith("ing") and len(word) > 5:
            word = word[:-3]
        elif word.endswith("ed") and len(word) > 4:
            word = word[:-2]
        elif word.endswith("s") and len(word) > 4:
            word = word[:-1]
        result.add(word)
    result.update(
        "label:" + match.group(1).casefold()
        for match in re.finditer(
            r"\b(?:method|model|variant|system|adapter|group)\s+([A-Z])\b",
            value, re.I,
        )
        if match.group(1).isupper()
    )
    result.update(
        "symbol:" + match.group().casefold()
        for match in re.finditer(r"\b[A-Z]\b", value)
        if match.group() not in {"A", "I"}
    )
    return result


def _sentences(value: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<!\d)(?<=[.!?])\s+(?!\d)", " ".join(value.split()))
        if sentence.strip()
    ]


def supported_sentence(claim: str, quote: str) -> str | None:
    """Return one source sentence carrying the claim's terms, actions, and qualifiers."""
    claim_words = _words(claim)
    if not claim_words:
        return None
    claim_numbers = {match.group().replace(",", "") for match in _NUMBER.finditer(claim)}
    required_actions = {
        name for name, pattern in _ACTIONS.items() if pattern.search(claim)
    }
    required_comparison = {
        name for name, pattern in _COMPARISON.items() if pattern.search(claim)
    }
    causal = _CAUSAL.search(claim)
    candidates = []
    for sentence in _sentences(quote):
        source_words = _words(sentence)
        coverage = len(claim_words & source_words) / len(claim_words)
        source_numbers = {
            match.group().replace(",", "") for match in _NUMBER.finditer(sentence)
        }
        if coverage < 0.8 or not claim_numbers <= source_numbers:
            continue
        if any(not _ACTIONS[action].search(sentence) for action in required_actions):
            continue
        if any(not _COMPARISON[direction].search(sentence) for direction in required_comparison):
            continue
        if causal:
            source_causal = _CAUSAL.search(sentence)
            if not source_causal or tuple(
                part.casefold() for part in source_causal.groups()
            ) != tuple(part.casefold() for part in causal.groups()):
                continue
        if bool(_NEGATION.search(claim)) != bool(_NEGATION.search(sentence)):
            continue
        if _HYPOTHETICAL.search(sentence) and not _HYPOTHETICAL.search(claim):
            continue
        if any(
            not re.search(rf"\b{re.escape(word)}\b", sentence, re.I)
            for word in _ABSOLUTE.findall(claim)
        ):
            continue
        candidates.append((coverage, -len(sentence), sentence))
    return max(candidates)[2] if candidates else None
