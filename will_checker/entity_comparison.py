"""Stage 2: Compare fact-pattern entities against will entities; flag mismatches."""

import re
from .models import DetectedError, ErrorCategory, ExtractedEntity

try:
    from thefuzz import fuzz
except ImportError as e:
    raise ImportError("thefuzz not installed. Run: pip install thefuzz python-Levenshtein") from e

# Per-label fuzzy thresholds (general mismatch detection)
_THRESHOLDS: dict[str, int] = {
    "PERSON": 92,
    "ASSET_REAL_PROPERTY": 88,
    "ASSET_VEHICLE": 88,
    "ASSET_FINANCIAL": 82,
    "ORG": 82,
    "MONEY": 82,
}
_DEFAULT_THRESHOLD = 82

_STREET_NUMBER_RE = re.compile(r"^\d+")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _first_name(full_name: str) -> str:
    """Extract first token (first name) from a full name."""
    tokens = _normalize(full_name).split()
    return tokens[0] if tokens else full_name


def _score(a: str, b: str) -> int:
    a, b = _normalize(a), _normalize(b)
    if len(a) < 30 and len(b) < 30:
        return fuzz.ratio(a, b)
    return fuzz.token_sort_ratio(a, b)


def _check_person_names(
    expected: str,
    candidate: ExtractedEntity,
) -> tuple[bool, int, str]:
    """
    Return (is_mismatch, confidence_score, explanation).
    For PERSON entities, any difference in first name tokens is a mismatch,
    since legal documents require exact name spelling.
    """
    exp_norm = _normalize(expected)
    cand_norm = _normalize(candidate.text)

    if exp_norm == cand_norm:
        return False, 0, ""

    # Compare first names — a single-character difference still matters legally
    exp_first = _first_name(expected)
    cand_first = _first_name(candidate.text)

    first_ratio = fuzz.ratio(exp_first, cand_first)
    overall_ratio = _score(expected, candidate.text)

    # Any non-exact match of the first name in a legal document is flagged
    if exp_first != cand_first:
        confidence = max(round(1.0 - first_ratio / 100.0, 2), 0.05)
        explanation = (
            f"Name mismatch (PERSON): expected '{expected}', found '{candidate.text}'. "
            f"First name '{exp_first}' ≠ '{cand_first}' (similarity {first_ratio}%)"
        )
        return True, confidence, explanation

    return False, 0, ""


def _check_address(
    expected: str,
    candidate: ExtractedEntity,
) -> tuple[bool, int, str]:
    """
    For ASSET_REAL_PROPERTY, compare the street number exactly.
    Transpositions like "841" vs "814" must be flagged regardless of overall similarity.
    """
    exp_norm = _normalize(expected)
    cand_norm = _normalize(candidate.text)

    if exp_norm == cand_norm:
        return False, 0, ""

    exp_num = _STREET_NUMBER_RE.match(exp_norm)
    cand_num = _STREET_NUMBER_RE.match(cand_norm)

    if exp_num and cand_num and exp_num.group(0) != cand_num.group(0):
        confidence = 0.90  # high confidence: house numbers must match exactly
        explanation = (
            f"Address mismatch (ASSET_REAL_PROPERTY): expected '{expected}', "
            f"found '{candidate.text}'. "
            f"Street number '{exp_num.group(0)}' ≠ '{cand_num.group(0)}'"
        )
        return True, confidence, explanation

    # Fall back to general fuzzy check
    score = _score(expected, candidate.text)
    threshold = _THRESHOLDS.get("ASSET_REAL_PROPERTY", _DEFAULT_THRESHOLD)
    if score < threshold:
        confidence = max(round(1.0 - score / 100.0, 2), 0.05)
        return True, confidence, (
            f"Address mismatch: expected '{expected}', found '{candidate.text}' "
            f"(similarity {score}%)"
        )

    return False, 0, ""


def _best_match(
    target: str, candidates: list[ExtractedEntity]
) -> tuple[ExtractedEntity, int]:
    best = max(candidates, key=lambda c: _score(target, c.text))
    return best, _score(target, best.text)


def compare_entities(
    fact_entities: list[ExtractedEntity],
    will_entities: list[ExtractedEntity],
    fuzzy_threshold: int = _DEFAULT_THRESHOLD,
) -> list[DetectedError]:
    """
    For each fact-pattern entity, find the best-matching will entity of the same label.
    Uses label-specific comparison strategies:
      - PERSON: exact first-name check (catches single-char typos like Sara/Sarah)
      - ASSET_REAL_PROPERTY: exact street number check (catches transpositions like 841/814)
      - Others: fuzzy ratio below threshold
    """
    errors: list[DetectedError] = []
    seen: set[tuple[str, str]] = set()  # (expected_text, will_text) dedup

    will_by_label: dict[str, list[ExtractedEntity]] = {}
    for ent in will_entities:
        will_by_label.setdefault(ent.label, []).append(ent)

    for fp_ent in fact_entities:
        candidates = will_by_label.get(fp_ent.label, [])
        if not candidates:
            continue

        best, score = _best_match(fp_ent.text, candidates)
        key = (_normalize(fp_ent.text), _normalize(best.text))

        is_mismatch = False
        confidence = 0.0
        explanation = ""

        if fp_ent.label == "PERSON":
            is_mismatch, confidence, explanation = _check_person_names(fp_ent.text, best)
        elif fp_ent.label == "ASSET_REAL_PROPERTY":
            is_mismatch, confidence, explanation = _check_address(fp_ent.text, best)
        else:
            threshold = _THRESHOLDS.get(fp_ent.label, fuzzy_threshold)
            if score < threshold:
                is_mismatch = True
                confidence = max(round(1.0 - score / 100.0, 2), 0.05)
                explanation = (
                    f"Entity mismatch ({fp_ent.label}): "
                    f"expected '{fp_ent.text}', found '{best.text}' "
                    f"(similarity {score}%, threshold {threshold}%)"
                )

        if is_mismatch and key not in seen:
            seen.add(key)
            errors.append(DetectedError(
                category=ErrorCategory.FACTUAL_ERROR,
                relevant_text=best.text,
                expected_text=fp_ent.text,
                confidence=confidence,
                section=best.section or "unknown",
                explanation=explanation,
            ))

    return errors
