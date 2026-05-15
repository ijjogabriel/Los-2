"""Stage 5: Cross-section consistency checks — detect internal contradictions."""

import re
from collections import defaultdict

from .models import DetectedError, ErrorCategory, SectionType, WillSection
from .ner_extraction import _normalize

try:
    from thefuzz import fuzz
except ImportError as e:
    raise ImportError("thefuzz not installed. Run: pip install thefuzz python-Levenshtein") from e

_WORD_TO_NUM: dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

# Jurisdiction-specific rules keyed by state code
_JURISDICTION_RULES: dict[str, dict] = {
    "MN": {"required_witnesses": 2, "state_name": "Minnesota", "statute": "Minn. Stat. § 524.2-502"},
    "CA": {"required_witnesses": 2, "state_name": "California"},
    "NY": {"required_witnesses": 2, "state_name": "New York"},
    "TX": {"required_witnesses": 2, "state_name": "Texas"},
    "FL": {"required_witnesses": 2, "state_name": "Florida"},
    "IL": {"required_witnesses": 2, "state_name": "Illinois"},
    "OH": {"required_witnesses": 2, "state_name": "Ohio"},
    "LA": {"required_witnesses": 2, "state_name": "Louisiana", "notarization_required": True},
}


def _get_known_persons(fact_pattern: dict) -> set[str]:
    """Collect all person names from the fact pattern."""
    names: set[str] = set()

    testator = fact_pattern.get("testator", {})
    if name := testator.get("name"):
        names.add(_normalize(name))

    if spouse := fact_pattern.get("spouse", {}):
        if name := spouse.get("name"):
            names.add(_normalize(name))

    for child in fact_pattern.get("children", []):
        if name := child.get("name"):
            names.add(_normalize(name))

    pr = fact_pattern.get("personal_representative", {})
    if name := pr.get("primary"):
        names.add(_normalize(name))
    if name := pr.get("alternate"):
        names.add(_normalize(name))

    for asset in fact_pattern.get("assets", []):
        beneficiary = asset.get("beneficiary")
        if isinstance(beneficiary, str):
            names.add(_normalize(beneficiary))
        elif isinstance(beneficiary, list):
            for b in beneficiary:
                names.add(_normalize(b))

    return names


def check_personal_representative_consistency(
    sections: list[WillSection],
) -> list[DetectedError]:
    """
    Find all PERSON entities in personal-representative contexts across sections.
    If different names are used for the primary representative role → inconsistency.
    """
    errors: list[DetectedError] = []
    # Find all sentences/phrases that name a personal representative
    rep_re = re.compile(
        r"I appoint\s+([A-Z][a-zA-Z\s]+?)(?:\s+as\s+(?:Personal\s+Representative|Executor|successor))",
        re.I,
    )
    successor_re = re.compile(
        r"(?:successor|alternate|if .{0,40}unable)\s+(?:Personal\s+Representative|Executor)[,\s]+I appoint\s+([A-Z][a-zA-Z\s]+)",
        re.I,
    )

    named_reps: list[tuple[str, str]] = []  # (name, section_type)

    full_text = "\n\n".join(s.text for s in sections)

    for m in rep_re.finditer(full_text):
        name = m.group(1).strip()
        named_reps.append((name, "primary"))

    # Also look for alternative formulation with successor/alternate
    alt_re = re.compile(
        r"(?:successor|alternate)\s+Personal\s+Representative[,\s]+(?:I appoint\s+)?([A-Z][a-zA-Z ]+?)(?:\s+as|\s*[,.])",
        re.I,
    )
    for m in alt_re.finditer(full_text):
        name = m.group(1).strip()
        named_reps.append((name, "successor"))

    # Look for "If X is unable ... I appoint Y" patterns
    unable_re = re.compile(
        r"If\s+([A-Z][a-zA-Z ]+?)\s+is\s+unable.*?I appoint\s+([A-Z][a-zA-Z ]+?)\s+as\s+(?:successor\s+)?Personal",
        re.I | re.DOTALL,
    )
    for m in unable_re.finditer(full_text):
        primary = m.group(1).strip()
        successor = m.group(2).strip()
        named_reps.append((primary, "primary"))
        named_reps.append((successor, "successor"))

    if len(named_reps) < 2:
        return errors

    # The primary rep should be consistent across all primary references
    primaries = [name for name, role in named_reps if role == "primary"]
    if len(primaries) >= 2:
        first = _normalize(primaries[0])
        for other in primaries[1:]:
            if fuzz.ratio(first, _normalize(other)) < 85:
                errors.append(DetectedError(
                    category=ErrorCategory.INTERNAL_INCONSISTENCY,
                    relevant_text=other,
                    expected_text=primaries[0],
                    confidence=0.90,
                    section=SectionType.PERSONAL_REPRESENTATIVE.value,
                    explanation=(
                        f"Conflicting personal representatives: '{primaries[0]}' "
                        f"named in one section, '{other}' named in another. "
                        "The same person should serve as primary representative throughout."
                    ),
                ))

    return errors


def check_unknown_persons(
    sections: list[WillSection],
    fact_pattern: dict,
) -> list[DetectedError]:
    """
    Find PERSON entities in the will not matching any known person in the fact pattern.
    Persons named in representative or beneficiary roles are flagged as inconsistencies.
    """
    errors: list[DetectedError] = []
    known = _get_known_persons(fact_pattern)

    # Extract PERSON entities from section entities
    all_persons: list[tuple[str, str]] = []
    for sec in sections:
        for ent in sec.entities:
            if ent.label == "PERSON":
                all_persons.append((ent.text, sec.section_type.value))

    seen_unknowns: set[str] = set()

    for person_text, section_name in all_persons:
        normalized = _normalize(person_text)
        # Skip very short tokens (articles, pronouns) and the testator themselves
        if len(normalized.split()) < 2:
            continue

        best_score = max(
            (fuzz.ratio(normalized, known_name) for known_name in known),
            default=0,
        )
        if best_score < 80 and normalized not in seen_unknowns:
            seen_unknowns.add(normalized)
            errors.append(DetectedError(
                category=ErrorCategory.INTERNAL_INCONSISTENCY,
                relevant_text=person_text,
                expected_text="No person with this name appears in the fact pattern",
                confidence=round(1.0 - best_score / 100.0, 2),
                section=section_name,
                explanation=(
                    f"Person '{person_text}' named in section '{section_name}' "
                    "does not match any person in the fact pattern "
                    f"(best fuzzy match score: {best_score}%). "
                    "This may be an introduced name or internal inconsistency."
                ),
            ))

    return errors


def check_beneficiary_percentage_sum(
    sections: list[WillSection],
) -> list[DetectedError]:
    """Flag distributions within a single section whose percentages sum to ≠ 100%."""
    errors: list[DetectedError] = []
    percent_re = re.compile(r"(\d+(?:\.\d+)?)\s*%")

    for sec in sections:
        if sec.section_type not in (
            SectionType.SPECIFIC_BEQUESTS, SectionType.RESIDUARY_CLAUSE
        ):
            continue
        percentages = [float(m.group(1)) for m in percent_re.finditer(sec.text)]
        if len(percentages) >= 2:
            total = sum(percentages)
            if total > 100.5:  # allow minor floating-point slack
                errors.append(DetectedError(
                    category=ErrorCategory.INTERNAL_INCONSISTENCY,
                    relevant_text=f"Percentages found: {percentages} (sum={total:.1f}%)",
                    expected_text="Beneficiary percentages must sum to 100%",
                    confidence=0.95,
                    section=sec.section_type.value,
                    explanation=(
                        f"Section '{sec.section_type.value}' contains percentage "
                        f"distributions that sum to {total:.1f}%, exceeding 100%."
                    ),
                ))

    return errors


def check_jurisdiction_rules(
    sections: list[WillSection],
    fact_pattern: dict,
) -> list[DetectedError]:
    """
    Verify witness count stated in the will matches the jurisdiction requirement.
    Also checks for three-witnesses vs two-witnesses discrepancies.
    """
    errors: list[DetectedError] = []
    jurisdiction = fact_pattern.get("jurisdiction", {})
    state_code = jurisdiction.get("state_code", "")
    rules = _JURISDICTION_RULES.get(state_code, {})

    required = rules.get("required_witnesses") or jurisdiction.get("required_witnesses")
    if not required:
        return errors

    statute = rules.get("statute", f"{jurisdiction.get('state', state_code)} law")

    # Search all sections for witness count claims
    witness_count_re = re.compile(
        r"(\w+)\s+\(\s*(\d+)\s*\)\s+witnesses?|"
        r"(\w+)\s+witnesses?\s+as\s+required",
        re.I,
    )

    for sec in sections:
        for m in witness_count_re.finditer(sec.text):
            # Try to extract word or digit count
            word_token = (m.group(1) or m.group(3) or "").lower()
            digit_token = m.group(2)

            stated: int | None = None
            if digit_token:
                stated = int(digit_token)
            elif word_token in _WORD_TO_NUM:
                stated = _WORD_TO_NUM[word_token]

            if stated is not None and stated != required:
                errors.append(DetectedError(
                    category=ErrorCategory.LEGAL_ERROR,
                    relevant_text=m.group(0),
                    expected_text=f"{required} witnesses as required by {statute}",
                    confidence=0.95,
                    section=sec.section_type.value,
                    explanation=(
                        f"Will states {stated} witness(es) are required but "
                        f"{jurisdiction.get('state', state_code)} law ({statute}) "
                        f"requires exactly {required}."
                    ),
                ))

    return errors


def check_consistency(
    sections: list[WillSection],
    fact_pattern: dict,
) -> list[DetectedError]:
    """Orchestrate all Stage 5 consistency sub-checks."""
    errors: list[DetectedError] = []
    errors.extend(check_personal_representative_consistency(sections))
    errors.extend(check_unknown_persons(sections, fact_pattern))
    errors.extend(check_beneficiary_percentage_sum(sections))
    errors.extend(check_jurisdiction_rules(sections, fact_pattern))
    return errors
