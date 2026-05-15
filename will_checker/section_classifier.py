"""Stage 3: Classify will text into named sections; flag missing required sections."""

import re
from .models import DetectedError, ErrorCategory, SectionType, WillSection

# Regex patterns that identify each section type. Multiple patterns per type; any match wins.
SECTION_PATTERNS: dict[SectionType, list[re.Pattern]] = {
    SectionType.PREAMBLE: [
        re.compile(r"last will and testament", re.I),
        re.compile(r"being of sound.{0,30}mind", re.I),
        re.compile(r"I,\s+[A-Z][a-z].*?declare this to be", re.I | re.DOTALL),
    ],
    SectionType.REVOCATION: [
        re.compile(r"hereby revoke", re.I),
        re.compile(r"revoke.{0,20}prior wills?", re.I),
        re.compile(r"revoke.{0,30}all.{0,30}wills?", re.I),
    ],
    SectionType.PERSONAL_REPRESENTATIVE: [
        re.compile(r"personal representative", re.I),
        re.compile(r"\bexecut(or|rix|ors)\b", re.I),
        re.compile(r"I appoint.{0,60}(representative|executor)", re.I | re.DOTALL),
    ],
    SectionType.SPECIFIC_BEQUESTS: [
        re.compile(r"\bgive.{0,20}bequeath\b", re.I),
        re.compile(r"\bI give\b.{0,40}(to|my)", re.I),
        re.compile(r"\bspecific bequest", re.I),
        re.compile(r"\bI devise\b", re.I),
    ],
    SectionType.RESIDUARY_CLAUSE: [
        re.compile(r"residuary estate", re.I),
        re.compile(r"rest,? residue", re.I),
        re.compile(r"all the rest.{0,40}estate", re.I),
        re.compile(r"remainder of my estate", re.I),
    ],
    SectionType.TRUST_PROVISIONS: [
        re.compile(r"\bheld in trust\b", re.I),
        re.compile(r"\btrustee shall\b", re.I),
        re.compile(r"\bestablish a.{0,20}trust\b", re.I),
        re.compile(r"\btrust for.{0,30}(minor|child|benefit)\b", re.I),
    ],
    SectionType.GUARDIANSHIP: [
        re.compile(r"\bguardian of\b", re.I),
        re.compile(r"\bguardianship\b", re.I),
        re.compile(r"\bnominate.{0,20}guardian\b", re.I),
    ],
    SectionType.ATTESTATION: [
        re.compile(r"\bin witness whereof\b", re.I),
        re.compile(r"\bsigned.{0,30}presence\b", re.I),
        re.compile(r"\bsubscribed.{0,30}testator\b", re.I),
        re.compile(r"\bwit(ness|nessed)\b.{0,60}(sign|subscri)", re.I | re.DOTALL),
    ],
    SectionType.SELF_PROVING_AFFIDAVIT: [
        re.compile(r"\bself.?proving\b", re.I),
        re.compile(r"\bnotary public\b", re.I),
        re.compile(r"\bsworn.{0,30}subscribed\b", re.I),
        re.compile(r"\baffidavit\b", re.I),
    ],
}

# Sections that must be present in a valid will.
REQUIRED_SECTIONS: set[SectionType] = {
    SectionType.PREAMBLE,
    SectionType.REVOCATION,
    SectionType.PERSONAL_REPRESENTATIVE,
    SectionType.SPECIFIC_BEQUESTS,
    SectionType.RESIDUARY_CLAUSE,
    SectionType.ATTESTATION,
}

# Detects "ARTICLE I — HEADING" lines (handles em-dash, en-dash, hyphen separators).
_ARTICLE_HEADING_RE = re.compile(
    r"^(?:ARTICLE|SECTION)\s+[IVXLCDM\d]+\s*[.):\-–—]?\s*[A-Z][A-Z\s\-–—]+$",
    re.MULTILINE,
)


def _split_into_chunks(will_text: str) -> list[tuple[int, int, str]]:
    """
    Split will text at article/section headings, keeping each heading attached
    to the body text that follows it.  Falls back to double-newline splitting
    if no headings are found.
    """
    boundaries = [0]
    for m in _ARTICLE_HEADING_RE.finditer(will_text):
        boundaries.append(m.start())
    boundaries.append(len(will_text))

    boundaries = sorted(set(boundaries))

    chunks = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        text = will_text[start:end].strip()
        if text:
            chunks.append((start, end, text))

    if len(chunks) <= 2:
        # Fallback: split on double newlines
        chunks = []
        pos = 0
        for part in re.split(r"\n\n+", will_text):
            text = part.strip()
            if text:
                start = will_text.find(text, pos)
                end = start + len(text)
                chunks.append((start, end, text))
                pos = end

    return chunks


def _score_chunk(text: str) -> SectionType:
    """Return the SectionType with the most pattern matches for this chunk text."""
    scores: dict[SectionType, int] = {st: 0 for st in SectionType}
    for section_type, patterns in SECTION_PATTERNS.items():
        for pat in patterns:
            if pat.search(text):
                scores[section_type] += 1

    best = max(scores, key=lambda k: scores[k])
    if scores[best] == 0:
        return SectionType.UNKNOWN
    return best


_TRIVIAL_RE = re.compile(
    r"^[\s_\-\*=]{3,}$|^Date:\s*_+$|^[A-Z][a-z]+\s+[A-Z][a-z]+,\s+Testator$",
    re.MULTILINE,
)


def classify_sections(will_text: str) -> list[WillSection]:
    """
    Classify will text into labeled WillSection objects.
    Merges consecutive chunks of the same type.
    Skips trivial lines (signature blanks, date lines).
    """
    chunks = _split_into_chunks(will_text)
    sections: list[WillSection] = []

    for start, end, text in chunks:
        # Skip trivial chunks: blank signature lines, date lines, testator signature
        stripped = text.strip()
        if len(stripped) < 5 or _TRIVIAL_RE.fullmatch(stripped):
            continue

        section_type = _score_chunk(text)
        # Merge with previous section if same type (except UNKNOWN)
        if (
            sections
            and sections[-1].section_type == section_type
            and section_type != SectionType.UNKNOWN
        ):
            prev = sections[-1]
            sections[-1] = WillSection(
                section_type=section_type,
                text=prev.text + "\n\n" + text,
                start_char=prev.start_char,
                end_char=end,
            )
        else:
            sections.append(WillSection(
                section_type=section_type,
                text=text,
                start_char=start,
                end_char=end,
            ))

    return sections


def find_missing_sections(sections: list[WillSection]) -> list[DetectedError]:
    """Emit a structural error for each required section absent from the will."""
    found_types = {s.section_type for s in sections}
    errors: list[DetectedError] = []

    for required in REQUIRED_SECTIONS:
        if required not in found_types:
            errors.append(DetectedError(
                category=ErrorCategory.STRUCTURAL_ERROR,
                relevant_text="(section absent from document)",
                expected_text=f"Required section '{required.value}' must be present",
                confidence=1.0,
                section=required.value,
                explanation=f"The required '{required.value}' section was not found in the will.",
            ))

    return errors
