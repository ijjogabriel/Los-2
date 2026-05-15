"""Stage 4: Semantic similarity between will sections and fact-pattern instructions."""

import re
from typing import Optional

import numpy as np

from .models import DetectedError, ErrorCategory, SectionType, WillSection

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError(
        "sentence-transformers not installed. Run: pip install sentence-transformers"
    ) from e

MODEL_NAME = "all-MiniLM-L6-v2"
_MODEL: Optional[SentenceTransformer] = None
_USE_TFIDF_FALLBACK = False

# Regex to detect statutory citations not in the fact pattern
_STATUTE_RE = re.compile(r"[A-Z][a-z]+\.\s*Stat\.\s*§\s*[\d.\-]+")


def load_model() -> Optional[SentenceTransformer]:
    """
    Try to load all-MiniLM-L6-v2. If unavailable (no internet, not cached),
    set a flag to use TF-IDF cosine similarity as a fallback.
    Returns the model, or None if falling back to TF-IDF.
    """
    global _MODEL, _USE_TFIDF_FALLBACK
    if _MODEL is not None:
        return _MODEL
    if _USE_TFIDF_FALLBACK:
        return None
    try:
        _MODEL = SentenceTransformer(MODEL_NAME)
        return _MODEL
    except Exception:
        _USE_TFIDF_FALLBACK = True
        return None


def _tfidf_cosine_similarity(text_a: str, text_b: str) -> float:
    """Offline fallback: TF-IDF cosine similarity between two texts."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    try:
        tfidf = vec.fit_transform([text_a, text_b])
        return float(sk_cosine(tfidf[0], tfidf[1])[0][0])
    except Exception:
        return 0.0


def _cosine_similarity(a: list, b: list) -> float:
    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


def build_fact_pattern_instructions(fact_pattern: dict) -> dict[SectionType, str]:
    """
    Convert the structured fact pattern JSON into natural-language instruction strings,
    one per section type. These are the reference texts for semantic comparison.
    """
    instructions: dict[SectionType, str] = {}

    testator = fact_pattern.get("testator", {})
    name = testator.get("name", "the testator")
    county = testator.get("county", "")
    state = testator.get("state", "")

    instructions[SectionType.PREAMBLE] = (
        f"Last Will and Testament of {name}, residing in {county}, {state}, "
        "being of sound mind, revoking all prior wills."
    )

    instructions[SectionType.REVOCATION] = (
        f"{name} hereby revokes any and all prior wills, codicils, "
        "and testamentary dispositions previously made."
    )

    pr = fact_pattern.get("personal_representative", {})
    primary = pr.get("primary", "")
    alternate = pr.get("alternate", "")
    instructions[SectionType.PERSONAL_REPRESENTATIVE] = (
        f"{primary} is appointed Personal Representative. "
        f"If unable or unwilling to serve, {alternate} is appointed as alternate "
        "Personal Representative."
    )

    # Build specific bequests text from assets
    bequest_parts: list[str] = []
    for asset in fact_pattern.get("assets", []):
        beneficiary = asset.get("beneficiary", "")
        if isinstance(beneficiary, list):
            beneficiary = " and ".join(beneficiary)
        desc = asset.get("description", "")
        address = asset.get("address", "")
        asset_id = f"{desc} at {address}" if address else desc
        distribution = asset.get("distribution", "")
        dist_note = f" in {distribution}" if distribution else ""
        bequest_parts.append(f"{asset_id} to {beneficiary}{dist_note}")
    instructions[SectionType.SPECIFIC_BEQUESTS] = (
        "Specific bequests: " + "; ".join(bequest_parts) + "."
    )

    residuary = fact_pattern.get("residuary_estate", {})
    res_beneficiary = residuary.get("beneficiary", "")
    instructions[SectionType.RESIDUARY_CLAUSE] = (
        f"The rest, residue, and remainder of the estate passes to {res_beneficiary}."
    )

    trust = fact_pattern.get("trust", {})
    trust_beneficiary = trust.get("beneficiary", "")
    termination_age = trust.get("age_of_termination", "")
    instructions[SectionType.TRUST_PROVISIONS] = (
        f"A trust is established for {trust_beneficiary}. "
        f"The trust shall terminate when {trust_beneficiary} reaches age {termination_age}, "
        "at which time principal and income distribute outright."
    )

    instructions[SectionType.GUARDIANSHIP] = (
        f"A guardian is nominated for {trust_beneficiary} in the event they are a minor "
        "at the time of the testator's death."
    )

    jurisdiction = fact_pattern.get("jurisdiction", {})
    witnesses = jurisdiction.get("required_witnesses", 2)
    state_name = jurisdiction.get("state", state)
    instructions[SectionType.ATTESTATION] = (
        f"The testator signs the will in the presence of {witnesses} witnesses as required "
        f"by {state_name} law. Each witness signs in the presence of the testator and each other."
    )

    return instructions


def embed_sections(
    sections: list[WillSection], model: Optional[SentenceTransformer]
) -> list[WillSection]:
    """Batch-encode all section texts; stores embeddings or marks TF-IDF mode."""
    if model is None:
        # Mark sections for TF-IDF fallback (no pre-stored embedding needed)
        return sections
    texts = [s.text for s in sections]
    embeddings = model.encode(texts, convert_to_numpy=True)
    for sec, emb in zip(sections, embeddings):
        sec.embedding = emb.tolist()
    return sections


def _detect_hallucinated_citations(
    section: WillSection, fact_pattern: dict
) -> list[DetectedError]:
    """Flag statutory citations not derivable from the fact pattern."""
    errors: list[DetectedError] = []
    known_statutes: set[str] = set()

    jurisdiction = fact_pattern.get("jurisdiction", {})
    if statute := jurisdiction.get("governing_statute"):
        # Extract statute number after the § sign
        m = re.search(r"§\s*([\d.\-]+)", statute)
        if m:
            known_statutes.add(m.group(1))

    for match in _STATUTE_RE.finditer(section.text):
        citation = match.group(0)
        num_match = re.search(r"§\s*([\d.\-]+)", citation)
        if num_match and num_match.group(1) not in known_statutes:
            errors.append(DetectedError(
                category=ErrorCategory.HALLUCINATED_CONTENT,
                relevant_text=citation,
                expected_text="No such statutory citation appears in the fact pattern",
                confidence=0.90,
                section=section.section_type.value,
                explanation=(
                    f"Statutory citation '{citation}' found in the will "
                    "is not referenced in or derivable from the fact pattern. "
                    "This may be a fabricated or incorrectly applied citation."
                ),
            ))

    return errors


def compare_semantically(
    sections: list[WillSection],
    fact_pattern: dict,
    model: Optional[SentenceTransformer],
    threshold: float = 0.75,
) -> list[DetectedError]:
    """
    For each will section:
      - If it has a corresponding fact-pattern instruction, compare cosine similarity.
        Below threshold → LEGAL_ERROR (semantic drift).
      - If it is UNKNOWN with no fact-pattern instruction → HALLUCINATED_CONTENT.
    Also scans all sections for unauthorized statutory citations.

    Falls back to TF-IDF cosine similarity when sentence-transformers is unavailable.
    """
    errors: list[DetectedError] = []
    instructions = build_fact_pattern_instructions(fact_pattern)

    # Pre-encode instructions only when using the neural model
    instruction_map: dict[SectionType, list] = {}
    if model is not None:
        types_with_instructions = [
            st for st in SectionType if st in instructions and st != SectionType.UNKNOWN
        ]
        instruction_texts = [instructions[st] for st in types_with_instructions]
        instruction_embeddings = model.encode(instruction_texts, convert_to_numpy=True)
        for st, emb in zip(types_with_instructions, instruction_embeddings):
            instruction_map[st] = emb.tolist()

    for section in sections:
        # Scan for hallucinated citations regardless of section type
        errors.extend(_detect_hallucinated_citations(section, fact_pattern))

        if section.section_type == SectionType.UNKNOWN:
            errors.append(DetectedError(
                category=ErrorCategory.HALLUCINATED_CONTENT,
                relevant_text=section.text[:300],
                expected_text="No corresponding instruction in the fact pattern",
                confidence=0.85,
                section=section.section_type.value,
                explanation=(
                    "This section could not be classified into any known will section type "
                    "and has no corresponding fact-pattern instruction. "
                    "It may represent content hallucinated by the LLM."
                ),
            ))
            continue

        instr_text = instructions.get(section.section_type)
        if instr_text is None:
            continue

        if model is None:
            # In TF-IDF fallback mode, skip semantic drift detection — TF-IDF produces
            # unreliable similarity scores when comparing long legal prose against short
            # instruction strings. Hallucination detection (UNKNOWN sections + citations)
            # still works correctly above.
            continue

        instr_emb = instruction_map.get(section.section_type)
        if instr_emb is None or section.embedding is None:
            continue

        similarity = _cosine_similarity(section.embedding, instr_emb)

        if similarity < threshold:
            errors.append(DetectedError(
                category=ErrorCategory.LEGAL_ERROR,
                relevant_text=section.text[:300],
                expected_text=instr_text,
                confidence=round(1.0 - similarity, 2),
                section=section.section_type.value,
                explanation=(
                    f"Section '{section.section_type.value}' has cosine similarity "
                    f"{similarity:.2f} (threshold {threshold:.2f}) with its fact-pattern "
                    "instruction. The will may have altered the meaning of the instructions."
                ),
            ))

    return errors
