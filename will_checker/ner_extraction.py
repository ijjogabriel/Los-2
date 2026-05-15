"""Stage 1: Extract named entities from the will and the fact pattern."""

import re
from typing import Optional

try:
    import spacy
    from spacy.language import Language
except ImportError as e:
    raise ImportError("spaCy not installed. Run: pip install spacy") from e

from .models import ExtractedEntity, WillSection

# Loaded once at import time; raises helpful error if model missing.
_NLP: Optional[Language] = None


def load_nlp_pipeline() -> Language:
    global _NLP
    if _NLP is not None:
        return _NLP
    try:
        nlp = spacy.load("en_core_web_lg")
    except OSError:
        raise OSError(
            "spaCy model 'en_core_web_lg' not found.\n"
            "Install it with:  python -m spacy download en_core_web_lg\n"
            "Or via pip:       pip install en-core-web-lg"
        )

    ruler = nlp.add_pipe("entity_ruler", before="ner", config={"overwrite_ents": False})
    ruler.add_patterns(_build_ruler_patterns())
    _NLP = nlp
    return nlp


def _build_ruler_patterns() -> list[dict]:
    return [
        # Real property: street addresses
        {
            "label": "ASSET_REAL_PROPERTY",
            "pattern": [
                {"IS_DIGIT": True},
                {"IS_ALPHA": True, "OP": "+"},
                {"LOWER": {"IN": [
                    "street", "st", "avenue", "ave", "drive", "dr",
                    "road", "rd", "lane", "ln", "boulevard", "blvd",
                    "court", "ct", "place", "pl", "way", "circle", "cir",
                ]}}
            ],
        },
        # Real property: descriptive phrases
        {"label": "ASSET_REAL_PROPERTY", "pattern": [{"LOWER": "family"}, {"LOWER": "home"}]},
        {"label": "ASSET_REAL_PROPERTY", "pattern": [{"LOWER": "real"}, {"LOWER": "property"}]},
        {"label": "ASSET_REAL_PROPERTY", "pattern": [{"LOWER": "real"}, {"LOWER": "estate"}]},
        {"label": "ASSET_REAL_PROPERTY", "pattern": [{"LOWER": "my"}, {"LOWER": "residence"}]},
        # Vehicles: year + make
        {
            "label": "ASSET_VEHICLE",
            "pattern": [
                {"SHAPE": "dddd"},
                {"LOWER": {"IN": [
                    "toyota", "ford", "honda", "chevrolet", "chevy", "bmw",
                    "subaru", "nissan", "hyundai", "kia", "volkswagen", "vw",
                    "dodge", "jeep", "ram", "tesla", "lexus", "acura",
                    "mercedes", "audi", "volvo", "mazda",
                ]}},
            ],
        },
        {"label": "ASSET_VEHICLE", "pattern": [{"LOWER": "automobile"}]},
        {"label": "ASSET_VEHICLE", "pattern": [{"LOWER": "my"}, {"LOWER": "vehicle"}]},
        # Financial accounts
        {
            "label": "ASSET_FINANCIAL",
            "pattern": [
                {"IS_ALPHA": True},
                {"LOWER": {"IN": [
                    "account", "ira", "401k", "403b", "brokerage",
                    "checking", "savings", "annuity",
                ]}},
            ],
        },
        {
            "label": "ASSET_FINANCIAL",
            "pattern": [
                {"LOWER": {"IN": [
                    "schwab", "fidelity", "vanguard", "merrill", "etrade",
                    "td ameritrade", "jpmorgan", "wells fargo",
                ]}},
            ],
        },
        {
            "label": "ASSET_FINANCIAL",
            "pattern": [
                {"LOWER": "investment"},
                {"LOWER": "account"},
            ],
        },
        # Personal representative / executor designations
        {
            "label": "REP_DESIGNATION",
            "pattern": [{"LOWER": "personal"}, {"LOWER": "representative"}],
        },
        {
            "label": "REP_DESIGNATION",
            "pattern": [{"LOWER": {"IN": ["executor", "executrix", "co-executor"]}}],
        },
        {
            "label": "REP_DESIGNATION",
            "pattern": [{"LOWER": "trustee"}],
        },
        {
            "label": "REP_DESIGNATION",
            "pattern": [{"LOWER": "guardian"}],
        },
    ]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _find_section_for_offset(
    char_offset: int, sections: list[WillSection]
) -> str:
    for sec in sections:
        if sec.start_char <= char_offset <= sec.end_char:
            return sec.section_type.value
    return "unknown"


def extract_entities_from_will(
    will_text: str,
    nlp: Language,
    sections: Optional[list[WillSection]] = None,
) -> list[ExtractedEntity]:
    """Run spaCy NER on will text and return ExtractedEntity list."""
    doc = nlp(will_text)
    entities: list[ExtractedEntity] = []

    for ent in doc.ents:
        # Map spaCy built-in labels to our canonical labels
        label = ent.label_
        if label in {"PERSON", "ORG", "MONEY", "DATE",
                     "ASSET_REAL_PROPERTY", "ASSET_VEHICLE",
                     "ASSET_FINANCIAL", "REP_DESIGNATION"}:
            section_name = (
                _find_section_for_offset(ent.start_char, sections)
                if sections
                else "unknown"
            )
            entities.append(ExtractedEntity(
                label=label,
                text=ent.text,
                normalized=_normalize(ent.text),
                source="will",
                section=section_name,
                start_char=ent.start_char,
                end_char=ent.end_char,
            ))

    return entities


def extract_entities_from_fact_pattern(fact_pattern: dict) -> list[ExtractedEntity]:
    """
    Extract ground-truth entities directly from the structured fact pattern JSON.
    No NLP is used here — the JSON is already structured.
    """
    entities: list[ExtractedEntity] = []

    def add(label: str, text: str):
        entities.append(ExtractedEntity(
            label=label,
            text=text,
            normalized=_normalize(text),
            source="fact_pattern",
            section="fact_pattern",
        ))

    # Testator
    testator = fact_pattern.get("testator", {})
    if name := testator.get("name"):
        add("PERSON", name)

    # Spouse
    if spouse := fact_pattern.get("spouse", {}):
        if name := spouse.get("name"):
            add("PERSON", name)

    # Children
    for child in fact_pattern.get("children", []):
        if name := child.get("name"):
            add("PERSON", name)

    # Personal representative
    pr = fact_pattern.get("personal_representative", {})
    if primary := pr.get("primary"):
        add("PERSON", primary)
    if alternate := pr.get("alternate"):
        add("PERSON", alternate)

    # Assets
    for asset in fact_pattern.get("assets", []):
        asset_type = asset.get("type", "")
        if asset_type == "real_property":
            if address := asset.get("address"):
                add("ASSET_REAL_PROPERTY", address)
        elif asset_type == "vehicle":
            if desc := asset.get("description"):
                add("ASSET_VEHICLE", desc)
        elif asset_type == "financial_account":
            if desc := asset.get("description"):
                add("ASSET_FINANCIAL", desc)

        # Beneficiary names within assets
        beneficiary = asset.get("beneficiary")
        if isinstance(beneficiary, str):
            add("PERSON", beneficiary)
        elif isinstance(beneficiary, list):
            for b in beneficiary:
                add("PERSON", b)

    # Trust beneficiary
    if trust := fact_pattern.get("trust", {}):
        if name := trust.get("beneficiary"):
            add("PERSON", name)

    # Note: witness count is checked in Stage 5 (consistency_checker), not via NER comparison

    return entities


def extract_all(
    fact_pattern: dict,
    will_text: str,
    sections: Optional[list[WillSection]] = None,
) -> tuple[list[ExtractedEntity], list[ExtractedEntity]]:
    """Return (fact_pattern_entities, will_entities)."""
    nlp = load_nlp_pipeline()
    fp_entities = extract_entities_from_fact_pattern(fact_pattern)
    will_entities = extract_entities_from_will(will_text, nlp, sections)
    return fp_entities, will_entities
