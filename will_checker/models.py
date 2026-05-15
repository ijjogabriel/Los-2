from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ErrorCategory(str, Enum):
    FACTUAL_ERROR = "factual_error"
    STRUCTURAL_ERROR = "structural_error"
    LEGAL_ERROR = "legal_error"
    HALLUCINATED_CONTENT = "hallucinated_content"
    INTERNAL_INCONSISTENCY = "internal_inconsistency"


class SectionType(str, Enum):
    PREAMBLE = "preamble"
    REVOCATION = "revocation_of_prior_wills"
    PERSONAL_REPRESENTATIVE = "personal_representative_appointment"
    SPECIFIC_BEQUESTS = "specific_bequests"
    RESIDUARY_CLAUSE = "residuary_clause"
    TRUST_PROVISIONS = "trust_provisions"
    GUARDIANSHIP = "guardianship_provisions"
    ATTESTATION = "attestation_clause"
    SELF_PROVING_AFFIDAVIT = "self_proving_affidavit"
    UNKNOWN = "unknown"


@dataclass
class ExtractedEntity:
    label: str
    text: str
    normalized: str
    source: str  # "fact_pattern" or "will"
    section: Optional[str] = None
    start_char: int = 0
    end_char: int = 0


@dataclass
class WillSection:
    section_type: SectionType
    text: str
    start_char: int
    end_char: int
    entities: list = field(default_factory=list)
    embedding: Optional[list] = None


@dataclass
class DetectedError:
    category: ErrorCategory
    relevant_text: str
    expected_text: str
    confidence: float
    section: Optional[str] = None
    explanation: str = ""


@dataclass
class CheckReport:
    errors: list = field(default_factory=list)
    counts_by_category: dict = field(default_factory=dict)

    def tally(self):
        self.counts_by_category = {}
        for e in self.errors:
            key = e.category.value
            self.counts_by_category[key] = self.counts_by_category.get(key, 0) + 1
