# will-checker

A Python CLI tool that detects and categorizes errors in LLM-generated wills by comparing them against structured attorney instructions.

## Overview

`will-checker` runs a five-stage NLP pipeline over two inputs — a structured **fact pattern** (JSON) and an **LLM-generated will** (plain text) — and produces a formatted error report. Every detected discrepancy is assigned one of five error categories:

| Category | What it means |
|---|---|
| `factual_error` | Names, addresses, or amounts in the will that don't match the fact pattern |
| `structural_error` | Required will sections that are missing entirely |
| `legal_error` | Provisions that contradict jurisdiction-specific legal requirements |
| `hallucinated_content` | Content with no basis in the fact pattern (invented pets, fabricated statutes) |
| `internal_inconsistency` | Contradictions within the will itself (different reps named in different sections) |

## Setup

**Requirements:** Python 3.11+

```bash
# Clone the repo, then install dependencies
pip install -r requirements.txt

# Install the package in editable mode (enables the `will-checker` command)
pip install -e .
```

The first run downloads two models automatically:
- **spaCy `en_core_web_lg`** (~560 MB) — named entity recognition
- **`all-MiniLM-L6-v2`** (~80 MB) — sentence embeddings for semantic comparison

If automatic download fails, install them manually:
```bash
python -m spacy download en_core_web_lg
```

## Usage

```bash
will-checker <fact_pattern.json> <will.txt> [options]
```

**Run the included sample:**
```bash
will-checker samples/sample_fact_pattern.json samples/sample_will_with_errors.txt
```

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--output FILE` | `will_checker_report.json` | Path for JSON output report |
| `--semantic-threshold F` | `0.75` | Cosine similarity threshold (0-1); lower = more errors flagged |
| `--fuzzy-threshold N` | `85` | Fuzzy match threshold (0-100); lower = more entities flagged |
| `--verbose` | off | Print per-stage progress |
| `--no-color` | off | Disable rich terminal colors |

**Example with options:**
```bash
will-checker fact.json will.txt --output report.json --semantic-threshold 0.70 --verbose
```

## Output

**Terminal:** A rich-formatted table listing every detected error with its category, section, the will text, what was expected, and a confidence score.

**JSON file** (default: `will_checker_report.json`):
```json
{
  "summary": {
    "total_errors": 7,
    "by_category": {
      "factual_error": 2,
      "structural_error": 1,
      "legal_error": 1,
      "hallucinated_content": 2,
      "internal_inconsistency": 2
    }
  },
  "errors": [
    {
      "id": 1,
      "category": "factual_error",
      "section": "specific_bequests",
      "relevant_text": "Sara Williams",
      "expected_text": "Sarah Williams",
      "confidence": 0.08,
      "explanation": "..."
    }
  ]
}
```

## Sample Files

The `samples/` directory includes a moderately complex estate scenario:

**`sample_fact_pattern.json`** — Testator John Robert Williams, 72, Hennepin County, Minnesota. Second marriage. Three children (David and Sarah Williams from first marriage; Michael Williams, age 14, from current marriage). Specific bequests: family home at 814 Elm Street to spouse Margaret; Schwab account ($340,000) split equally between David and Sarah; 2019 Toyota Camry to Michael. Residuary estate to Margaret. Personal representative: Margaret, alternate David. Trust for Michael until age 25. Two witnesses required per Minnesota law.

**`sample_will_with_errors.txt`** — An LLM-generated will based on the above, containing one error from each category:

| Error type | What's wrong |
|---|---|
| Factual | "Sara Williams" instead of "Sarah Williams"; "841 Elm Street" instead of "814 Elm Street" |
| Structural | Attestation clause with witness signature lines is absent |
| Legal | "three (3) witnesses" — Minnesota requires exactly two |
| Hallucinated | Pet trust for "family dog, Max" + citation `Minn. Stat. § 524.3-407` not in fact pattern |
| Internal inconsistency | Margaret Williams as personal rep in Article V, but "Elizabeth Williams" named as successor (Elizabeth is not in the fact pattern) |

## Pipeline Stages and NLP Concepts

### Stage 1 — NER Extraction (`ner_extraction.py`)

**NLP concept: Named Entity Recognition (NER)**

Uses spaCy's `en_core_web_lg` model to extract entities with labels `PERSON`, `ORG`, `MONEY`, and `DATE`. A custom `EntityRuler` component is added *before* the built-in NER pipe to recognize legal-domain entities: `ASSET_REAL_PROPERTY` (street addresses, "family home"), `ASSET_VEHICLE` (year + make patterns), `ASSET_FINANCIAL` (brokerage/account descriptions), and `REP_DESIGNATION` (executor/personal representative phrases).

Fact-pattern entities are extracted directly from the JSON structure — no NLP needed there, giving Stage 2 a reliable ground truth.

### Stage 2 — Entity Comparison (`entity_comparison.py`)

**NLP concept: Fuzzy string matching; Precision and Recall tradeoffs**

Compares each fact-pattern entity against will entities of the same label using the `thefuzz` library. Short tokens (names, addresses) use `fuzz.ratio` which is sensitive to single-character transpositions. Longer strings use `fuzz.token_sort_ratio` which is order-insensitive. Per-label thresholds are tuned: names use a threshold of 92% (catching "Sara" vs "Sarah"), addresses use 90% (catching "841" vs "814"). Below threshold → `factual_error`.

**Precision vs. recall tradeoff:** A lower threshold catches more errors (higher recall) but flags more false positives (lower precision). The defaults are calibrated for legal documents where false negatives (missed errors) are more costly than false positives.

### Stage 3 — Section Classification (`section_classifier.py`)

**NLP concept: Text classification via rule-based pattern matching**

Splits the will into chunks at article/section headings and classifies each chunk by scoring it against a library of regex patterns. Each section type has distinctive vocabulary: attestation clauses contain "in witness whereof" and "subscribed"; residuary clauses contain "rest, residue, and remainder." The classification with the most pattern matches wins. After classification, the set of detected section types is diffed against a list of required sections — any absence is flagged as a `structural_error`.

### Stage 4 — Semantic Comparison (`semantic_comparison.py`)

**NLP concept: Dense sentence embeddings and cosine similarity**

Loads `sentence-transformers/all-MiniLM-L6-v2`, a 22M-parameter transformer fine-tuned on semantic textual similarity tasks. Each will section and its corresponding fact-pattern instruction are encoded into 384-dimensional dense vectors. Cosine similarity measures the angle between them: 1.0 = identical meaning, 0.0 = orthogonal.

Sections scoring below the configurable threshold (default 0.75) are flagged as `legal_error` (semantic drift — the will may have changed the meaning of the instructions). Sections with no matching fact-pattern instruction are flagged as `hallucinated_content`. Statutory citations not derivable from the fact pattern are also flagged here via regex.

**Why cosine similarity:** It's scale-invariant (ignores vector magnitude) and effective for comparing sentence embeddings in high-dimensional spaces. A threshold of 0.75 corresponds roughly to "related topic but meaningfully different content."

### Stage 5 — Internal Consistency Check (`consistency_checker.py`)

**NLP concept: Cross-document entity resolution and constraint satisfaction**

Four sub-checks:

1. **Representative consistency** — regex-extracts all named personal representatives across sections; flags if different names appear in the primary role
2. **Unknown persons** — fuzzy-matches every PERSON entity in the will against the fact pattern's known-persons list; below 80% match → person is unknown to the fact pattern
3. **Percentage sums** — regex-extracts percentage distributions within bequest sections; flags if they sum above 100%
4. **Jurisdiction rules** — looks up the jurisdiction's required witness count (from an embedded rules table), finds witness count claims in the will via word-to-number conversion, and flags mismatches as `legal_error`

## Project Structure

```
will_checker/
├── __init__.py
├── cli.py                  # argparse entry point, pipeline orchestrator
├── models.py               # dataclasses: DetectedError, WillSection, ExtractedEntity, CheckReport
├── ner_extraction.py       # Stage 1: spaCy NER + EntityRuler
├── entity_comparison.py    # Stage 2: exact + fuzzy entity matching
├── section_classifier.py   # Stage 3: regex-based section classification
├── semantic_comparison.py  # Stage 4: sentence-transformers cosine similarity
├── consistency_checker.py  # Stage 5: cross-section contradiction detection
└── reporter.py             # rich terminal output + JSON serialization
samples/
├── sample_fact_pattern.json
└── sample_will_with_errors.txt
```

## Running the Module Directly

If you haven't installed with `pip install -e .`:

```bash
python -m will_checker.cli samples/sample_fact_pattern.json samples/sample_will_with_errors.txt
```
