"""Entry point for the will-checker CLI."""

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console

from .models import CheckReport
from .section_classifier import classify_sections, find_missing_sections
from .ner_extraction import extract_entities_from_fact_pattern, extract_entities_from_will, load_nlp_pipeline
from .entity_comparison import compare_entities
from .semantic_comparison import compare_semantically, embed_sections, load_model
from .consistency_checker import check_consistency
from .reporter import render_terminal_report, save_json_report


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="will-checker",
        description=(
            "Detect and categorize errors in LLM-generated wills by comparing them "
            "against structured attorney fact patterns."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  will-checker samples/sample_fact_pattern.json samples/sample_will_with_errors.txt
  will-checker fact.json will.txt --output report.json --semantic-threshold 0.70
  will-checker fact.json will.txt --no-color --verbose
        """,
    )
    parser.add_argument("fact_pattern", help="Path to fact pattern JSON file")
    parser.add_argument("will_text", help="Path to LLM-generated will text file")
    parser.add_argument(
        "--output", "-o",
        default="will_checker_report.json",
        metavar="FILE",
        help="Path for JSON output report (default: will_checker_report.json)",
    )
    parser.add_argument(
        "--fuzzy-threshold",
        type=int,
        default=85,
        metavar="N",
        help="Fuzzy match threshold for entity comparison, 0-100 (default: 85)",
    )
    parser.add_argument(
        "--semantic-threshold",
        type=float,
        default=0.75,
        metavar="F",
        help="Cosine similarity threshold for semantic drift, 0-1 (default: 0.75)",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable rich terminal colors",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-stage progress and debug info",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    console = Console(no_color=args.no_color)

    # Load inputs
    fact_pattern_path = Path(args.fact_pattern)
    will_text_path = Path(args.will_text)

    if not fact_pattern_path.exists():
        console.print(f"[red]Error:[/red] Fact pattern file not found: {fact_pattern_path}")
        return 1
    if not will_text_path.exists():
        console.print(f"[red]Error:[/red] Will text file not found: {will_text_path}")
        return 1

    try:
        fact_pattern = json.loads(fact_pattern_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        console.print(f"[red]Error:[/red] Invalid JSON in fact pattern: {e}")
        return 1

    will_text = will_text_path.read_text(encoding="utf-8")

    if args.verbose:
        console.print("[dim]Inputs loaded successfully.[/dim]")

    # ── Stage 3: Section Classification (must run before NER for section assignment) ──
    if args.verbose:
        console.print("[bold cyan]Stage 3:[/bold cyan] Classifying will sections...")
    sections = classify_sections(will_text)
    structural_errors = find_missing_sections(sections)
    if args.verbose:
        console.print(
            f"  Found {len(sections)} section(s); "
            f"{len(structural_errors)} structural error(s)"
        )

    # ── Stage 1: NER Extraction ──
    if args.verbose:
        console.print("[bold cyan]Stage 1:[/bold cyan] Extracting named entities...")
    try:
        nlp = load_nlp_pipeline()
    except OSError as e:
        console.print(f"[red]Error:[/red] {e}")
        return 1

    fp_entities = extract_entities_from_fact_pattern(fact_pattern)
    will_entities = extract_entities_from_will(will_text, nlp, sections)

    # Assign entities back to their sections
    for sec in sections:
        sec.entities = [
            e for e in will_entities
            if e.section == sec.section_type.value
        ]

    if args.verbose:
        console.print(
            f"  Fact pattern: {len(fp_entities)} entities; "
            f"Will: {len(will_entities)} entities"
        )

    # ── Stage 2: Entity Comparison ──
    if args.verbose:
        console.print("[bold cyan]Stage 2:[/bold cyan] Comparing entities...")
    factual_errors = compare_entities(fp_entities, will_entities, args.fuzzy_threshold)
    if args.verbose:
        console.print(f"  {len(factual_errors)} factual error(s) detected")

    # ── Stage 4: Semantic Comparison ──
    if args.verbose:
        console.print("[bold cyan]Stage 4:[/bold cyan] Loading sentence-transformers model...")
    st_model = load_model()
    if st_model is None and args.verbose:
        console.print(
            "  [yellow]Warning:[/yellow] sentence-transformers model unavailable "
            "(no internet / not cached). Using TF-IDF cosine similarity as fallback."
        )

    if args.verbose:
        console.print("[bold cyan]Stage 4:[/bold cyan] Running semantic comparison...")
    sections = embed_sections(sections, st_model)
    semantic_errors = compare_semantically(
        sections, fact_pattern, st_model, args.semantic_threshold
    )
    if args.verbose:
        console.print(f"  {len(semantic_errors)} semantic/hallucination error(s) detected")

    # ── Stage 5: Consistency Check ──
    if args.verbose:
        console.print("[bold cyan]Stage 5:[/bold cyan] Checking internal consistency...")
    consistency_errors = check_consistency(sections, fact_pattern)
    if args.verbose:
        console.print(f"  {len(consistency_errors)} consistency error(s) detected")

    # ── Combine and deduplicate ──
    all_errors = structural_errors + factual_errors + semantic_errors + consistency_errors

    # Deduplicate on (category, relevant_text) — keep highest confidence
    seen: dict[tuple, DetectedError] = {}
    from .models import DetectedError
    for err in all_errors:
        key = (err.category.value, err.relevant_text[:60])
        if key not in seen or err.confidence > seen[key].confidence:
            seen[key] = err
    all_errors = list(seen.values())

    # Sort: structural first, then factual, legal, hallucinated, inconsistency
    _order = {
        "structural_error": 0,
        "factual_error": 1,
        "legal_error": 2,
        "hallucinated_content": 3,
        "internal_inconsistency": 4,
    }
    all_errors.sort(key=lambda e: _order.get(e.category.value, 99))

    report = CheckReport(errors=all_errors)
    report.tally()

    # ── Output ──
    render_terminal_report(report, console)

    try:
        save_json_report(report, args.output)
        console.print(f"[green]JSON report saved to:[/green] {args.output}")
    except OSError as e:
        console.print(f"[yellow]Warning:[/yellow] Could not save JSON report: {e}")

    return 0 if not all_errors else 1


if __name__ == "__main__":
    sys.exit(main())
