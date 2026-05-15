"""Rich terminal report and JSON output for the will-checker error report."""

import dataclasses
import json
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .models import CheckReport, ErrorCategory, DetectedError

_CATEGORY_COLORS: dict[str, str] = {
    ErrorCategory.FACTUAL_ERROR.value: "yellow",
    ErrorCategory.STRUCTURAL_ERROR.value: "bold red",
    ErrorCategory.LEGAL_ERROR.value: "red",
    ErrorCategory.HALLUCINATED_CONTENT.value: "magenta",
    ErrorCategory.INTERNAL_INCONSISTENCY.value: "dark_orange",
}

_CATEGORY_LABELS: dict[str, str] = {
    ErrorCategory.FACTUAL_ERROR.value: "FACTUAL",
    ErrorCategory.STRUCTURAL_ERROR.value: "STRUCTURAL",
    ErrorCategory.LEGAL_ERROR.value: "LEGAL",
    ErrorCategory.HALLUCINATED_CONTENT.value: "HALLUCINATED",
    ErrorCategory.INTERNAL_INCONSISTENCY.value: "INCONSISTENCY",
}


def _bar(count: int, max_count: int, width: int = 20) -> str:
    if max_count == 0:
        return ""
    filled = round(count / max_count * width)
    return "█" * filled + "░" * (width - filled)


def render_terminal_report(report: CheckReport, console: Console) -> None:
    """Print the full error report to the terminal using rich."""
    total = len(report.errors)
    counts = report.counts_by_category

    console.print()
    console.print(Panel(
        "[bold white]WILL CHECKER — ERROR REPORT[/bold white]",
        box=box.DOUBLE,
        expand=False,
        padding=(0, 4),
    ))

    if total == 0:
        console.print("\n[bold green]No errors detected.[/bold green]\n")
        return

    console.print(f"\n[bold]Total errors detected:[/bold] [bold red]{total}[/bold red]\n")

    # Summary table
    summary_table = Table(title="Error Summary", box=box.SIMPLE_HEAD, show_edge=False)
    summary_table.add_column("Category", style="bold", min_width=22)
    summary_table.add_column("Count", justify="right", min_width=6)
    summary_table.add_column("Distribution", min_width=24)

    max_count = max(counts.values()) if counts else 1
    for cat in ErrorCategory:
        count = counts.get(cat.value, 0)
        color = _CATEGORY_COLORS.get(cat.value, "white")
        label = _CATEGORY_LABELS.get(cat.value, cat.value)
        bar = _bar(count, max_count)
        summary_table.add_row(
            f"[{color}]{label}[/{color}]",
            f"[{color}]{count}[/{color}]",
            f"[{color}]{bar}[/{color}]",
        )

    console.print(summary_table)
    console.print()

    # Detailed errors table
    detail_table = Table(
        title="Detected Errors",
        box=box.ROUNDED,
        show_lines=True,
        expand=True,
    )
    detail_table.add_column("#", justify="right", style="dim", width=3)
    detail_table.add_column("Category", min_width=14)
    detail_table.add_column("Section", min_width=14)
    detail_table.add_column("Found in Will", min_width=20)
    detail_table.add_column("Expected", min_width=20)
    detail_table.add_column("Conf.", justify="right", width=6)

    for i, error in enumerate(report.errors, 1):
        color = _CATEGORY_COLORS.get(error.category.value, "white")
        label = _CATEGORY_LABELS.get(error.category.value, error.category.value)
        section = (error.section or "").replace("_", " ")
        will_text = error.relevant_text[:80] + ("…" if len(error.relevant_text) > 80 else "")
        expected = error.expected_text[:80] + ("…" if len(error.expected_text) > 80 else "")
        conf_str = f"{error.confidence:.0%}"

        detail_table.add_row(
            str(i),
            f"[{color}]{label}[/{color}]",
            section,
            will_text,
            expected,
            conf_str,
        )

    console.print(detail_table)

    # Explanations section
    console.print("\n[bold]Explanations:[/bold]")
    for i, error in enumerate(report.errors, 1):
        if error.explanation:
            color = _CATEGORY_COLORS.get(error.category.value, "white")
            label = _CATEGORY_LABELS.get(error.category.value, error.category.value)
            console.print(f"  [{color}][{i}] {label}:[/{color}] {error.explanation}")

    console.print()


def _build_json_payload(report: CheckReport) -> dict:
    return {
        "summary": {
            "total_errors": len(report.errors),
            "by_category": report.counts_by_category,
        },
        "errors": [
            {
                "id": i,
                "category": error.category.value,
                "section": error.section,
                "relevant_text": error.relevant_text,
                "expected_text": error.expected_text,
                "confidence": error.confidence,
                "explanation": error.explanation,
            }
            for i, error in enumerate(report.errors, 1)
        ],
    }


def save_json_report(report: CheckReport, output_path: str) -> None:
    payload = _build_json_payload(report)
    Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
