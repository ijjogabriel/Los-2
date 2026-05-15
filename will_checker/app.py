"""Interactive Streamlit UI for will-checker."""

import json
import re
from pathlib import Path
from typing import Optional

import streamlit as st

from .models import CheckReport, ErrorCategory, SectionType
from .section_classifier import classify_sections, find_missing_sections
from .ner_extraction import (
    extract_entities_from_fact_pattern,
    extract_entities_from_will,
    load_nlp_pipeline,
)
from .entity_comparison import compare_entities
from .semantic_comparison import compare_semantically, embed_sections, load_model
from .consistency_checker import check_consistency

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Will Checker",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Constants ─────────────────────────────────────────────────────────────────

SAMPLES_DIR = Path(__file__).parent.parent / "samples"
SAMPLE_FP_PATH = SAMPLES_DIR / "sample_fact_pattern.json"
SAMPLE_WILL_PATH = SAMPLES_DIR / "sample_will_with_errors.txt"

_CATEGORY_COLORS = {
    ErrorCategory.FACTUAL_ERROR.value: "#d97706",       # amber
    ErrorCategory.STRUCTURAL_ERROR.value: "#dc2626",    # red
    ErrorCategory.LEGAL_ERROR.value: "#b91c1c",         # dark red
    ErrorCategory.HALLUCINATED_CONTENT.value: "#7c3aed", # violet
    ErrorCategory.INTERNAL_INCONSISTENCY.value: "#c2410c", # orange
}

_CATEGORY_LABELS = {
    ErrorCategory.FACTUAL_ERROR.value: "Factual Error",
    ErrorCategory.STRUCTURAL_ERROR.value: "Structural Error",
    ErrorCategory.LEGAL_ERROR.value: "Legal Error",
    ErrorCategory.HALLUCINATED_CONTENT.value: "Hallucinated Content",
    ErrorCategory.INTERNAL_INCONSISTENCY.value: "Internal Inconsistency",
}

_CATEGORY_ICONS = {
    ErrorCategory.FACTUAL_ERROR.value: "📋",
    ErrorCategory.STRUCTURAL_ERROR.value: "🏗️",
    ErrorCategory.LEGAL_ERROR.value: "⚖️",
    ErrorCategory.HALLUCINATED_CONTENT.value: "👻",
    ErrorCategory.INTERNAL_INCONSISTENCY.value: "🔄",
}

# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading spaCy model (en_core_web_lg)…")
def get_nlp():
    return load_nlp_pipeline()


@st.cache_resource(show_spinner="Loading sentence-transformers model…")
def get_st_model():
    return load_model()  # returns None gracefully if not available


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(
    fact_pattern: dict,
    will_text: str,
    fuzzy_threshold: int,
    semantic_threshold: float,
) -> tuple[CheckReport, list]:
    """Run all five stages and return (report, sections)."""
    nlp = get_nlp()
    st_model = get_st_model()

    # Stage 3 first (section ranges needed for entity→section assignment)
    sections = classify_sections(will_text)
    structural_errors = find_missing_sections(sections)

    # Stage 1
    fp_entities = extract_entities_from_fact_pattern(fact_pattern)
    will_entities = extract_entities_from_will(will_text, nlp, sections)
    for sec in sections:
        sec.entities = [e for e in will_entities if e.section == sec.section_type.value]

    # Stage 2
    factual_errors = compare_entities(fp_entities, will_entities, fuzzy_threshold)

    # Stage 4
    sections = embed_sections(sections, st_model)
    semantic_errors = compare_semantically(sections, fact_pattern, st_model, semantic_threshold)

    # Stage 5
    consistency_errors = check_consistency(sections, fact_pattern)

    all_errors = structural_errors + factual_errors + semantic_errors + consistency_errors

    # Deduplicate
    seen: dict[tuple, object] = {}
    for err in all_errors:
        key = (err.category.value, err.relevant_text[:60])
        if key not in seen or err.confidence > seen[key].confidence:
            seen[key] = err

    _order = {
        "structural_error": 0, "factual_error": 1, "legal_error": 2,
        "hallucinated_content": 3, "internal_inconsistency": 4,
    }
    deduped = sorted(seen.values(), key=lambda e: _order.get(e.category.value, 99))

    report = CheckReport(errors=list(deduped))
    report.tally()
    return report, sections


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_sample_fp() -> str:
    return SAMPLE_FP_PATH.read_text(encoding="utf-8") if SAMPLE_FP_PATH.exists() else "{}"


def _load_sample_will() -> str:
    return SAMPLE_WILL_PATH.read_text(encoding="utf-8") if SAMPLE_WILL_PATH.exists() else ""


def _parse_fp(text: str) -> tuple[Optional[dict], Optional[str]]:
    """Return (parsed_dict, error_message)."""
    try:
        return json.loads(text), None
    except json.JSONDecodeError as e:
        return None, str(e)


def _fp_summary(fp: dict) -> str:
    """Return a compact human-readable summary of the fact pattern."""
    lines = []
    testator = fp.get("testator", {})
    if name := testator.get("name"):
        age = testator.get("age", "")
        county = testator.get("county", "")
        state = testator.get("state", "")
        lines.append(f"**Testator:** {name}, {age}, {county}, {state}")

    if spouse := fp.get("spouse", {}).get("name"):
        lines.append(f"**Spouse:** {spouse}")

    children = fp.get("children", [])
    if children:
        names = [f"{c.get('name','')}{'*' if c.get('minor') else ''}" for c in children]
        lines.append(f"**Children:** {', '.join(names)}  *(\\* = minor)*")

    pr = fp.get("personal_representative", {})
    if primary := pr.get("primary"):
        alt = pr.get("alternate", "")
        lines.append(f"**Personal Rep:** {primary}" + (f", alt: {alt}" if alt else ""))

    jx = fp.get("jurisdiction", {})
    if state := jx.get("state"):
        witnesses = jx.get("required_witnesses", "?")
        lines.append(f"**Jurisdiction:** {state} — {witnesses} witnesses required")

    assets = fp.get("assets", [])
    if assets:
        lines.append(f"**Assets:** {len(assets)} bequest(s)")

    return "\n\n".join(lines) if lines else "*No recognizable fields found.*"


def _badge(text: str, color: str) -> str:
    return (
        f'<span style="background:{color};color:white;padding:2px 8px;'
        f'border-radius:4px;font-size:0.75rem;font-weight:600;">{text}</span>'
    )


def _confidence_bar(confidence: float) -> str:
    pct = int(confidence * 100)
    color = "#dc2626" if pct >= 70 else "#d97706" if pct >= 40 else "#16a34a"
    return (
        f'<div style="display:flex;align-items:center;gap:6px;">'
        f'<div style="background:#e5e7eb;border-radius:4px;height:8px;width:100px;">'
        f'<div style="background:{color};width:{pct}%;height:100%;border-radius:4px;"></div>'
        f'</div>'
        f'<span style="font-size:0.8rem;color:#6b7280;">{pct}%</span>'
        f'</div>'
    )


# ── Main UI ───────────────────────────────────────────────────────────────────

def main():
    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="padding:1rem 0 0.5rem">
          <h1 style="margin:0;font-size:1.8rem">⚖️ Will Checker</h1>
          <p style="color:#6b7280;margin:0.25rem 0 0">
            Detect and categorize errors in LLM-generated wills against attorney fact patterns
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Session state defaults ────────────────────────────────────────────────
    if "fp_text" not in st.session_state:
        st.session_state.fp_text = _load_sample_fp()
    if "will_text" not in st.session_state:
        st.session_state.will_text = _load_sample_will()
    if "report" not in st.session_state:
        st.session_state.report = None
        st.session_state.sections = []

    # ── Layout: left inputs | right results ───────────────────────────────────
    left, right = st.columns([2, 3], gap="large")

    # ════════════════════════════════════════════════════════
    # LEFT COLUMN — Inputs
    # ════════════════════════════════════════════════════════
    with left:

        # ── Fact Pattern ──────────────────────────────────────────────────────
        st.markdown("### 📋 Fact Pattern")

        fp_col1, fp_col2 = st.columns([1, 1])
        with fp_col1:
            if st.button("Load sample", key="load_sample_fp", use_container_width=True):
                st.session_state.fp_text = _load_sample_fp()
                st.rerun()
        with fp_col2:
            uploaded_fp = st.file_uploader(
                "Upload JSON", type=["json"], key="fp_upload", label_visibility="collapsed"
            )
            if uploaded_fp:
                st.session_state.fp_text = uploaded_fp.read().decode("utf-8")
                st.rerun()

        fp_text = st.text_area(
            "Fact pattern JSON",
            value=st.session_state.fp_text,
            height=280,
            key="fp_textarea",
            label_visibility="collapsed",
            help="Edit the JSON directly. Fields: testator, spouse, children, assets, personal_representative, trust, jurisdiction",
        )
        st.session_state.fp_text = fp_text

        # Validation + preview
        fp_dict, fp_error = _parse_fp(fp_text)
        if fp_error:
            st.error(f"⚠️ Invalid JSON: {fp_error}")
        else:
            with st.expander("📌 Parsed preview", expanded=False):
                st.markdown(_fp_summary(fp_dict))

        st.markdown("")

        # ── Will Text ─────────────────────────────────────────────────────────
        st.markdown("### 📄 Will Text")

        will_col1, will_col2 = st.columns([1, 1])
        with will_col1:
            if st.button("Load sample", key="load_sample_will", use_container_width=True):
                st.session_state.will_text = _load_sample_will()
                st.rerun()
        with will_col2:
            uploaded_will = st.file_uploader(
                "Upload TXT", type=["txt"], key="will_upload", label_visibility="collapsed"
            )
            if uploaded_will:
                st.session_state.will_text = uploaded_will.read().decode("utf-8")
                st.rerun()

        will_text = st.text_area(
            "Will text",
            value=st.session_state.will_text,
            height=280,
            key="will_textarea",
            label_visibility="collapsed",
            help="Paste the full text of the LLM-generated will here",
        )
        st.session_state.will_text = will_text

        st.markdown("")

        # ── Settings ──────────────────────────────────────────────────────────
        with st.expander("⚙️ Analysis Settings", expanded=False):
            fuzzy_threshold = st.slider(
                "Fuzzy match threshold (entity names/addresses)",
                min_value=50, max_value=99, value=85,
                help="Lower = flag more entity mismatches. Default 85.",
            )
            semantic_threshold = st.slider(
                "Semantic similarity threshold (section meaning)",
                min_value=0.30, max_value=0.99, value=0.75, step=0.01,
                help="Lower = flag more semantic drift. Requires neural model.",
            )

        st.markdown("")

        # ── Analyze button ────────────────────────────────────────────────────
        analyze_disabled = fp_dict is None or not will_text.strip()
        if st.button(
            "🔍 Analyze Will",
            type="primary",
            use_container_width=True,
            disabled=analyze_disabled,
        ):
            if fp_dict and will_text.strip():
                with st.spinner("Running five-stage NLP pipeline…"):
                    try:
                        report, sections = run_pipeline(
                            fp_dict, will_text, fuzzy_threshold, semantic_threshold
                        )
                        st.session_state.report = report
                        st.session_state.sections = sections
                    except Exception as e:
                        st.error(f"Pipeline error: {e}")
                        st.session_state.report = None

    # ════════════════════════════════════════════════════════
    # RIGHT COLUMN — Results
    # ════════════════════════════════════════════════════════
    with right:
        report: Optional[CheckReport] = st.session_state.report

        if report is None:
            # Empty state
            st.markdown(
                """
                <div style="display:flex;flex-direction:column;align-items:center;
                justify-content:center;height:500px;color:#9ca3af;text-align:center;">
                  <div style="font-size:4rem;margin-bottom:1rem">⚖️</div>
                  <div style="font-size:1.1rem;font-weight:500">No analysis yet</div>
                  <div style="font-size:0.9rem;margin-top:0.5rem">
                    Load the sample files or paste your own content,<br>
                    then click <strong>Analyze Will</strong> to see results.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            _render_results(report)


def _render_results(report: CheckReport):
    """Render the full error report in the results panel."""
    total = len(report.errors)
    counts = report.counts_by_category

    # ── Summary metrics ───────────────────────────────────────────────────────
    st.markdown("### 📊 Results")

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Total Errors", total)
    with m2:
        st.metric("Factual", counts.get("factual_error", 0))
    with m3:
        st.metric("Structural", counts.get("structural_error", 0))
    with m4:
        st.metric("Legal", counts.get("legal_error", 0))
    with m5:
        st.metric(
            "Hallucinated + Inconsistency",
            counts.get("hallucinated_content", 0) + counts.get("internal_inconsistency", 0),
        )

    if total == 0:
        st.success("✅ No errors detected in this will.")
        return

    # ── Category filter ───────────────────────────────────────────────────────
    all_cats = sorted({e.category.value for e in report.errors})
    cat_labels = {c: f"{_CATEGORY_ICONS.get(c,'🔸')} {_CATEGORY_LABELS.get(c, c)}" for c in all_cats}

    selected_cats = st.multiselect(
        "Filter by category",
        options=all_cats,
        default=all_cats,
        format_func=lambda c: cat_labels[c],
        label_visibility="collapsed",
    )

    filtered = [e for e in report.errors if e.category.value in selected_cats]

    st.markdown(
        f"<div style='color:#6b7280;font-size:0.85rem;margin-bottom:0.5rem'>"
        f"Showing {len(filtered)} of {total} error(s)</div>",
        unsafe_allow_html=True,
    )

    # ── Error cards ───────────────────────────────────────────────────────────
    for i, error in enumerate(filtered, 1):
        cat = error.category.value
        color = _CATEGORY_COLORS.get(cat, "#6b7280")
        label = _CATEGORY_LABELS.get(cat, cat)
        icon = _CATEGORY_ICONS.get(cat, "🔸")
        section = (error.section or "").replace("_", " ").title()

        header = (
            f"{icon} &nbsp;"
            f"{_badge(label, color)}"
            f"&nbsp;&nbsp;<span style='color:#6b7280;font-size:0.8rem'>· {section}</span>"
        )

        with st.expander(
            label=f"{icon} [{i}] {label}  —  {section}",
            expanded=(i <= 3),  # auto-expand first 3
        ):
            # Main comparison
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Found in will:**")
                st.code(error.relevant_text[:400], language=None)
            with col_b:
                st.markdown("**Expected (fact pattern):**")
                st.code(error.expected_text[:400], language=None)

            # Confidence + explanation
            conf_pct = int(error.confidence * 100)
            conf_color = "#dc2626" if conf_pct >= 70 else "#d97706" if conf_pct >= 40 else "#16a34a"

            st.markdown(
                f"<div style='display:flex;align-items:center;gap:12px;margin-top:0.5rem'>"
                f"<span style='font-size:0.85rem;color:#6b7280'>Confidence:</span>"
                f"<div style='background:#e5e7eb;border-radius:4px;height:8px;width:120px;'>"
                f"<div style='background:{conf_color};width:{conf_pct}%;height:100%;border-radius:4px;'></div>"
                f"</div>"
                f"<span style='font-size:0.85rem;font-weight:600;color:{conf_color};'>{conf_pct}%</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            if error.explanation:
                st.info(error.explanation, icon="ℹ️")

    st.divider()

    # ── Download + summary ────────────────────────────────────────────────────
    col_dl, col_sum = st.columns([1, 2])
    with col_dl:
        payload = {
            "summary": {"total_errors": total, "by_category": counts},
            "errors": [
                {
                    "id": i,
                    "category": e.category.value,
                    "section": e.section,
                    "relevant_text": e.relevant_text,
                    "expected_text": e.expected_text,
                    "confidence": e.confidence,
                    "explanation": e.explanation,
                }
                for i, e in enumerate(report.errors, 1)
            ],
        }
        st.download_button(
            label="⬇️ Download JSON report",
            data=json.dumps(payload, indent=2),
            file_name="will_checker_report.json",
            mime="application/json",
            use_container_width=True,
        )

    with col_sum:
        # Category breakdown bar chart using native Streamlit
        if counts:
            import pandas as pd
            df = pd.DataFrame(
                [
                    {"Category": _CATEGORY_LABELS.get(k, k), "Count": v}
                    for k, v in sorted(counts.items(), key=lambda x: -x[1])
                ]
            )
            st.bar_chart(df.set_index("Category"), height=180, use_container_width=True)


# ── Entry point ───────────────────────────────────────────────────────────────
# Streamlit executes module-level code on every rerun, so call main() directly.
main()
