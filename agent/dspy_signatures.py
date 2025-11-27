"""DSPy Signatures and Modules for the Retail Analytics Copilot.

This module defines:

- RouterSignature / RouterModule: classify question as `rag`, `sql`, or `hybrid`.
- NL2SQLSignature / NL2SQLModule: generate SQLite queries, *and* inline a small
  planner that extracts constraints (campaign dates, KPI, category) from the
  question + retrieved docs.
- SynthSignature / SynthModule: synthesize the final typed answer with
  explanation, confidence, and citations.

The inline planner logic is kept simple and deterministic, tailored to the
assignment docs (marketing calendar, KPI definitions, catalog).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import json
import re

import dspy


# ---------------------------------------------------------------------------
# Inline planner helpers (constraints extraction)
# ---------------------------------------------------------------------------

@dataclass
class Constraints:
    """Structured planning output used by NL→SQL and Synthesizer.

    All fields are optional; downstream components can decide what to use.
    """

    date_from: Optional[str] = None
    date_to: Optional[str] = None
    campaign_name: Optional[str] = None
    kpi: Optional[str] = None
    metric_type: Optional[str] = None
    category: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_KNOWN_CATEGORIES = [
    "beverages",
    "condiments",
    "confections",
    "dairy products",
    "grains/cereals",
    "meat/poultry",
    "produce",
    "seafood",
]


def _detect_kpi_from_question(q_lower: str, constraints: Constraints) -> None:
    if "average order value" in q_lower or "aov" in q_lower:
        constraints.kpi = "AOV"
        constraints.metric_type = "AverageOrderValue"
        return

    if "gross margin" in q_lower or "top customer by margin" in q_lower or "margin" in q_lower:
        constraints.kpi = "GrossMargin"
        constraints.metric_type = "GrossMargin"
        return

    if "revenue" in q_lower:
        constraints.kpi = "Revenue"
        constraints.metric_type = "Revenue"
        return


def _detect_category_from_question(q_lower: str, constraints: Constraints) -> None:
    for cat in _KNOWN_CATEGORIES:
        if cat in q_lower:
            if "/" in cat:
                parts = cat.split("/")
                constraints.category = "/".join(p.title() for p in parts)
            else:
                constraints.category = cat.title()
            return


def _extract_date_range_from_text(text: str) -> tuple[Optional[str], Optional[str]]:
    pattern = r"(19\d{2}-\d{2}-\d{2})\s+to\s+(19\d{2}-\d{2}-\d{2})"
    m = re.search(pattern, text)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def _extract_campaign_dates_from_docs(
    retrieved_chunks: List[Dict[str, Any]],
    q_lower: str,
    constraints: Constraints,
) -> None:
    if constraints.date_from or constraints.date_to:
        return

    for chunk in retrieved_chunks:
        source = str(chunk.get("source", "")).lower()
        text = chunk.get("text", "")

        if "marketing_calendar" not in source:
            continue

        lines = text.splitlines()
        campaign_name: Optional[str] = None
        for line in lines:
            if line.strip().startswith("## "):
                campaign_name = line.strip().lstrip("#").strip()
                break

        date_from, date_to = _extract_date_range_from_text(text)
        if not date_from or not date_to:
            continue

        if campaign_name and campaign_name.lower() in q_lower:
            constraints.campaign_name = campaign_name
            constraints.date_from = date_from
            constraints.date_to = date_to
            return

        if "summer beverages" in q_lower and "summer beverages" in (campaign_name or "").lower():
            constraints.campaign_name = campaign_name
            constraints.date_from = date_from
            constraints.date_to = date_to
            return

        if "winter classics" in q_lower and "winter classics" in (campaign_name or "").lower():
            constraints.campaign_name = campaign_name
            constraints.date_from = date_from
            constraints.date_to = date_to
            return


def _maybe_record_year_hint(q_lower: str, constraints: Constraints) -> None:
    if constraints.date_from or constraints.date_to:
        return
    m = re.search(r"(19\d{2})", q_lower)
    if m:
        year = m.group(1)
        if constraints.notes:
            constraints.notes += f"; year_hint={year}"
        else:
            constraints.notes = f"year_hint={year}"


def extract_constraints(question: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Inline planner: derive constraints from question + docs.

    Returns a plain dict that can be serialized or directly passed into
    DSPy modules / LangGraph state.
    """

    q_lower = question.lower()
    constraints = Constraints()

    _detect_kpi_from_question(q_lower, constraints)
    _detect_category_from_question(q_lower, constraints)
    _extract_campaign_dates_from_docs(retrieved_chunks, q_lower, constraints)
    _maybe_record_year_hint(q_lower, constraints)

    return constraints.to_dict()


# ---------------------------------------------------------------------------
# DSPy Signatures
# ---------------------------------------------------------------------------


class RouterSignature(dspy.Signature):
    """Classify the question into a routing label: rag | sql | hybrid."""

    question: str = dspy.InputField()
    format_hint: str = dspy.InputField()
    route: str = dspy.OutputField(desc="One of: rag, sql, hybrid")


class NL2SQLSignature(dspy.Signature):
    """Generate a SQLite query for the Northwind DB.

    The `constraints_json` field encodes planner output (dates, KPI, category).
    """

    question: str = dspy.InputField()
    schema: str = dspy.InputField()
    constraints_json: str = dspy.InputField()
    sql: str = dspy.OutputField(desc="A valid SQLite SELECT query.")


class SynthSignature(dspy.Signature):
    """Synthesize the final typed answer with citations."""

    question: str = dspy.InputField()
    format_hint: str = dspy.InputField()
    sql_result_json: str = dspy.InputField()
    docs_json: str = dspy.InputField()
    constraints_json: str = dspy.InputField()

    final_answer: str = dspy.OutputField(desc="Answer encoded as JSON or scalar; must match format_hint.")
    explanation: str = dspy.OutputField(desc="Short, 1-2 sentence explanation.")
    citations_json: str = dspy.OutputField(desc="JSON-encoded list of citations (tables + doc chunk ids).")
    confidence: float = dspy.OutputField(desc="Confidence score between 0 and 1.")


# ---------------------------------------------------------------------------
# DSPy Modules
# ---------------------------------------------------------------------------


class RouterModule(dspy.Module):
    """Wrapper around a DSPy classifier for routing.

    You can later optimize this with a small labeled set and MIPROv2 or
    BootstrapFewShot.
    """

    def __init__(self, lm: Optional[dspy.LM] = None):
        super().__init__()
        self.program = dspy.Predict(RouterSignature, lm=lm)

    def forward(self, question: str, format_hint: str) -> str:
        pred = self.program(question=question, format_hint=format_hint)
        # Normalize route a bit
        route = (pred.route or "").strip().lower()
        if route not in {"rag", "sql", "hybrid"}:
            # simple heuristic fallback
            if "return window" in question.lower() or "policy" in question.lower():
                route = "rag"
            elif "top 3" in question.lower() or "top" in question.lower() and "product" in question.lower():
                route = "sql"
            else:
                route = "hybrid"
        return route


class NL2SQLModule(dspy.Module):
    """NL→SQL module with inline constraints planner.

    Usage:

        nl2sql = NL2SQLModule(lm)
        sql, constraints = nl2sql(
            question=question,
            schema=schema_str,
            retrieved_chunks=retrieved_chunks,
        )
    """

    def __init__(self, lm: Optional[dspy.LM] = None):
        super().__init__()
        self.program = dspy.Predict(NL2SQLSignature, lm=lm)

    def forward(
        self,
        question: str,
        schema: str,
        retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Return dict with `sql` and `constraints`.

        `constraints` is the dict produced by the inline planner.
        """

        if retrieved_chunks is None:
            retrieved_chunks = []

        constraints = extract_constraints(question, retrieved_chunks)
        constraints_json = json.dumps(constraints, ensure_ascii=False)

        pred = self.program(
            question=question,
            schema=schema,
            constraints_json=constraints_json,
        )

        return {
            "sql": pred.sql,
            "constraints": constraints,
        }


class SynthModule(dspy.Module):
    """Synthesize final answer, explanation, confidence, and citations.

    The LM returns everything as strings; we parse JSON where appropriate in
    the LangGraph node, not inside this module, to keep evaluation explicit.
    """

    def __init__(self, lm: Optional[dspy.LM] = None):
        super().__init__()
        self.program = dspy.Predict(SynthSignature, lm=lm)

    def forward(
        self,
        question: str,
        format_hint: str,
        sql_result: List[Dict[str, Any]],
        docs: List[Dict[str, Any]],
        constraints: Dict[str, Any],
    ) -> Dict[str, Any]:
        sql_result_json = json.dumps(sql_result, ensure_ascii=False)
        docs_json = json.dumps(docs, ensure_ascii=False)
        constraints_json = json.dumps(constraints, ensure_ascii=False)

        pred = self.program(
            question=question,
            format_hint=format_hint,
            sql_result_json=sql_result_json,
            docs_json=docs_json,
            constraints_json=constraints_json,
        )

        # The caller is responsible for parsing final_answer / citations_json
        # according to `format_hint`.
        return {
            "final_answer": pred.final_answer,
            "explanation": pred.explanation,
            "citations_json": pred.citations_json,
            "confidence": float(pred.confidence),
        }


__all__ = [
    "Constraints",
    "extract_constraints",
    "RouterSignature",
    "NL2SQLSignature",
    "SynthSignature",
    "RouterModule",
    "NL2SQLModule",
    "SynthModule",
]
