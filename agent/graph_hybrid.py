"""LangGraph-based hybrid agent for the Retail Analytics Copilot.

This file wires together:
- Router (DSPy-based) → decides: rag | sql | hybrid
- Retriever (TF-IDF over docs/)
- NL→SQL (DSPy-based, with inline planner in dspy_signatures.py)
- SQL Executor (SQLite over Northwind)
- Synthesizer (DSPy-based) with strict format enforcement + citations
- Repair loop (up to 2 attempts)
- Simple JSON logging/trace per step

The graph is compiled and can be invoked from run_agent_hybrid.py.
"""

from __future__ import annotations

import json
import logging
import dspy
from typing import Any, Dict, List, Optional, Literal

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END

from agent.rag.retrieval import get_retriever
from agent.tools.sqlite_tool import execute_sql, get_schema_description
from agent.dspy_signatures import (
    RouterModule,
    NL2SQLModule,
    SynthModule,
)


# ---------------------------------------------------------------------------
# Logging / tracing setup
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)


def log_step(step: str, state: "AgentState") -> None:
    """Log a minimal JSON snapshot of the agent state for tracing.

    This acts as a simple, replayable execution trace for the assignment's
    "checkpointer/trace" requirement.
    """

    snapshot = {
        "step": step,
        "id": state.get("id"),
        "route": state.get("route"),
        "attempts": state.get("attempts", 0),
        "sql": state.get("sql_query"),
        "sql_error": state.get("sql_error"),
        "synth_error": state.get("synth_error"),
    }
    logging.info(json.dumps(snapshot, default=str))


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------


class AgentState(TypedDict, total=False):
    # Input
    id: str
    question: str
    format_hint: str

    # Routing
    route: Literal["rag", "sql", "hybrid"]

    # Retrieval / planning
    retrieved_chunks: List[Dict[str, Any]]
    constraints: Dict[str, Any]

    # SQL pipeline
    schema: str
    sql_query: str
    sql_result: List[Dict[str, Any]]
    sql_error: Optional[str]

    # Final answer
    final_answer: Any
    explanation: str
    citations: List[str]
    confidence: float
    synth_error: Optional[str]

    # Repair loop
    attempts: int


# ---------------------------------------------------------------------------
# Helper functions: answer coercion + citation augmentation
# ---------------------------------------------------------------------------


_KNOWN_TABLES = [
    "Orders",
    "Order Details",
    "Products",
    "Customers",
    "Categories",
    "Suppliers",
]


def _coerce_answer(format_hint: str, answer_raw: str) -> Any:
    """Convert LM-produced answer string into the expected Python type.

    Supports:
      - "int"
      - "float"
      - "{category:str, quantity:int}"
      - "list[{product:str, revenue:float}]" (and similar)
    """

    fh = format_hint.strip().lower()
    text = answer_raw.strip()

    def _try_json() -> Optional[Any]:
        try:
            return json.loads(text)
        except Exception:
            return None

    # Simple scalars
    if fh == "int":
        try:
            return int(text)
        except ValueError:
            try:
                return int(float(text))
            except ValueError as e:
                raise ValueError(f"Failed to coerce answer to int: {text}") from e

    if fh == "float":
        try:
            return float(text)
        except ValueError as e:
            raise ValueError(f"Failed to coerce answer to float: {text}") from e

    # List[...] → JSON array
    if fh.startswith("list["):
        data = _try_json()
        if isinstance(data, list):
            return data
        raise ValueError(
            f"Expected JSON array for format_hint '{format_hint}', got: {text}"
        )

    # Object-like {foo:str, bar:int} → JSON object
    if "{" in fh and "}" in fh:
        data = _try_json()
        if isinstance(data, dict):
            return data
        raise ValueError(
            f"Expected JSON object for format_hint '{format_hint}', got: {text}"
        )

    # Fallback: try JSON, else return raw string
    data = _try_json()
    if data is not None:
        return data

    return text


def _augment_citations(
    base: List[str], sql_query: str, docs: List[Dict[str, Any]]
) -> List[str]:
    """Ensure DB table names and doc chunk IDs appear in the citations list."""

    citations: List[str] = list(dict.fromkeys(base))  # dedupe

    sql_lower = (sql_query or "").lower()

    # Add tables if they appear in the SQL
    for table in _KNOWN_TABLES:
        patterns = [table.lower(), f'"{table.lower()}"']
        if any(p in sql_lower for p in patterns):
            if table not in citations:
                citations.append(table)

    # Add doc chunk IDs
    for doc in docs:
        cid = doc.get("id")
        if cid and cid not in citations:
            citations.append(str(cid))

    return citations


# ---------------------------------------------------------------------------
# Global singletons (LM-backed modules and retriever)
# ---------------------------------------------------------------------------

dspy.configure(
    lm=dspy.LM(
        "ollama/phi3.5:3.8b-mini-instruct-q4_K_M",
        api_base="http://localhost:11434",
    )
)

router_module = RouterModule(lm=None)
nl2sql_module = NL2SQLModule(lm=None)
synth_module = SynthModule(lm=None)

retriever = get_retriever("./docs")
schema_str = get_schema_description()


# ---------------------------------------------------------------------------
# LangGraph nodes
# ---------------------------------------------------------------------------


def router_node(state: AgentState) -> AgentState:
    route = router_module(
        question=state["question"],
        format_hint=state["format_hint"],
    )
    state["route"] = route  # "rag" | "sql" | "hybrid"
    log_step("router", state)
    return state


def retriever_node(state: AgentState) -> AgentState:
    # Only run retrieval for rag / hybrid routes
    q = state["question"]
    chunks = retriever.retrieve_topk(q, k=5)
    state["retrieved_chunks"] = chunks
    log_step("retriever", state)
    return state


def nl2sql_node(state: AgentState) -> AgentState:
    # Inline planner is inside NL2SQLModule
    retrieved_chunks = state.get("retrieved_chunks", [])

    result = nl2sql_module(
        question=state["question"],
        schema=schema_str,
        retrieved_chunks=retrieved_chunks,
    )

    state["schema"] = schema_str
    state["sql_query"] = result["sql"]
    state["constraints"] = result["constraints"]
    state["sql_error"] = None

    log_step("nl2sql", state)
    return state


def sql_executor_node(state: AgentState) -> AgentState:
    sql = state.get("sql_query", "")
    try:
        rows = execute_sql(sql)
        state["sql_result"] = rows
        state["sql_error"] = None
    except Exception as e:
        state["sql_result"] = []
        state["sql_error"] = str(e)

    log_step("sql_exec", state)
    return state


def synthesizer_node(state: AgentState) -> AgentState:
    question = state["question"]
    format_hint = state["format_hint"]
    sql_result = state.get("sql_result", [])
    docs = state.get("retrieved_chunks", [])
    constraints = state.get("constraints", {})
    sql_query = state.get("sql_query", "")

    try:
        raw = synth_module(
            question=question,
            format_hint=format_hint,
            sql_result=sql_result,
            docs=docs,
            constraints=constraints,
        )

        answer_raw = raw.get("final_answer", "")
        explanation = raw.get("explanation", "")
        citations_json = raw.get("citations_json", "")
        confidence = float(raw.get("confidence", 0.0))

        # Enforce format_hint → Python object
        final_answer = _coerce_answer(format_hint, answer_raw)

        # Parse LM citations
        base_citations: List[str] = []
        if citations_json:
            try:
                parsed = json.loads(citations_json)
                if isinstance(parsed, list):
                    base_citations = [str(x) for x in parsed]
            except Exception:
                base_citations = []

        # Augment citations with tables + doc chunks
        citations = _augment_citations(base_citations, sql_query, docs)

        state["final_answer"] = final_answer
        state["explanation"] = explanation
        state["citations"] = citations
        state["confidence"] = confidence
        state["synth_error"] = None

    except Exception as e:
        state["synth_error"] = str(e)

    log_step("synthesizer", state)
    return state


# ---------------------------------------------------------------------------
# Repair loop
# ---------------------------------------------------------------------------

MAX_ATTEMPTS = 2


def repair_node(state: AgentState) -> AgentState:
    attempts = state.get("attempts", 0)
    state["attempts"] = attempts + 1
    log_step("repair", state)
    return state


def repair_condition(state: AgentState) -> str:
    attempts = state.get("attempts", 0)

    # SQL error → try to regenerate SQL
    if state.get("sql_error"):
        if attempts < MAX_ATTEMPTS:
            return "retry_sql"
        return "give_up"

    # Synth error → try to resynthesize
    if state.get("synth_error"):
        if attempts < MAX_ATTEMPTS:
            return "retry_synth"
        return "give_up"

    # No errors → success
    return "success"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph() -> StateGraph[AgentState]:
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("router", router_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("nl2sql", nl2sql_node)
    graph.add_node("sql_exec", sql_executor_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("repair", repair_node)

    graph.set_entry_point("router")

    # Routing after router
    def _router_decision(state: AgentState) -> str:
        route = state.get("route", "hybrid")
        if route == "rag":
            return "rag_only"
        if route == "sql":
            return "sql_only"
        return "hybrid"

    graph.add_conditional_edges(
        "router",
        _router_decision,
        {
            "rag_only": "retriever",
            "sql_only": "nl2sql",  # no retrieval
            "hybrid": "retriever",
        },
    )

    # After retriever → NL2SQL (for hybrid) or Synthesizer (for rag-only)
    def _post_retriever_decision(state: AgentState) -> str:
        route = state.get("route", "hybrid")
        if route == "rag":
            return "rag"
        return "needs_sql"

    graph.add_conditional_edges(
        "retriever",
        _post_retriever_decision,
        {
            "rag": "synthesizer",
            "needs_sql": "nl2sql",
        },
    )

    # SQL pipeline
    graph.add_edge("nl2sql", "sql_exec")
    graph.add_edge("sql_exec", "synthesizer")

    # After synthesizer → repair
    graph.add_edge("synthesizer", "repair")

    # Repair decisions
    graph.add_conditional_edges(
        "repair",
        repair_condition,
        {
            "retry_sql": "nl2sql",
            "retry_synth": "synthesizer",
            "success": END,
            "give_up": END,
        },
    )

    return graph.compile()


__all__ = ["AgentState", "build_graph"]
