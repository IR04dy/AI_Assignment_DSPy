"""Main entrypoint for the Retail Analytics Copilot hybrid agent.

Usage (as required by the assignment):

    python run_agent_hybrid.py \
        --batch sample_questions_hybrid_eval.jsonl \
        --out outputs_hybrid.jsonl

This script:
- loads the JSONL batch file (each line: {id, question, format_hint}),
- runs the LangGraph-based hybrid agent for each question, and
- writes outputs_hybrid.jsonl with one JSON object per line matching the
  Output Contract:

    {
      "id": "...",
      "final_answer": <matches format_hint>,
      "sql": "<last executed SQL or empty if RAG-only>",
      "confidence": 0.0,
      "explanation": "<= 2 sentences",
      "citations": [ ... ]
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import click

from agent.graph_hybrid import build_graph, AgentState


def _truncate_explanation(text: str, max_sentences: int = 2) -> str:
    """Ensure explanation is at most `max_sentences` sentences.

    This is a light post-process; the Synthesizer should already aim for
    <= 2 sentences, but we enforce it here for safety.
    """

    text = (text or "").strip()
    if not text:
        return ""

    sentences = []
    current = []
    for ch in text:
        current.append(ch)
        if ch in ".!?":
            s = "".join(current).strip()
            if s:
                sentences.append(s)
            current = []
    # Add any trailing fragment
    trailing = "".join(current).strip()
    if trailing:
        sentences.append(trailing)

    if len(sentences) <= max_sentences:
        return text

    return " ".join(sentences[:max_sentences])


def run_one(graph, item: Dict[str, Any]) -> Dict[str, Any]:
    """Run the hybrid agent on a single question item and return output object."""

    state: AgentState = {
        "id": item["id"],
        "question": item["question"],
        "format_hint": item["format_hint"],
        "attempts": 0,
    }

    final_state = graph.invoke(state)

    # Build output contract
    out: Dict[str, Any] = {
        "id": final_state.get("id", item["id"]),
        "final_answer": final_state.get("final_answer"),
        "sql": final_state.get("sql_query", "") or "",
        "confidence": float(final_state.get("confidence", 0.0)),
        "explanation": _truncate_explanation(final_state.get("explanation", "")),
        "citations": final_state.get("citations", []),
    }

    return out


@click.command()
@click.option("--batch", type=click.Path(exists=True, dir_okay=False), required=True,
              help="Path to input JSONL with questions.")
@click.option("--out", "out_path", type=click.Path(dir_okay=False), required=True,
              help="Path to output JSONL file.")
def main(batch: str, out_path: str) -> None:
    """CLI entrypoint: run the hybrid agent over a batch of questions."""

    batch_path = Path(batch)
    out_file = Path(out_path)

    # Build the graph once
    graph = build_graph()

    # Read input JSONL
    items = []
    # NOTE: use utf-8-sig to strip BOM if present (Windows editors)
    with batch_path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))

    # Run agent per item and write outputs
    with out_file.open("w", encoding="utf-8") as f_out:
        for item in items:
            result = run_one(graph, item)
            json_line = json.dumps(result, ensure_ascii=False)
            f_out.write(json_line + "\n")


if __name__ == "__main__":
    main()
