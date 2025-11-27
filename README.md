# Retail Analytics Copilot — DSPy + LangGraph (Hybrid RAG + SQL Agent)

This project implements a hybrid agent that answers retail analytics questions using both RAG (TF-IDF) and SQL execution over the Northwind dataset.
Built using DSPy, LangGraph, SQLite, and a local Ollama model (Phi-3.5 Mini).
### •	Gross margin calculated by 30% of the unit price.

Run the Agent
```python
python run_agent_hybrid.py --batch sample_questions_hybrid_eval.jsonl --out outputs_hybrid.jsonl
```

Repo Structure

```
agent/
    dspy_signatures.py
    graph_hybrid.py
    retriever_tfidf.py
    sql_tool.py
run_agent_hybrid.py
sample_questions_hybrid_eval.jsonl
outputs_hybrid.jsonl
AI_Assignment_Report.docx / .pdf
```

## Description
The system:
Routes each question (SQL / Hybrid / RAG)
Retrieves relevant policy or KPI docs via TF-IDF
Generates SQL using DSPy modules
Executes against SQLite Northwind DB
Repairs SQL on errors (retry loop)
Produces structured output with final answer, SQL, confidence, and citations
