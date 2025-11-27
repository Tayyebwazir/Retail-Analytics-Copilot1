# your_project — Hybrid RAG + SQL Agent Skeleton

This repository is a small skeleton showing how to structure a hybrid agent combining RAG-document search and SQL execution.

Structure
```
your_project/
├─ agent/
│  ├─ graph_hybrid.py
│  ├─ dspy_signatures.py
│  ├─ rag/retrieval.py
│  └─ tools/sqlite_tool.py
├─ data/
│  └─ northwind.sqlite (create with data/create_sample_db.py)
├─ docs/
│  ├─ marketing_calendar.md
│  ├─ kpi_definitions.md
│  ├─ catalog.md
│  └─ product_policy.md
├─ sample_questions_hybrid_eval.jsonl
├─ run_agent_hybrid.py
└─ requirements.txt
```

Quick start

1. From `your_project` folder install the environment:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

2. Create a sample DB for local tests:

```powershell
python data/create_sample_db.py
```

3. Run the lightweight runner (no LM required):

```powershell
python run_agent_hybrid.py -q "What were the total revenue and average order value for June 2024?"
```

Notes
- The agent modules in `agent/` are copies placed in this skeleton so they don't modify your existing files.
- The `run_agent_hybrid.py` uses a small heuristic for SQL and retrieval to enable quick local testing without an LM.
