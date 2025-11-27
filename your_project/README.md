# 🚀 Retail Analytics Copilot — Hybrid RAG + SQL Agent (LangGraph + DSPy)

A full AI-powered analytics assistant using Hybrid Retrieval, SQL execution, DSPy optimization, and LangGraph pipelines.

# 📌 📖 Project Overview

This project is a Hybrid RAG + SQL AI Agent designed for retail analytics.
It can answer questions using:

✔ RAG (text documents)
✔ SQL (northwind.sqlite database)
✔ Hybrid reasoning (RAG + SQL together)

The agent:

Reads business documents (marketing calendar, KPIs, catalog, policies)

Searches documents using TF-IDF retriever

Reads SQL data from Northwind dataset

Converts user questions → SQL queries using NL→SQL module

Uses LangGraph nodes with repair loops

Generates final answers with citations

It behaves like a smart retail analytics assistant capable of answering:

“What was the revenue during Black Friday?”

“Give me the AOV for beverages in Q2.”

“What is our return policy for dairy?”

“Compare sales performance during Winter Sale vs Spring Promotion.”

# 📁 🗂 Project Structure
your_project/
├─ agent/
│  ├─ graph_hybrid.py             # LangGraph pipeline (≥ 6 nodes + repair loop)
│  ├─ dspy_signatures.py          # DSPy modules (Router, NL→SQL, Synthesizer)
│  ├─ rag/
│  │   └─ retrieval.py            # TF-IDF retriever (chunking + search)
│  └─ tools/
│      └─ sqlite_tool.py          # Safe SQL executor + schema introspection
│
├─ data/
│  └─ northwind.sqlite            # Retail SQL database (Orders, Products, etc.)
│
├─ docs/
│  ├─ marketing_calendar.md       # RAG document 1
│  ├─ kpi_definitions.md          # RAG document 2
│  ├─ catalog.md                  # RAG document 3
│  └─ product_policy.md           # RAG document 4
│
├─ sample_questions_hybrid_eval.jsonl   # Test questions for evaluation
├─ run_agent_hybrid.py                  # Main script: runs the Hybrid Agent
└─ requirements.txt                     # Python dependencies

# 🧠 ✨ Key Features
🔹 1. Hybrid RAG + SQL Agent

The system uses both text retrieval + SQL queries to answer questions accurately.

🔹 2. LangGraph Workflow (≥ 6 Nodes)

Includes nodes for:

Routing

Retrieval

Planning

SQL generation

SQL execution

Hybrid fusion

Synthesis

Repair loop for SQL errors

# 🔹 3. DSPy Optimization

DSPy modules trained for:

Question Routing (RAG / SQL / Hybrid)

NL → SQL generation

Answer synthesis

# 🔹 4. TF-IDF Retriever

Document search using:

Chunking

TF-IDF vectorization

Top-k retrieval

Document citations

# 🔹 5. SQLite Integration

Executes SQL queries safely with:

Schema reading

Error handling

Query repair on failures
