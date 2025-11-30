from pathlib import Path
import json
import streamlit as st

# Import helpers from the existing runner (no file modifications done)
from run_agent_hybrid import (
    ensure_db,
    get_retriever,
    get_db_tool,
    heuristic_route,
    simple_sql_for_question,
    synthesize_answer,
)

ROOT = Path(__file__).parent
DB_PATH = ROOT / "data" / "northwind.sqlite"

st.set_page_config(page_title="Hybrid Agent Demo", layout="centered")
st.title("Retail Analytics Copilot")

st.markdown(
    "This UI runs the local demo runner from the project without changing other files.\n"
    "It supports single-shot queries and an interactive prompt flow."
)

# Sidebar: DB creation and sample queries
with st.sidebar:
    st.header("Setup & Samples")
    if st.button("(Re)create sample DB"):
        # create or replace
        ensure_db(create_if_missing=True)
        st.success(f"Sample DB created at {DB_PATH}")

    st.markdown("---")
    st.subheader("Sample questions")
    sample_qs = [
        "What is our return policy for beverages?",
        "List the top 3 best selling products by quantity in July 2024.",
        "Calculate AOV for June 2024 and provide any related KPI definitions from our docs.",
        "How many units of 'Chai' were sold?",
        "Show me units per transaction for the last quarter and link any related product policy references.",
    ]
    choice = st.selectbox("Pick sample", ["(none)"] + sample_qs)

# Main UI: ask a question
st.header("Ask a question")
q_input = st.text_area("Question", height=120, value="" if choice == "(none)" else choice)
run_button = st.button("Ask")

if run_button and q_input.strip():
    # Ensure DB and components available
    ensure_db(create_if_missing=True)
    try:
        retriever = get_retriever(docs_dir=str(ROOT / "docs"))
        db_tool = get_db_tool(db_path=str(DB_PATH))
    except Exception as e:
        st.error("Failed to initialize retriever or DB tool: " + str(e))
        st.stop()

    question = q_input.strip()

    # Route decision
    route = heuristic_route(question)
    st.info(f"Routed to: {route}")

    # If retrieval is part of the route, show matched chunks
    doc_context = None
    if route in ("rag", "hybrid"):
        chunks = retriever.search(question, top_k=5)
        if not chunks:
            st.warning("No document chunks matched the question")
        else:
            st.write(f"Retrieved {len(chunks)} document chunks:")
            for chunk, score in chunks:
                st.markdown(f"**{chunk.full_id}** — score {score:.3f}\n> {chunk.content}")
        doc_context = [f"{c.full_id}: {c.content}" for c, s in chunks]

    # If SQL is needed, generate and run
    rows = None
    if route in ("sql", "hybrid"):
        st.write("Generating SQL (heuristic)")
        sql = simple_sql_for_question(question)
        st.code(sql, language="sql")

        result = db_tool.execute_query(sql)
        if result.success:
            rows = result.rows
            st.success(f"SQL returned {len(rows)} rows")
            st.dataframe(rows)
        else:
            st.error("SQL error: " + str(result.error))

    # Synthesize and show final answer
    final = synthesize_answer(route, rows, docs=doc_context)
    st.header("Final answer")
    if isinstance(final, str) and final.startswith("{"):
        try:
            parsed = json.loads(final)
            st.json(parsed)
        except Exception:
            st.text(final)
    else:
        st.text(final)

# Tips and quick-run
st.markdown("---")
st.write("Quick usage:")
st.code("streamlit run your_project/app.py")

st.caption("Note: Streamlit must be installed in your environment. If you get an import error for streamlit, install with `pip install streamlit`.")
