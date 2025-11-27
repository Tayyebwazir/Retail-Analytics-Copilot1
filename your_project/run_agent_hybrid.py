"""
Small CLI runner for the hybrid agent skeleton in your_project.
This runner contains a lightweight, deterministic fallback logic so it can run without a configured LM.

CLI:
    python run_agent_hybrid.py --question "Your question here"

If `data/northwind.sqlite` is missing the script will offer to create a small sample DB automatically.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "northwind.sqlite"

# Import local components
try:
    from agent.rag.retrieval import get_retriever
    from agent.tools.sqlite_tool import get_db_tool
except Exception as e:
    # Provide a clear error message if package imports fail
    print("Failed to import agent modules. Make sure you're running from the `your_project` directory.")
    raise


def ensure_db(create_if_missing: bool = True):
    if not DB_PATH.exists():
        if create_if_missing:
            print("Database not found — creating a small sample DB for testing...")
            import data.create_sample_db as creator
            creator.create_db()
        else:
            raise FileNotFoundError("data/northwind.sqlite is required. Run data/create_sample_db.py")


def heuristic_route(question: str) -> str:
    q = question.lower()
    if any(k in q for k in ("policy", "return", "warranty", "how to")):
        return 'rag'
    if any(k in q for k in ("revenue", "top", "best", "average", "aov", "sum", "total")) and any(k in q for k in ("product","unit","order","orders","order")):
        return 'sql'
    # default - hybrid
    return 'hybrid'


def simple_extract_dates(question: str):
    # naive extraction of YYYY or YYYY-MM formats
    import re
    ranges = re.findall(r"(\d{4}(?:-\d{2})?(?:-\d{2})?)", question)
    if not ranges:
        return 'none'
    if len(ranges) >= 2:
        return f"{ranges[0]} to {ranges[1]}"
    return ranges[0]


def simple_sql_for_question(question: str):
    q = question.lower()
    # Top-selling products by quantity in a month range
    if 'top' in q and 'best' in q and 'product' in q:
        # Choose a simple SQL for sample DB
        return (
            "SELECT p.ProductName as product, SUM(od.Quantity) as quantity "
            "FROM OrderDetails od JOIN Products p ON od.ProductID=p.ProductID "
            "JOIN Orders o ON od.OrderID=o.OrderID "
            "GROUP BY p.ProductID ORDER BY quantity DESC LIMIT 3"
        )
    # Total revenue or AOV for a month
    if 'total revenue' in q or 'average order value' in q or 'aov' in q:
        return (
            "SELECT SUM(od.Quantity * od.UnitPrice) as total_revenue, "
            "(SUM(od.Quantity * od.UnitPrice) / COUNT(DISTINCT o.OrderID)) as aov "
            "FROM OrderDetails od JOIN Orders o ON od.OrderID=o.OrderID"
        )
    # Fallback: sample product list
    if 'list' in q and 'product' in q:
        return "SELECT ProductName, UnitPrice FROM Products LIMIT 10"

    # If undetermined, just return a safe sample
    return "SELECT ProductName, UnitPrice FROM Products LIMIT 5"


def synthesize_answer(route, rows, docs=None):
    if route == 'rag':
        if not docs:
            return "No relevant documents found."
        return "\n--- DOCUMENT EXCERPTS ---\n" + "\n\n".join(docs)
    else:
        if rows is None:
            return "No SQL results"
        if not rows:
            return "No rows returned by query"
        try:
            return json.dumps(rows, indent=2)
        except Exception:
            return str(rows)


def main():
    ap = argparse.ArgumentParser(prog="run_agent_hybrid.py")
    ap.add_argument("--question", "-q", required=False, help="Natural language question to ask the hybrid agent (omit to enter interactive mode)")
    ap.add_argument("--create-db/--no-create-db", dest="create_db", default=True, help="Auto-create sample DB if missing")

    args = ap.parse_args()
    # If --question/-q was provided use it, otherwise prompt interactively
    question = args.question.strip() if args.question else None

    if not question:
        # Simple interactive REPL: keep prompting until user enters blank line
        try:
            while True:
                try:
                    question = input("Enter your question (blank to quit): ").strip()
                except EOFError:
                    # When input is piped and there's no more data, exit cleanly
                    print('\nNo more input — exiting.')
                    return
                if not question:
                    print("No question entered — exiting.")
                    return

                # Ensure DB exists and components are initialized for each prompt
                ensure_db(create_if_missing=args.create_db)

                retriever = get_retriever(docs_dir=str(ROOT / 'docs'))
                db_tool = get_db_tool(db_path=str(DB_PATH))

                route = heuristic_route(question)
                print(f"→ Routed to: {route}")

                # Document search when route is rag or hybrid
                doc_context = None
                if route in ('rag', 'hybrid'):
                    docs_found = retriever.search(question, top_k=5)
                    doc_context = [f"{c.full_id}: {c.content[:200]}" for c, s in docs_found]
                    print(f"→ Retrieved {len(doc_context)} document chunks\n")

                # If SQL is relevant, generate a simple SQL and run
                rows = None
                if route in ('sql', 'hybrid'):
                    print("→ Generating SQL (simple heuristic)")
                    sql = simple_sql_for_question(question)
                    print(f"   SQL: {sql[:140]}")
                    result = db_tool.execute_query(sql)
                    if result.success:
                        rows = result.rows
                        print(f"→ SQL returned {len(rows)} rows")
                    else:
                        print(f"→ SQL error: {result.error}")

                # Synthesize and print
                answer = synthesize_answer(route, rows, doc_context)
                print("\n=== FINAL ANSWER ===\n")
                print(answer)

                print('\n--- Ask another question or press Enter to exit ---\n')

        except KeyboardInterrupt:
            print('\nInterrupted — exiting.')
            return

    # If we got here, a question was provided on the command line — run a single request
    ensure_db(create_if_missing=args.create_db)

    retriever = get_retriever(docs_dir=str(ROOT / 'docs'))
    db_tool = get_db_tool(db_path=str(DB_PATH))

    route = heuristic_route(question)
    print(f"→ Routed to: {route}")

    # Document search when route is rag or hybrid
    doc_context = None
    if route in ('rag', 'hybrid'):
        docs_found = retriever.search(question, top_k=5)
        doc_context = [f"{c.full_id}: {c.content[:200]}" for c, s in docs_found]
        print(f"→ Retrieved {len(doc_context)} document chunks\n")

    # If SQL is relevant, generate a simple SQL and run
    rows = None
    if route in ('sql', 'hybrid'):
        print("→ Generating SQL (simple heuristic)")
        sql = simple_sql_for_question(question)
        print(f"   SQL: {sql[:140]}")
        result = db_tool.execute_query(sql)
        if result.success:
            rows = result.rows
            print(f"→ SQL returned {len(rows)} rows")
        else:
            print(f"→ SQL error: {result.error}")

    # Synthesize and print
    answer = synthesize_answer(route, rows, doc_context)
    print("\n=== FINAL ANSWER ===\n")
    print(answer)


if __name__ == '__main__':
    main()
