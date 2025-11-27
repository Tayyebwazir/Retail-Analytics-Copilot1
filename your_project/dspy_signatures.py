"""
DSPy signatures for the retail analytics agent.
"""
import dspy
from typing import Literal


class RouteQuery(dspy.Signature):
    """
    Classify whether a question requires:
    - 'rag': Only document search (policies, definitions, dates)
    - 'sql': Only database query (pure numerical analysis)
    - 'hybrid': Both documents and database (e.g., KPI calculation with date constraints)
    """
    
    question: str = dspy.InputField(desc="The user's question")
    route: Literal['rag', 'sql', 'hybrid'] = dspy.OutputField(
        desc="Classification: 'rag', 'sql', or 'hybrid'"
    )


class ExtractConstraints(dspy.Signature):
    """
    Extract structured constraints from question and document context.
    Identifies: date ranges, KPI formulas, product categories, entities.
    """
    
    question: str = dspy.InputField(desc="The user's question")
    doc_context: str = dspy.InputField(desc="Retrieved document chunks")
    
    date_range: str = dspy.OutputField(desc="Date range in YYYY-MM-DD format, or 'none'")
    kpi_formula: str = dspy.OutputField(desc="KPI calculation formula if mentioned, or 'none'")
    categories: str = dspy.OutputField(desc="Comma-separated product categories, or 'none'")
    entities: str = dspy.OutputField(desc="Relevant entities (customers, products), or 'none'")


class GenerateSQL(dspy.Signature):
    """
    Generate valid SQLite query from natural language question.
    Uses database schema and extracted constraints.
    """
    
    question: str = dspy.InputField(desc="The user's question")
    schema: str = dspy.InputField(desc="Database schema with table and column info")
    constraints: str = dspy.InputField(desc="Extracted constraints (dates, categories, etc.)")
    
    sql: str = dspy.OutputField(desc="Valid SQLite SELECT query")
    reasoning: str = dspy.OutputField(desc="Brief explanation of the query logic")


class RepairSQL(dspy.Signature):
    """
    Fix a SQL query that failed to execute.
    Uses the error message to diagnose and correct the issue.
    """
    
    original_sql: str = dspy.InputField(desc="The SQL query that failed")
    error_message: str = dspy.InputField(desc="The error message from execution")
    schema: str = dspy.InputField(desc="Database schema")
    
    fixed_sql: str = dspy.OutputField(desc="Corrected SQL query")
    changes_made: str = dspy.OutputField(desc="What was fixed")


class SynthesizeAnswer(dspy.Signature):
    """
    Format the final answer matching the required format_hint.
    Combines SQL results and document context with proper citations.
    """
    
    question: str = dspy.InputField(desc="Original question")
    sql_results: str = dspy.InputField(desc="Results from SQL query (JSON format)")
    doc_chunks: str = dspy.InputField(desc="Retrieved document chunks with IDs")
    format_hint: str = dspy.InputField(desc="Expected format: 'int', 'float', '{key:type}', etc.")
    
    answer: str = dspy.OutputField(desc="Final answer matching format_hint exactly")
    confidence: str = dspy.OutputField(desc="Confidence score 0.0-1.0")
    explanation: str = dspy.OutputField(desc="Brief explanation (max 2 sentences)")
    citations: str = dspy.OutputField(desc="Comma-separated: table names and doc chunk IDs")


# DSPy Modules (wrappers around signatures)

class Router(dspy.Module):
    """Route incoming questions to the appropriate processing path."""
    
    def __init__(self):
        super().__init__()
        self.classify = dspy.ChainOfThought(RouteQuery)
    
    def forward(self, question: str) -> str:
        result = self.classify(question=question)
        return result.route


class Planner(dspy.Module):
    """Extract constraints from question and documents."""
    
    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(ExtractConstraints)
    
    def forward(self, question: str, doc_context: str) -> dspy.Prediction:
        return self.extract(question=question, doc_context=doc_context)


class SQLGenerator(dspy.Module):
    """Generate SQL from natural language."""
    
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(GenerateSQL)
    
    def forward(self, question: str, schema: str, constraints: str) -> dspy.Prediction:
        return self.generate(question=question, schema=schema, constraints=constraints)


class SQLRepairer(dspy.Module):
    """Fix broken SQL queries."""
    
    def __init__(self):
        super().__init__()
        self.repair = dspy.ChainOfThought(RepairSQL)
    
    def forward(self, original_sql: str, error_message: str, schema: str) -> dspy.Prediction:
        return self.repair(
            original_sql=original_sql,
            error_message=error_message,
            schema=schema
        )


class Synthesizer(dspy.Module):
    """Synthesize final formatted answer with citations."""
    
    def __init__(self):
        super().__init__()
        self.synthesize = dspy.ChainOfThought(SynthesizeAnswer)
    
    def forward(self, question: str, sql_results: str, 
                doc_chunks: str, format_hint: str) -> dspy.Prediction:
        return self.synthesize(
            question=question,
            sql_results=sql_results,
            doc_chunks=doc_chunks,
            format_hint=format_hint
        )


if __name__ == "__main__":
    print("✅ DSPy signatures and modules defined successfully")
    print("\nAvailable modules:")
    print("  - Router: Classifies questions")
    print("  - Planner: Extracts constraints")
    print("  - SQLGenerator: Creates SQL queries")
    print("  - SQLRepairer: Fixes broken SQL")
    print("  - Synthesizer: Formats final answers")