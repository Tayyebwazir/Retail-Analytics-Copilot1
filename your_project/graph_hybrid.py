"""
LangGraph hybrid agent with RAG + SQL capabilities.
Implements a stateful graph with routing, retrieval, planning, SQL generation,
execution, repair, and synthesis.
"""
import json
from typing import TypedDict, Literal, List, Dict, Any, Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import dspy
from dspy import LM

from agent.rag.retrieval import get_retriever
from agent.tools.sqlite_tool import get_db_tool
from agent.dspy_signatures import (
    Router, Planner, SQLGenerator, SQLRepairer, Synthesizer
)


# State definition for the graph
class AgentState(TypedDict):
    """State that flows through the graph."""
    question: str
    format_hint: str
    route: str  # 'rag', 'sql', 'hybrid'
    
    # Retrieval
    doc_chunks: List[Dict[str, Any]]
    doc_context: str
    
    # Planning
    constraints: Dict[str, str]
    
    # SQL
    sql: str
    sql_results: List[Dict[str, Any]]
    sql_error: str
    repair_count: int
    
    # Final
    final_answer: Any
    confidence: float
    explanation: str
    citations: List[str]
    
    # Trace
    trace: List[str]


def add_trace(state: AgentState, message: str) -> AgentState:
    """Helper to add trace messages."""
    if 'trace' not in state:
        state['trace'] = []
    state['trace'].append(message)
    return state


class HybridAgent:
    """
    Main agent orchestrating RAG + SQL with LangGraph.
    """
    
    def __init__(self, model_name: str = "ollama_chat/phi3.5:3.8b-mini-instruct-q4_K_M"):
        """
        Initialize the agent with local Phi-3.5 model.
        
        Args:
            model_name: Ollama model identifier
        """
        print("🚀 Initializing Hybrid Agent...")
        
        # Initialize DSPy with local Ollama
        self.lm = LM(model=model_name, max_tokens=1000, temperature=0.3)
        dspy.configure(lm=self.lm)
        
        # Initialize components
        self.retriever = get_retriever()
        self.db_tool = get_db_tool()
        
        # Initialize DSPy modules
        self.router = Router()
        self.planner = Planner()
        self.sql_generator = SQLGenerator()
        self.sql_repairer = SQLRepairer()
        self.synthesizer = Synthesizer()
        
        # Build the graph
        self.graph = self._build_graph()
        
        print("✅ Agent initialized successfully")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        
        # Create graph with state
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("router", self.node_router)
        workflow.add_node("retriever", self.node_retriever)
        workflow.add_node("planner", self.node_planner)
        workflow.add_node("sql_generator", self.node_sql_generator)
        workflow.add_node("executor", self.node_executor)
        workflow.add_node("repair", self.node_repair)
        workflow.add_node("synthesizer", self.node_synthesizer)
        
        # Define edges
        workflow.set_entry_point("router")
        
        # Router conditionally routes to retriever based on route type
        workflow.add_conditional_edges(
            "router",
            self.should_retrieve,
            {
                "retrieve": "retriever",
                "skip_retrieve": "planner"
            }
        )
        
        workflow.add_edge("retriever", "planner")
        
        # Planner routes to SQL generator or directly to synthesizer
        workflow.add_conditional_edges(
            "planner",
            self.should_generate_sql,
            {
                "generate_sql": "sql_generator",
                "skip_sql": "synthesizer"
            }
        )
        
        workflow.add_edge("sql_generator", "executor")
        
        # Executor routes to repair on error, or synthesizer on success
        workflow.add_conditional_edges(
            "executor",
            self.should_repair,
            {
                "repair": "repair",
                "synthesize": "synthesizer"
            }
        )
        
        workflow.add_edge("repair", "executor")  # Repair loops back to executor
        workflow.add_edge("synthesizer", END)
        
        return workflow.compile(checkpointer=MemorySaver())
    
    # ============================================================
    # GRAPH NODES
    # ============================================================
    
    def node_router(self, state: AgentState) -> AgentState:
        """Node 1: Route the question."""
        add_trace(state, "🔀 ROUTER: Classifying question")
        
        try:
            route = self.router(question=state['question'])
            state['route'] = route.lower() if isinstance(route, str) else str(route).lower()
        except Exception as e:
            print(f"⚠️ Router error: {e}, defaulting to 'hybrid'")
            state['route'] = 'hybrid'
        
        add_trace(state, f"   Route: {state['route']}")
        return state
    
    def node_retriever(self, state: AgentState) -> AgentState:
        """Node 2: Retrieve relevant document chunks."""
        add_trace(state, "📚 RETRIEVER: Searching documents")
        
        results = self.retriever.search(state['question'], top_k=5)
        
        state['doc_chunks'] = [
            {
                'id': chunk.full_id,
                'content': chunk.content,
                'score': score
            }
            for chunk, score in results
        ]
        
        # Create formatted context for downstream modules
        context_parts = []
        for chunk_data in state['doc_chunks']:
            context_parts.append(
                f"[{chunk_data['id']}] {chunk_data['content']}"
            )
        state['doc_context'] = "\n\n".join(context_parts)
        
        add_trace(state, f"   Found {len(state['doc_chunks'])} relevant chunks")
        return state
    
    def node_planner(self, state: AgentState) -> AgentState:
        """Node 3: Extract constraints from question and docs."""
        add_trace(state, "🗺️  PLANNER: Extracting constraints")
        
        doc_context = state.get('doc_context', 'No document context available.')
        
        try:
            result = self.planner(
                question=state['question'],
                doc_context=doc_context
            )
            
            state['constraints'] = {
                'date_range': result.date_range,
                'kpi_formula': result.kpi_formula,
                'categories': result.categories,
                'entities': result.entities
            }
        except Exception as e:
            print(f"⚠️ Planner error: {e}")
            state['constraints'] = {
                'date_range': 'none',
                'kpi_formula': 'none',
                'categories': 'none',
                'entities': 'none'
            }
        
        add_trace(state, f"   Constraints: {state['constraints']}")
        return state
    
    def node_sql_generator(self, state: AgentState) -> AgentState:
        """Node 4: Generate SQL query."""
        add_trace(state, "🔧 SQL_GENERATOR: Creating query")
        
        schema = self.db_tool.get_schema()
        constraints_str = json.dumps(state['constraints'], indent=2)
        
        try:
            result = self.sql_generator(
                question=state['question'],
                schema=schema,
                constraints=constraints_str
            )
            
            state['sql'] = result.sql
            add_trace(state, f"   Generated SQL: {result.sql[:100]}...")
        except Exception as e:
            print(f"⚠️ SQL generation error: {e}")
            state['sql'] = ""
            state['sql_error'] = str(e)
        
        return state
    
    def node_executor(self, state: AgentState) -> AgentState:
        """Node 5: Execute SQL query."""
        add_trace(state, "⚡ EXECUTOR: Running SQL")
        
        if not state.get('sql'):
            state['sql_error'] = "No SQL query to execute"
            state['sql_results'] = []
            return state
        
        result = self.db_tool.execute_query(state['sql'])
        
        if result.success:
            state['sql_results'] = result.rows
            state['sql_error'] = ""
            add_trace(state, f"   ✅ Success: {len(result.rows)} rows")
        else:
            state['sql_error'] = result.error
            state['sql_results'] = []
            add_trace(state, f"   ❌ Error: {result.error}")
        
        return state
    
    def node_repair(self, state: AgentState) -> AgentState:
        """Node 7: Repair failed SQL."""
        repair_count = state.get('repair_count', 0)
        state['repair_count'] = repair_count + 1
        
        add_trace(state, f"🔨 REPAIR: Attempt {state['repair_count']}/2")
        
        schema = self.db_tool.get_schema()
        
        try:
            result = self.sql_repairer(
                original_sql=state['sql'],
                error_message=state['sql_error'],
                schema=schema
            )
            
            state['sql'] = result.fixed_sql
            add_trace(state, f"   Changes: {result.changes_made}")
        except Exception as e:
            print(f"⚠️ Repair error: {e}")
            add_trace(state, f"   Repair failed: {e}")
        
        return state
    
    def node_synthesizer(self, state: AgentState) -> AgentState:
        """Node 6: Synthesize final answer."""
        add_trace(state, "🎯 SYNTHESIZER: Formatting answer")
        
        # Prepare inputs
        sql_results_str = json.dumps(state.get('sql_results', []), indent=2)
        doc_chunks_str = state.get('doc_context', 'No documents retrieved')
        
        try:
            result = self.synthesizer(
                question=state['question'],
                sql_results=sql_results_str,
                doc_chunks=doc_chunks_str,
                format_hint=state['format_hint']
            )
            
            # Parse answer
            state['final_answer'] = self._parse_answer(
                result.answer, 
                state['format_hint']
            )
            
            # Parse confidence
            try:
                state['confidence'] = float(result.confidence)
            except:
                state['confidence'] = 0.5
            
            state['explanation'] = result.explanation
            
            # Parse citations
            citations_raw = result.citations.split(',')
            state['citations'] = [c.strip() for c in citations_raw if c.strip()]
            
            # Add table names from SQL if used
            if state.get('sql'):
                tables = self.db_tool.get_table_names()
                for table in tables:
                    if table in state['sql'] and table not in state['citations']:
                        state['citations'].append(table)
            
            # Add doc chunk IDs
            for chunk in state.get('doc_chunks', []):
                if chunk['id'] not in state['citations']:
                    state['citations'].append(chunk['id'])
            
            add_trace(state, f"   Answer: {state['final_answer']}")
            add_trace(state, f"   Confidence: {state['confidence']}")
            
        except Exception as e:
            print(f"⚠️ Synthesizer error: {e}")
            state['final_answer'] = "Error synthesizing answer"
            state['confidence'] = 0.0
            state['explanation'] = f"Synthesis failed: {e}"
            state['citations'] = []
        
        return state