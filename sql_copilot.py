from typing import Literal, Dict, Any

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from data import LlmConfigManager
from node.analyze_node import AnalyzeDataNode
from node.deep_state import DeepState
from node.query_analyze import QueryAnalyzeNode, get_user_question
from node.query_intent import QueryIntentNode
from node.sql_coder import SqlCoderNode
from node.sql_fix import SqlFixNode
from node.table_search import TableSearchNode


def intent_path(state: DeepState):
    """
    Determine the next step based on the intent type.
    
    Args:
        state: Current workflow state
        
    Returns:
        Next node name or END
    """
    if state.intent_type == "other":
        return END
    return "query_analyze"

def get_sql_copilot(database: str, checkpointer: Any = None) -> CompiledStateGraph:
    """
    Construct and compile the SQL Copilot state graph.
    
    Args:
        database: Target database ID
        llm: Initialized ChatOpenAI instance
        checkpointer: Optional checkpointer instance
        
    Returns:
        Compiled state graph ready for execution
    """
    # Get LLM configuration
    config_manager = LlmConfigManager()
    llm = config_manager.get_llm("default")

    # Initialize nodes
    analyzer = AnalyzeDataNode(llm)
    query_intent = QueryIntentNode(llm)
    query_analyze = QueryAnalyzeNode(llm)
    sql_coder = SqlCoderNode(llm, database)
    table_search = TableSearchNode(database, llm)
    sql_fix = SqlFixNode(llm, database)
    
    # Create graph
    workflow = StateGraph(DeepState)

    # Add nodes
    workflow.add_node("query_intent", query_intent.parse_intent)
    workflow.add_node("query_analyze", query_analyze.analyze_query)
    workflow.add_node("table_search", table_search.search)
    workflow.add_node("sql_coder", sql_coder.code_sql)
    workflow.add_node("sql_fix", sql_fix.fix_sql)
    workflow.add_node("analyzer", analyzer.analyze_data)
    
    # Set entry point
    workflow.set_entry_point("query_intent")
    
    # Add edges and conditional edges
    workflow.add_conditional_edges(
        "query_intent",
        intent_path
    )
    
    def analyze_path_map(state: DeepState):
        if get_user_question(state):
            return "query_analyze"
        else:
            return "table_search"
            
    workflow.add_conditional_edges(
        "query_analyze",
        analyze_path_map
    )
    
    workflow.add_edge("table_search", "sql_coder")
    workflow.add_edge("sql_coder", "sql_fix")

    def fix_path(state: DeepState):
        if state.query_data:
            return "analyzer" if state.intent_type == "analyze" else END
        else:
            return "sql_fix"
    workflow.add_conditional_edges(
        "sql_fix",
        fix_path
    )
    
    workflow.set_finish_point("analyzer")
    
    # Compile graph with checkpointer
    if checkpointer is None:
        checkpointer = InMemorySaver()
    return workflow.compile(checkpointer)


