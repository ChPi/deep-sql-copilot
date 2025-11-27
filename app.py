from time import sleep
from typing import Dict, Any

from langgraph.types import Command

from data import DatabaseManager, LlmConfigManager
from data.knowledge_manager import KnowledgeManager
from sql_copilot import get_sql_copilot
from utils.exceptions import (
    LLMServiceError, WorkflowExecutionError,
    ConfigurationError, InvalidQueryError
)
from utils.logger import setup_logger, get_logger

# Setup application logger
setup_logger()
logger = get_logger(__name__)


def init(database_id: str) -> None:
    """
    Initialize database by storing target database table column information to system tables.
    
    Args:
        database_id: Target database ID
        
    Raises:
        ConfigurationError: If database configuration is not found
        DatabaseConnectionError: If database connection fails
        WorkflowExecutionError: If initialization process fails
    """
    try:
        logger.info(f"Starting database initialization for {database_id}")
        database_manager = DatabaseManager()
        
        # Get database configuration
        db_config = database_manager.databases.get(database_id)
        if not db_config:
            raise ConfigurationError(f"Database configuration not found: {database_id}")
        
        logger.info(f"Initializing database {database_id} ({db_config.name})")
        
        # Save table structure information to system database
        database_manager.save_table_schemas_to_system(database_id)
        
        # Initialize KnowledgeManager
        knowledge_manager = KnowledgeManager(database_id)
        df = database_manager.get_column(database_id)
        if not df.empty:
            knowledge_manager.add_column(df[['id', 'column_comment']].to_dict('records'))
        
        logger.info(f"Database {database_id} initialization completed successfully")
        
    except ConfigurationError:
        logger.error(f"Database configuration error for {database_id}")
        raise
    except Exception as e:
        logger.error(f"Database initialization failed for {database_id}: {str(e)}")
        raise WorkflowExecutionError(f"Database initialization failed: {str(e)}") from e

# Global checkpointer for state persistence
from langgraph.checkpoint.memory import InMemorySaver
checkpointer = InMemorySaver()

def process_chat(query: str, database_id: str = "chenjie", session_id: str = "default", resume_input: str = None) -> Dict[str, Any]:
    """
    Process a chat message or resume from interruption without blocking.
    
    Args:
        query: Natural language query (used if starting new query)
        database_id: Target database ID
        session_id: Session ID for conversation tracking
        resume_input: User input to resume execution (used if resuming)
        
    Returns:
        Dictionary containing result, interruption info, or error
    """
    try:


        # Create graph with shared checkpointer
        graph = get_sql_copilot(database_id, checkpointer=checkpointer)

        config = {"configurable": {"thread_id": session_id}}

        if resume_input is not None:
            # Resume execution with user input
            logger.info(f"Resuming execution for session {session_id} with input: {resume_input}")
            result = graph.invoke(Command(resume=resume_input), config=config)
        else:
            # Start new execution
            if not query or not query.strip():
                raise InvalidQueryError("Query cannot be empty")
                
            logger.info(f"Processing query: '{query}' for database: {database_id}")
            
            initial_state = {
                "session_id": session_id,
                "user_input": query,
                "database": database_id,
                "intent_ambiguous": {},
                "messages": [],
                "intent_type": "other",
                "schema_context": "",
                "final_sql": "",
                "answer": "",
                "cnt": 0,
                "actual_query": "",
                "tables": []
            }
            result = graph.invoke(initial_state, config=config)

        # Check result state
        if "answer" in result and result["answer"]:
            return {
                "success": True,
                "answer": result["answer"],
                "sql": result.get("final_sql"),
                "query_data": result.get("query_data"),
                "session_id": session_id,
                "status": "completed"
            }
        
        if "__interrupt__" in result and result["__interrupt__"]:
            question = result["__interrupt__"][0].value["content"]
            return {
                "success": True,
                "question": question,
                "session_id": session_id,
                "status": "interrupted"
            }
            
        # Fallback for unexpected state
        return {
            "success": False,
            "error": "Unexpected workflow state",
            "status": "error"
        }
        
    except Exception as e:
        logger.error(f"Error in process_chat: {str(e)}")
        raise

def stream_chat(query: str, database_id: str = "chenjie", session_id: str = "default", resume_input: str = None):
    """
    Stream chat events from the LangGraph workflow.
    
    Yields:
        Dict containing event data (node name, content, etc.)
    """
    try:

        # Create graph with shared checkpointer
        graph = get_sql_copilot(database_id, checkpointer=checkpointer)

        config = {"configurable": {"thread_id": session_id}, "recursion_limit": 100}

        if resume_input is not None:
            # Resume execution with user input
            logger.info(f"Resuming execution for session {session_id} with input: {resume_input}")
            input_data = Command(resume=resume_input)
        else:
            # Start new execution
            if not query or not query.strip():
                raise InvalidQueryError("Query cannot be empty")
                
            logger.info(f"Processing query: '{query}' for database: {database_id}")
            
            input_data = {
                "session_id": session_id,
                "user_input": query,
                "database": database_id,
                "intent_ambiguous": {},
                "messages": [],
                "intent_type": "other",
                "schema_context": "",
                "final_sql": "",
                "answer": "",
                "cnt": 0,
                "actual_query": "",
                "tables": []
            }

        # Stream events from the graph
        # stream_mode="updates" gives us the state updates from each node
        final_state = None
        for event in graph.stream(input_data, config=config, stream_mode="updates" ):
            for node_name, state_update in event.items():
                final_state = state_update
                if node_name == "__interrupt__":
                     # Handle interruption (ambiguity)
                    question = state_update[0].value["content"]
                    yield {
                        "type": "interrupt",
                        "node": "system",
                        "content": question
                    }
                else:
                    # Determine content based on node type and state update
                    content = ""
                    # This logic might need adjustment based on what each node actually returns in state_update
                    # For now, we try to extract meaningful info
                    
                    if "messages" in state_update and state_update["messages"]:
                        last_msg = state_update["messages"][-1]
                        node_name = last_msg.role
                        if hasattr(last_msg, "content"):
                            content = last_msg.content
                        else:
                            content = str(last_msg)
                    elif "answer" in state_update:
                        content = state_update["answer"]
                    elif "final_sql" in state_update:
                        content = f"Generated SQL: \n```sql\n{state_update['final_sql']}\n```"
                    elif "question" in state_update: # For semantic router or ambiguity
                         content = f"Clarification needed: {state_update['question']}"
                    
                    # Yield the event
                    if content:
                        yield {
                            "type": "chunk",
                            "node": node_name,
                            "content": content,
                        }
                    
        # If we have a final answer, yield a completion event
        if "answer" in final_state and final_state["answer"]:
             yield {
                "type": "complete",
                "node": "system",
                "content": final_state["answer"],
                "sql": final_state.get("final_sql")
            }

    except Exception as e:
        logger.error(f"Error in stream_chat: {str(e)}", e)
        yield {
            "type": "error",
            "node": "system",
            "content": str(e)
        }

def ask(query: str, database_id: str = "chenjie", session_id: str = "default") -> Dict[str, Any]:
    """
    CLI wrapper for process_chat that handles interruptions interactively.
    """
    # Initial call
    result = process_chat(query, database_id, session_id)
    
    # Handle interruptions loop
    while result.get("status") == "interrupted":
        question = result.get("question")
        logger.info(f"Requesting user clarification: {question}")
        print("input----")
        user_input = input(question)
        
        # Resume execution
        result = process_chat(None, database_id, session_id, resume_input=user_input)
        
    return result

def main():
    """Main entry point for the SQL Copilot application."""
    logger.info("Starting SQL Copilot application")

    # Initialize database (commented out for demo)
    # logger.info("Initializing database...")
    init("chenjie")

    # Example query
    # query = "在各公司所有品牌收入排名中，给出每一个品牌，其所在公司以及收入占该公司的总收入比例，同时给出该公司的年营业额"
    # # query = "分析下每个品牌收入"
    # logger.info(f"Executing example query: {query}")
    #
    # result = ask(query)
    #
    # logger.info(result)
    # print(result["answer"])
    
    return 0


if __name__ == '__main__':
    exit(main())


