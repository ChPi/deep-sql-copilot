from enum import Enum
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field


class IntentType(str, Enum):
    """Enumeration of possible user intents."""
    QUERY = "query"
    ANALYZE = "analyze"
    OTHER = "other"


class QueryMessage(BaseModel):
    """Represents a message in the conversation history."""
    role: str = Field(..., description="Role of the message sender (user, assistant, system)")
    content: str = Field(..., description="Content of the message")


class TableSchema(BaseModel):
    """Represents a database table schema."""
    table_name: str = Field(..., description="Name of the table")
    columns: List[Dict[str, Any]] = Field(..., description="List of column definitions")
    table_comment: str = Field("", description="Comment/description of the table")


class DeepState(BaseModel):
    """
    Main state model for SQL Copilot workflow.
    
    Tracks the entire state of the conversation and workflow execution.
    """

    # User input and session information
    user_input: str = Field(..., description="Original user input query")
    session_id: str = Field(..., description="Unique session identifier")
    database: str = Field(..., description="Target database identifier")
    actual_query: str = Field("", description="Processed/refined query after analysis")

    # Conversation history
    messages: List[QueryMessage] = Field(default_factory=list, description="Conversation message history")

    # Intent analysis
    intent_type: IntentType = Field(IntentType.OTHER, description="Classified intent of the user query")

    # Ambiguity resolution
    intent_ambiguous: Dict[str, Optional[str]] = Field(
        default_factory=dict,
        description="Dictionary of ambiguous concepts and their resolved values"
    )

    # Schema and table information
    tables: List[str] = Field(default_factory=list, description="List of relevant table names")
    table_schema: str = Field("", description="Raw table schema information")
    schema_context: str = Field("", description="Contextual schema information for SQL generation")

    # SQL generation and execution
    final_sql: Optional[str] = Field(None, description="Final generated SQL query")
    query_data: str = Field("", description="Query execution results in markdown format")

    # Response and output
    answer: str = Field("", description="Final answer to present to the user")

    # Execution tracking
    cnt: int = Field(0, description="Counter for tracking execution attempts")

    # Metadata
    execution_time: Optional[float] = Field(None, description="Total execution time in seconds")
    error_message: Optional[str] = Field(None, description="Error message if execution failed")

    class Config:
        """Pydantic model configuration."""
        use_enum_values = True
        validate_assignment = True
        extra = "forbid"  # Prevent extra fields


def create_initial_state(
        user_input: str,
        session_id: str,
        database: str,
        **kwargs
) -> DeepState:
    """
    Create an initial state for the workflow.
    
    Args:
        user_input: Original user query
        session_id: Session identifier
        database: Target database
        **kwargs: Additional state fields
        
    Returns:
        Initialized DeepState instance
    """
    return DeepState(
        user_input=user_input,
        session_id=session_id,
        database=database,
        actual_query=user_input,  # Start with original input
        **kwargs
    )
