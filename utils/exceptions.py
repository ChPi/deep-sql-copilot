"""
Custom exceptions for SQL Copilot.
"""


class SQLCopilotError(Exception):
    """Base exception for SQL Copilot application."""
    pass


class ConfigurationError(SQLCopilotError):
    """Raised when there are issues with configuration."""
    pass


class DatabaseConnectionError(SQLCopilotError):
    """Raised when database connection fails."""
    pass


class DatabaseQueryError(SQLCopilotError):
    """Raised when database query execution fails."""
    pass


class LLMServiceError(SQLCopilotError):
    """Raised when LLM service encounters errors."""
    pass


class VectorSearchError(SQLCopilotError):
    """Raised when vector search operations fail."""
    pass


class WorkflowExecutionError(SQLCopilotError):
    """Raised when workflow execution encounters errors."""
    pass


class InvalidQueryError(SQLCopilotError):
    """Raised when user query is invalid or cannot be processed."""
    pass
