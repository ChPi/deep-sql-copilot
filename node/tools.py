import logging
from typing import List, Tuple, Any

import pandas as pd
from langchain_core.tools import tool

from data import DatabaseManager
from data.knowledge_manager import KnowledgeManager

logger = logging.getLogger(__name__)


def create_schema(databaseManager: DatabaseManager, database: str):
    """
    Create a tool for retrieving table schema information.
    
    Args:
        databaseManager: Database manager instance
        database: Database identifier
        
    Returns:
        Tool function for schema retrieval
    """

    @tool
    def get_schema(table: str) -> str:
        """
        Retrieve the schema information for a specified table.
        
        This tool helps analyze table structure including column names,
        data types, and constraints to assist in SQL generation and debugging.
        
        Args:
            table: Name of the table to get schema for
            
        Returns:
            String containing table schema information
            
        Example:
            get_schema("users") -> "Table: users\nColumns: id (int), name (varchar), ..."
        """
        try:
            logger.info(f"Retrieving schema for table: {table}")
            schema = databaseManager.get_schema(database, table)
            return f"Schema for table '{table}':\n{schema}"
        except Exception as e:
            error_msg = f"Error retrieving schema for table '{table}': {str(e)}"
            logger.error(error_msg)
            return error_msg

    return get_schema


def create_exec_sql(databaseManager: DatabaseManager, database: str):
    """
    Create a tool for executing SQL queries and returning results.
    
    Args:
        databaseManager: Database manager instance
        database: Database identifier
        
    Returns:
        Tool function for SQL execution
    """

    @tool
    def exec_sql(sql: str) -> str:
        """
        Execute an SQL query and return the results in markdown format.
        
        This tool is used for testing SQL queries, debugging, and validating
        query results. It executes the SQL against the target database and
        returns formatted results.
        
        Args:
            sql: SQL query to execute
            
        Returns:
            Query results in markdown format or error message
            
        Example:
            exec_sql("SELECT * FROM users LIMIT 5") -> "| id | name | ..."
        """
        try:
            logger.info(f"Executing SQL: {sql}")
            engine = databaseManager.get_engine(database)
            df = pd.read_sql(sql, engine)

            if df.empty:
                return "Query executed successfully but returned no results."

            markdown_result = df.to_markdown(index=False)
            return markdown_result

        except Exception as e:
            return str(e)

    return exec_sql


def create_search_entity(knowledgeManager: KnowledgeManager):
    """
    Create a tool for searching entity values in the database.
    
    Args:
        knowledgeManager: Knowledge manager instance
        
    Returns:
        Tool function for entity value search
    """

    @tool
    def search_entity(text: str) -> List[Tuple[Any, Any, Any]]:
        """
        Search for entity string values by keyword and return matching fields with similarity scores.
        
        This tool helps find specific values in the database and identifies
        which columns contain those values, along with similarity scores
        to rank the matches.
        
        Args:
            text: Keyword or value to search for, must be string
            
        Returns:
            List of tuples containing (column_id, value, similarity_score)
            
        Example:
            search_entity("john") -> [(123, "john_doe", 0.95), (456, "john_smith", 0.87)]
        """
        try:
            logger.info(f"Searching entities for: {text}")
            results = knowledgeManager.search_entity(text)

            if not results:
                return [("No matching entities found", "", 0.0)]

            # Format results for better readability
            formatted_results = []
            for result in results:
                if len(result) >= 3:
                    formatted_results.append((
                        result[0],  # column_id
                        str(result[1]),  # value
                        float(result[2]) if isinstance(result[2], (int, float)) else 0.0  # similarity
                    ))

            return formatted_results

        except Exception as e:
            error_msg = f"Entity search error for '{text}': {str(e)}"
            logger.error(error_msg)
            return [(error_msg, "", 0.0)]

    return search_entity


def create_search_and_embedding(knowledgeManager: KnowledgeManager):
    """
    Create a tool for searching database columns using semantic embedding.
    
    Args:
        knowledgeManager: Knowledge manager instance
        
    Returns:
        Tool function for column search with embeddings
    """

    @tool
    def search_col(text: str) -> List[Tuple[Any, Any]]:
        """
        Search for database columns using semantic similarity with embeddings.
        
        This tool uses advanced embedding techniques to find columns that
        are semantically related to the search text, even if the exact
        keywords don't match. Useful for discovering relevant fields
        for complex queries.
        
        Args:
            text: Search text describing the desired column functionality
            
        Returns:
            List of tuples containing (column_id, column_comment/description)
            
        Example:
            search_col("customer name") -> [(123, "customer_full_name"), (456, "client_name")]
        """
        try:
            logger.info(f"Searching columns for: {text}")
            results = knowledgeManager.search_and_embedding_col(text)

            if not results:
                return [("No matching columns found", "Try different search terms")]

            return results

        except Exception as e:
            error_msg = f"Column search error for '{text}': {str(e)}"
            logger.error(error_msg)
            return [(error_msg, "Search failed")]

    return search_col
