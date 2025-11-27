# SQL Copilot

SQL Copilot is an intelligent natural language to SQL interface powered by LangGraph and LLM, featuring **knowledge base management** and **self-evolution capabilities**. The system not only allows users to query databases using natural language but also continuously learns from historical interactions to optimize knowledge base structure and query performance.

![video](web.mp4)
## ✨ Core Features

### 🧠 Knowledge Base Management
- **Intelligent Knowledge Storage**: Automatically stores successful query patterns, table sql
- **Hybrid Search RAG**: Conducts multi vector similarity search with a rerank for rearrangement
- **Pattern Optimization**: Dynamically adjusts knowledge base weights based on query frequency and success rates

### 💡 Intelligent Query
- **Natural Language Understanding**: Converts complex user questions into accurate SQL queries
- **Graph Workflow**: Uses state graphs to manage query analysis, intent recognition, SQL generation, and error recovery
- **Schema Awareness**: Automatically understands database structure and relationships
- **Human-in-the-Loop**: Actively incorporates human feedback and guidance throughout the query process
- **Self-Correction**: Automatically fixes invalid SQL queries and learns correction strategies

## ⚡ Technical Dependencies

- LangGraph: Workflow orchestration
- LangChain: LLM integration
- Milvus: Vector database (for knowledge base storage)
- SQLAlchemy: Database abstraction
- Pandas: Data manipulation

## ⚡ Dependencies

- LangGraph: Workflow orchestration
- LangChain: LLM integration
- Milvus: Vector database
- SQLAlchemy: Database abstraction
- Pandas: Data manipulation

## 🚀Quick Start

### 1. Environment Configuration

#### Configure LLM Service
Create configuration file `config/llm_config.json`:
```json
{
  "chat": {
    "model_name": "gpt-4",
    "api_key": "your-openai-api-key",
    "base_url": "https://api.openai.com/v1",
    "temperature": 0.1,
    "max_tokens": 2000
  }
}
```

#### Configure Database Connection
Create `config/database_config.json`:
```json
{
  "chenjie": {
    "name": "Example Database",
    "type": "mysql",
    "host": "localhost",
    "port": 3306,
    "username": "your_username",
    "password": "your_password",
    "database": "your_database"
  }
}
```

### 5. Initialize System
```bash
# Initialize database structure
python -c "from app import init; init('chenjie')"
```

### 6. Launch Web Application
```bash
python web_app.py

# Or use interactive mode
python app.py
```

### 7. Start Using
Open your browser and navigate to http://localhost:5123. Enter your natural language queries in the chat interface, for example:
- "Query the product with the highest sales"
- "Analyze sales trends across regions"
- "Find the most active users from last month"
