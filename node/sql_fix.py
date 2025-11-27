import pandas as pd
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from data import DatabaseManager
from data.knowledge_manager import KnowledgeManager
from node.deep_state import DeepState, QueryMessage
from node.tools import create_schema, create_exec_sql


class SqlFixNode:
    def __init__(self, llm: ChatOpenAI, database):
        self.llm = llm
        self.database = database
        self.knowledgeManager = KnowledgeManager(database)
        self.database_manager = DatabaseManager()

    def fix_sql(self, state: DeepState) -> DeepState:
        """Fix SQL errors and provide alternative solutions or explanations."""

        if state.answer:
            state.messages.append(QueryMessage(role=f"SQL修复{state.cnt}", content = ""))
            return state



        sql = state.final_sql
        state.cnt += 1

        # Try to execute the SQL
        engine = self.database_manager.get_engine(self.database)
        try:
            df = pd.read_sql(sql, engine)
            state.query_data = df.to_markdown(index=False)
            if state.intent_type == "query":
                state.answer = state.query_data
            return state
        except Exception as e:
            error = str(e)

        # Check retry limit
        if state.cnt > 10:
            state.answer = "SQL修复尝试次数过多，请检查查询条件或联系管理员。"
            return state

        # Enhanced system prompt for SQL debugging
        system_prompt = f"""
# SQL调试与修复专家

你是一个专业的SQL调试专家，负责分析SQL错误并修复。

## 可用数据库表结构
{state.table_schema}

## 任务要求

### 1. 错误分析
- 分析SQL语法错误
- 识别表名、字段名错误
- 检查数据类型不匹配
- 验证JOIN条件和关联关系

### 2. 修复策略
- 优先修复语法错误
- 使用get_schema、exec_sql工具确保表名和字段名正确
- 验证数据类型转换
- 优化查询逻辑

### 3. 输出格式

请严格使用以下格式之一：

**情况1: 成功修复SQL**
```
[sql]修复后的SQL语句
```

**情况2: 无法修复，需要回复用户**
```
[回复]给用户的回复内容
```

### 4. 示例

**示例1 (成功修复):**
[sql]SELECT user_id, username FROM users WHERE status = 'active'

**示例2 (无法修复):**
[回复]无法修复该SQL语句，可能是因为查询条件过于复杂或数据库结构不支持。

## 注意事项
- 只需要输出修复后的SQL或给用户的回复
- 确保修复后的SQL可以直接执行
- 如果无法修复，提供清晰的解释
- 考虑数据库性能和最佳实践
- 如果遇到表结构或者数据问题，使用工具排查修复
"""

        tools = [
            create_schema(self.database_manager, self.database),
            create_exec_sql(self.database_manager, self.database)
        ]

        agent = create_agent(
            model=self.llm,
            tools=tools,
            system_prompt=system_prompt,
            name="sql_fix_agent",
        )

        # Enhanced user prompt with structured error information
        prompt = f"""
## 需要修复的SQL语句
```sql
{sql}
```

## 执行错误信息
```
{error}
```

"""
        msg = {"messages": [{"role": "user", "content": prompt}]}
        result = agent.invoke(msg)
        result_text = result["messages"][-1].text
        state.messages.append(QueryMessage(role=f"SQL修复{state.cnt}", content = f"```sql\n{sql}\n```\n修复错误\n ```\n{error}\n```"))
        result_text = result_text.split("</think>")[-1].replace("```sql", "").replace("```", "").strip()

        # Parse the result
        if "[sql]" in result_text:
            fixed_sql = result_text[result_text.index("[sql]"):].replace("[sql]", "").strip()
            state.final_sql = fixed_sql
            # Recursively try the fixed SQL
            return state
        elif result_text.startswith("[回复]"):
            state.answer = result_text.replace("[回复]", "").strip()
        else:
            # Fallback: if format is not followed, treat as answer
            state.answer = f"SQL修复失败: {result_text}"

        return state
