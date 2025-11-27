import ast
import json
import re

from langchain.agents import create_agent
from pydantic import BaseModel, Field

from data import DatabaseManager
from data.knowledge_manager import KnowledgeManager
from node.deep_state import DeepState, QueryMessage
from node.tools import create_search_entity, create_search_and_embedding


class ValueSearch(BaseModel):
    though: str = Field(..., description="思考过程")


class TableSearchNode:

    def __init__(self, database, llm):
        self.database = database
        self.llm = llm
        self.knowledgeManager = KnowledgeManager(database)
        self.database_manager = DatabaseManager()

    def search(self, state: DeepState) -> DeepState:
        """Search for relevant database columns based on user query."""
        if state.answer:
            state.messages.append(QueryMessage(role="知识搜索", content = ""))
            return state

        tools = [
            create_search_entity(self.knowledgeManager),
            create_search_and_embedding(self.knowledgeManager)
        ]

        # Enhanced system prompt for table search
        system_prompt = """
# 数据库表结构搜索专家

你是一个专业的数据库表结构搜索专家，负责根据用户查询找到构建SQL所需的所有相关字段，只需要严格遵守格式，回答最终结果。

## 任务要求

### 1. 字段搜索原则
- **全面性**: 找到所有可能相关的字段，宁可多不可少
- **相关性**: 包括ID字段、维度字段、指标字段、关联条件等
- **完整性**: 确保输出的字段尽量全面、完整，如果有相关id字段，必须输出

### 2. 字段类型考虑
- **主键字段**: 用于唯一标识记录的字段
- **外键字段**: 用于表关联的字段
- **维度字段**: 用于分组、筛选的字段（如日期、类别、地区等）
- **指标字段**: 用于计算、聚合的字段（如数量、金额、评分等）
- **关联字段**: 用于JOIN操作的字段

### 3. 输出格式

**情况1: 找到相关字段**
[column_id1, column_id2, column_id3, ...]

**情况2: 未找到相关字段**
无法找到满足查询需求的数据库字段，请检查查询条件或数据库结构。

### 4. 示例

**示例1 (成功找到字段):**
[123, 456, 789, 101, 112]

**示例2 (未找到字段):**
无法根据当前查询找到相关的数据库字段，请提供更具体的查询条件。

## 注意事项
- **只需要回答字段ID列表或给用户的回复**
- 确保字段列表足够完整以构建SQL
- 优先选择与查询语义最相关的字段
- 考虑字段在SQL中的各种用途（SELECT、WHERE、GROUP BY、JOIN等）
"""

        agent = create_agent(
            model=self.llm,
            tools=tools,
            system_prompt=system_prompt,
            name="table_search_agent",
        )

        user_input = state.actual_query
        msg = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]
        }
        result = agent.invoke(msg)
        result_text = result["messages"][-1].text

        result_text = result_text.split("</think>")[-1].replace("```", "").strip()
        pattern = r'\[[\d,\s]+\]'
        match = re.search(pattern, result_text)
        if match:
            result_text = match.group()
        # Parse the result
        if result_text.startswith("["):
            try:
                col_list = json.loads(result_text)
                # Validate that we have a list of IDs
                if isinstance(col_list, list) and all(isinstance(item, (int, str)) for item in col_list):
                    df = self.database_manager.get_by_column_id(col_list)
                    state.schema_context = df.to_markdown(index=False)
                    cdf = df[["table_name","column_name","column_comment"]].rename(
                        columns={
                            "table_name": "表名",
                            "column_name": "列名",
                            "column_comment": "注释"
                        }
                    )
                    state.messages.append(QueryMessage(role="知识搜索", content = cdf.to_markdown(index=False)))

                else:
                    state.answer = "搜索到的字段格式不正确，请重试。"
            except json.JSONDecodeError:
                state.answer = f"字段搜索结果解析失败: {result_text}"
        else:
            state.answer = result_text

        return state
