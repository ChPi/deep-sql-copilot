from langgraph.types import interrupt

from typing import List

from langgraph.types import interrupt
from pydantic import BaseModel, Field

from data import DatabaseManager
from node.deep_state import DeepState, QueryMessage


class QueryIntent(BaseModel):
    """Schema for query intent analysis response."""
    thought: str = Field(..., description="思考过程")
    intent_ambiguous: List[str] = Field(..., description="模糊的地方，需要向用户澄清")
    cannot_answer: str = Field(..., description="数据库不满足，直接回答用户的内容")
    actual_query: str = Field(..., description="用户真实问题")


def get_user_question(state: DeepState) -> List[str]:
    """Get unresolved ambiguous concepts from state."""
    if state.intent_ambiguous:
        return [key for key, value in state.intent_ambiguous.items() if value is None]
    return []


class QueryAnalyzeNode:
    def __init__(self, llm):
        self.llm = llm
        self.database_manager = DatabaseManager()

    def analyze_query(self, state: DeepState) -> DeepState:
        """Parse user intent and identify ambiguous concepts, direct answers, queries, and analysis."""
        # Use actual_query if available, otherwise use original user_input
        user_input = state.actual_query if state.actual_query else state.user_input

        # Get database schema
        schema = self.database_manager.get_table_schemas(state.database)
        if schema is None:
            state.answer = "数据库中没有找到表模型"
            return state

        # Handle unresolved ambiguous concepts
        question_list = get_user_question(state)
        if question_list:
            for question in question_list:
                updated = interrupt({
                    "instruction": "需要澄清以下模糊概念",
                    "content": question,
                })
                state.intent_ambiguous[question] = updated

        # Prepare context information
        info = "无"
        if state.intent_ambiguous:
            info = "\n".join(f"{key}: {value}" for key, value in state.intent_ambiguous.items())

        # Prepare table schema information
        table_schema = ""
        for table_name, table_info in schema.items():
            comment = table_info.table_comment
            table_schema = f"{table_schema}\n{table_name}: {comment}"

        state.table_schema = table_schema

        # Enhanced prompt with structured approach
        prompt = f"""
# 查询分析专家任务

你是一个专业的数据查询分析专家，负责分析用户查询并确定数据库是否与查询相关。

## 用户查询
{user_input}

## 已知澄清信息
{info}

## 可用数据库表
{table_schema}

## 分析步骤

### 1. 数据库满足性评估
- 检查数据库的表是否与用户查询相关
- 只要表具备部分相关性，那就默认包含了相关所有数据，满足查询条件

### 2. 模糊概念处理
- 检查用户查询中是否存在模糊概念
- 如果已知信息中已澄清，直接使用澄清内容
- 如果存在未澄清的模糊点，需要输出澄清问题

### 3. 查询解析
- 如果一切明确，输出最终的自然语言查询
- 确保查询准确反映用户意图

## 输出格式要求

请严格按照以下JSON格式输出：

```json
{{
    "thought": "详细的分析思考过程",
    "intent_ambiguous": ["需要澄清的问题1", "需要澄清的问题2"],
    "cannot_answer": "如果数据库不满足时的回复内容",
    "actual_query": "解析后的用户查询"
}}
```

## 注意事项
- 如果数据库与查询相关，`cannot_answer` 字段留空
- 只有在真正需要澄清时才在 `intent_ambiguous` 中添加问题
- 确保 `actual_query` 是清晰、明确的自然语言查询
- 用中文回复
"""
        result: QueryIntent = self.llm.with_structured_output(QueryIntent).invoke(prompt)
        # Update state with analysis results
        if result.cannot_answer:
            state.answer = result.cannot_answer
        state.actual_query = result.actual_query
        state.messages.append(QueryMessage(role="查询分析", content = result.thought))

        # Add new ambiguous concepts to state
        if result.intent_ambiguous and len(result.intent_ambiguous) > 0:
            for ambiguous_concept in result.intent_ambiguous:
                state.intent_ambiguous[ambiguous_concept] = None

        return state
