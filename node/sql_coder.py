import logging
from typing import List, Tuple

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from data.knowledge_manager import KnowledgeManager
from node.deep_state import DeepState, QueryMessage

logger = logging.getLogger(__name__)


class SqlCoderNode:
    def __init__(self, llm: ChatOpenAI, database: str):
        self.llm = llm
        self.database = database
        self.knowledgeManager = KnowledgeManager(database)

    def code_sql(self, state: DeepState) -> DeepState:
        """
        Generate SQL code based on user query and database context.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with generated SQL
            
        Raises:
            Exception: If SQL generation fails
        """
        if state.answer:
            logger.info("SQL generation skipped - answer already exists")
            state.messages.append(QueryMessage(role="SQL生成", content = ""))
            return state

        user_input = state.actual_query
        schema_context = state.schema_context

        logger.info(f"Starting SQL generation for query: {user_input}")

        # Get SQL history
        sql_his: List[str] = self.knowledgeManager.search_sql(state.tables)
        sql_his_text = "\n".join(sql_his) if sql_his else "无参考SQL"

        # Get query history
        query_his: List[Tuple[str, str]] = self.knowledgeManager.search_query(user_input)
        query_his_text = "\n".join(f"{i[0]}: {i[1]}" for i in query_his) if query_his else "无参考查询"

        # Enhanced prompt for SQL generation
        prompt = f"""
    # SQL生成专家任务
    
    你是一个专业的数据SQL专家，负责根据用户查询生成可直接执行的SQL语句。
    
    ## 用户查询
    {user_input}
    
    ## 数据库结构信息
    {schema_context}
    
    ## 参考查询示例
    {query_his_text}
    
    ## 参考SQL语句
    {sql_his_text}
    
    ## SQL生成要求
    
    ### 1. 基本原则
    - 生成可直接在数据库执行的SQL语句
    - 优先使用最合适的表和字段
    - 考虑查询性能和可读性
    - 遵循SQL最佳实践
    - 别名使用中文，通熟易懂
    
    ### 2. 字段选择规则
    - **优先级**: 主键 > 外键 > 维度字段 > 指标字段
    - **语义匹配**: 选择与用户查询语义最接近的字段
    - **数据类型**: 确保字段类型与操作匹配
    - **关联关系**: 正确使用JOIN条件连接相关表
    
    ### 3. SQL结构规范
    - 使用标准的SELECT语句结构
    - 包含必要的WHERE条件
    - 使用适当的聚合函数（如COUNT, SUM, AVG等）
    - 添加有意义的别名
    - 包含必要的排序和分组
    
    ### 4. 输出格式
    - 只输出SQL语句，不要包含任何解释
    - 去除SQL代码块标记（如```sql）
    - 确保SQL语法正确
    - 使用标准SQL格式
    
    """
        res = self.llm.invoke([HumanMessage(prompt)]).text
        state.messages.append(QueryMessage(role="SQL生成", content = f"```sql\n{res}\n```"))

    # Clean up the response
        res = res.split("</think>")[-1].strip()
        res = res.replace("```sql", "").replace("```", "").strip()

        # Validate that we have a SQL statement
        if res and any(keyword in res.upper() for keyword in ["SELECT", "INSERT", "UPDATE", "DELETE"]):
            state.final_sql = res
            logger.info(f"SQL generated successfully: {res[:100]}...")
        else:
            state.answer = "无法生成有效的SQL语句，请检查查询条件和数据库结构。"
            logger.warning("Generated SQL was invalid or empty")

        return state
