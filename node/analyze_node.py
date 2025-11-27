import logging

from langchain_openai import ChatOpenAI

from node.deep_state import DeepState, QueryMessage

logger = logging.getLogger(__name__)


class AnalyzeDataNode:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def analyze_data(self, state: DeepState) -> DeepState:
        """
        Analyze data and generate insights based on user query and data.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with analysis results
            
        Raises:
            Exception: If analysis fails
        """
        if state.answer:
            logger.info("Analysis skipped - answer already exists")
            state.messages.append(QueryMessage(role="报告分析", content = ""))
            return state

        user_input = state.actual_query
        data = state.query_data

        logger.info(f"Starting data analysis for query: {user_input}")

        prompt = f"""
    # 数据分析专家任务
    你是一个专业的数据分析师，负责根据用户查询和提供的数据生成深入的分析报告。
    
    ## 用户查询
    {user_input}
    
    ## 数据详情
    {data}
    
    ## 分析要求
    请按照以下结构生成分析报告：
    
    ### 1. 数据概览
    - 数据的基本统计信息
    - 数据质量和完整性评估
    - 关键指标识别
    
    ### 2. 核心发现
    - 主要趋势和模式
    - 异常值或值得注意的数据点
    - 关键洞察和结论
    
    ### 3. 深度分析
    - 数据之间的关联性分析
    - 时间序列趋势（如果适用）
    - 比较分析（如果适用）
    
    ### 4. 业务意义
    - 对业务决策的启示
    - 潜在的风险或机会
    - 建议的后续行动
    
    ### 5. 局限性说明
    - 分析中的假设
    - 数据的局限性
    - 需要进一步验证的方面
    
    请确保分析报告：
    - 基于数据事实，避免主观臆断
    - 使用清晰、专业的语言
    - 包含具体的数字和指标支持
    - 提供可操作的业务建议
    """
        result = self.llm.invoke(prompt).text
        result = result.split("</think>")[-1].strip()
        state.answer = result

        logger.info("Data analysis completed successfully")
        return state
