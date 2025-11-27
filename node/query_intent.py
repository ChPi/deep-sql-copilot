from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from node.deep_state import DeepState, IntentType, QueryMessage
from utils.exceptions import LLMServiceError
from utils.logger import get_logger

logger = get_logger(__name__)

intent_map = {
    "query": "问数",
    "analyze": "分析",
    "other": "其他"
}

class QueryIntentNode:
    """Node for analyzing user intent in natural language queries."""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self._setup_prompts()

    def _setup_prompts(self):
        """Setup prompt templates for intent analysis."""
        self.intent_prompt = ChatPromptTemplate.from_messages([
            ("system", """# 数据意图分析专家

你是一个专业的数据意图分析专家，负责准确分类用户输入的意图类型。

## 意图分类标准

### 1. 数据查询 (query)
- **定义**: 用户请求从数据库中检索特定数据记录或信息
- **特征**: 涉及数据提取、筛选、排序、聚合等操作
- **示例**:
  - "显示销售额前10的产品"
  - "查询上个月的订单详情"
  - "找出库存量低于100的商品"
  - "统计每个地区的客户数量"

### 2. 数据分析 (analyze)
- **定义**: 用户请求对数据进行解释、洞察、趋势分析或深度解读
- **特征**: 涉及数据解释、原因分析、趋势识别、业务洞察
- **示例**:
  - "分析销售趋势"
  - "解释为什么这个月收入下降"
  - "对比不同产品的市场表现"
  - "预测下个季度的销售情况"

### 3. 其他 (other)
- **定义**: 不涉及数据查询或分析的请求
- **特征**: 系统问题、问候、配置问题、不相关的问题
- **示例**:
  - "你好"、"谢谢"
  - "系统如何使用"
  - "修改数据库连接"
  - "不相关的问题"

## 输出格式要求

请严格遵循以下输出规则：

### 情况1: 数据查询意图
只输出：`query`

### 情况2: 数据分析意图
只输出：`analyze`

### 情况3: 其他意图
直接提供有帮助的回答，例如：
- 对于问候："您好！我是数据分析助手，可以帮助您查询和分析数据。"
- 对于系统问题："我主要专注于数据查询和分析，系统配置问题请咨询管理员。"
- 对于不明确的问题："您的问题不太明确，请提供更具体的数据查询或分析需求。"

## 用户输入
{user_input}

## 你的分析
请基于上述分类标准进行准确判断：""")
        ])

    def parse_intent(self, state: DeepState) -> DeepState:
        """
        Parse user intent and identify ambiguous concepts.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with intent analysis
            
        Raises:
            LLMServiceError: If LLM service fails
        """
        try:
            logger.info(f"Analyzing intent for query: {state.user_input}")

            # Prepare prompt
            prompt = self.intent_prompt.format_messages(user_input=state.user_input)

            # Get LLM response
            response = self.llm.invoke(prompt)
            result_text = self._extract_response_text(response)
            state.messages.append(QueryMessage(role="路由", content = intent_map.get(result_text, "其他")))

            # Parse intent
            intent_type, answer = self._parse_intent_result(result_text)

            # Update state
            state.intent_type = intent_type
            if intent_type == IntentType.OTHER and answer:
                state.answer = answer
                logger.info(f"Intent classified as 'other', providing answer: {answer[:100]}...")
            else:
                logger.info(f"Intent classified as: {intent_type.value}")

            return state

        except Exception as e:
            logger.error(f"Intent analysis failed: {str(e)}")
            raise LLMServiceError(f"Intent analysis failed: {str(e)}") from e

    def _extract_response_text(self, response) -> str:
        """Extract and clean response text from LLM."""
        if hasattr(response, 'text'):
            text = response.text
        elif hasattr(response, 'content'):
            text = response.content
        else:
            text = str(response)

        # Remove thinking tags if present
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()

        return text.strip()

    def _parse_intent_result(self, result_text: str) -> tuple:
        """
        Parse the intent classification result.
        
        Args:
            result_text: Raw LLM response text
            
        Returns:
            Tuple of (intent_type, answer)
        """
        result_text_lower = result_text.lower().strip()

        # Handle exact matches first
        if result_text_lower == "query":
            logger.info("Intent classified as QUERY")
            return IntentType.QUERY, None
        elif result_text_lower == "analyze":
            logger.info("Intent classified as ANALYZE")
            return IntentType.ANALYZE, None
        else:
            # Check for partial matches in case of formatting issues
            if "query" in result_text_lower and len(result_text_lower) < 20:
                logger.info("Intent classified as QUERY (partial match)")
                return IntentType.QUERY, None
            elif "analyze" in result_text_lower and len(result_text_lower) < 20:
                logger.info("Intent classified as ANALYZE (partial match)")
                return IntentType.ANALYZE, None
            else:
                # If it's not a clear intent classification, treat as "other"
                # and use the response as the answer
                logger.info(f"Intent classified as OTHER: {result_text[:100]}...")
                return IntentType.OTHER, result_text
