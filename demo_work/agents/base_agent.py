# agents/base_agent.py

from langgraph.prebuilt import create_react_agent
from langchain_qwq import ChatQwQ

class BaseAgent:
    """
    通用 Agent 基类，内部使用 LangGraph 的 create_react_agent 实现 ReAct 流程，
    并在内部创建 ChatQwQ 模型实例。
    """
    DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def __init__(
        self,
        model: str,
        tools: list,
        prompt: str = "",
        base_url: str | None = None,
        **kwargs
    ):
        """
        :param model: ChatQwQ 模型名称（如 "qwen-plus"）
        :param tools: 工具列表，每个工具应使用 @tool 装饰
        :param prompt: 初始系统提示
        :param base_url: ChatQwQ 接口地址，默认为 DEFAULT_BASE_URL
        :param kwargs: 其他 ChatQwQ 支持的参数（如 max_tokens, timeout, max_retries 等）
        """
        # 使用默认 base_url，除非用户显式传入
        url = base_url or self.DEFAULT_BASE_URL

        # 内部创建 ChatQwQ 实例，除 model 和 base_url 外的所有参数都从 kwargs 读取
        self.llm = ChatQwQ(
            model=model,
            base_url=url,
            **kwargs
        )

        # 使用 LangGraph 的预构建 ReAct agent
        self.agent = create_react_agent(
            self.llm,
            tools,
            prompt=prompt,
        )

    def invoke(self, messages: list[dict]) -> list[dict]:
        """
        通过 ReAct agent 发送消息并获取响应。
        :param messages: 消息列表，格式同 OpenAI Chat API
        :return: 响应消息列表
        """
        result = self.agent.invoke({"messages": messages})
        return result["messages"]
