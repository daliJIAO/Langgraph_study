# agents/plus_agent.py

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from demo_work.agents.base_agent import BaseAgent
from demo_work.tools.plus_tool import plus_tool

class PlusAgent(BaseAgent):
    """
    专用的加法 Agent，内部只暴露 plus 工具。
    """
    def __init__(self, **kwargs):
        super().__init__(
            model="qwen-plus",
            tools=[plus_tool],  # 传递函数对象
            prompt="你是一个专门用来计算两数之和的加法助手。只返回最终数字结果，不要输出任何解释或单位。"
                   "如果输入的括号中已经没有更多的运算符号则直接不再返回括号。",
            **kwargs
        )

# 如果你希望在导入时就能直接拿到实例，也可以加一行：
# agent = PlusAgent()
