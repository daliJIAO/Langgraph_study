# agents/subtract_agent.py

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from demo_work.agents.base_agent import BaseAgent
from demo_work.tools.subtract_tool import subtract_tool

class SubtractAgent(BaseAgent):
    """
    专用的减法 Agent，内部只暴露 subtract 工具。
    """
    def __init__(self, **kwargs):
        super().__init__(
            model="qwen-plus",
            tools=[subtract_tool],  # 传递函数对象
            prompt="你是一个专门用来计算两数之差的减法助手。只返回最终数字结果，不要输出任何解释或单位。"
                   "如果输入的括号中已经没有更多的运算符号则直接不再返回括号。",
            **kwargs
        )

# 同样可以在文件末尾加：
# agent = SubtractAgent()
