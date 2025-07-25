"""
计算代理模块
该模块定义了基本的数学计算工具，包括加法、乘法和除法操作
这些工具可以被LLM调用来执行数学计算任务
"""

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI  # 导入LLM模型

# 初始化语言模型
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 定义计算工具集

@tool
def multiply(a: int, b: int) -> int:
    """
    乘法运算工具

    执行两个整数的乘法运算

    Args:
        a: 第一个整数（被乘数）
        b: 第二个整数（乘数）

    Returns:
        int: 乘法运算结果
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """
    加法运算工具

    执行两个整数的加法运算

    Args:
        a: 第一个整数（加数）
        b: 第二个整数（加数）

    Returns:
        int: 加法运算结果
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """
    除法运算工具

    执行两个整数的除法运算

    Args:
        a: 第一个整数（被除数）
        b: 第二个整数（除数）

    Returns:
        float: 除法运算结果

    Note:
        当b为0时会抛出ZeroDivisionError异常
    """
    return a / b


# 将工具与LLM进行集成

# 工具列表：包含所有可用的数学计算工具
tools = [add, multiply, divide]

# 工具字典：通过工具名称快速查找工具对象
tools_by_name = {tool.name: tool for tool in tools}

# 增强型LLM：绑定了计算工具的语言模型
# 这使得LLM可以在对话中调用这些数学计算工具
llm_with_tools = llm.bind_tools(tools)