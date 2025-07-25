# tools/subtract_tool.py

from pydantic import BaseModel
from typing import Any
from demo_work.tools.base_tool import register_tool

class SubtractSchema(BaseModel):
    """
    参数模型：两个浮点数 a 和 b
    """
    a: float
    b: float

@register_tool("subtract", "计算两个浮点数的差，并保留小数点后三位", SubtractSchema)
def subtract_tool(a: float, b: float) -> Any:
    """
    计算两个浮点数之和，并保留小数点后三位。
    """
    result = round(a - b, 3)
    return result

if __name__ == "__main__":
    from base_tool import invoke_tool

    msg = invoke_tool("subtract", {"a": 2.6, "b": 2.3})
    print(msg)
    # ToolMessage(tool='subtract', success=True, result=0.3)
