# tools/plus_tool.py

from pydantic import BaseModel
from typing import Any
from demo_work.tools.base_tool import register_tool

class PlusSchema(BaseModel):
    """
    参数模型：两个浮点数 a 和 b
    """
    a: float
    b: float

@register_tool("plus", "计算两个浮点数的和，并保留小数点后三位", PlusSchema)
def plus_tool(a: float, b: float) -> Any:
    """
    计算两个浮点数之和，并保留小数点后三位。
    """
    result = round(a + b, 3)
    return result

if __name__ == "__main__":
    from base_tool import invoke_tool

    msg = invoke_tool("plus", {"a": 1.2345, "b": 2.3456})
    print(msg)
    # ToolMessage(tool='plus', success=True, result=3.58)
