# tools/base_tool.py

from typing import Any, Callable, Type, Dict
from pydantic import BaseModel, ValidationError
from dataclasses import dataclass

# 全局工具注册表
tool_registry: Dict[str, Callable] = {}

def register_tool(name: str, description: str, schema: Type[BaseModel]):
    """
    装饰器：将函数注册为工具，并赋予属性。
    用法：
      @register_tool("my_tool", "描述", MySchemaModel)
      def my_tool(**kwargs): ...
    """
    def decorator(func: Callable) -> Callable:
        func.name = name
        func.description = description
        func.schema = schema
        tool_registry[name] = func
        return func
    return decorator

# ---------- 工具调用支持 ----------

class ToolException(Exception):
    """工具调用过程中的统一异常类型。"""
    pass

@dataclass
class ToolMessage:
    """
    工具调用返回的结构化消息。
    """
    tool: str
    success: bool
    result: Any = None
    error: str = None

def invoke_tool(name: str, args: dict) -> ToolMessage:
    """
    通过工具名称和参数调用已注册的工具并返回 ToolMessage。
    :param name: 在 tool_registry 中注册的工具名
    :param args: 原始参数字典
    :return: ToolMessage 实例，包含成功标志、结果或错误信息
    :raises ToolException: 若工具不存在或参数验证失败
    """
    if name not in tool_registry:
        raise ToolException(f"Tool '{name}' is not registered.")
    tool = tool_registry[name]

    # 参数校验
    try:
        validated = tool.schema(**args)
    except ValidationError as ve:
        raise ToolException(f"参数校验失败: {ve}")

    # 执行工具
    try:
        output = tool(**validated.dict())
        return ToolMessage(tool=name, success=True, result=output)
    except Exception as e:
        # 捕获执行过程中的任何异常
        return ToolMessage(tool=name, success=False, error=str(e))