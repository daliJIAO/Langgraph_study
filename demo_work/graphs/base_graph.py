# graphs/base_graph.py
"""
LangGraph 的 StateGraph 的轻量级包装器，允许你
通过简洁的 Node / Edge 描述符构建图结构，
并在所有具体函数准备好后再进行编译。

Usage example
-------------
from typing_extensions import TypedDict
from graphs.base_graph import Node, Edge, BaseGraph
from langgraph.graph import START, END

# 1. 定义 State
class State(TypedDict):
    msg: str

# 2. 定义节点函数
def hello(state: State):
    return {"msg": "Hello"}

def world(state: State):
    return {"msg": state["msg"] + " World!"}

# 3. 创建节点、边
n1 = Node("hello", func=hello)
n2 = Node("world", func=world)
e1 = Edge(START, "hello")
e2 = Edge("hello", "world")
e3 = Edge("world", END)

# 4. 组装并编译
graph = BaseGraph(State, nodes=[n1, n2], edges=[e1, e2, e3]).compile()
print(graph.invoke({}))           # {'msg': 'Hello World!'}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence, Type, Optional, Any, Dict, List

from langgraph.graph import StateGraph, START, END  # type: ignore
from langgraph.constants import START as _START_CONST, END as _END_CONST  # 保证字符串常量一致

# ---------------------------------------------------------------------- #
# Node                                                                  #
# ---------------------------------------------------------------------- #
@dataclass
class Node:
    """
    图节点的包装器描述。
    你可以用 func=None 初始化，然后在编译图之前通过 ``set_func`` 方法设置具体的可调用对象。
    """
    name: str
    func: Optional[Callable[..., Mapping[str, Any]]] = None
    # extra metadata placeholder
    meta: Dict[str, Any] = field(default_factory=dict)

    def set_func(self, func: Callable[..., Mapping[str, Any]]) -> None:
        """Attach or replace the underlying runnable."""
        self.func = func

# ---------------------------------------------------------------------- #
# Edge                                                                  #
# ---------------------------------------------------------------------- #
@dataclass
class Edge:
    """
    Wrapper describing a graph edge.

    Parameters
    ----------
    source : str | list[str]
        起始节点（或节点列表 / START 常量）
    target : str | None
        目标节点（普通边必须提供；条件边可留空）
    route_func : Callable | None
        条件路由函数；若提供则使用 ``add_conditional_edges``，
        否则使用 ``add_edge``。
    mapping : dict | None
        可选的路由映射字典，传给 ``add_conditional_edges``。
    """
    source: str | Sequence[str]
    target: Optional[str] = None
    route_func: Optional[Callable[..., Any]] = None
    mapping: Optional[dict] = None


# ---------------------------------------------------------------------- #
# BaseGraph                                                             #
# ---------------------------------------------------------------------- #
class BaseGraph:
    """
    组装一个 LangGraph 的 StateGraph，使用 Node / Edge 对象，将*结构*定义（拓扑）与*功能*细节��节点可调用对象）分离。

    参数
    ----
    state_schema : Type
        描述图状态的 TypedDict 或 Pydantic 模型。
    nodes : Iterable[Node]
        所有节点。它们的 ``func`` 初始时可以为 None，编译时若有节点缺少可调用对象会报错。
    edges : Iterable[Edge]
        有向边（普通或条件边）。
    input_schema / output_schema : Type | None
        可选，传递给 StateGraph 的输入/输出 schema。
    """

    def __init__(
        self,
        state_schema: Type,
        *,
        nodes: Iterable[Node] | None = None,
        edges: Iterable[Edge] | None = None,
        input_schema: Optional[Type] = None,
        output_schema: Optional[Type] = None,
    ) -> None:
        self.state_schema = state_schema
        self.input_schema = input_schema
        self.output_schema = output_schema
        self._nodes: Dict[str, Node] = {n.name: n for n in (nodes or [])}
        self._edges: List[Edge] = list(edges or [])

    # ---------- public helpers ---------- #
    def add_node(self, node: Node) -> None:
        self._nodes[node.name] = node

    def add_edge(self, edge: Edge) -> None:
        self._edges.append(edge)

    def add_conditional_edges(self, source: str, route_func: Callable, mapping: dict) -> None:
        """添加条件边的便捷方法"""
        edge = Edge(source=source, route_func=route_func, mapping=mapping)
        self.add_edge(edge)

    # ---------- compilation ---------- #
    def compile(self, **compile_kwargs):
        """
         构建 StateGraph，添加节点和边，然后��译并返回。

          异常
           ----
           ValueError
             如果有节点的 ``func`` 仍为 None，则抛出该异常。
       """
        builder = StateGraph(
            self.state_schema,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
        )

        # 1) add nodes
        for node in self._nodes.values():
            if node.func is None:
                raise ValueError(f"Node '{node.name}' has no callable attached.")
            builder.add_node(node.name, node.func)

        # 2) add edges
        for edge in self._edges:
            if edge.route_func is not None:
                builder.add_conditional_edges(
                    edge.source, edge.route_func, edge.mapping or {}
                )
            else:
                if edge.target is None:
                    raise ValueError(
                        f"Normal edge from '{edge.source}' must specify 'target'."
                    )
                builder.add_edge(edge.source, edge.target)

        # 3) compile
        return builder.compile(**compile_kwargs)

    # ---------- convenience ---------- #
    def __repr__(self) -> str:  # pragma: no cover
        return f"<BaseGraph nodes={list(self._nodes)} edges={len(self._edges)}>"

    # expose START / END so用户代码可直接引用
    START = START
    END = END
