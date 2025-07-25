# graphs/compute_graph.py
import sys
from pathlib import Path
import re
from typing_extensions import TypedDict
from demo_work.graphs.base_graph import Node, Edge, BaseGraph

# 将项目根目录添加到Python路径
sys.path.append(str(Path(__file__).parent.parent))


def parse_next_op(expr: str):
    """解析表达式中的下一个运算操作"""
    # 首先处理括号内的表达式
    bracket_match = re.search(r'\(([^()]+)\)', expr)
    if bracket_match:
        inner_expr = bracket_match.group(1)
        # 在括号内查找运算
        op_match = re.search(r'(-?\d+(?:\.\d+)?)\s*([+-])\s*(-?\d+(?:\.\d+)?)', inner_expr)
        if op_match:
            left, op, right = op_match.groups()
            # 返回括号内运算的信息和括号的位置
            return str(left), op, str(right), bracket_match.span(), "bracket"

    # 如果没有括号，匹配第一个加法或减法操作
    match = re.search(r'(-?\d+(?:\.\d+)?)\s*([+-])\s*(-?\d+(?:\.\d+)?)', expr)
    if match:
        left, op, right = match.groups()
        return str(left), op, str(right), match.span(), "normal"

    return None, None, None, None, None

def extract_number(text):
    """从文本中提取数字"""
    match = re.search(r'-?\d+(?:\.\d+)?', text)
    if match:
        return match.group()
    return text  # fallback

class CalcState(TypedDict, total=False):
    expr: str          # 当前表达式
    left: str          # 左操作数
    right: str         # 右操作数
    route: str         # 路由标识（plus/subtract/bracket/end）
    span: tuple        # 操作在表达式中的位置
    result: str        # 最终结果
    op_type: str       # 操作类型（bracket/normal）
    step: int          # 当前步骤编号
    operation_log: str # 当前操作的详细描述

# 构建计算图
def build_compute_graph(plus_agent, subtract_agent):

    def router_node(state: CalcState) -> CalcState:
        """路由节点：分析表达式并决定下一步操作"""
        expr = state["expr"]
        step = state.get("step", 0) + 1

        print(f"\n{'='*60}")
        print(f"🚀 节点执行: ROUTER (路由器)")
        print(f"📍 步骤 {step}: 分析表达式 '{expr}'")
        print(f"{'='*60}")

        left, op, right, span, op_type = parse_next_op(expr)

        # 如果无法再解析运算，检查是否为最终结果
        if op is None:
            if re.fullmatch(r'-?\d+(?:\.\d+)?', expr.strip()):
                print(f"✅ 路由器决策: 计算完成！最终结果: {expr.strip()}")
                print(f"➡️  下一步: 结束流程 (END)")
                return {"result": expr.strip(), "route": "end", "step": step}
            else:
                print(f"✅ 路由器决策: 计算完成！最终结果: {expr}")
                print(f"➡️  下一步: 结束流程 (END)")
                return {"result": expr, "route": "end", "step": step}

        # 根据操作符和类型设置路由
        if op == "+":
            route = "plus_bracket" if op_type == "bracket" else "plus"
            operation_desc = f"{'括号内' if op_type == 'bracket' else ''}加法运算: {left} + {right}"
        elif op == "-":
            route = "subtract_bracket" if op_type == "bracket" else "subtract"
            operation_desc = f"{'括号内' if op_type == 'bracket' else ''}减法运算: {left} - {right}"
        else:
            route = "end"
            operation_desc = "无法识别的运算"

        print(f"🔍 路由器分析: {operation_desc}")
        print(f"📍 路由器决策: 分发到 {route.upper()} 节点")
        print(f"➡️  下一步: 执行 {route} 节点")

        return {
            "route": route,
            "left": left,
            "right": right,
            "expr": expr,
            "span": span,
            "op_type": op_type,
            "step": step,
            "operation_log": operation_desc
        }

    def condition_router(state: CalcState) -> str:
        """条件路由函数：根据route字段决定下一个节点"""
        route = state.get("route", "end")
        return route

    def run_plus(state: CalcState) -> CalcState:
        """执行加法运算"""
        if not state.get("route", "").startswith("plus"):
            return {}

        left, right = state['left'], state['right']
        op_type = state.get('op_type', 'normal')
        route = state.get('route', 'plus')

        print(f"\n{'='*60}")
        print(f"🚀 节点执行: {route.upper()} ({'括号内' if op_type == 'bracket' else '普通'}加法节点)")
        print(f"🧮 准备计算: {left} + {right}")
        print(f"{'='*60}")

        print(f"📞 调用 Agent: PLUS_AGENT")
        print(f"📝 输入参数: \"{left}+{right}\"")

        try:
            agent_output = str(plus_agent.invoke([
                {"role": "user", "content": f"{left}+{right}"}
            ])[-1].content)
            result = extract_number(agent_output)

            print(f"🤖 Agent 原始输出: {agent_output}")
            print(f"💡 提取的计算结果: {result}")
            print(f"✅ Agent 调用成功!")

        except Exception as e:
            print(f"❌ Agent 调用失败: {e}")
            result = str(float(left) + float(right))  # fallback
            print(f"🔄 使用备用计算: {result}")

        updated_state = _update_expression(state, result)
        print(f"🔄 表达式更新: '{state['expr']}' → '{updated_state['expr']}'")
        print(f"➡️  下一步: 返回 ROUTER 节点继续分析")

        return updated_state

    def run_subtract(state: CalcState) -> CalcState:
        """执行减法运算"""
        if not state.get("route", "").startswith("subtract"):
            return {}

        left, right = state['left'], state['right']
        op_type = state.get('op_type', 'normal')
        route = state.get('route', 'subtract')

        print(f"\n{'='*60}")
        print(f"🚀 节点执行: {route.upper()} ({'括号内' if op_type == 'bracket' else '普通'}减法节点)")
        print(f"🧮 准备计算: {left} - {right}")
        print(f"{'='*60}")

        print(f"📞 调用 Agent: SUBTRACT_AGENT")
        print(f"📝 输入参数: \"{left}-{right}\"")

        try:
            agent_output = str(subtract_agent.invoke([
                {"role": "user", "content": f"{left}-{right}"}
            ])[-1].content)
            result = extract_number(agent_output)

            print(f"🤖 Agent 原始输出: {agent_output}")
            print(f"💡 提取的计算结果: {result}")
            print(f"✅ Agent 调用成功!")

        except Exception as e:
            print(f"❌ Agent 调用失败: {e}")
            result = str(float(left) - float(right))  # fallback
            print(f"🔄 使用备用计算: {result}")

        updated_state = _update_expression(state, result)
        print(f"🔄 表达式更新: '{state['expr']}' → '{updated_state['expr']}'")
        print(f"➡️  下一步: 返回 ROUTER 节点继续分析")

        return updated_state

    def _update_expression(state: CalcState, result: str) -> CalcState:
        """更新表达式：将计算结果替换到原表达式中"""
        expr = state["expr"]
        start, end = state["span"]

        if state.get("op_type") == "bracket":
            # 如果是括号运算，需要特殊处理
            # 找到括号的位置并替换整个括号内容
            bracket_match = re.search(r'\([^()]+\)', expr)
            if bracket_match:
                new_expr = expr[:bracket_match.start()] + str(result) + expr[bracket_match.end():]
            else:
                new_expr = expr[:start] + str(result) + expr[end:]
        else:
            # 普通运算直接替换
            new_expr = expr[:start] + str(result) + expr[end:]

        return {"expr": new_expr}

    # 创建节点
    router = Node("router", func=router_node)
    plus_node = Node("plus", func=run_plus)
    subtract_node = Node("subtract", func=run_subtract)
    plus_bracket_node = Node("plus_bracket", func=run_plus)
    subtract_bracket_node = Node("subtract_bracket", func=run_subtract)

    # 创建边连接
    edges = [
        # 从开始到路由器
        Edge(BaseGraph.START, "router"),

        # 从各个运算节点回到路由器
        Edge("plus", "router"),
        Edge("subtract", "router"),
        Edge("plus_bracket", "router"),
        Edge("subtract_bracket", "router"),
    ]

    # 创建计算图实例
    compute_graph = BaseGraph(
        CalcState,
        nodes=[router, plus_node, subtract_node, plus_bracket_node, subtract_bracket_node],
        edges=edges
    )

    # 添加条件边：从路由器根据条件分发到不同的运算节点
    compute_graph.add_conditional_edges(
        "router",
        condition_router,
        {
            "plus": "plus",
            "subtract": "subtract",
            "plus_bracket": "plus_bracket",
            "subtract_bracket": "subtract_bracket",
            "end": BaseGraph.END
        }
    )

    return compute_graph.compile()
