# demo_work/test_compute_graph.py

import sys
import os
from pathlib import Path
from dotenv import load_dotenv
# 将项目根目录添加到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from demo_work.agents.plus_agent import PlusAgent
from demo_work.agents.subtract_agent import SubtractAgent
from demo_work.graphs.compute_graph import build_compute_graph

# 加载.env文件
load_dotenv(Path(__file__).parent / '.env')

# 从环境变量中获取API key
api_key = os.getenv('DASHSCOPE_API_KEY')
if not api_key:
    raise ValueError('DASHSCOPE_API_KEY not found in .env file')

# 实例化 Agent
plus_agent = PlusAgent(api_key=api_key)
subtract_agent = SubtractAgent(api_key=api_key)

# 构建计算图
graph = build_compute_graph(plus_agent, subtract_agent)


def stream_graph_updates(expr: str):
    """
    对于给定的表达式 expr，逐步打印 graph 的每次 super-step 更新结果。
    每个 event 是一个 dict，key 为节点名，value 为该节点执行后返回的状态片段。
    """
    print(f"\n{'='*80}")
    print(f"🧮 开始计算表达式: {expr}")
    print(f"🏗️  计算图流程追踪")
    print(f"{'='*80}")

    step_counter = 0
    node_execution_order = []

    # 使用流式处理来显示每个节点的执行过程
    events = []
    for event in graph.stream({"expr": expr}):
        events.append(event)
        for node_name, node_output in event.items():
            if node_name != "__end__":
                step_counter += 1
                node_execution_order.append(node_name)

                print(f"\n🔄 流程步骤 {step_counter}")
                print(f"   当前执行节点: {node_name.upper()}")

                # 显示节点输出的关键信息
                if "route" in node_output:
                    print(f"   节点决策: 下一步 → {node_output['route']}")
                if "operation_log" in node_output:
                    print(f"   操作类型: {node_output['operation_log']}")
                if "expr" in node_output:
                    print(f"   更新后表达式: {node_output['expr']}")
                if "result" in node_output and node_output.get("route") == "end":
                    print(f"   🎯 最终结果: {node_output['result']}")

    print(f"\n{'='*80}")
    print(f"📊 执行摘要")
    print(f"{'='*80}")
    print(f"🔢 总执行步骤: {step_counter}")
    print(f"🗺️  节点执行顺序: {' → '.join(node_execution_order)}")
    print(f"✨ 计算完成！")
    print(f"{'='*80}")

    return events


def calculate_with_steps(expr: str):
    """
    计算表达式并展示详细步骤
    """
    print(f"\n🚀 启动智能计算器...")
    print(f"📝 输入表达式: {expr}")

    # 显示计算图结构信息
    print(f"\n🏗️  计算图架构:")
    print(f"   - ROUTER: 表达式解析和路由分发")
    print(f"   - PLUS: 普通加法运算 (调用 PLUS_AGENT)")
    print(f"   - SUBTRACT: 普通减法运算 (调用 SUBTRACT_AGENT)")
    print(f"   - PLUS_BRACKET: 括号内加法运算 (调用 PLUS_AGENT)")
    print(f"   - SUBTRACT_BRACKET: 括号内减法运算 (调用 SUBTRACT_AGENT)")

    # 执行计算并显示步骤
    events = stream_graph_updates(expr)

    # 获取最终结果
    final_result = graph.invoke({"expr": expr})

    print(f"\n📈 计���统计:")
    print(f"   原始表达式: {expr}")
    print(f"   最终结果: {final_result.get('result', '未知')}")
    print(f"   执行步骤数: {final_result.get('step', '未知')}")
    print(f"   图执行轮次: {len(events)}")

    return final_result


def interactive_calculator():
    """
    交互式计算器
    """
    print("\n🎉 欢迎使用智能表达式计算器！")
    print("💡 支持括号运算，例如: (3+5)-2+1")
    print("💡 输入 'q' 退出")

    while True:
        try:
            expr = input("\n请输入数学表达式: ").strip()

            if expr.lower() == 'q':
                print("👋 再见！")
                break

            if not expr:
                print("❌ 请输入有效的表达式")
                continue

            calculate_with_steps(expr)

        except KeyboardInterrupt:
            print("\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")


if __name__ == "__main__":
    # 测试几个示例表达式
    test_expressions = [
        "((3+5)-2)+1",
        "((3+5-2)-2)+1"
    ]

    print("🧪 运行测试用例...")
    for expr in test_expressions:
        calculate_with_steps(expr)
        print("\n" + "-"*40)

    # 启动交互式计算器
    interactive_calculator()
