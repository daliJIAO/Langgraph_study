"""
多代理循环代码改进系统

这个模块实现了一个基于LangGraph的多代理循环系统，用于代码的自动化迭代改进：
1. 代码编写代理（Code Writer）- 根据反馈生成或改进代码
2. 代码测试代理（Code Tester）- 执行测试用例并提供反馈
3. 循环控制机制 - 根据测试结果决定是否继续迭代

系统会自动循环运行，直到代码通过所有测试或达到最大迭代次数。
"""

from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
import textwrap


# 工作流状态跟踪类
class EvaluationState(Dict[str, Any]):
    """
    评估状态类

    继承自Dict，用于跟踪整个代码改进工作流的状态信息
    包含代码内容、测试反馈、通过状态、迭代次数等关键信息
    """
    code: str = ""                    # 当前代码内容
    feedback: str = ""                # 测试反馈信息
    passed: bool = False              # 是否通过所有测试
    iteration: int = 0                # 当前迭代次数
    max_iterations: int = 3           # 最大迭代次数限制
    history: List[Dict] = []          # 历史记录列表

    def __init__(self, *args, **kwargs):
        """
        初始化评估状态

        设置所有字段的默认值，确保状态对象的完整性
        """
        super().__init__(*args, **kwargs)

        # 为所有字段设置默认值
        self.setdefault("code", "")
        self.setdefault("feedback", "")
        self.setdefault("passed", False)
        self.setdefault("iteration", 0)
        self.setdefault("max_iterations", 3)
        self.setdefault("history", [])


# 代理1：代码编写者
def code_writer_agent(state: EvaluationState, config: RunnableConfig) -> Dict[str, Any]:
    """
    代码编写代理

    根据测试反馈智能生成或改进代码：
    - 第一次迭代：生成基础版本（带有已知缺陷）
    - 后续迭代：根据具体测试失败信息进行针对性修复

    Args:
        state (EvaluationState): 当前工作流状态
        config (RunnableConfig): 运行时配置

    Returns:
        Dict[str, Any]: 包含新代码、反馈和迭代次数的字典
    """
    print(f"第 {state['iteration'] + 1} 轮迭代 - 代码编写代理：开始生成代码")
    print(f"第 {state['iteration'] + 1} 轮迭代 - 代码编写代理：收到反馈: {state['feedback']}")

    iteration = state["iteration"] + 1
    feedback = state["feedback"]

    if iteration == 1:
        # 初始尝试：基础阶乘函数（存在缺陷：未处理0和负数情况）
        code = textwrap.dedent("""
        def factorial(n):
            result = 1
            for i in range(1, n + 1):
                result *= i
            return result
        """)
        writer_feedback = "生成初始代码。"

    elif "factorial(0)" in feedback.lower():
        # 修复零值情况
        code = textwrap.dedent("""
        def factorial(n):
            if n == 0:
                return 1
            result = 1
            for i in range(1, n + 1):
                result *= i
            return result
        """)
        writer_feedback = "修复了 n=0 的处理。"

    elif "factorial(-1)" in feedback.lower() or "negative" in feedback.lower():
        # 修复负数输入问题
        code = textwrap.dedent("""
        def factorial(n):
            if n < 0:
                raise ValueError("Factorial not defined for negative numbers")
            if n == 0:
                return 1
            result = 1
            for i in range(1, n + 1):
                result *= i
            return result
        """)
        writer_feedback = "添加了负数输入的错误处理。"

    else:
        # 没有���一步的改进需求
        code = state["code"]
        writer_feedback = "未识别到进一步的改进需求。"

    print(f"第 {iteration} 轮迭代 - 代码编写代理：代码生成完成")

    return {
        "code": code,
        "feedback": writer_feedback,
        "iteration": iteration
    }


# 代理2：代码测试者
def code_tester_agent(state: EvaluationState, config: RunnableConfig) -> Dict[str, Any]:
    """
    代码测试代理

    执行全面的测试用例来验证代码的正确性：
    - 测试正常情况：factorial(0), factorial(1), factorial(5)
    - 测试异常情况：factorial(-1) 应抛出ValueError
    - 收集所有测试失败信息并提供详细反馈

    Args:
        state (EvaluationState): 当前工作流状态
        config (RunnableConfig): 运行时配置

    Returns:
        Dict[str, Any]: 包含测试结果、反馈和历史记录的字典
    """
    print(f"第 {state['iteration']} 轮迭代 - 代码测试代理：开始测试代码")

    code = state["code"]

    try:
        # 定义全面的测试用例
        test_cases = [
            (0, 1),      # factorial(0) = 1 (数学定义)
            (1, 1),      # factorial(1) = 1
            (5, 120),    # factorial(5) = 120
            (-1, None),  # 应该抛出 ValueError
        ]

        # 在安全的命名空间中执行代码
        namespace = {}
        exec(code, namespace)
        factorial = namespace.get('factorial')

        # 检查函数是否存在且可调用
        if not callable(factorial):
            return {"passed": False, "feedback": "未找到 factorial 函数。"}

        feedback_parts = []
        passed = True

        # 运行所有测试用例并收集失败信息
        for input_val, expected in test_cases:
            try:
                result = factorial(input_val)

                if expected is None:  # 期望抛出异常
                    passed = False
                    feedback_parts.append(f"测试失败: factorial({input_val}) 应该抛出错误。")
                elif result != expected:
                    passed = False
                    feedback_parts.append(
                        f"测试失败: factorial({input_val}) 返回 {result}，期望 {expected}。")

            except ValueError as ve:
                if expected is not None:
                    passed = False
                    feedback_parts.append(
                        f"测试失败: factorial({input_val}) 意外抛出 ValueError: {str(ve)}")
                # 如果期望抛出ValueError，则测试通过

            except Exception as e:
                passed = False
                feedback_parts.append(f"测试失败: factorial({input_val}) 导致错误: {str(e)}")

        # 生成测试反馈
        feedback = "所有测试通过！" if passed else "\n".join(feedback_parts)

        print(f"第 {state['iteration']} 轮迭代 - 代码测试代理：测试完成 - {'通过' if passed else '失败'}")

        # 记录本次尝试到历史中
        history = state["history"]
        history.append({
            "iteration": state["iteration"],
            "code": code,
            "feedback": feedback,
            "passed": passed
        })

        return {
            "passed": passed,
            "feedback": feedback,
            "history": history
        }

    except Exception as e:
        print(f"第 {state['iteration']} 轮迭代 - 代码测试代理：测试失败")
        return {"passed": False, "feedback": f"测试过程中出错: {str(e)}"}


# 条件边函数：决定是否继续循环
def should_continue(state: EvaluationState) -> str:
    """
    循环控制决策函数

    根据当前状态决定工作流的下一步：
    - 如果测试通过或达到最大迭代次数：结束循环
    - 否则：继续下一轮迭代

    Args:
        state (EvaluationState): 当前工作流状态

    Returns:
        str: 下一个节点名称 ("end" 或 "code_writer")
    """
    if state["passed"] or state["iteration"] >= state["max_iterations"]:
        if state["passed"]:
            print(f"第 {state['iteration']} 轮迭代 - 循环结束：测试通过")
        else:
            print(f"第 {state['iteration']} 轮迭代 - 循环结束：达到最大迭代次数")
        return "end"

    print(f"第 {state['iteration']} 轮迭代 - 循环继续：测试失败")
    return "code_writer"


# 构建LangGraph工作流
print("🏗️ 构建多代理循环改进工作流...")

# 创建状态图
workflow = StateGraph(EvaluationState)

# 添加代理节点
workflow.add_node("code_writer", code_writer_agent)
workflow.add_node("code_tester", code_tester_agent)

print("➕ 已添加代理节点：代码编写者、代码测试者")

# 添加边连接
workflow.set_entry_point("code_writer")  # 从代码编写者开始
workflow.add_edge("code_writer", "code_tester")  # 编写完成后进行测试

# 添加条件边：根据测试结果决定下一步
workflow.add_conditional_edges(
    "code_tester",
    should_continue,
    {
        "code_writer": "code_writer",  # 测试失败，继续改进
        "end": END                     # 测试通过或达到最大次数，结束
    }
)

print("🔗 已添加工作流边连接")

# 编译图
app = workflow.compile()
print("✅ 工作流编译完成")


# 主执行函数
def main():
    """
    主执行函数

    启动代码改进循环工作流，并显示最终结果和迭代历史
    """
    print("🚀 开始多代理循环代码改进演示")
    print("=" * 60)

    # 创建初始状态
    initial_state = EvaluationState()

    # 执行工作流
    result = app.invoke(initial_state)

    # 显示最终结果
    print("\n📋 === 评估结果 ===")
    final_status = "通过" if result['passed'] else "失败"
    print(f"🎯 最终状态: {final_status}，共进行 {result['iteration']} 轮迭代")

    print(f"\n📝 最终代码:")
    print(result['code'])

    print(f"💬 最终反馈:")
    print(result['feedback'])

    print(f"\n📚 迭代历史:")
    for record in result.get('history', []):
        status = "✅ 通过" if record['passed'] else "❌ 失败"
        print(f"  第 {record['iteration']} 轮: {status}")
        print(f"    反馈: {record['feedback']}")

    print("\n✨ 代码改进流程完成!")


# 程序入口点
if __name__ == "__main__":
    main()