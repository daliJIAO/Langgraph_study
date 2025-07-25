# 评估优化工作流程示例 - 使用LangGraph构建自我优化的内容生成系统
# 功能：生成笑话 → 评估质量 → 根据反馈改进 → 循环优化直到满意

import os
import getpass
from typing_extensions import Literal, TypedDict
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
from dotenv import load_dotenv
from langchain_qwq import ChatQwQ
from pydantic import BaseModel, Field

# 加载环境变量配置
load_dotenv()

# 初始化通义千问LLM模型
llm = ChatQwQ(
    model="qwen3-4b",                                                    # 使用qwen3-4b模型
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",      # 阿里云API地址
    max_tokens=3_000,                                                   # 最大生成token数
    timeout=None,                                                       # 超时设置
    max_retries=2,                                                      # 最大重试次数
    # other params...
)

# 检查并设置API密钥
if not os.getenv("DASHSCOPE_API_KEY"):
    os.environ["DASHSCOPE_API_KEY"] = getpass.getpass("Enter your Dashscope API key: ")

# 状态定义 - 在工作流中传递的数据结构
class State(TypedDict):
    joke: str           # 生成的笑话内容
    topic: str          # 笑话主题
    feedback: str       # 评估反馈信息
    funny_or_not: str   # 评估结果（funny/not funny）

# 评估反馈的结构化输出模式定义
class Feedback(BaseModel):
    grade: Literal["funny", "not funny"] = Field(
        description="判断笑话是否有趣"
    )
    feedback: str = Field(
        description="如果笑话不够有趣，提供改进建议"
    )

# 增强LLM以支持结构化输出，用于笑话评估
evaluator = llm.with_structured_output(Feedback)

# 节点函数定义 - 每个节点负责特定的处理任务

def llm_call_generator(state: State):
    """笑话生成器节点 - 生成或改进笑话"""
    if state.get("feedback"):
        # 如果有反馈，根据反馈改进笑话
        msg = llm.invoke(
            f"写一个关于{state['topic']}的笑话，但要考虑以下反馈意见：{state['feedback']}"
        )
    else:
        # 首次生成笑话
        msg = llm.invoke(f"写一个关于{state['topic']}的笑话")

    return {"joke": msg.content}

def llm_call_evaluator(state: State):
    """笑话评估器节点 - 评估笑话质量并提供反馈"""
    # 使用结构化输出评估笑话
    grade = evaluator.invoke(f"评估这个笑话：{state['joke']}")

    return {
        "funny_or_not": grade.grade,
        "feedback": grade.feedback
    }

# 条件边函数 - 根据评估结果决定下一步操作
def route_joke(state: State):
    """
    路由决策函数 - 根据评估结果决定是结束还是继续改进
    返回值决定工作流的下一步走向
    """
    if state["funny_or_not"] == "funny":
        return "Accepted"                    # 笑话被接受，结束流程
    elif state["funny_or_not"] == "not funny":
        return "Rejected + Feedback"        # 笑话被拒绝，返回生成器改进
    return None


# 构建优化工作流程图
optimizer_builder = StateGraph(State)

# 添加节点到图中
optimizer_builder.add_node("llm_call_generator", llm_call_generator)    # 笑话生成器节点
optimizer_builder.add_node("llm_call_evaluator", llm_call_evaluator)    # 笑话评估器节点

# 添加边连接节点 - 定义执行流程
optimizer_builder.add_edge(START, "llm_call_generator")                 # 从START开始到生成器
optimizer_builder.add_edge("llm_call_generator", "llm_call_evaluator")  # 生成器到评估器

# 添加条件边 - 根据评估结果决定下一步
optimizer_builder.add_conditional_edges(
    "llm_call_evaluator",    # 来源节点
    route_joke,              # 条件函数
    {  # 路由映射：route_joke返回的名称 : 要访问的下一个节点名称
        "Accepted": END,                        # 接受 → 结束
        "Rejected + Feedback": "llm_call_generator",  # 拒绝 → 重新生成
    },
)

# 编译工作流
optimizer_workflow = optimizer_builder.compile()

def simple_llm_chat(user_input: str):
    """
    简单的LLM对话 - 直接与用户对话
    参数: user_input - 用户输入的消息
    """
    response = llm.invoke(user_input)
    print("Assistant:", response.content)

def demo_optimizer_workflow():
    """演示评估优化工作流程 - 自动优化笑话质量"""
    print("=== 评估优化工作流程演示 ===")
    print("正在显示工作流图形...")

    # 显示工作流图形
    try:
        display(Image(optimizer_workflow.get_graph().draw_mermaid_png()))
    except Exception as e:
        print(f"无法显示图形: {e}")

    # 获取用户输入的笑话主题
    topic = input("\n请输入笑话主题 (或直接按回车使用默认主题'Cats'): ").strip()
    if not topic:
        topic = "Cats"
        print(f"使用默认主题: {topic}")

    # 执行优化工作流
    print(f"\n正在生成关于'{topic}'的笑话并进行质量优化...")
    print("流程: 生成笑话 → 评估质量 → 如需要则根据反馈改进 → 重复直到满意")

    state = optimizer_workflow.invoke({"topic": topic})

    print(f"\n优化完成!")
    print(f"最终评估结果: {state.get('funny_or_not', '未知')}")
    print("\n" + "="*50)
    print("最终笑话:")
    print("="*50)
    print(state["joke"])

    # 显示反馈信息（如果有）
    if state.get("feedback"):
        print(f"\n最后一次反馈: {state['feedback']}")

def interactive_chat():
    """交互式聊天功能"""
    print("\n=== 交互式聊天模式 ===")
    print("输入 'quit', 'exit' 或 'q' 退出聊天")
    print("-" * 50)

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye! 再见!")
            break
        simple_llm_chat(user_input)
        print()

def interactive_joke_optimizer():
    """交互式笑话优化器"""
    print("\n=== 交互式笑话优化器 ===")
    print("系统会自动生成笑话并进行质量评估和优化")
    print("输入 'quit', 'exit' 或 'q' 退出")
    print("-" * 50)

    while True:
        topic = input("请输入笑话主题: ").strip()
        if topic.lower() in ["quit", "exit", "q"]:
            print("Goodbye! 再见!")
            break

        if not topic:
            print("请输入有效的笑话主题")
            continue

        print(f"\n正在为'{topic}'生成并优化笑话...")
        try:
            state = optimizer_workflow.invoke({"topic": topic})
            print(f"\n优化完成! 评估结果: {state.get('funny_or_not', '未知')}")
            print("\n" + "="*50)
            print("优化后的笑话:")
            print("="*50)
            print(state["joke"])

            if state.get("feedback"):
                print(f"\n改进过程中的反馈: {state['feedback']}")
            print()
        except Exception as e:
            print(f"生成笑话时出现错误: {e}")
        print()

def main():
    """主函数 - 提供多种功能选择"""
    print("🚀 LangGraph 评估优化工作流程和笑话生成系统")
    print("=" * 65)

    while True:
        print("\n请选择功能:")
        print("1. 演示评估优化工作流程 (单次笑话优化)")
        print("2. 交互式笑话优化器 (连续优化笑话)")
        print("3. 简单聊天模式")
        print("4. 退出")

        choice = input("\n请输入选择 (1-4): ").strip()

        if choice == "1":
            demo_optimizer_workflow()
        elif choice == "2":
            interactive_joke_optimizer()
        elif choice == "3":
            interactive_chat()
        elif choice == "4":
            print("感谢使用! 再见!")
            break
        else:
            print("无效选择，请输入 1-4 之间的数字")

if __name__ == "__main__":
    main()
