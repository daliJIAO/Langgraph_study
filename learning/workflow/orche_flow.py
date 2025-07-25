# 编排工作流程示例 - 使用LangGraph构建多工作者协作的报告生成系统
# 功能：自动生成报告计划，并行编写各个章节，最后合成完整报告

import os
import getpass
import operator
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
from IPython.display import Image, display, Markdown
from dotenv import load_dotenv
from langchain_qwq import ChatQwQ
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

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

# 报告章节结构定义
class Section(BaseModel):
    name: str = Field(description="章节名称")
    description: str = Field(description="章节描述和要求")

# 报告计划结构定义
class Plan(BaseModel):
    sections: list[Section] = Field(description="报告章节列表")

# 增强LLM以支持结构化输出，用于报告规划
planner = llm.with_structured_output(Plan)

# 主状态定义 - 在整个工作流中传递的数据结构
class State(TypedDict):
    topic: str                                      # 报告主题
    sections: list[Section]                         # 报告章节列表
    completed_sections: Annotated[list, operator.add]  # 所有工作者并行写入的完成章节
    final_report: str                               # 最终报告

# 工作者状态定义 - 每个工作者处理单个章节时使用的状态
class WorkerState(TypedDict):
    section: Section                                # 要处理的章节
    completed_sections: Annotated[list, operator.add]  # 完成的章节列表

# 节点函数定义 - 每个节点负责特定的处理任务

def orchestrator(state: State):
    """编排者节点 - 生成报告计划并分解为多个章节"""
    # 使用规划器生成报告章节
    report_sections = planner.invoke(
        [
            SystemMessage(content="为报告生成一个详细的计划，包括各个章节的名称和描述。"),
            HumanMessage(content=f"报告主题是: {state['topic']}"),
        ]
    )
    return {"sections": report_sections.sections}

def llm_call(state: WorkerState):
    """工作者节点 - 编写报告的单个章节"""
    # 生成章节内容
    section = llm.invoke(
        [
            SystemMessage(
                content="根据提供的章节名称和描述编写报告章节。不要包含章节前言。使用markdown格式。"
            ),
            HumanMessage(
                content=f"章节名称: {state['section'].name}\n章节描述: {state['section'].description}"
            ),
        ]
    )
    # 将完成的章节写入到完成章节列表中
    return {"completed_sections": [section.content]}

def synthesizer(state: State):
    """合成器节点 - 将所有完成的章节合成为完整报告"""
    # 获取完成的章节列表
    completed_sections = state["completed_sections"]

    # 将完成的章节格式化为字符串，用作最终报告的上下文
    completed_report_sections = "\n\n---\n\n".join(completed_sections)

    return {"final_report": completed_report_sections}

# 条件边函数 - 为每个章节创建工作者
def assign_workers(state: State):
    """为计划中的每个章节分配一个工作者"""
    # 通过Send() API并行启动章节编写任务
    return [Send("llm_call", {"section": s}) for s in state["sections"]]

# 构建编排工作流程图
orchestrator_worker_builder = StateGraph(State)

# 添加节点到图中
orchestrator_worker_builder.add_node("orchestrator", orchestrator)    # 编排者节点
orchestrator_worker_builder.add_node("llm_call", llm_call)            # 工作者节点（可并行）
orchestrator_worker_builder.add_node("synthesizer", synthesizer)      # 合成器节点

# 添加边连接节点 - 定义执行流程
orchestrator_worker_builder.add_edge(START, "orchestrator")           # 从START开始到编排者

# 添加条件边 - 根据章节数量动态分配工作者
orchestrator_worker_builder.add_conditional_edges(
    "orchestrator",      # 来源节点
    assign_workers,      # 分配函数
    ["llm_call"]         # 目标节点列表
)

orchestrator_worker_builder.add_edge("llm_call", "synthesizer")       # 所有工作者完成后到合成器
orchestrator_worker_builder.add_edge("synthesizer", END)              # 合成器完成后结束

# 编译工作流
orchestrator_worker = orchestrator_worker_builder.compile()

def simple_llm_chat(user_input: str):
    """
    简单的LLM对话 - 直接与用户对话
    参数: user_input - 用户输入的消息
    """
    response = llm.invoke(user_input)
    print("Assistant:", response.content)

def demo_orchestrator_workflow():
    """演示编排工作流程 - 自动生成结构化报告"""
    print("=== 编排工作流程演示 ===")
    print("正在显示工作流图形...")

    # 显示工作流图形
    try:
        display(Image(orchestrator_worker.get_graph().draw_mermaid_png()))
    except Exception as e:
        print(f"无法显示图形: {e}")

    # 获取用户输入的报告主题
    topic = input("\n请输入报告主题 (或直接按回车使用默认主题): ").strip()
    if not topic:
        topic = "Create a report on LLM scaling laws"
        print(f"使用默认主题: {topic}")

    # 执行编排工作流
    print(f"\n正在生成关于'{topic}'的报告...")
    print("步骤1: 编排者正在制定报告计划...")
    print("步骤2: 多个工作者并行编写章节...")
    print("步骤3: 合成器正在整合最终报告...")

    state = orchestrator_worker.invoke({"topic": topic})

    print(f"\n报告生成完成! 共生成 {len(state.get('completed_sections', []))} 个章节")
    print("\n" + "="*60)
    print("最终报告:")
    print("="*60)

    # 在控制台显示报告内容
    print(state["final_report"])

    # 如果在Jupyter环境中，也可以显示Markdown格式
    try:
        display(Markdown(state["final_report"]))
    except:
        pass  # 非Jupyter环境中忽略

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

def interactive_report_generator():
    """交互式报告生成器"""
    print("\n=== 交互式报告生成器 ===")
    print("输入 'quit', 'exit' 或 'q' 退出")
    print("-" * 50)

    while True:
        topic = input("请输入报告主题: ").strip()
        if topic.lower() in ["quit", "exit", "q"]:
            print("Goodbye! 再见!")
            break

        if not topic:
            print("请输入有效的报告主题")
            continue

        print(f"\n正在生成关于'{topic}'的报告...")
        try:
            state = orchestrator_worker.invoke({"topic": topic})
            print(f"\n报告生成完成! 共生成 {len(state.get('completed_sections', []))} 个章节")
            print("\n" + "="*60)
            print("最终报告:")
            print("="*60)
            print(state["final_report"])
            print("\n" + "="*60)
        except Exception as e:
            print(f"生成报告时出现错误: {e}")
        print()

def main():
    """主函数 - 提供多种功能选择"""
    print("🚀 LangGraph 编排工作流程和报告生成系统")
    print("=" * 60)

    while True:
        print("\n请选择功能:")
        print("1. 演示编排工作流程 (单次报告生成)")
        print("2. 交互式报告生成器 (连续生成报告)")
        print("3. 简单聊天模式")
        print("4. 退出")

        choice = input("\n请输入选择 (1-4): ").strip()

        if choice == "1":
            demo_orchestrator_workflow()
        elif choice == "2":
            interactive_report_generator()
        elif choice == "3":
            interactive_chat()
        elif choice == "4":
            print("感谢使用! 再见!")
            break
        else:
            print("无效选择，请输入 1-4 之间的数字")

if __name__ == "__main__":
    main()
