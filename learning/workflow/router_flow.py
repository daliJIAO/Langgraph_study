# 路由工作流程示例 - 使用LangGraph构建智能路由的内容生成器
# 功能：根据用户输入自动判断并生成相应类型的内容（故事、笑话、诗歌）

import os
import getpass
from typing_extensions import Literal, TypedDict
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
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

# 路由决策的结构化输出模式定义
class Route(BaseModel):
    step: Literal["poem", "story", "joke"] = Field(
        None, description="路由过程中的下一步：诗歌、故事或笑话"
    )

# 增强LLM以支持结构化输出，用于路由逻辑
router = llm.with_structured_output(Route)

# 状态定义 - 定义工作流中传递的数据结构
class State(TypedDict):
    input: str      # 用户输入
    decision: str   # 路由决策结果
    output: str     # 最终生成的内容

# 节点函数定义 - 每个节点负责生成特定类型的内容

def llm_call_1(state: State):
    """LLM调用1 - 生成故事"""
    result = llm.invoke(f"写一个关于以下内容的故事: {state['input']}")
    return {"output": result.content}

def llm_call_2(state: State):
    """LLM调用2 - 生成笑话"""
    result = llm.invoke(f"写一个关于以下内容的笑话: {state['input']}")
    return {"output": result.content}

def llm_call_3(state: State):
    """LLM调用3 - 生成诗歌"""
    result = llm.invoke(f"写一首关于以下内容的诗: {state['input']}")
    return {"output": result.content}

def llm_call_router(state: State):
    """路由器节点 - 根据输入内容决定生成哪种类型的内容"""
    # 使用增强的LLM进行结构化输出，作为路由逻辑
    decision = router.invoke(
        [
            SystemMessage(
                content="根据用户的请求，将输入路由到故事(story)、笑话(joke)或诗歌(poem)。"
            ),
            HumanMessage(content=state["input"]),
        ]
    )
    return {"decision": decision.step}

# 条件边函数 - 根据路由决策选择下一个节点
def route_decision(state: State):
    """
    路由决策函数 - 返回要访问的下一个节点名称
    根据decision字段的值决定调用哪个LLM节点
    """
    if state["decision"] == "story":
        return "llm_call_1"      # 生成故事
    elif state["decision"] == "joke":
        return "llm_call_2"      # 生成笑话
    elif state["decision"] == "poem":
        return "llm_call_3"      # 生成诗歌
    return None


# 构建路由工作流程图
router_builder = StateGraph(State)

# 添加节点到图中
router_builder.add_node("llm_call_1", llm_call_1)          # 故事生成节点
router_builder.add_node("llm_call_2", llm_call_2)          # 笑话生成节点
router_builder.add_node("llm_call_3", llm_call_3)          # 诗歌生成节点
router_builder.add_node("llm_call_router", llm_call_router) # 路由决策节点

# 添加边连接节点 - 定义执行流程
router_builder.add_edge(START, "llm_call_router")          # 从START开始到路由器

# 添加条件边 - 根据路由决策分发到不同的内容生成节点
router_builder.add_conditional_edges(
    "llm_call_router",    # 来源节点
    route_decision,       # 条件函数
    {  # 路由映射：route_decision返回的名称 : 要访问的下一个节点名称
        "llm_call_1": "llm_call_1",  # 故事
        "llm_call_2": "llm_call_2",  # 笑话
        "llm_call_3": "llm_call_3",  # 诗歌
    },
)

# 所有内容生成节点都连接到END
router_builder.add_edge("llm_call_1", END)
router_builder.add_edge("llm_call_2", END)
router_builder.add_edge("llm_call_3", END)

# 编译工作流
router_workflow = router_builder.compile()

def simple_llm_chat(user_input: str):
    """
    简单的LLM对话 - 直接与用户对话
    参数: user_input - 用户输入的消息
    """
    response = llm.invoke(user_input)
    print("Assistant:", response.content)

def demo_router_workflow():
    """演示路由工作流程 - 智能内容生成"""
    print("=== 智能路由工作流程演示 ===")
    print("正在显示工作流图形...")

    # 显示工作流图形
    try:
        display(Image(router_workflow.get_graph().draw_mermaid_png()))
    except Exception as e:
        print(f"无法显示图形: {e}")

    # 获取用户输入
    user_input = input("\n请输入您想要的内容 (系统会自动判断生成故事、笑话或诗歌): ").strip()
    if not user_input:
        user_input = "Write me a joke about cats"
        print(f"使用默认输入: {user_input}")

    # 执行路由工作流
    print(f"\n正在分析并生成内容...")
    state = router_workflow.invoke({"input": user_input})
    print(f"\n路由决策: {state.get('decision', '未知')}")
    print("\n生成结果:")
    print(state["output"])

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

def interactive_router():
    """交互式路由功能"""
    print("\n=== 交互式智能路由模式 ===")
    print("系统会自动判断您的输入并生成相应的故事、笑话或诗歌")
    print("输入 'quit', 'exit' 或 'q' 退出")
    print("-" * 50)

    while True:
        user_input = input("请描述您想要的内容: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye! 再见!")
            break

        print("\n正在分析并生成内容...")
        state = router_workflow.invoke({"input": user_input})
        print(f"路由决策: {state.get('decision', '未知')}")
        print("\n生成结果:")
        print(state["output"])
        print()

def main():
    """主函数 - 提供多种功能选择"""
    print("🚀 LangGraph 智能路由工作流程和聊天系统")
    print("=" * 60)

    while True:
        print("\n请选择功能:")
        print("1. 演示智能路由工作流程 (自动判断生成故事/笑话/诗歌)")
        print("2. 交互式智能路由 (连续使用路由功能)")
        print("3. 简单聊天模式")
        print("4. 退出")

        choice = input("\n请输入选择 (1-4): ").strip()

        if choice == "1":
            demo_router_workflow()
        elif choice == "2":
            interactive_router()
        elif choice == "3":
            interactive_chat()
        elif choice == "4":
            print("感谢使用! 再见!")
            break
        else:
            print("无效选择，请输入 1-4 之间的数字")

if __name__ == "__main__":
    main()
