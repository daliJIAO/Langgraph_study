# 并行工作流程示例 - 使用LangGraph构建并行执行的内容生成器
# 功能：根据给定主题，并行生成笑话、故事和诗歌，最后合并输出

import os
import getpass
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
from dotenv import load_dotenv
from langchain_qwq import ChatQwQ

# 加载环境变量配置
load_dotenv()

# 初始化通义千问LLM模型
llm = ChatQwQ(
    model="qwen3-8b",                                                    # 使用qwen3-8b模型
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",      # 阿里云API地址
    max_tokens=3_000,                                                   # 最大生成token数
    timeout=None,                                                       # 超时设置
    max_retries=2,                                                      # 最大重试次数
    # other params...
)

# 检查并设置API密钥
if not os.getenv("DASHSCOPE_API_KEY"):
    os.environ["DASHSCOPE_API_KEY"] = getpass.getpass("Enter your Dashscope API key: ")

# 图状态定义 - 定义工作流中传递的数据结构
class State(TypedDict):
    topic: str          # 输入主题
    joke: str           # 生成的笑话
    story: str          # 生成的故事
    poem: str           # 生成的诗歌
    combined_output: str # 合并后的最终输出

# 节点函数定义 - 每个节点负责特定的处理任务

def call_llm_1(state: State):
    """第一个LLM调用 - 生成笑话"""
    msg = llm.invoke(f"Write a joke about {state['topic']}")
    return {"joke": msg.content}

def call_llm_2(state: State):
    """第二个LLM调用 - 生成故事"""
    msg = llm.invoke(f"Write a story about {state['topic']}")
    return {"story": msg.content}

def call_llm_3(state: State):
    """第三个LLM调用 - 生成诗歌"""
    msg = llm.invoke(f"Write a poem about {state['topic']}")
    return {"poem": msg.content}

def aggregator(state: State):
    """聚合器 - 将笑话、故事和诗歌合并为单个输出"""
    combined = f"Here's a story, joke, and poem about {state['topic']}!\n\n"
    combined += f"STORY:\n{state['story']}\n\n"
    combined += f"JOKE:\n{state['joke']}\n\n"
    combined += f"POEM:\n{state['poem']}"
    return {"combined_output": combined}

# 构建工作流程图
parallel_builder = StateGraph(State)

# 添加节点到图中
parallel_builder.add_node("call_llm_1", call_llm_1)  # 笑话生成节点
parallel_builder.add_node("call_llm_2", call_llm_2)  # 故事生成节点
parallel_builder.add_node("call_llm_3", call_llm_3)  # 诗歌生成节点
parallel_builder.add_node("aggregator", aggregator)   # 聚合节点

# 添加边连接节点 - 定义执行流程
# 从START开始，三个LLM调用并行执行
parallel_builder.add_edge(START, "call_llm_1")
parallel_builder.add_edge(START, "call_llm_2")
parallel_builder.add_edge(START, "call_llm_3")

# 三个LLM调用的结果都汇聚到聚合器
parallel_builder.add_edge("call_llm_1", "aggregator")
parallel_builder.add_edge("call_llm_2", "aggregator")
parallel_builder.add_edge("call_llm_3", "aggregator")

# 聚合器处理完后结束
parallel_builder.add_edge("aggregator", END)

# 编译工作流
parallel_workflow = parallel_builder.compile()

def stream_graph_updates(user_input: str):
    """
    简单的LLM调用 - 直接与用户对话
    参数: user_input - 用户输入的消息
    """
    # 直接调用LLM进行对话
    response = llm.invoke(user_input)
    print("Assistant:", response.content)

def demo_parallel_workflow():
    """演示并行工作流程 - 生成笑话、故事和诗歌"""
    print("=== 并行工作流程演示 ===")
    print("正在显示工作流图形...")

    # 显示工作流图形
    try:
        display(Image(parallel_workflow.get_graph().draw_mermaid_png()))
    except Exception as e:
        print(f"无法显示图形: {e}")

    # 获取用户输入的主题
    topic = input("\n请输入一个主题来生成内容 (或直接按回车使用默认主题'cats'): ").strip()
    if not topic:
        topic = "cats"

    # 执行工作流
    print(f"\n正在生成关于'{topic}'的内容...")
    state = parallel_workflow.invoke({"topic": topic})
    print("\n生成结果:")
    print(state["combined_output"])

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
        stream_graph_updates(user_input)
        print()

def main():
    """主函数 - 提供多种功能选择"""
    print("🚀 LangGraph 并行工作流程和聊天系统")
    print("=" * 50)

    while True:
        print("\n请选择功能:")
        print("1. 演示并行工作流程 (生成笑话、故事、诗歌)")
        print("2. 交互式聊天")
        print("3. 退出")

        choice = input("\n请输入选择 (1-3): ").strip()

        if choice == "1":
            demo_parallel_workflow()
        elif choice == "2":
            interactive_chat()
        elif choice == "3":
            print("感谢使用! 再见!")
            break
        else:
            print("无效选择，请输入 1、2 或 3")

if __name__ == "__main__":
    main()
