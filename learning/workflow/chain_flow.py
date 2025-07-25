# 导入必要的库
import getpass  # 用于安全地获取用户输入的密码
import os  # 用于操作系统环境变量

from dotenv import load_dotenv  # 从.env文件加载环境变量
from langchain_qwq import ChatQwQ  # 通义千问的聊天模型
from typing_extensions import TypedDict  # 类型注解支持
from langgraph.graph import StateGraph, START, END  # LangGraph图构建组件
from IPython.display import Image, display  # Jupyter显示功能

# 加载环境变量配置
load_dotenv()

# 初始化通义千问聊天模型
llm = ChatQwQ(
    model="qwen3-4b",  # 使用qwen3-4b模型
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # API接口地址
    max_tokens=3_000,  # 最大生成token数量
    timeout=None,  # 请求超时时间
    max_retries=2,  # 最大重试次数
    # other params...
)


try:
    # 测试API连接
    print("Testing API connection...")
    test_msg = llm.invoke("Hello")
    print("API connection successful")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()



# 检查并设置API密钥
if not os.getenv("DASHSCOPE_API_KEY"):
    os.environ["DASHSCOPE_API_KEY"] = getpass.getpass("Enter your Dashscope API key: ")

# 定义状态类型，包含整个工作流程中需要传递的数据
class State(TypedDict):
    topic: str          # 笑话主题
    joke: str           # 初始生成的笑话
    improved_joke: str  # 改进后的笑话
    final_joke: str     # 最终润色的笑话

# 生成笑话的节点函数
def generate_joke(state: State):
    """根据给定主题生成一个简短的笑话"""
    msg = llm.invoke(f"Write a short joke about {state['topic']}")
    return {"joke": msg.content}

# 检查笑话质量的条件函数
def check_punchline(state: State):
    """检查笑话是否包含问号或感叹号，判断是否需要改进"""
    return "Fail" if "?" in state["joke"] or "!" in state["joke"] else "Pass"

# 改进笑话的节点函数
def improve_joke(state: State):
    """通过添加文字游戏来让笑话更有趣"""
    msg = llm.invoke(f"Make this joke funnier by adding wordplay: {state['joke']}")
    return {"improved_joke": msg.content}

# 润色笑话的节点函数
def polish_joke(state: State):
    """为改进后的笑话添加意外的转折"""
    msg = llm.invoke(f"Add a surprising twist to this joke: {state['improved_joke']}")
    return {"final_joke": msg.content}

# 构建工作流程图
workflow = StateGraph(State)

# 添加节点到工作流程
workflow.add_node("generate_joke", generate_joke)  # 生成笑话节点
workflow.add_node("improve_joke", improve_joke)    # 改进笑话节点
workflow.add_node("polish_joke", polish_joke)      # 润色笑话节点

# 添加边连接，定义执行流程
workflow.add_edge(START, "generate_joke")  # 从开始节点到生成笑话节点

# 添加条件边：根据笑话质量检查结果决定下一步
# 如果检查失败，进入改进笑话节点；如果通过，直接结束
workflow.add_conditional_edges("generate_joke", check_punchline, {"Fail": "improve_joke", "Pass": END})
workflow.add_edge("improve_joke", "polish_joke")  # 从改进笑话节点到润色笑话节点
workflow.add_edge("polish_joke", END)  # 从润色笑话节点到结束节点

# 编译并可视化工作流程
chain = workflow.compile()
# display(Image(chain.get_graph().draw_mermaid_png()))

def stream_graph_updates(user_input: str):
    """流式处理用户输入的笑话主题"""
    # 构造初始state，包含用户输入的主题
    input_state = {"topic": user_input}
    # chain.stream会逐步返回节点输出
    for event in chain.stream(input=input_state):
        print("Processing event:", event)
        for value in event.values():
            # 打印每个节点的输出结果
            if "joke" in value and "improved_joke" not in value and "final_joke" not in value:
                print("Initial joke:", value["joke"])
            elif "improved_joke" in value:
                print("Improved joke:", value["improved_joke"])
            elif "final_joke" in value:
                print("Final joke:", value["final_joke"])

# 交互式循环处理用户输入
while True:
    user_input = input("Enter a topic for joke generation (or 'quit'/'exit'/'q' to stop): ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    stream_graph_updates(user_input)
