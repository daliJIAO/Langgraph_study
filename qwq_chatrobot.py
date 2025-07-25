import os
import getpass
from typing import Annotated
from langgraph.graph import START, END
from dotenv import load_dotenv
from langchain_qwq import ChatQwQ
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph
from IPython.display import Image, display


load_dotenv()

llm = ChatQwQ(
    model="qwen3-4b",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    max_tokens=3_000,
    timeout=None,
    max_retries=2,
    # other params...
)

if not os.getenv("DASHSCOPE_API_KEY"):
    os.environ["DASHSCOPE_API_KEY"] = getpass.getpass("Enter your Dashscope API key: ")

class State(TypedDict):
    # messages是一个列表，保存所有对话消息
    # Annotated的作用是指定它的“合并方式”，这里采用add_messages（追加而不是覆盖）
    messages: Annotated[list, add_messages]


def chatbot(state: State):
    # 取出历史消息，通过llm生成回复，返回新的消息（以列表形式包裹）
    return {"messages": [llm.invoke(state["messages"])]}


graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")  # 起点到chatbot节点
graph_builder.add_edge("chatbot", END)    # chatbot节点到终点
graph = graph_builder.compile()

def stream_graph_updates(user_input: str):
    # 构造初始state，messages里只有你这次的用户输入
    input = {"messages": [{"role": "user", "content": user_input}]}
    # graph.stream会逐步返回节点输出
    for event in graph.stream(input=input):
        for value in event.values():
            # 打印最后一条AI消息
            print("Assistant:", value["messages"][-1].content)

while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    stream_graph_updates(user_input)


