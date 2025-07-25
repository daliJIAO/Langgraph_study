import getpass
import os
from dotenv import load_dotenv
from langchain_qwq import ChatQwQ
from langgraph.prebuilt import create_react_agent

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

message = {"messages":
               [{"role": "system", "content": "你可以做整数的加法运算"},
                {"role": "user", "content": "计算3加5"}, ]}

def plus(a: int, b: int):
    """计算输入的两个整数之和"""
    return a + b

agent = create_react_agent(
    model=llm,
    tools=[plus],
    prompt="You are a helpful assistant"
)

try:
    response = agent.invoke(message)
    print(response)
except Exception as e:
    print(f"Agent调用失败: {e}")