import sys
from pathlib import Path

# 将项目根目录添加到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
import os
from demo_work.agents.plus_agent import PlusAgent
from demo_work.agents.subtract_agent import SubtractAgent

# 加载.env文件
load_dotenv(Path(__file__).parent / '.env')

# 从环境变量中获取API key
api_key = os.getenv('DASHSCOPE_API_KEY')
if not api_key:
    raise ValueError('DASHSCOPE_API_KEY not found in .env file')


# 传入API key实例化Agent
plus_agent = PlusAgent(api_key=api_key)
subtract_agent = SubtractAgent(api_key=api_key)

# 调用示例
resp1 = plus_agent.invoke([{"role":"user","content":"请计算1.234+2.345"}])
resp2 = subtract_agent.invoke([{"role":"user","content":"请计算5.678-2.345"}])

print(f"resp1{resp1}")
print(f"resp2{resp2}")