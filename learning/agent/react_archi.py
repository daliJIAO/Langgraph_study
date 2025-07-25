"""
ReAct (Reasoning and Acting) 架构代理系统

ReAct是一种结合推理和行动的智能代理架构，通过思考-行动-暂停-观察的循环来解决问题。
该模块实现了一个基于通义千问模型的ReAct代理，支持数学计算和犬种体重查询功能。
"""

import getpass
import os
import re
from dotenv import load_dotenv
from langchain_qwq import ChatQwQ

# 加载环境变量配置
load_dotenv()

# 初始化通义千问语言模型
llm = ChatQwQ(
    model="qwen3-4b",  # 使用通义千问3-4B模型
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 阿里云灵积平台API端点
    max_tokens=3_000,  # 最大生成令牌数
    timeout=None,  # 请求超时时间（无限制）
    max_retries=2,  # 最大重试次数
)

# 检查并设置API密钥
if not os.getenv("DASHSCOPE_API_KEY"):
    os.environ["DASHSCOPE_API_KEY"] = getpass.getpass("Enter your Dashscope API key: ")


class Agent:
    """
    ReAct架构代理类

    实现了思考-行动-暂停-观察的循环模式，能够通过对话历史记录
    来维持上下文，并执行推理和行动任务。
    """

    def __init__(self, system=""):
        """
        初始化代理

        Args:
            system (str): 系统提示词，用于设定代理的行为模式和角色
        """
        self.system = system  # 存储系统提示词
        self.messages = []  # 对话历史记录列表

        # 如果提供了系统提示词，则添加到消息列表开头
        if self.system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message):
        """
        代理调用方法（使对象可调用）

        处理用户输入消息，执行推理，并返回代理的响应

        Args:
            message (str): 用户输入的消息

        Returns:
            str: 代理的响应结果
        """
        # 将用户消息添加到对话历史
        self.messages.append({"role": "user", "content": message})

        # 执行推理和生成响应
        result = self.execute()

        # 将代理响应添加到对话历史
        self.messages.append({"role": "assistant", "content": result})

        return result

    def execute(self):
        """
        执行代理推理

        调用语言模型生成响应，基于当前的对话历史记录

        Returns:
            str: 模型生成的响应内容
        """
        # 使用已定义的llm对象生成响应
        response = llm.invoke(self.messages)
        return response.content


# ReAct模式的系统提示词模板
# 定义了代理的工作流程和可用工具
REACT_PROMPT = """
你运行在一个思考(Thought)、行动(Action)、暂停(PAUSE)、观察(Observation)的循环中。

在循环结束时，你输出一个答案(Answer)。

使用思考(Thought)来描述你对被问问题的想法。

使用行动(Action)来运行你可以使用的操作之一 - 然后返回暂停(PAUSE)。

观察(Observation)将是运行这些操作的结果。

你可以使用的操作有：

calculate:
例如: calculate: 4 * 7 / 3
运行计算并返回数字 - 使用Python语法，如果需要请确保使用浮点语法

average_dog_weight:
例如: average_dog_weight: Collie
给定品种时返回狗的平均体重

示例会话：

问题：斗牛犬重多少？
思考：我应该使用average_dog_weight查询狗的体重
行动：average_dog_weight: 斗牛犬
暂停

你将再次被调用，带着这个：

观察：斗牛犬的重量是51磅

然后你输出：

答案：斗牛犬的重量是51磅
""".strip()


def calculate(what):
    """
    执行数学计算

    使用Python的eval函数计算表达式，并返回结果

    Args:
        what (str): 要计算的数学表达式

    Returns:
        float: 计算结果

    Note:
        使用eval存在安全风险，在生产环境中应使用更安全的替代方案
    """
    return eval(what)


def average_dog_weight(name):
    """
    查询犬种的平均体重

    根据给定的犬种名称，返回该犬种的平均体重描述

    Args:
        name (str): 犬种名称

    Returns:
        str: 平均体重描述
    """
    # 犬种体重数据库（简化版）
    if "Scottish Terrier" in name or "苏格兰梗" in name:
        return "苏格兰梗的平均体重是20磅"
    elif "Border Collie" in name or "边境牧羊犬" in name:
        return "边境牧羊犬的平均体重是37磅"
    elif "Toy Poodle" in name or "玩具贵宾犬" in name:
        return "玩具贵宾犬的平均体重是7磅"
    else:
        return "一般狗的平均体重是50磅"


# 用于匹配各个阶段的正则表达式
thought_re = re.compile(r"^思考[：:](.*)$", re.MULTILINE)
action_re = re.compile(r"^行动[：:]\s*(\w+)[：:]\s*(.*)$", re.MULTILINE)
pause_re = re.compile(r"^暂停", re.MULTILINE)
observation_re = re.compile(r"^观察[：:](.*)$", re.MULTILINE)
answer_re = re.compile(r"^答案[：:](.*)$", re.MULTILINE)


def parse_react_response(response):
    """
    解析ReAct响应，提取思考、行动、暂停、观察等阶段

    Args:
        response (str): 代理的完整响应

    Returns:
        dict: 包含各阶段内容的字典
    """
    stages = {
        'thought': None,
        'action': None,
        'action_input': None,
        'pause': False,
        'observation': None,
        'answer': None
    }

    # 提取思考内容
    thought_match = thought_re.search(response)
    if thought_match:
        stages['thought'] = thought_match.group(1).strip()

    # 提取行动内容
    action_match = action_re.search(response)
    if action_match:
        stages['action'] = action_match.group(1).strip()
        stages['action_input'] = action_match.group(2).strip()

    # 检查是否有暂停
    if pause_re.search(response):
        stages['pause'] = True

    # 提取观察内容
    observation_match = observation_re.search(response)
    if observation_match:
        stages['observation'] = observation_match.group(1).strip()

    # 提取答案内容
    answer_match = answer_re.search(response)
    if answer_match:
        stages['answer'] = answer_match.group(1).strip()

    return stages


def display_stage(stage_name, content, color_code=None):
    """
    格式化显示ReAct阶段信息

    Args:
        stage_name (str): 阶段名称
        content (str): 阶段内容
        color_code (str): 颜色代码（可选）
    """
    if color_code:
        print(f"\n{color_code}[{stage_name}]{content}\033[0m")
    else:
        print(f"\n[{stage_name}] {content}")


def query(question, max_turns=5):
    """
    查询接口

    与智能体进行交互式查询，支持多轮对话，实现ReAct循环
    显示详细的思考、行动、暂停、观察过程

    Args:
        question (str): 用户问题
        max_turns (int): 最多轮次，防止过长对话（默认5轮）

    Returns:
        None: 结果直接打印输出
    """
    print("🤖 开始ReAct代理推理...")
    print("📝 用户问题:", question)
    print("=" * 60)

    i = 0
    bot = Agent(REACT_PROMPT)
    next_prompt = question

    while i < max_turns:
        i += 1
        print(f"\n🔄 第 {i} 轮推理:")
        print("-" * 40)

        # 获取代理响应
        result = bot(next_prompt)

        # 解析响应的各个阶段
        stages = parse_react_response(result)

        # 显示原始响应（调试用）
        print(f"\n📋 原始响应:\n{result}")
        print("-" * 40)

        # 显示思考阶段
        if stages['thought']:
            display_stage("💭 思考", stages['thought'], "\033[94m")  # 蓝色

        # 显示行动阶段
        if stages['action'] and stages['action_input']:
            action_text = f"{stages['action']}: {stages['action_input']}"
            display_stage("⚡ 行动", action_text, "\033[93m")  # 黄色

            # 执行行动
            if stages['action'] in known_actions:
                print(f"\n🔧 正在执行行动: {stages['action']} {stages['action_input']}")
                try:
                    observation = known_actions[stages['action']](stages['action_input'])
                    display_stage("👁️ 观察", str(observation), "\033[92m")  # 绿色

                    # 为下一轮准备观察结果
                    next_prompt = f"观察: {observation}"

                except Exception as e:
                    error_msg = f"执行行动时出错: {str(e)}"
                    display_stage("❌ 错误", error_msg, "\033[91m")  # 红色
                    return
            else:
                error_msg = f"未知行动: {stages['action']}"
                display_stage("❌ 错误", error_msg, "\033[91m")  # 红色
                return

        # 显示暂停状态
        if stages['pause']:
            display_stage("⏸️ 暂停", "等待观察结果...", "\033[96m")  # 青色

        # 显示答案（如果有）
        if stages['answer']:
            display_stage("✅ 答案", stages['answer'], "\033[95m")  # 紫色
            print("\n🎉 推理完成!")
            return

        # 如果没有行动指令且没有答案，可能是推理结束
        if not stages['action'] and not stages['answer']:
            print("\n🔚 没有更多行动指令，推理结束")
            return

    print(f"\n⚠️ 达到最大轮次限制 ({max_turns})，推理结束")


# 定义已知的操作及其对应的处理函数
known_actions = {
    "calculate": calculate,
    "average_dog_weight": average_dog_weight
}


# 示例使用
if __name__ == "__main__":
    question = """
    我有2只狗，一只边境牧羊犬和一只苏格兰梗。
    它们的总重量是多少？
    """

    print("开始ReAct代理查询...")
    print("问题:", question)
    print("=" * 50)

    query(question)
