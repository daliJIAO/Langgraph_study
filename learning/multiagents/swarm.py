"""
多代理群体（Swarm）系统

这个模块实现了一个基于LangGraph的智能代理��体系统，其中：
1. 多个代理可以动态决定下一个执行的代理
2. 使用Command模式进行代理间的路由和状态更新
3. LLM负责智能决策下一步应该执行哪个代理

这种模式适合需要复杂决策流程和动态路由的多代理协作场景。
"""

from typing import Literal
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command

# 初始化语言模型
model = ChatOpenAI()

def agent_1(state: MessagesState) -> Command[Literal["agent_2", "agent_3", END]]:
    """
    代理1 - 群体协调者

    作为群体中的第一个代理，负责分析当前状态并决定下一步的执行路径。
    这个代理可以选择将任务分配给agent_2、agent_3或者直接结束流程。

    Args:
        state (MessagesState): 当前的消息状态，包含对话历史

    Returns:
        Command: 包含下一个代理路由信息和状态更新的命令对象
    """

    # 可以将状态的相关部分传递给LLM（例如，state["messages"]）
    # 来确定下一个调用哪个代理。常见模式是使用结构化输出调用模型
    # （例如强制返回包含"next_agent"字段的输出）

    # 构建给LLM的提示，让其根据当前对话决定下一步行动
    system_prompt = """
    你是一个智能代理协调器。根据当前的对话内容，决定���一步应该：
    - 选择 'agent_2' 如果需要专业技术处理
    - 选择 'agent_3' 如果需要创意性工作  
    - 选择 '__end__' 如果任务已完成
    
    请返回JSON格式：{"next_agent": "选择的代理", "content": "你的回复"}
    """

    messages = [
        {"role": "system", "content": system_prompt},
        *state["messages"]
    ]

    response = model.invoke(messages)

    # 根据LLM的决策路由到其中一个代理或退出
    # 如果LLM返回"__end__"，图将完成执行
    return Command(
        goto=response["next_agent"],  # 路由到指定的下一个代理
        update={"messages": [response["content"]]},  # 更新消息状态
    )


def agent_2(state: MessagesState) -> Command[Literal["agent_1", "agent_3", END]]:
    """
    代理2 - 技术专家

    专门处理技术相关的任务和问题。可以选择将复杂任务交回给协调者，
    或者转给创意代理，或者完成任务。

    Args:
        state (MessagesState): 当前的消息状态

    Returns:
        Command: 路由命令和状态更新
    """

    system_prompt = """
    你是一个技术专家代理。专门处理技术问题、编程任务等。
    根据当前对话，决定：
    - 选择 'agent_1' 如果需要重新协调
    - 选择 'agent_3' 如果需要创意支持
    - 选择 '__end__' 如果技术问题已解决
    
    请返回JSON格式：{"next_agent": "选择的代理", "content": "你的技术回复"}
    """

    messages = [
        {"role": "system", "content": system_prompt},
        *state["messages"]
    ]

    response = model.invoke(messages)

    return Command(
        goto=response["next_agent"],
        update={"messages": [response["content"]]},
    )


def agent_3(state: MessagesState) -> Command[Literal["agent_1", "agent_2", END]]:
    """
    代理3 - 创意专家

    专门处理创意性工作、设计思考等任务。可以选择将任务交回给协调者，
    或者转给技术专家，或者完成任务。

    Args:
        state (MessagesState): 当前的消息状态

    Returns:
        Command: 路由命令和状态更新
    """

    system_prompt = """
    你是一个创意专家代理。专门处理创意工作、设计、文案等任务。
    根据当前对话，决定：
    - 选择 'agent_1' 如果需要重新协调
    - 选择 'agent_2' 如果需要技术实现
    - 选择 '__end__' 如果创意工作已完成
    
    请返回JSON格式：{"next_agent": "选择的代理", "content": "你的创意回复"}
    """

    messages = [
        {"role": "system", "content": system_prompt},
        *state["messages"]
    ]

    response = model.invoke(messages)

    return Command(
        goto=response["next_agent"],
        update={"messages": [response["content"]]},
    )


# 构建多代理群体图
def build_swarm_graph():
    """
    构建多代理群体工作流图

    创建一个支持动态路由的多代理协作系统：
    - 每个代理都可以决定下一个执行的代理
    - 支持循环和条件分支
    - 智能结束条件判断

    Returns:
        CompiledGraph: 编译后的可执行图
    """
    print("🏗️ 构建多代理群体工作流...")

    # 创建状态图
    graph = StateGraph(MessagesState)

    # 添加代理节点
    graph.add_node("agent_1", agent_1)  # 协调者代理
    graph.add_node("agent_2", agent_2)  # 技术专家代理
    graph.add_node("agent_3", agent_3)  # 创意专家代理

    print("➕ 已添加所有代理节点")

    # 设置入口点为协调者
    graph.set_entry_point("agent_1")
    print("🎯 设置入口点：agent_1（协调者）")

    # 由于使用Command模式，所有的路由逻辑都在代理函数内部处理
    # 不需要显式添加边，LangGraph会根据Command的goto字段自动路由

    print("🔗 代理间动态路由已配置（基于Command模式）")
    print("✅ 群体工作流构建完成")

    return graph.compile()


# 主执行函数
def main():
    """
    主演示函数

    展示多代理群体系统的智能协作能力
    """
    print("🚀 多代理群体（Swarm）系统演示")
    print("=" * 80)

    # 初始化状态
    initial_state = {
        "messages": [
            {
                "role": "user",
                "content": "我需要开发一个创意性的网页应用，既要有技术实现又要有好的设计"
            }
        ]
    }

    print("📝 用户需求:")
    print(f"   {initial_state['messages'][0]['content']}")
    print("-" * 60)

    # 构建并执行群体图
    swarm = build_swarm_graph()

    print("🔄 开始多代理协作...")
    start_time = time.time()

    result = swarm.invoke(initial_state)

    total_time = time.time() - start_time

    # 显示协作结果
    print("\n📋 === 群体协作结果 ===")
    print("🔄 代理协作流程:")

    for i, message in enumerate(result["messages"], 1):
        role = message.get("role", "assistant")
        content = message.get("content", "")
        print(f"  {i}. [{role}]: {content}")

    print(f"\n⏱️ 总协作时间: {total_time:.2f} 秒")
    print("✨ 群体协作完成!")


# 辅助函数：创建带有特定角色的代理
def create_specialized_agent(role: str, specialization: str):
    """
    创建专业化代理的工厂函数

    Args:
        role (str): 代理角色名称
        specialization (str): 专业领域描述

    Returns:
        function: 配置好的代理函数
    """
    def agent_function(state: MessagesState) -> Command:
        system_prompt = f"""
        你是一个{role}，专门负责{specialization}。
        根据当前对话内容，智能决定下一步的协作流程。
        
        请返回JSON格式：{{"next_agent": "选择的代理", "content": "你的专业回复"}}
        """

        messages = [
            {"role": "system", "content": system_prompt},
            *state["messages"]
        ]

        response = model.invoke(messages)

        return Command(
            goto=response.get("next_agent", "__end__"),
            update={"messages": [response.get("content", "")]},
        )

    return agent_function


# 程序入口点
if __name__ == "__main__":
    import time
    main()
