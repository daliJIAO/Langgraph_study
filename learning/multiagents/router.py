"""
智能客服工单路由系统

这个模块实现了一个基于LangGraph的多代理工单路由系统，用于自动化客服工单处理：
1. 路由代理（Router Agent）- 分析工单内容并自动分类
2. 专业团队代理 - 根据分类将工单分配给对应的专业团队
   - 账单团队（Billing Team）
   - 技术支持团队（Technical Team）
   - 一般咨询团队（General Team）
   - 人工审核（Manual Review）

系统通过关键词匹配进行智能分类，并将工单路由到相应的处理团队。
"""

from typing import Dict, Any, TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
import re
import time


# 步骤1：定义状态结构
class TicketState(TypedDict):
    """
    工单状态定义

    保存工单信息和处理结果的状态结构
    """
    ticket_text: str        # 工单内容文本
    category: str          # 确定的分类（账单、技术、一般咨询或未知）
    resolution: str        # 支持团队提供的解决方案
    processing_time: float # 处理工单所需的时间（秒）


# 步骤2：定义路由代理
def router_agent(state: TicketState) -> Dict[str, Any]:
    """
    路由分析代理

    分析工单内容并确定其分类类别：
    - 使用关键词匹配进行简单分类（可替换为LLM或机器学习模型）
    - 支持账单、技术、一般咨询和未知四个类别

    Args:
        state (TicketState): 当前工单状态

    Returns:
        Dict[str, Any]: 包含分类结果和处理时间的字典
    """
    print("🔍 路由代理：正在分析工单...")

    start_time = time.time()
    ticket_text = state["ticket_text"].lower()

    # 基于关键词的简单分类逻辑（可替换为更高级的LLM或ML模型）
    if any(keyword in ticket_text for keyword in ["billing", "payment", "invoice", "charge", "账单", "付款", "发票", "收费"]):
        category = "Billing"
    elif any(keyword in ticket_text for keyword in ["technical", "bug", "error", "crash", "技术", "错误", "故障", "崩溃"]):
        category = "Technical"
    elif any(keyword in ticket_text for keyword in ["general", "question", "inquiry", "info", "一般", "问题", "咨询", "信息"]):
        category = "General"
    else:
        category = "Unknown"

    processing_time = time.time() - start_time

    print(f"🔍 路由代理：分类为 '{category}'，耗时 {processing_time:.2f} 秒")

    return {
        "category": category,
        "processing_time": processing_time
    }


# 步骤3：定义专业支持团队代理
# 每个代理处理特定类别的工单

# 账单团队代理
def billing_team_agent(state: TicketState) -> Dict[str, Any]:
    """
    账单团队处理代理

    专门处理与账单、付款、发票相关的工单

    Args:
        state (TicketState): 当前工单状态

    Returns:
        Dict[str, Any]: 包含解决方案和总处理时间的字典
    """
    print("💰 账单团队代理：正在处理工单...")

    start_time = time.time()
    ticket_text = state["ticket_text"]

    # 生成账单相关的标准回复
    resolution = f"账单团队：已审核工单 '{ticket_text}'。请检查您的发票详情或联系我们的账单部门获取进一步帮助。"

    processing_time = time.time() - start_time
    time.sleep(1)  # 模拟处理时间

    print(f"💰 账单团队代理：处理完成，耗时 {processing_time:.2f} 秒")

    return {
        "resolution": resolution,
        "processing_time": state["processing_time"] + processing_time
    }


# 技术支持团队代理
def technical_team_agent(state: TicketState) -> Dict[str, Any]:
    """
    技术支持团队处理代理

    专门处理技术问题、错误报���、故障排除等工单

    Args:
        state (TicketState): 当前工单状态

    Returns:
        Dict[str, Any]: 包含技术解决方案和总处理时间的字典
    """
    print("🔧 技术支持团队代理：���在处理工单...")

    start_time = time.time()
    ticket_text = state["ticket_text"]

    # 生成技术支持相关的标准回复
    resolution = f"技术团队：已审核工单 '{ticket_text}'。请尝试重启您的设备或提交详细的错误日志以进一步调查。"

    processing_time = time.time() - start_time
    time.sleep(1.5)  # 模拟处理时间（技术问题通常需要更多时间）

    print(f"🔧 技术支持团队代理：处理���成，耗时 {processing_time:.2f} 秒")

    return {
        "resolution": resolution,
        "processing_time": state["processing_time"] + processing_time
    }


# 一般咨询团队代理
def general_team_agent(state: TicketState) -> Dict[str, Any]:
    """
    一般咨询团队处理代理

    处理一般性问题、咨询和信息请求

    Args:
        state (TicketState): 当前工单状态

    Returns:
        Dict[str, Any]: 包含一般咨询回复和总处理时间的字典
    """
    print("📋 一般咨询团队代理：正在处理工单...")

    start_time = time.time()
    ticket_text = state["ticket_text"]

    # 生成一般咨询相关的标准回复
    resolution = f"一般咨询团队：已审核工单 '{ticket_text}'。如需更多信息，请参考我们的常见问题解答或通过邮件联系我们。"

    processing_time = time.time() - start_time
    time.sleep(0.8)  # 模拟处理时间

    print(f"📋 一般咨询团队代理：处理完成，耗时 {processing_time:.2f} 秒")

    return {
        "resolution": resolution,
        "processing_time": state["processing_time"] + processing_time
    }


# 人工审核代理（处理未知类别）
def manual_review_agent(state: TicketState) -> Dict[str, Any]:
    """
    人工审核代理

    处理无法自动分类的工单，标记为需要人工审核

    Args:
        state (TicketState): 当前工单状态

    Returns:
        Dict[str, Any]: 包含人工审核标记和总处理时间的字典
    """
    print("👤 人工审核代理：正在处理工单...")

    start_time = time.time()
    ticket_text = state["ticket_text"]

    # 生成人工审核标记的回复
    resolution = f"人工审核：工单 '{ticket_text}' 无法自动分类。已标记为需要人工审核，请手动分配给相应团队。"

    processing_time = time.time() - start_time
    time.sleep(0.5)  # 模拟处理时间

    print(f"👤 人工审核代理：处理完成，耗时 {processing_time:.2f} 秒")

    return {
        "resolution": resolution,
        "processing_time": state["processing_time"] + processing_time
    }


# 步骤4：定义路由决策函数
def route_ticket(state: TicketState) -> Literal["billing_team", "technical_team", "general_team", "manual_review"]:
    """
    工单路由决策函数

    根据工单分类结果决定将工单路由到哪个处理团队

    Args:
        state (TicketState): 当前工单状态

    Returns:
        Literal: 下一个处理节点的名称
    """
    category = state["category"]
    print(f"🚦 路由决策：工单类别为 '{category}'")

    # 根据分类结果进行路由
    if category == "Billing":
        return "billing_team"
    elif category == "Technical":
        return "technical_team"
    elif category == "General":
        return "general_team"
    else:
        return "manual_review"


# 步骤5：构建LangGraph工作流
def build_router_graph():
    """
    构建工单路由图

    创建包含路由逻辑和专业团队处理的完整工作流

    Returns:
        CompiledGraph: 编译后的可执行图
    """
    print("🏗️ 构建工单路由工作流...")

    # 创建状态图
    workflow = StateGraph(TicketState)

    # 添加路由代理节点
    workflow.add_node("router", router_agent)

    # 添加各专业团队��理节点
    workflow.add_node("billing_team", billing_team_agent)
    workflow.add_node("technical_team", technical_team_agent)
    workflow.add_node("general_team", general_team_agent)
    workflow.add_node("manual_review", manual_review_agent)

    print("➕ 已添加所有处理节点")

    # 设置入口点为路由代理
    workflow.set_entry_point("router")

    # 添加条件路由边：从路由代理到各专业团队
    workflow.add_conditional_edges(
        "router",
        route_ticket,
        {
            "billing_team": "billing_team",
            "technical_team": "technical_team",
            "general_team": "general_team",
            "manual_review": "manual_review"
        }
    )

    # 所有专业团队处理完成后结束流程
    workflow.add_edge("billing_team", END)
    workflow.add_edge("technical_team", END)
    workflow.add_edge("general_team", END)
    workflow.add_edge("manual_review", END)

    print("🔗 已添加路由边连接")
    print("✅ 工作流构��完成")

    return workflow.compile()


# 主执行函数
def main():
    """
    主演示函数

    使用多个示例工单测试路由系统的完整功能
    """
    print("🚀 智能客服工单路由系统演示")
    print("=" * 80)

    # 定义测试工单样例
    test_tickets = [
        "我的账单有问题，付款没有成功",
        "应用程序一直崩溃，无法正常使用",
        "请问你们的服务时间是什么？",
        "这是一个无法分类的奇怪问题"
    ]

    # 处理每个测试工单
    for ticket_text in test_tickets:
        # 初始化工单状态
        initial_state = {
            "ticket_text": ticket_text,
            "category": "",
            "resolution": "",
            "processing_time": 0.0
        }

        print(f"\n=== 处理工单: '{ticket_text}' ===")

        # 构建并执行路由图
        app = build_router_graph()
        start_time = time.time()
        result = app.invoke(initial_state, config=RunnableConfig())
        total_time = time.time() - start_time

        # 显示处理结果
        print("\n📋 === 工单处理结果 ===")
        print(f"🏷️  分类: {result['category']}")
        print(f"💬 解决方案: {result['resolution']}")
        print(f"⏱️  处理时间: {result['processing_time']:.2f} 秒")
        print(f"🕐 总耗时: {total_time:.2f} 秒")
        print("-" * 50)


# 程序入口点
if __name__ == "__main__":
    main()
