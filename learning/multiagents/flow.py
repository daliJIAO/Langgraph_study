"""
多代理审批流程系统

这个模块实现了一个基于LangGraph的多代理审批流程，模拟企业项目提案的层级审批：
1. 团队负责人（Team Lead）- 初步审核
2. 部门经理（Department Manager）- 预算审核
3. 财务总监（Finance Director）- 最终审批

流程采用条件路由，每个代理根据审核结果决定下一步流向。
"""

from typing import Dict
from langgraph.graph import StateGraph, MessagesState, END
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage
import json


# 代理1：团队负责人
def team_lead_agent(state: MessagesState, config: RunnableConfig) -> Dict:
    """
    团队负责人审核代理

    负责项目提案的初步审核，检查提案的完整性：
    - 验证提案标题是否存在
    - 验证预算金额是否有效

    Args:
        state (MessagesState): 当前消息状态，包含项目提案
        config (RunnableConfig): 运行时配置

    Returns:
        Dict: 更新后的消息状态，包含审核结果
    """
    print("代理（团队负责人）：开始审核")

    messages = state["messages"]

    # 解析原始提案内容
    proposal = json.loads(messages[0].content)
    title = proposal.get("title", "")
    amount = proposal.get("amount", 0.0)

    # 基本完整性检查
    if not title or amount <= 0:
        status = "拒绝"
        comment = "团队负责人：提案因缺少标题或无效金额而被拒绝。"
        goto = END  # 直接结束流程
    else:
        status = "团队负责人批准"
        comment = "团队负责人：提案完整且通过审核。"
        goto = "dept_manager"  # 转发给部门经理

    print(f"代理（团队负责人）：审核完成 - {status}")

    # 将审核结果添加到消息历史
    messages.append(AIMessage(
        content=json.dumps({"status": status, "comment": comment}),
        additional_kwargs={"agent": "team_lead", "goto": goto}
    ))

    return {"messages": messages}


# 代理2：部门经理
def dept_manager_agent(state: MessagesState, config: RunnableConfig) -> Dict:
    """
    部门经理审核代理

    负责项目提案的预算审核：
    - 检查团队负责人的审核结果
    - 验证预算是否超过部门限额（100,000）

    Args:
        state (MessagesState): 当前消息状态，包含项目提案和前序审核结果
        config (RunnableConfig): 运行时配置

    Returns:
        Dict: 更新后的消息状态，包含审核结果
    """
    print("代理（部门经理）：开始审核")

    messages = state["messages"]

    # 查找团队负责人的审核结果
    team_lead_msg = next((m for m in messages if m.additional_kwargs.get("agent") == "team_lead"), None)

    # 获取提案金额
    proposal = json.loads(messages[0].content)
    amount = proposal.get("amount", 0.0)

    # 检查前序审核结果和预算限制
    if json.loads(team_lead_msg.content)["status"] != "团队负责人批准":
        status = "拒绝"
        comment = "部门经理：因团队负责人拒绝而跳过审核。"
        goto = END
    elif amount > 100000:
        status = "拒绝"
        comment = "部门经理：预算超出限额。"
        goto = END
    else:
        status = "部门经理批准"
        comment = "部门经理：预算在限额范围内。"
        goto = "finance_director"  # 转发给财务总监

    print(f"代理（部门经理）：审核完成 - {status}")

    # 将审核结果添加到消息历史
    messages.append(AIMessage(
        content=json.dumps({"status": status, "comment": comment}),
        additional_kwargs={"agent": "dept_manager", "goto": goto}
    ))

    return {"messages": messages}


# 代理3：财务总监
def finance_director_agent(state: MessagesState, config: RunnableConfig) -> Dict:
    """
    财务总监审核代理

    负责项目提案的最终财务审批：
    - 检查部门经理的审核结果
    - 验证预算是否在财务可承受范围内（50,000）

    Args:
        state (MessagesState): 当前消息状态，包含项目提案和前序审核结果
        config (RunnableConfig): 运行时配置

    Returns:
        Dict: 更新后的消息状态，包含最终审核结果
    """
    print("代理（财务总监）：开始审核")

    messages = state["messages"]

    # 查找部门经理的审核结果
    dept_msg = next((m for m in messages if m.additional_kwargs.get("agent") == "dept_manager"), None)

    # 获取提案金额
    proposal = json.loads(messages[0].content)
    amount = proposal.get("amount", 0.0)

    # 检查前序审核结果和财务限制
    if json.loads(dept_msg.content)["status"] != "部门经理批准":
        status = "拒绝"
        comment = "财务总监：因部门经理拒绝而跳过审核。"
    elif amount > 50000:
        status = "拒绝"
        comment = "财务总监：预算不足。"
    else:
        status = "批准"
        comment = "财务总监：批准且可行。"

    print(f"代理（财务总监）：审核完成 - {status}")

    # 将最终审核结果添加到消息历史
    messages.append(AIMessage(
        content=json.dumps({"status": status, "comment": comment}),
        additional_kwargs={"agent": "finance_director", "goto": END}
    ))

    return {"messages": messages}


# 路由决策函数
def route_step(state: MessagesState) -> str:
    """
    条件路由函数

    根据最新代理的审核结果决定下一个执行节点：
    - 检查消息历史中最新的goto指令
    - 返回相应的下一个节点名称或END

    Args:
        state (MessagesState): 当前消息状态

    Returns:
        str: 下一个节点的名称或END
    """
    # 从最新的消息开始向前查找goto指令
    for msg in reversed(state["messages"]):
        goto = msg.additional_kwargs.get("goto")
        if goto:
            print(f"路由：代理 {msg.additional_kwargs.get('agent')} 设置下一步为 {goto}")
            return goto

    return END


# 构建LangGraph工作流
print("🏗️ 构建多代理审批工作流...")

# 创建状态图
builder = StateGraph(MessagesState)

# 添加代理节点
builder.add_node("team_lead", team_lead_agent)
builder.add_node("dept_manager", dept_manager_agent)
builder.add_node("finance_director", finance_director_agent)

print("➕ 已添加所有代理节点")

# 设置入口点为团队负责人
builder.set_entry_point("team_lead")
print("🎯 设置入口点：team_lead")

# 添加条件边：团队负责人 -> 部门经理或结束
builder.add_conditional_edges("team_lead", route_step, {
    "dept_manager": "dept_manager",
    END: END
})

# 添加条件边：部门经理 -> 财务总监或结束
builder.add_conditional_edges("dept_manager", route_step, {
    "finance_director": "finance_director",
    END: END
})

# 添加条件边：财务总监 -> 结束
builder.add_conditional_edges("finance_director", route_step, {
    END: END
})

print("🔗 已添加所有条件路由边")

# 编译工作流
workflow = builder.compile()
print("✅ 工作流编译完成")


# 主执行函数
def main():
    """
    主执行函数

    创建示例项目提案并执行完整的审批流程，
    然后解析和显示最终的审批结果。
    """
    print("🚀 开始多代理审批流程演示")
    print("=" * 60)

    # 创建初始项目提案
    initial_state = {
        "messages": [
            HumanMessage(
                content=json.dumps({
                    "title": "新设备采购",
                    "amount": 40000.0,
                    "department": "工程部"
                })
            )
        ]
    }

    print("📄 项目提案:")
    proposal_data = json.loads(initial_state["messages"][0].content)
    print(f"   标题: {proposal_data['title']}")
    print(f"   金额: ¥{proposal_data['amount']:,.2f}")
    print(f"   部门: {proposal_data['department']}")
    print("-" * 60)

    # 执行审批流程
    result = workflow.invoke(initial_state)
    messages = result["messages"]

    # 解析审批结果
    proposal = json.loads(messages[0].content)

    print("\n📋 === 审批结果 ===")
    print(f"📝 提案标题: {proposal['title']}")

    # 收集所有审核状态和评论
    final_status = "未知"
    comments = []

    # 遍历所有AI消息，提取审核信息
    for msg in messages[1:]:
        if isinstance(msg, AIMessage):
            try:
                data = json.loads(msg.content)
                if "status" in data:
                    final_status = data["status"]
                if "comment" in data:
                    comments.append(data["comment"])
            except Exception:
                # 忽略JSON解析错误
                continue

    # 显示最终结果
    print(f"🎯 最终状态: {final_status}")
    print("💬 审核意见:")
    for comment in comments:
        print(f"  - {comment}")

    print("\n✨ 审批流程完成!")


# 程序入口点
if __name__ == "__main__":
    main()
