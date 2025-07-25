"""
并行多代理文本处理系统

这个模块实现了一个基于LangGraph的并行多代理系统，用于同时处理文本的多个任务：
- 文本摘要（Summarization）
- 文本翻译（Translation）
- 情感分析（Sentiment Analysis）

通过并行执行这些任务，可以提高处理效率和用户体验。
"""

from typing import Dict, Any, TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
from textblob import TextBlob
import re
import time


# 定义系统状态结构
class AgentState(TypedDict):
    """
    代理状态定义

    定义了在图执行过程中各个节点之间传递的状态数据结构
    """
    text: str              # 原始输入文本
    summary: str           # 文本摘要结果
    translation: str       # 翻译结果
    sentiment: str         # 情感分析结果
    summary_time: float    # 摘要处理耗时（秒）
    translation_time: float # 翻译处理耗时（秒）
    sentiment_time: float   # 情感分析处理耗时（秒）


# 文本摘要代理
def summarize_agent(state: AgentState) -> Dict[str, Any]:
    """
    文本摘要处理代理

    使用简单的句子评分算法提取文本中的关键句子作为摘要
    评分标准：句子长度（单词数量）

    Args:
        state (AgentState): 当前状态，包含待处理的文本

    Returns:
        Dict[str, Any]: 包含摘要结果和处理时间的字典
    """
    print("📝 摘要代理：开始运行")
    start_time = time.time()

    try:
        text = state["text"]

        # 检查输入文本是否为空
        if not text.strip():
            return {
                "summary": "未提供文本进行摘要处理。",
                "summary_time": 0.0
            }

        # 模拟处理时间
        time.sleep(2)

        # 按标点符号分割句子
        sentences = re.split(r'(?<=[.!?]) +', text.strip())

        # 为每个句子打分（基于单词数量）
        scored_sentences = [(s, len(s.split())) for s in sentences if s]

        # 选择得分最高的前两个句子作为摘要
        top_sentences = [s for s, _ in sorted(scored_sentences, key=lambda x: x[1], reverse=True)[:2]]

        # 生成最终摘要
        summary = " ".join(top_sentences) if top_sentences else "文本太短无法生成摘要。"

        processing_time = time.time() - start_time
        print(f"📝 摘要代理：完成处理，耗时 {processing_time:.2f} 秒")

        return {
            "summary": summary,
            "summary_time": processing_time
        }

    except Exception as e:
        return {
            "summary": f"摘要处理出错: {str(e)}",
            "summary_time": 0.0
        }


# 文本翻译代理
def translate_agent(state: AgentState) -> Dict[str, Any]:
    """
    文本翻译处理代理

    将英文文本翻译为西班牙语（这里使用模拟翻译）
    在实际应用中，这里可以集成真实的翻译API

    Args:
        state (AgentState): 当前状态，包含待翻译的文本

    Returns:
        Dict[str, Any]: 包含翻译结果和处理时间的字典
    """
    print("🌐 翻译代理：开始运行")
    start_time = time.time()

    try:
        text = state["text"]

        # 检查输入文本是否为空
        if not text.strip():
            return {
                "translation": "未提供文本进行翻译处理。",
                "translation_time": 0.0
            }

        # 模拟翻译处理时间（比摘要稍长）
        time.sleep(3)

        # 模拟的西班牙语翻译结果
        # 在实际应用中，这里应该调用真实的翻译服务
        translation = (
            "El nuevo parque en la ciudad es una maravillosa adición. "
            "Las familias disfrutan de los espacios abiertos, y a los niños les encanta el parque infantil. "
            "Sin embargo, algunas personas piensan que el área de estacionamiento es demasiado pequeña."
        )

        processing_time = time.time() - start_time
        print(f"🌐 翻译代理：完成处理，耗时 {processing_time:.2f} 秒")

        return {
            "translation": translation,
            "translation_time": processing_time
        }

    except Exception as e:
        return {
            "translation": f"翻译处理出错: {str(e)}",
            "translation_time": 0.0
        }


# 情感分析代理
def sentiment_agent(state: AgentState) -> Dict[str, Any]:
    """
    情感分析处理代理

    使用TextBlob库分析文本的情感倾向
    返回情感极性（正面/负面/中性）和主观性分数

    Args:
        state (AgentState): 当前状态，包含待分析的文本

    Returns:
        Dict[str, Any]: 包含情感分析结果和处理时间的字典
    """
    print("😊 情感分析代理：开始运行")
    start_time = time.time()

    try:
        text = state["text"]

        # 检查输入文本是否为空
        if not text.strip():
            return {
                "sentiment": "未提供文本进行情感分析。",
                "sentiment_time": 0.0
            }

        # 模拟处理时间（最短）
        time.sleep(1.5)

        # 使用TextBlob进行情感分析
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity      # 情感极性：-1(负面) 到 1(正面)
        subjectivity = blob.sentiment.subjectivity  # 主观性：0(客观) 到 1(主观)

        # 根据极性值确定情感类别
        if polarity > 0:
            sentiment = "正面"
        elif polarity < 0:
            sentiment = "负面"
        else:
            sentiment = "中性"

        # 格式化结果字符串
        result = f"{sentiment} (极性: {polarity:.2f}, 主观性: {subjectivity:.2f})"

        processing_time = time.time() - start_time
        print(f"😊 情感分析代理：完成处理，耗时 {processing_time:.2f} 秒")

        return {
            "sentiment": result,
            "sentiment_time": processing_time
        }

    except Exception as e:
        return {
            "sentiment": f"情感分析出错: {str(e)}",
            "sentiment_time": 0.0
        }


# 并行结果合并节点
def join_parallel_results(state: AgentState) -> AgentState:
    """
    合并并行处理结果

    这是一个简单的合并函数，直接返回状态
    在更复杂的场景中，这里可以进行结果的后处理或整合

    Args:
        state (AgentState): 包含所有并行处理结果的状态

    Returns:
        AgentState: 合并后的最终状态
    """
    print("🔗 合并节点：整合并行处理结果")
    return state


# 构建并行处理图
def create_parallel_workflow():
    """
    创建并行文本处理工作流

    构建一个支持并行执行的LangGraph工作流：
    1. 从分支节点开始
    2. 并行执行三个处理任务（摘要、翻译、情感分析）
    3. 在合并节点收集所有结果
    4. 结束流程

    Returns:
        CompiledGraph: 编译后的可执行图
    """
    print("🏗️ 构建并行处理工作流...")

    # 创建状态图
    workflow = StateGraph(AgentState)

    # 定义并行分支：每个分支对应一个处理任务
    parallel_branches = {
        "summarize_node": summarize_agent,    # 摘要处理分支
        "translate_node": translate_agent,     # 翻译处理分支
        "sentiment_node": sentiment_agent      # 情感分析分支
    }

    # 添加并行处理节点到工作流
    for name, agent in parallel_branches.items():
        workflow.add_node(name, agent)
        print(f"➕ 添加节点: {name}")

    # 添加分支和合并节点
    workflow.add_node("branch", lambda state: state)  # 简化的分支函数，直接传递状态
    workflow.add_node("join", join_parallel_results)   # 合并节点

    print("➕ 添加分支节点和合并节点")

    # 设置工作流入口点
    workflow.set_entry_point("branch")
    print("🎯 设置入口点: branch")

    # 添加并行执行的边：从分支节点到各个处理节点
    for name in parallel_branches:
        workflow.add_edge("branch", name)
        print(f"🔗 添加边: branch -> {name}")

    # 添加从各个处理节点到合并节点的边
    for name in parallel_branches:
        workflow.add_edge(name, "join")
        print(f"🔗 添加边: {name} -> join")

    # 添加从合并节点到结束的边
    workflow.add_edge("join", END)
    print("🔗 添加边: join -> END")

    print("✅ 工作流构建完成")
    return workflow.compile()


# 执行并行处理的主函数
def run_parallel_processing(input_text: str):
    """
    运行并行文本处理

    Args:
        input_text (str): 待处理的输入文本

    Returns:
        Dict: 包含所有处理结果的字典
    """
    print("🚀 开始并行文本处理...")
    print(f"📄 输入文本: {input_text[:100]}..." if len(input_text) > 100 else f"📄 输入文本: {input_text}")
    print("=" * 80)

    # 创建工作流
    workflow = create_parallel_workflow()

    # 初始化状态
    initial_state = {
        "text": input_text,
        "summary": "",
        "translation": "",
        "sentiment": "",
        "summary_time": 0.0,
        "translation_time": 0.0,
        "sentiment_time": 0.0
    }

    # 执行工作流
    start_time = time.time()
    final_state = workflow.invoke(initial_state)
    total_time = time.time() - start_time

    print("=" * 80)
    print("🎉 并行处理完成!")
    print(f"⏱️ 总耗时: {total_time:.2f} 秒")
    print("\n📊 处理结果:")
    print("-" * 50)
    print(f"📝 摘要 (耗时 {final_state['summary_time']:.2f}s):")
    print(f"   {final_state['summary']}")
    print(f"\n🌐 翻译 (耗时 {final_state['translation_time']:.2f}s):")
    print(f"   {final_state['translation']}")
    print(f"\n😊 情感分析 (耗时 {final_state['sentiment_time']:.2f}s):")
    print(f"   {final_state['sentiment']}")

    return final_state


# 示例使用
if __name__ == "__main__":
    # 测试文本
    sample_text = """
    The new park in the city is a wonderful addition. 
    Families enjoy the open spaces, and children love the playground. 
    However, some people think the parking area is too small.
    """

    print("🔥 并行多代理文本处理系统演示")
    print("=" * 80)

    # 运行并行处理
    result = run_parallel_processing(sample_text.strip())

    print("\n✨ 演示完成!")
