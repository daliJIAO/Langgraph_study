"""
社交媒体情感分析聚合系统

这个模块实现了一个基于LangGraph的多代理���交媒体情感分析聚合系统，用于：
1. 从多个社交媒体平台收集帖子内容（Twitter、Instagram、Reddit）
2. 对各平台的帖子进行情感分析
3. 聚合分析结果生成综合报告

系统采用并行处理架构，能够同时处理多个平台的数据，并将结果聚合到统一的状态中。
"""

from typing import Dict, Any, TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
from textblob import TextBlob
import time
from typing_extensions import Annotated
from operator import add


# 步骤1：定义状态结构
class SocialMediaState(TypedDict):
    """
    社交媒体分析状态定义

    存储各个社交媒体平台的帖子内容、情感分析结果和最终报告
    """
    twitter_posts: List[str]           # Twitter帖子列表
    instagram_posts: List[str]         # Instagram帖子列表
    reddit_posts: List[str]            # Reddit帖子列表
    twitter_sentiment: Dict[str, float]   # Twitter情感分析结果
    instagram_sentiment: Dict[str, float] # Instagram情感分析结果
    reddit_sentiment: Dict[str, float]    # Reddit情感分析结果
    final_report: str                  # 最终聚合分析报告
    processing_time: Annotated[float, add]  # 累计处理时间（使用add操作符聚合）


# 步骤2：定义帖子收集代理
# 每个代理负责从特定社交媒体平台收集帖子

def collect_twitter_posts(state: SocialMediaState) -> Dict[str, Any]:
    """
    Twitter帖子收集代理

    模拟从Twitter平台收集品牌相关帖子
    在实际应用中，这里会调用Twitter API获取真实数据

    Args:
        state (SocialMediaState): 当前状态

    Returns:
        Dict[str, Any]: 包含Twitter帖子和处理时间的字典
    """
    print("🐦 Twitter代理：正在收集帖子...")

    start_time = time.time()

    # 模拟Twitter帖子数据（包含正面和负面评价）
    posts = [
        "Loving the new product from this brand! Amazing quality.",  # 正面评价
        "Terrible customer service from this brand. Very disappointed."  # 负面评价
    ]

    time.sleep(1)  # 模拟API调用延迟
    processing_time = time.time() - start_time

    print(f"🐦 Twitter代理：收集完成，耗时 {processing_time:.2f} 秒")

    return {
        "twitter_posts": posts,
        "processing_time": processing_time
    }


def collect_instagram_posts(state: SocialMediaState) -> Dict[str, Any]:
    """
    Instagram帖子收集代理

    模拟从Instagram平台收集品牌相关帖子和评论

    Args:
        state (SocialMediaState): 当前状态

    Returns:
        Dict[str, Any]: 包含Instagram帖子和处理时间的字典
    """
    print("📷 Instagram代理：正在收集帖子...")

    start_time = time.time()

    # 模拟Instagram帖子数据（通常更加视觉化和情感化）
    posts = [
        "Beautiful design by this brand! #loveit",  # 正面评价带话题标签
        "Not impressed with the latest release. Expected better."  # 负��评价
    ]

    time.sleep(1.2)  # 模拟API调用延迟（稍长，因为需要处理图片数据）
    processing_time = time.time() - start_time

    print(f"📷 Instagram代理：收集完成，耗时 {processing_time:.2f} 秒")

    return {
        "instagram_posts": posts,
        "processing_time": processing_time
    }


def collect_reddit_posts(state: SocialMediaState) -> Dict[str, Any]:
    """
    Reddit帖子收集代理

    模拟从Reddit平台收集品牌相关讨论和评论

    Args:
        state (SocialMediaState): 当前状态

    Returns:
        Dict[str, Any]: 包含Reddit帖子和处理时间的字典
    """
    print("🔗 Reddit代理：正在收集帖子...")

    start_time = time.time()

    # 模拟Reddit帖子数据（通常更加详细和讨论性）
    posts = [
        "This brand is awesome! Great value for money.",  # 正面评价
        "Had a bad experience with their support team. Not happy."  # 负面评价
    ]

    time.sleep(0.8)  # 模拟API调用延迟
    processing_time = time.time() - start_time

    print(f"🔗 Reddit代理：收集完成，耗时 {processing_time:.2f} 秒")

    return {
        "reddit_posts": posts,
        "processing_time": processing_time
    }


# 步骤3：定义情感分析代理
# 每个代理负责分析特定平台帖子的情感倾向

def analyze_twitter_sentiment(state: SocialMediaState) -> Dict[str, Any]:
    """
    Twitter情感分析代理

    使用TextBlob对Twitter帖子进行情感分析
    计算平均情感极性值

    Args:
        state (SocialMediaState): 包含Twitter帖子的当前状态

    Returns:
        Dict[str, Any]: 包含情感分析结果和处理时间的字典
    """
    print("🔍 Twitter情感分析代理：正在分析情感...")

    start_time = time.time()
    posts = state["twitter_posts"]

    # 使用TextBlob计算每个帖子的情感极性
    polarities = [TextBlob(post).sentiment.polarity for post in posts]

    # 计算平均情感极���（-1为最负面，1为最正面，0为中性）
    avg_polarity = sum(polarities) / len(polarities) if polarities else 0.0

    time.sleep(0.5)  # 模拟情感分析处理时间
    processing_time = time.time() - start_time

    print(f"🔍 Twitter情感分析代理：分析完成，耗时 {processing_time:.2f} 秒")

    return {
        "twitter_sentiment": {
            "average_polarity": avg_polarity,
            "num_posts": len(posts)
        },
        "processing_time": processing_time
    }


def analyze_instagram_sentiment(state: SocialMediaState) -> Dict[str, Any]:
    """
    Instagram情感分析代理

    使用TextBlob对Instagram帖子进行情感分析

    Args:
        state (SocialMediaState): 包含Instagram帖子的当前状态

    Returns:
        Dict[str, Any]: 包含情感分析结果和处理时间的字典
    """
    print("🔍 Instagram情感分析代理：正在分析情感...")

    start_time = time.time()
    posts = state["instagram_posts"]

    # 计算情感极性
    polarities = [TextBlob(post).sentiment.polarity for post in posts]
    avg_polarity = sum(polarities) / len(polarities) if polarities else 0.0

    time.sleep(0.6)  # 模拟处理时间
    processing_time = time.time() - start_time

    print(f"🔍 Instagram情感分析代理：分析完成，耗时 {processing_time:.2f} 秒")

    return {
        "instagram_sentiment": {
            "average_polarity": avg_polarity,
            "num_posts": len(posts)
        },
        "processing_time": processing_time
    }


def analyze_reddit_sentiment(state: SocialMediaState) -> Dict[str, Any]:
    """
    Reddit情感分析代理

    使用TextBlob对Reddit帖子进行情感分析

    Args:
        state (SocialMediaState): 包含Reddit帖子的当前状态

    Returns:
        Dict[str, Any]: 包含情感分析结果和处理时间的字典
    """
    print("🔍 Reddit情感分析代理：正在分��情感...")

    start_time = time.time()
    posts = state["reddit_posts"]

    # 计算情感极性
    polarities = [TextBlob(post).sentiment.polarity for post in posts]
    avg_polarity = sum(polarities) / len(polarities) if polarities else 0.0

    time.sleep(0.4)  # 模拟处理时间
    processing_time = time.time() - start_time

    print(f"🔍 Reddit情感分析代理：分析完成，耗时 {processing_time:.2f} 秒")

    return {
        "reddit_sentiment": {
            "average_polarity": avg_polarity,
            "num_posts": len(posts)
        },
        "processing_time": processing_time
    }


# 步骤4：定义报告聚合代理
def generate_final_report(state: SocialMediaState) -> Dict[str, Any]:
    """
    最终报告生成代理

    聚合所有平台的情感分析结果，生成综合分析报告

    Args:
        state (SocialMediaState): 包含所有分析结果的状态

    Returns:
        Dict[str, Any]: 包含最终报告和处理时间的字典
    """
    print("📊 报告生成代理：正在生成最终报告...")

    start_time = time.time()

    # 提取各平台的情感分析结果
    twitter_sentiment = state.get("twitter_sentiment", {})
    instagram_sentiment = state.get("instagram_sentiment", {})
    reddit_sentiment = state.get("reddit_sentiment", {})

    # 计算整体情感倾向
    all_polarities = []
    total_posts = 0

    platform_results = []

    # 处理Twitter结果
    if twitter_sentiment:
        twitter_polarity = twitter_sentiment.get("average_polarity", 0)
        twitter_posts = twitter_sentiment.get("num_posts", 0)
        all_polarities.append(twitter_polarity)
        total_posts += twitter_posts

        sentiment_label = "正面" if twitter_polarity > 0.1 else "负面" if twitter_polarity < -0.1 else "中性"
        platform_results.append(f"🐦 Twitter: {sentiment_label} (极性: {twitter_polarity:.3f}, 帖子数: {twitter_posts})")

    # 处理Instagram结果
    if instagram_sentiment:
        instagram_polarity = instagram_sentiment.get("average_polarity", 0)
        instagram_posts = instagram_sentiment.get("num_posts", 0)
        all_polarities.append(instagram_polarity)
        total_posts += instagram_posts

        sentiment_label = "正面" if instagram_polarity > 0.1 else "负面" if instagram_polarity < -0.1 else "中性"
        platform_results.append(f"📷 Instagram: {sentiment_label} (极性: {instagram_polarity:.3f}, 帖子数: {instagram_posts})")

    # 处理Reddit结果
    if reddit_sentiment:
        reddit_polarity = reddit_sentiment.get("average_polarity", 0)
        reddit_posts = reddit_sentiment.get("num_posts", 0)
        all_polarities.append(reddit_polarity)
        total_posts += reddit_posts

        sentiment_label = "正面" if reddit_polarity > 0.1 else "负面" if reddit_polarity < -0.1 else "中性"
        platform_results.append(f"🔗 Reddit: {sentiment_label} (极性: {reddit_polarity:.3f}, 帖子数: {reddit_posts})")

    # 计算整体平均情感
    overall_polarity = sum(all_polarities) / len(all_polarities) if all_polarities else 0.0
    overall_sentiment = "正面" if overall_polarity > 0.1 else "负面" if overall_polarity < -0.1 else "中性"

    # 生成最终报告
    report = f"""
📊 社交媒体情感分析综合报告
{'='*50}

🎯 整体情感倾向: {overall_sentiment}
📈 整体情感极性: {overall_polarity:.3f}
📝 总帖子数量: {total_posts}

📱 各平台详细分析:
{chr(10).join(platform_results)}

⏱️ 总处理时间: {state.get('processing_time', 0):.2f} 秒

💡 建议:
"""

    # 根据整体情感添加建议
    if overall_polarity > 0.1:
        report += "品牌在社交媒体上获得了积极反馈，建议继续保持当前策略并扩大正面影响。"
    elif overall_polarity < -0.1:
        report += "品牌在社交媒体上存在负面反馈，建议深入了解用户关切并改进产品/服务。"
    else:
        report += "品牌在社交媒体上反馈较为中性，建议加强品牌宣传和用户互动。"

    time.sleep(0.3)  # 模拟报告生成时间
    processing_time = time.time() - start_time

    print(f"📊 报告生成代理：报告生成完成，耗时 {processing_time:.2f} 秒")

    return {
        "final_report": report,
        "processing_time": processing_time
    }


# 步骤5：构建聚合工作流
def build_social_media_analysis_graph():
    """
    构建社交媒体分析聚合图

    创建包含数据收集、情感分析和报告生成的完整工作流

    Returns:
        CompiledGraph: 编译后的可执行图
    """
    print("🏗️ 构建社交媒体分析聚合工作流...")

    # 创建状态图
    workflow = StateGraph(SocialMediaState)

    # 添加数据收集节点
    workflow.add_node("collect_twitter", collect_twitter_posts)
    workflow.add_node("collect_instagram", collect_instagram_posts)
    workflow.add_node("collect_reddit", collect_reddit_posts)

    # 添加情感分析节点
    workflow.add_node("analyze_twitter", analyze_twitter_sentiment)
    workflow.add_node("analyze_instagram", analyze_instagram_sentiment)
    workflow.add_node("analyze_reddit", analyze_reddit_sentiment)

    # 添加报告生成节点
    workflow.add_node("generate_report", generate_final_report)

    print("➕ 已添加所有处理节点")

    # 设置入口点（并行开始）
    workflow.set_entry_point("collect_twitter")
    workflow.set_entry_point("collect_instagram")
    workflow.set_entry_point("collect_reddit")

    # 添加数据收集到情感分析的边
    workflow.add_edge("collect_twitter", "analyze_twitter")
    workflow.add_edge("collect_instagram", "analyze_instagram")
    workflow.add_edge("collect_reddit", "analyze_reddit")

    # 所有情感分析完成后生成报告
    workflow.add_edge("analyze_twitter", "generate_report")
    workflow.add_edge("analyze_instagram", "generate_report")
    workflow.add_edge("analyze_reddit", "generate_report")

    # 报告生成后结束
    workflow.add_edge("generate_report", END)

    print("🔗 已添加工作流边连接")
    print("✅ 工作流构建完成")

    return workflow.compile()


# 主执行函数
def main():
    """
    主演示函数

    执行完整的社交媒体情感分析聚合流程
    """
    print("🚀 社交媒体情感分析聚合系统演示")
    print("=" * 80)

    # 初始化空状态
    initial_state = {
        "twitter_posts": [],
        "instagram_posts": [],
        "reddit_posts": [],
        "twitter_sentiment": {},
        "instagram_sentiment": {},
        "reddit_sentiment": {},
        "final_report": "",
        "processing_time": 0.0
    }

    print("📱 开始从多个社交媒体平台收集和分析数据...")
    print("-" * 60)

    # 构建并执行聚合图
    app = build_social_media_analysis_graph()
    start_time = time.time()
    result = app.invoke(initial_state, config=RunnableConfig())
    total_time = time.time() - start_time

    # 显示最终结果
    print("\n📋 === 分析结果 ===")
    print(result["final_report"])

    print(f"\n🕐 实际总耗时: {total_time:.2f} 秒")

    print("\n✨ 社交媒体分析完成!")


# 程序入口点
if __name__ == "__main__":
    main()