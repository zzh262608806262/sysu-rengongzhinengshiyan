"""搜索工具 - 使用DuckDuckGo进行网络搜索"""
from langchain_community.tools import DuckDuckGoSearchResults

def create_search_tool():
    """创建搜索工具"""
    search = DuckDuckGoSearchResults(
        num_results=5,
        backend="api"
    )
    return search

def search_debate_arguments(topic: str, side: str) -> str:
    """搜索辩论论据
    
    Args:
        topic: 辩论主题
        side: 立场（affirmative/negative）
    
    Returns:
        搜索结果字符串
    """
    try:
        search = DuckDuckGoSearchResults(num_results=5)
        query = f"{topic} {'pros benefits advantages' if side == 'affirmative' else 'cons disadvantages problems'}"
        result = search.run(query)
        return result
    except Exception as e:
        return f"搜索失败: {str(e)}，将使用已有知识进行辩论"
