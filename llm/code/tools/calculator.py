"""计算工具 - 用于统计论点和评分计算"""

def count_arguments(arguments: str) -> dict:
    """统计论点数量
    
    Args:
        arguments: 论点文本
    
    Returns:
        包含统计信息的字典
    """
    # 简单统计：计算句子数量作为论点数量的近似
    sentences = [s.strip() for s in arguments.split('.') if len(s.strip()) > 10]
    return {
        "total_points": len(sentences),
        "word_count": len(arguments.split())
    }

def calculate_debate_score(
    affirmative_points: int,
    negative_points: int,
    affirmative_rebuttals: int = 0,
    negative_rebuttals: int = 0
) -> dict:
    """计算辩论评分
    
    Args:
        affirmative_points: 正方论点数量
        negative_points: 反方论点数量
        affirmative_rebuttals: 正方反驳数量
        negative_rebuttals: 反方反驳数量
    
    Returns:
        评分结果
    """
    total_affirmative = affirmative_points + affirmative_rebuttals
    total_negative = negative_points + negative_rebuttals
    total = total_affirmative + total_negative
    
    if total == 0:
        return {
            "affirmative_score": 50,
            "negative_score": 50,
            "winner": "tie"
        }
    
    affirmative_score = (total_affirmative / total) * 100
    negative_score = (total_negative / total) * 100
    
    if affirmative_score > negative_score:
        winner = "affirmative"
    elif negative_score > affirmative_score:
        winner = "negative"
    else:
        winner = "tie"
    
    return {
        "affirmative_score": round(affirmative_score, 2),
        "negative_score": round(negative_score, 2),
        "winner": winner,
        "total_arguments": total
    }
