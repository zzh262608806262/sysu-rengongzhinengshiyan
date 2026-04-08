from puzzle import Puzzle

def ida_star(initial_state, heuristic):
    """IDA*算法解决15-Puzzle问题"""
    initial_puzzle = Puzzle(initial_state)
    
    if initial_puzzle.is_goal():
        return [], 0, 0
    
    threshold = heuristic(initial_state)
    expanded_nodes = 0
    
    while True:
        # IDA*不需要visited集合，依靠阈值剪枝
        result, new_threshold, nodes, path = dfs(
            initial_puzzle, 0, threshold, heuristic, []
        )
        expanded_nodes += nodes
        
        if result:
            return path, threshold, expanded_nodes
        
        if new_threshold == float('inf'):
            return None, -1, expanded_nodes  # 无解
        
        threshold = new_threshold

def dfs(node, g, threshold, heuristic, path):
    """
    深度优先搜索，带阈值限制
    返回: (是否找到, 新阈值, 扩展节点数, 路径)
    """
    f = g + heuristic(node.state)
    
    if f > threshold:
        return False, f, 1, None
    
    if node.is_goal():
        return True, 0, 1, path + [node.state]
    
    min_threshold = float('inf')
    expanded = 1
    
    for neighbor in node.get_neighbors():
        # 简单防止循环：检查当前路径中是否已有相同状态
        # 不需要全局visited，只需要检查当前路径
        if any(neighbor.state == s for s in path):
            continue
            
        result, new_threshold, nodes, sub_path = dfs(
            neighbor, g + 1, threshold, heuristic, path + [node.state]
        )
        expanded += nodes
        
        if result:
            return True, 0, expanded, sub_path
        
        if new_threshold < min_threshold:
            min_threshold = new_threshold
    
    return False, min_threshold, expanded, None