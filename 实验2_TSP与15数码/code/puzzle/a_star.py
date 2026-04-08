import heapq
from puzzle import Puzzle

def a_star(initial_state, heuristic):
    """A*算法解决15-Puzzle问题"""
    initial_puzzle = Puzzle(initial_state)
    if initial_puzzle.is_goal():
        return [], 0, 0
    
    # 优先队列: (f, counter, puzzle, g, path)
    open_set = []
    counter = 0
    initial_f = heuristic(initial_state)
    heapq.heappush(open_set, (initial_f, counter, initial_puzzle, 0, []))
    
    # 已访问的状态
    closed_set = set()
    
    expanded_nodes = 0
    max_expanded = 10000  # 最大扩展节点数，防止无限循环
    
    while open_set:
        if expanded_nodes > max_expanded:
            # 超过最大扩展节点数，认为无解
            return None, -1, expanded_nodes
        
        f, _, current, g, path = heapq.heappop(open_set)
        expanded_nodes += 1
        
        if current.is_goal():
            # 返回实际的路径
            return path, g, expanded_nodes
        
        if current in closed_set:
            continue
        
        closed_set.add(current)
        
        for neighbor in current.get_neighbors():
            if neighbor not in closed_set:
                new_g = g + 1
                new_f = new_g + heuristic(neighbor.state)
                new_path = path + [neighbor]
                counter += 1
                heapq.heappush(open_set, (new_f, counter, neighbor, new_g, new_path))
    
    # 优先队列为空，未找到解
    return None, -1, expanded_nodes