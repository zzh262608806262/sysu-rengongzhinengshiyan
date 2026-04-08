def misplaced_tiles(state):
    """计算错位棋子数"""
    goal = [[1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 0]]
    count = 0
    for i in range(4):
        for j in range(4):
            if state[i][j] != 0 and state[i][j] != goal[i][j]:
                count += 1
    return count

def manhattan_distance(state):
    """计算曼哈顿距离"""
    distance = 0
    for i in range(4):
        for j in range(4):
            value = state[i][j]
            if value != 0:
                # 计算目标位置
                goal_i = (value - 1) // 4
                goal_j = (value - 1) % 4
                distance += abs(i - goal_i) + abs(j - goal_j)
    return distance

def linear_conflict(state):
    """计算线性冲突"""
    conflict = 0
    
    # 检查行冲突
    for i in range(4):
        row = state[i]
        max_val = -1
        for j in range(4):
            value = row[j]
            if value != 0 and (value - 1) // 4 == i:
                if value > max_val:
                    max_val = value
                else:
                    conflict += 2
    
    # 检查列冲突
    for j in range(4):
        col = [state[i][j] for i in range(4)]
        max_val = -1
        for i in range(4):
            value = col[i]
            if value != 0 and (value - 1) % 4 == j:
                if value > max_val:
                    max_val = value
                else:
                    conflict += 2
    
    return conflict

def manhattan_with_conflict(state):
    """曼哈顿距离加上线性冲突"""
    return manhattan_distance(state) + linear_conflict(state)