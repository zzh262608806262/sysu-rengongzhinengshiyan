print("开始测试Puzzle类的基本功能...")

# 直接在脚本中定义Puzzle类，避免导入问题
class Puzzle:
    def __init__(self, state, empty_pos=None):
        self.state = state
        self.size = 4
        if empty_pos is None:
            self.empty_pos = self.find_empty()
        else:
            self.empty_pos = empty_pos
    
    def find_empty(self):
        for i in range(self.size):
            for j in range(self.size):
                if self.state[i][j] == 0:
                    return (i, j)
        return None
    
    def get_neighbors(self):
        neighbors = []
        i, j = self.empty_pos
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.size and 0 <= nj < self.size:
                new_state = [row.copy() for row in self.state]
                new_state[i][j], new_state[ni][nj] = new_state[ni][nj], new_state[i][j]
                neighbors.append(Puzzle(new_state, (ni, nj)))
        
        return neighbors
    
    def is_goal(self):
        goal = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 0]]
        return self.state == goal
    
    def __eq__(self, other):
        if not isinstance(other, Puzzle):
            return False
        return self.state == other.state
    
    def __hash__(self):
        return hash(tuple(tuple(row) for row in self.state))
    
    def __str__(self):
        result = ""
        for row in self.state:
            result += " ".join(f"{num:2d}" if num != 0 else "  " for num in row)
            result += "\n"
        return result

# 测试用例
initial_state = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 15, 14, 0]
]

print("测试1: 创建Puzzle实例")
try:
    puzzle = Puzzle(initial_state)
    print("✓ 成功创建Puzzle实例")
    print("初始状态:")
    print(puzzle)
    print(f"空白位置: {puzzle.empty_pos}")
except Exception as e:
    print(f"✗ 创建Puzzle实例失败: {e}")

print("\n测试2: 检查是否是目标状态")
try:
    is_goal = puzzle.is_goal()
    print(f"✓ 成功检查目标状态: {is_goal}")
except Exception as e:
    print(f"✗ 检查目标状态失败: {e}")

print("\n测试3: 获取邻居")
try:
    neighbors = puzzle.get_neighbors()
    print(f"✓ 成功获取邻居: {len(neighbors)} 个")
    for i, neighbor in enumerate(neighbors):
        print(f"  邻居 {i+1}:")
        print(neighbor)
except Exception as e:
    print(f"✗ 获取邻居失败: {e}")

print("\n测试4: 测试__eq__和__hash__方法")
try:
    puzzle2 = Puzzle(initial_state)
    print(f"✓ 成功测试__eq__方法: {puzzle == puzzle2}")
    s = set()
    s.add(puzzle)
    s.add(puzzle2)
    print(f"✓ 成功测试__hash__方法: 集合大小 = {len(s)}")
except Exception as e:
    print(f"✗ 测试__eq__和__hash__方法失败: {e}")

print("\n测试5: 测试目标状态")
try:
    goal_state = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 0]
    ]
    goal_puzzle = Puzzle(goal_state)
    is_goal = goal_puzzle.is_goal()
    print(f"✓ 成功测试目标状态: {is_goal}")
except Exception as e:
    print(f"✗ 测试目标状态失败: {e}")

print("\n所有测试完成!")