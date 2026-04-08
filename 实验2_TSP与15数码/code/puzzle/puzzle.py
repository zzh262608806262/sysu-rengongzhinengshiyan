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
    
    def is_solvable(self):
        """判断15-Puzzle是否有解"""
        # 计算逆序数
        inversions = 0
        flat_state = []
        for row in self.state:
            for num in row:
                if num != 0:
                    flat_state.append(num)
        
        for i in range(len(flat_state)):
            for j in range(i + 1, len(flat_state)):
                if flat_state[i] > flat_state[j]:
                    inversions += 1
        
        # 计算空白格从底部数的行数（从1开始）
        empty_row_from_bottom = self.size - self.empty_pos[0]
        
        # 对于4×4的棋盘（偶数），规则是：
        # 空白格在奇数行（从底部数），逆序数必须为偶数
        # 空白格在偶数行（从底部数），逆序数必须为奇数
        if self.size % 2 == 0:
            if empty_row_from_bottom % 2 == 1:
                return inversions % 2 == 0
            else:
                return inversions % 2 == 1
        else:
            # 对于奇数大小的棋盘，逆序数必须为偶数
            return inversions % 2 == 0