import time
import sys
sys.path.insert(0, '.')

print("开始测试15-Puzzle问题...")

try:
    from a_star import a_star
    from ida_star import ida_star
    from heuristics import misplaced_tiles, manhattan_distance, manhattan_with_conflict
    print("成功导入所有模块")
except Exception as e:
    print(f"导入模块失败: {e}")
    sys.exit(1)

# 从test_cases.py导入测试用例
from test_cases import test_cases

# 启发式函数列表
heuristics = [
    ("错位棋子数", misplaced_tiles),
    ("曼哈顿距离", manhattan_distance),
    ("曼哈顿距离+线性冲突", manhattan_with_conflict)
]

# 测试所有用例
for case_name, initial_state in test_cases:
    print(f"\n测试用例: {case_name}")
    print("初始状态:")
    for row in initial_state:
        print(" ".join(f"{num:2d}" if num != 0 else "  " for num in row))

    # 检测是否有解
    from puzzle import Puzzle
    puzzle = Puzzle(initial_state)
    if not puzzle.is_solvable():
        print("-" * 60)
        print("该测试用例无解，跳过测试")
        print("-" * 60)
        continue
    
    print("-" * 60)
    print("该测试用例有解，开始测试...")

    # 测试A*算法
    print("\nA*算法:")
    for name, heuristic in heuristics:
        print(f"  测试 {name}...")
        try:
            start_time = time.time()
            path, steps, expanded = a_star(initial_state, heuristic)
            end_time = time.time()
            if path is not None:
                print(f"  {name}: 步数={steps}, 扩展节点数={expanded}, 时间={end_time - start_time:.6f}s")
                # 输出成功的路径（前5步和后5步）
                if steps > 0:
                    print(f"  路径长度: {steps}")
                    if steps <= 10:
                        print("  完整路径:")
                        for i, step in enumerate(path):
                            print(f"  步骤 {i+1}:")
                            print(step)
                    else:
                        print("  路径前5步:")
                        for i, step in enumerate(path[:5]):
                            print(f"  步骤 {i+1}:")
                            print(step)
                        print("  ...")
                        print(f"  路径后5步:")
                        for i, step in enumerate(path[-5:], steps-4):
                            print(f"  步骤 {i}:")
                            print(step)
            else:
                print(f"  {name}: 无解")
        except Exception as e:
            print(f"  {name} 测试失败: {e}")

    # 测试IDA*算法
    print("\nIDA*算法:")
    for name, heuristic in heuristics:
        print(f"  测试 {name}...")
        try:
            start_time = time.time()
            path, steps, expanded = ida_star(initial_state, heuristic)
            end_time = time.time()
            if path is not None:
                print(f"  {name}: 步数={steps}, 扩展节点数={expanded}, 时间={end_time - start_time:.6f}s")
                # 输出成功的路径
                if steps > 0:
                    print(f"  路径长度: {steps}")
                    if steps <= 10:
                        print("  完整路径:")
                        for i, step in enumerate(path):
                            print(f"  步骤 {i+1}:")
                            for row in step:
                                print("  " + " ".join(f"{num:2d}" if num != 0 else "  " for num in row))
                    else:
                        print("  路径前5步:")
                        for i, step in enumerate(path[:5]):
                            print(f"  步骤 {i+1}:")
                            for row in step:
                                print("  " + " ".join(f"{num:2d}" if num != 0 else "  " for num in row))
                        print("  ...")
                        print(f"  路径后5步:")
                        for i, step in enumerate(path[-5:], steps-4):
                            print(f"  步骤 {i}:")
                            for row in step:
                                print("  " + " ".join(f"{num:2d}" if num != 0 else "  " for num in row))
            else:
                print(f"  {name}: 无解")
        except Exception as e:
            print(f"  {name} 测试失败: {e}")

    print("-" * 60)

print("-" * 60)
print("测试完成!")