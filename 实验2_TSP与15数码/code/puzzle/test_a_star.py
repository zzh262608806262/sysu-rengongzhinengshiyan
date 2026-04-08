import time
import sys
sys.path.insert(0, '.')

print("开始测试A*算法...")

try:
    from a_star import a_star
    from heuristics import manhattan_distance
    print("成功导入模块")
except Exception as e:
    print(f"导入模块失败: {e}")
    sys.exit(1)

# 测试用例：简单案例
initial_state = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 0],
    [13, 14, 15, 12]
]

print("初始状态:")
for row in initial_state:
    print(" ".join(f"{num:2d}" if num != 0 else "  " for num in row))

print("\n测试A*算法...")
try:
    start_time = time.time()
    path, steps, expanded = a_star(initial_state, manhattan_distance)
    end_time = time.time()
    if path is not None:
        print(f"成功找到解！")
        print(f"步数: {steps}")
        print(f"扩展节点数: {expanded}")
        print(f"运行时间: {end_time - start_time:.6f}s")
    else:
        print("无解")
except Exception as e:
    print(f"测试失败: {e}")
    import traceback
    traceback.print_exc()

print("测试完成!")