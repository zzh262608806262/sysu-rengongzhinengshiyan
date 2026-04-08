import time
from tsp import TSP
from genetic import GeneticAlgorithm

print("开始测试TSP问题...")

# 测试用例1：简单城市集
def test_simple_cities():
    print("\n测试用例1：简单城市集")
    # 简单的5个城市
    cities = [(0, 0), (1, 3), (2, 1), (3, 4), (4, 2)]
    tsp = TSP(cities)
    
    print(f"城市数量: {tsp.num_cities}")
    
    # 测试遗传算法
    ga = GeneticAlgorithm(tsp, population_size=50, mutation_rate=0.01, crossover_rate=0.8, generations=500)
    
    start_time = time.time()
    best_route, best_distance = ga.evolve()
    end_time = time.time()
    
    print(f"\n最佳路线: {best_route}")
    print(f"最佳距离: {best_distance:.2f}")
    print(f"运行时间: {end_time - start_time:.2f}s")

# 测试用例2：从文件读取城市数据
def test_file_data():
    print("\n测试用例2：从文件读取城市数据")
    # 创建一个简单的测试文件
    test_file = "test_cities.txt"
    with open(test_file, 'w') as f:
        f.write("1 0.0 0.0\n")
        f.write("2 1.0 3.0\n")
        f.write("3 2.0 1.0\n")
        f.write("4 3.0 4.0\n")
        f.write("5 4.0 2.0\n")
    
    tsp = TSP([])
    cities = tsp.read_cities_from_file(test_file)
    
    print(f"读取到的城市数量: {tsp.num_cities}")
    
    # 测试遗传算法
    ga = GeneticAlgorithm(tsp, population_size=50, mutation_rate=0.01, crossover_rate=0.8, generations=500)
    
    start_time = time.time()
    best_route, best_distance = ga.evolve()
    end_time = time.time()
    
    print(f"\n最佳路线: {best_route}")
    print(f"最佳距离: {best_distance:.2f}")
    print(f"运行时间: {end_time - start_time:.2f}s")

# 测试不同参数设置
def test_different_parameters():
    print("\n测试用例3：不同参数设置")
    cities = [(0, 0), (1, 3), (2, 1), (3, 4), (4, 2), (5, 5), (6, 3), (7, 1), (8, 4), (9, 2)]
    tsp = TSP(cities)
    
    parameters = [
        (50, 0.01, 0.8),    # 默认参数
        (100, 0.01, 0.8),   # 更大的种群
        (50, 0.05, 0.8),    # 更高的变异率
        (50, 0.01, 0.9),    # 更高的交叉率
    ]
    
    for pop_size, mut_rate, cross_rate in parameters:
        print(f"\n参数: 种群大小={pop_size}, 变异率={mut_rate}, 交叉率={cross_rate}")
        ga = GeneticAlgorithm(tsp, population_size=pop_size, mutation_rate=mut_rate, crossover_rate=cross_rate, generations=300)
        
        start_time = time.time()
        best_route, best_distance = ga.evolve()
        end_time = time.time()
        
        print(f"最佳距离: {best_distance:.2f}")
        print(f"运行时间: {end_time - start_time:.2f}s")

if __name__ == "__main__":
    test_simple_cities()
    test_file_data()
    test_different_parameters()
    print("\n测试完成!")