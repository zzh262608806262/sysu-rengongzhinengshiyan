import time
import os
print("开始测试真实TSP数据集...")

try:
    from tsp import TSP
    from genetic import GeneticAlgorithm
    from test_cases import test_datasets, test_parameters
    print("成功导入模块")
except Exception as e:
    print(f"导入模块失败: {e}")
    import sys
    sys.exit(1)

# 测试所有数据集和参数设置
for dataset_name, file_path in test_datasets:
    # 确保文件路径是绝对路径
    file_path = os.path.join(os.path.dirname(__file__), file_path)
    print(f"检查数据集文件: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"数据集文件 {file_path} 不存在")
        continue
    
    print(f"\n测试数据集: {dataset_name}")
    
    # 读取数据集
    try:
        tsp = TSP([])
        cities = tsp.read_cities_from_file(file_path)
        print(f"成功读取数据集，城市数量: {tsp.num_cities}")
    except Exception as e:
        print(f"读取数据集失败: {e}")
        continue
    
    # 测试不同参数设置
    for params in test_parameters:
        print(f"\n参数: 种群大小={params['population_size']}, 迭代次数=100, 变异率=0.01")
        
        try:
            # 创建遗传算法实例
            ga = GeneticAlgorithm(tsp, population_size=params['population_size'], mutation_rate=0.01, crossover_rate=0.8, generations=100)
            
            # 运行算法
            print("开始运行遗传算法...")
            start_time = time.time()
            best_route, best_distance = ga.evolve()
            end_time = time.time()
            
            # 输出结果
            print(f"最佳距离: {best_distance:.2f}")
            print(f"运行时间: {end_time - start_time:.2f}s")
            print(f"最佳路线长度: {len(best_route)}")
        except Exception as e:
            print(f"运行算法失败: {e}")

print("\n测试完成!")