import time
import os
import sys

# 重定向输出到文件
output_file = open('test_results.txt', 'w', encoding='utf-8')
original_stdout = sys.stdout
sys.stdout = output_file

print("开始自动测试TSP数据集...")

try:
    from tsp import TSP
    from genetic import GeneticAlgorithm
    from test_cases import test_datasets
    print("成功导入模块")
except Exception as e:
    print(f"导入模块失败: {e}")
    import sys
    sys.exit(1)

# 多组测试参数配置
test_parameters = [
    {"population_size": 50, "mutation_rate": 0.01, "crossover_rate": 0.8, "generations": 1000, "early_stopping": 50, "name": "配置1: 小种群"},
    {"population_size": 100, "mutation_rate": 0.01, "crossover_rate": 0.8, "generations": 1000, "early_stopping": 50, "name": "配置2: 标准参数"},
    {"population_size": 200, "mutation_rate": 0.01, "crossover_rate": 0.8, "generations": 1000, "early_stopping": 50, "name": "配置3: 大种群"},
    {"population_size": 100, "mutation_rate": 0.05, "crossover_rate": 0.8, "generations": 1000, "early_stopping": 50, "name": "配置4: 高变异率"},
    {"population_size": 100, "mutation_rate": 0.01, "crossover_rate": 0.9, "generations": 1000, "early_stopping": 50, "name": "配置5: 高交叉率"},
    {"population_size": 100, "mutation_rate": 0.01, "crossover_rate": 0.8, "generations": 2000, "early_stopping": 100, "name": "配置6: 长迭代"},
]

# 存储所有结果
all_results = []

# 测试所有数据集和参数设置
for dataset_name, file_path in test_datasets:
    # 确保文件路径是绝对路径
    file_path = os.path.join(os.path.dirname(__file__), file_path)
    print(f"\n{'='*60}")
    print(f"数据集: {dataset_name}")
    print(f"文件路径: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"数据集文件 {file_path} 不存在")
        continue
    
    # 读取数据集
    try:
        tsp = TSP([])
        cities = tsp.read_cities_from_file(file_path)
        print(f"成功读取数据集，城市数量: {tsp.num_cities}")
    except Exception as e:
        print(f"读取数据集失败: {e}")
        continue
    
    dataset_results = {
        'dataset_name': dataset_name,
        'num_cities': tsp.num_cities,
        'results': []
    }
    
    # 测试不同参数配置
    for params in test_parameters:
        print(f"\n{'-'*40}")
        print(f"测试参数: {params['name']}")
        print(f"种群大小={params['population_size']}, 变异率={params['mutation_rate']}, 交叉率={params['crossover_rate']}, 迭代次数={params['generations']}, 早停代数={params['early_stopping']}")
        
        try:
            # 创建遗传算法实例
            ga = GeneticAlgorithm(
                tsp, 
                population_size=params['population_size'], 
                mutation_rate=params['mutation_rate'], 
                crossover_rate=params['crossover_rate'], 
                generations=params['generations'],
                early_stopping=params['early_stopping']
            )
            
            # 运行算法
            start_time = time.time()
            best_route, best_distance = ga.evolve()
            end_time = time.time()
            
            runtime = end_time - start_time
            
            # 输出结果
            print(f"最佳距离: {best_distance:.2f}")
            print(f"运行时间: {runtime:.2f}s")
            print(f"最佳路线长度: {len(best_route)}")
            
            # 保存结果
            dataset_results['results'].append({
                'config_name': params['name'],
                'population_size': params['population_size'],
                'mutation_rate': params['mutation_rate'],
                'crossover_rate': params['crossover_rate'],
                'generations': params['generations'],
                'early_stopping': params['early_stopping'],
                'best_distance': best_distance,
                'runtime': runtime
            })
            
        except Exception as e:
            print(f"运行算法失败: {e}")
    
    all_results.append(dataset_results)

# 打印汇总结果
print(f"\n{'='*60}")
print("测试完成！汇总结果：")
print(f"{'='*60}")

for dataset_result in all_results:
    print(f"\n数据集: {dataset_result['dataset_name']} ({dataset_result['num_cities']} 个城市)")
    print(f"{'参数配置':<20} {'种群大小':<10} {'变异率':<10} {'交叉率':<10} {'最佳距离':<15} {'运行时间(s)':<12}")
    print("-" * 80)
    for result in dataset_result['results']:
        print(f"{result['config_name']:<20} {result['population_size']:<10} {result['mutation_rate']:<10} {result['crossover_rate']:<10} {result['best_distance']:<15.2f} {result['runtime']:<12.2f}")

# 关闭文件
output_file.close()
sys.stdout = original_stdout
print("测试完成！结果已保存到 test_results.txt")
