import time
import os
import re
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

def get_user_parameters():
    """获取用户输入的遗传算法参数"""
    # 默认参数
    default_population_size = 100
    default_mutation_rate = 0.01
    default_crossover_rate = 0.8
    default_generations = 1000
    default_early_stopping = 50
    
    print("\n=== 遗传算法参数设置 ===")
    print(f"默认值: 种群大小={default_population_size}, 变异率={default_mutation_rate}, 交叉率={default_crossover_rate}, 迭代次数={default_generations}, 早停代数={default_early_stopping}")
    print("按Enter键使用默认值，或输入新值")
    
    # 获取种群大小
    while True:
        pop_size_input = input(f"种群大小 (默认: {default_population_size}): ")
        if not pop_size_input:
            population_size = default_population_size
            break
        try:
            population_size = int(pop_size_input)
            if population_size > 0:
                break
            else:
                print("请输入正整数")
        except ValueError:
            print("请输入有效的整数")
    
    # 获取变异率
    while True:
        mut_rate_input = input(f"变异率 (默认: {default_mutation_rate}): ")
        if not mut_rate_input:
            mutation_rate = default_mutation_rate
            break
        try:
            mutation_rate = float(mut_rate_input)
            if 0 <= mutation_rate <= 1:
                break
            else:
                print("请输入0-1之间的数")
        except ValueError:
            print("请输入有效的小数")
    
    # 获取交叉率
    while True:
        cross_rate_input = input(f"交叉率 (默认: {default_crossover_rate}): ")
        if not cross_rate_input:
            crossover_rate = default_crossover_rate
            break
        try:
            crossover_rate = float(cross_rate_input)
            if 0 <= crossover_rate <= 1:
                break
            else:
                print("请输入0-1之间的数")
        except ValueError:
            print("请输入有效的小数")
    
    # 获取迭代次数
    while True:
        generations_input = input(f"迭代次数 (默认: {default_generations}): ")
        if not generations_input:
            generations = default_generations
            break
        try:
            generations = int(generations_input)
            if generations > 0:
                break
            else:
                print("请输入正整数")
        except ValueError:
            print("请输入有效的整数")
    
    # 获取早停代数
    while True:
        early_stopping_input = input(f"早停代数 (默认: {default_early_stopping}): ")
        if not early_stopping_input:
            early_stopping = default_early_stopping
            break
        try:
            early_stopping = int(early_stopping_input)
            if early_stopping > 0:
                break
            else:
                print("请输入正整数")
        except ValueError:
            print("请输入有效的整数")
    
    return {
        'population_size': population_size,
        'mutation_rate': mutation_rate,
        'crossover_rate': crossover_rate,
        'generations': generations,
        'early_stopping': early_stopping
    }

# 获取用户输入的参数
user_params = get_user_parameters()
print(f"\n使用参数: 种群大小={user_params['population_size']}, 变异率={user_params['mutation_rate']}, 交叉率={user_params['crossover_rate']}, 迭代次数={user_params['generations']}, 早停代数={user_params['early_stopping']}")

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
    
    # 使用用户参数运行算法
    try:
        # 创建遗传算法实例
        ga = GeneticAlgorithm(
            tsp, 
            population_size=user_params['population_size'], 
            mutation_rate=user_params['mutation_rate'], 
            crossover_rate=user_params['crossover_rate'], 
            generations=user_params['generations'],
            early_stopping=user_params['early_stopping']
        )
        
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