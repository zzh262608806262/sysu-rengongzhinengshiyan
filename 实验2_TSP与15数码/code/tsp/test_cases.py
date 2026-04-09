# TSP测试用例

# 数据集路径
data_dir = "data"

# 测试数据集列表
test_datasets = [
    ("Djibouti (38 cities)", f"{data_dir}/dj38.tsp"),
    ("Western Sahara (29 cities)", f"{data_dir}/wi29.tsp")
]

# 测试参数设置
test_parameters = [
    {"population_size": 50, "generations": 100, "mutation_rate": 0.01},
    {"population_size": 100, "generations": 200, "mutation_rate": 0.01},
    {"population_size": 150, "generations": 300, "mutation_rate": 0.01}
]