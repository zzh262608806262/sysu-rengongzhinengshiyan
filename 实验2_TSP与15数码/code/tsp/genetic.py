import random
import numpy as np

class GeneticAlgorithm:
    def __init__(self, tsp, population_size=100, mutation_rate=0.01, crossover_rate=0.8, generations=1000, early_stopping=50):
        self.tsp = tsp
        # 确保种群大小为偶数
        self.population_size = population_size if population_size % 2 == 0 else population_size + 1
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations
        self.early_stopping = early_stopping  # 早停代数
        self.population = self.initialize_population()
    
    def initialize_population(self):
        """初始化种群"""
        population = []
        for _ in range(self.population_size):
            route = list(range(self.tsp.num_cities))
            random.shuffle(route)
            population.append(route)
        return population
    
    def calculate_fitness(self, route):
        """计算适应度（距离的倒数）"""
        distance = self.tsp.calculate_route_distance(route)
        return 1 / distance
    
    def selection(self):
        """轮盘赌选择"""
        fitness_values = [self.calculate_fitness(route) for route in self.population]
        total_fitness = sum(fitness_values)
        probabilities = [f / total_fitness for f in fitness_values]
        selected = random.choices(self.population, weights=probabilities, k=self.population_size)
        return selected
    
    def crossover(self, parent1, parent2):
        """部分映射交叉（PMX）"""
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        
        # 创建子代
        child = [None] * size
        # 复制父1的基因片段
        child[start:end+1] = parent1[start:end+1]
        
        # 创建父1基因位置映射，提高查找效率
        parent1_map = {gene: idx for idx, gene in enumerate(parent1)}
        
        # 处理父2的基因
        for i in range(size):
            if i < start or i > end:
                gene = parent2[i]
                while gene in child:
                    # 找到基因在父1中的位置（O(1)）
                    idx = parent1_map[gene]
                    # 用父2中对应位置的基因替换
                    gene = parent2[idx]
                child[i] = gene
        
        return child
    
    def mutate(self, route):
        """交换变异"""
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(route)), 2)
            route[i], route[j] = route[j], route[i]
        return route
    
    def evolve(self):
        """进化过程"""
        best_route = None
        best_distance = float('inf')
        no_improvement_generations = 0  # 记录无改善的代数
        
        for generation in range(self.generations):
            # 选择
            selected = self.selection()
            # 交叉和变异
            new_population = []
            for i in range(0, self.population_size, 2):
                parent1 = selected[i]
                parent2 = selected[i+1]  # 由于种群大小为偶数，不需要处理边界情况
                
                if random.random() < self.crossover_rate:
                    child1 = self.crossover(parent1, parent2)
                    child2 = self.crossover(parent2, parent1)
                else:
                    child1 = parent1.copy()
                    child2 = parent2.copy()
                
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            # 精英主义：保留上一代最好的个体
            if best_route is not None:
                # 计算新种群中每个个体的距离
                distances = [self.tsp.calculate_route_distance(route) for route in new_population]
                # 找到最差的个体索引
                worst_idx = distances.index(max(distances))
                # 用最好的个体替换最差的个体
                new_population[worst_idx] = best_route.copy()
            
            self.population = new_population
            
            # 评估当前种群
            current_best = min(self.population, key=lambda x: self.tsp.calculate_route_distance(x))
            current_distance = self.tsp.calculate_route_distance(current_best)
            
            if current_distance < best_distance:
                best_distance = current_distance
                best_route = current_best.copy()
                no_improvement_generations = 0  # 重置无改善代数
            else:
                no_improvement_generations += 1  # 增加无改善代数
            
            # 早停检查
            if no_improvement_generations >= self.early_stopping:
                print(f"Early stopping at generation {generation+1}: No improvement for {self.early_stopping} generations")
                break
            
            # 每100代打印一次
            if (generation + 1) % 100 == 0:
                print(f"Generation {generation+1}: Best distance = {best_distance:.2f}")
        
        return best_route, best_distance