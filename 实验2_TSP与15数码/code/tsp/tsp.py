import math

class TSP:
    def __init__(self, cities):
        self.cities = cities
        self.num_cities = len(cities)
        self.distance_matrix = self.calculate_distance_matrix()
    
    def calculate_distance_matrix(self):
        """计算城市间的距离矩阵"""
        matrix = []
        for i in range(self.num_cities):
            row = []
            for j in range(self.num_cities):
                if i == j:
                    row.append(0)
                else:
                    x1, y1 = self.cities[i]
                    x2, y2 = self.cities[j]
                    distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                    row.append(distance)
            matrix.append(row)
        return matrix
    
    def calculate_route_distance(self, route):
        """计算给定路线的总距离"""
        total_distance = 0
        for i in range(self.num_cities - 1):
            total_distance += self.distance_matrix[route[i]][route[i+1]]
        # 回到起点
        total_distance += self.distance_matrix[route[-1]][route[0]]
        return total_distance
    
    def read_cities_from_file(self, file_path):
        """从文件中读取城市坐标"""
        cities = []
        read_coords = False
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line == 'NODE_COORD_SECTION':
                    read_coords = True
                    continue
                elif line == 'EOF':
                    break
                elif read_coords or not any(keyword in line.upper() for keyword in ['NAME:', 'COMMENT', 'TYPE:', 'DIMENSION:', 'EDGE_WEIGHT_TYPE:']):
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            x = float(parts[1])
                            y = float(parts[2])
                            cities.append((x, y))
                        except ValueError:
                            pass
        self.cities = cities
        self.num_cities = len(cities)
        self.distance_matrix = self.calculate_distance_matrix()
        return cities