"""
多层感知机（MLP）模型
使用NumPy从零实现
"""

import numpy as np


class MLP:
    """
    多层感知机模型
    支持任意深度的隐藏层
    """

    def __init__(self, input_size, hidden_sizes, output_size=1, activation='relu'):
        """
        初始化MLP

        参数:
            input_size: 输入层大小
            hidden_sizes: 隐藏层大小列表，如 [64, 32] 表示两层隐藏层
            output_size: 输出层大小
            activation: 激活函数类型 ('relu', 'sigmoid', 'tanh')
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation = activation

        # 初始化权重和偏置（使用He初始化）
        self.weights = []
        self.biases = []

        # 输入层 -> 第一层隐藏层
        prev_size = input_size
        for hidden_size in hidden_sizes:
            # He初始化：适合ReLU激活函数
            std = np.sqrt(2.0 / prev_size)
            self.weights.append(np.random.randn(prev_size, hidden_size) * std)
            self.biases.append(np.zeros((1, hidden_size)))
            prev_size = hidden_size

        # 最后一层隐藏层 -> 输出层（Xavier初始化用于输出层）
        std = np.sqrt(2.0 / prev_size) if activation == 'relu' else np.sqrt(1.0 / prev_size)
        self.weights.append(np.random.randn(prev_size, output_size) * 0.01)
        self.biases.append(np.zeros((1, output_size)))

    def _activate(self, z):
        """激活函数"""
        if self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        elif self.activation == 'tanh':
            return np.tanh(z)
        else:
            return z

    def _activate_derivative(self, z):
        """激活函数的导数"""
        if self.activation == 'relu':
            return (z > 0).astype(float)
        elif self.activation == 'sigmoid':
            s = self._activate(z)
            return s * (1 - s)
        elif self.activation == 'tanh':
            return 1 - np.tanh(z) ** 2
        else:
            return np.ones_like(z)

    def forward(self, X):
        """
        前向传播

        参数:
            X: 输入数据，形状为 (batch_size, input_size)

        返回:
            output: 模型输出
            cache: 用于反向传播的缓存
        """
        self.cache = {'a0': X}
        a = X

        # 隐藏层
        for i in range(len(self.weights) - 1):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            a = self._activate(z)
            self.cache[f'z{i+1}'] = z
            self.cache[f'a{i+1}'] = a

        # 输出层（线性输出，用于回归）
        z_out = np.dot(a, self.weights[-1]) + self.biases[-1]
        self.cache[f'z{len(self.weights)}'] = z_out
        self.cache[f'a{len(self.weights)}'] = z_out

        return z_out

    def backward(self, y, learning_rate, clip_threshold=1.0):
        """
        反向传播

        参数:
            y: 真实标签，形状为 (batch_size, 1)
            learning_rate: 学习率
            clip_threshold: 梯度裁剪阈值
        """
        m = y.shape[0]
        n_layers = len(self.weights)

        # 输出层误差 (MSE损失函数的导数)
        # dL/dz_out = (a_out - y)
        dz_out = self.cache[f'a{n_layers}'] - y.reshape(-1, 1)

        # 梯度裁剪
        dz_out = np.clip(dz_out, -clip_threshold, clip_threshold)

        # 反向传播
        for i in range(n_layers - 1, -1, -1):
            a_prev = self.cache[f'a{i}'] if i > 0 else self.cache['a0']

            # 梯度
            dW = np.dot(a_prev.T, dz_out) / m
            db = np.mean(dz_out, axis=0, keepdims=True)

            # 梯度裁剪
            dW = np.clip(dW, -clip_threshold, clip_threshold)
            db = np.clip(db, -clip_threshold, clip_threshold)

            # 更新权重和偏置
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * db

            # 计算前一层的误差（用于链式法则）
            if i > 0:
                dz_prev = np.dot(dz_out, self.weights[i].T) * self._activate_derivative(self.cache[f'z{i}'])
                # 梯度裁剪
                dz_prev = np.clip(dz_prev, -clip_threshold, clip_threshold)
                dz_out = dz_prev

    def predict(self, X):
        """预测"""
        return self.forward(X)

    def get_weights(self):
        """获取所有权重和偏置"""
        return self.weights, self.biases

    def set_weights(self, weights, biases):
        """设置权重和偏置"""
        self.weights = weights
        self.biases = biases
