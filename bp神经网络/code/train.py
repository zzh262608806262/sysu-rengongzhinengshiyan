"""
训练模块和可视化
"""

import numpy as np
import matplotlib.pyplot as plt
from mlp_model import MLP


def mean_squared_error(y_true, y_pred):
    """计算均方误差"""
    return np.mean((y_true - y_pred) ** 2)


def train(model, X_train, y_train, X_val=None, y_val=None,
          epochs=1000, learning_rate=0.01, batch_size=32, print_every=100, clip_threshold=1.0):
    """
    训练MLP模型

    参数:
        model: MLP模型实例
        X_train: 训练数据
        y_train: 训练标签
        X_val: 验证数据（可选）
        y_val: 验证标签（可选）
        epochs: 训练轮数
        learning_rate: 学习率
        batch_size: 批量大小
        print_every: 打印间隔
        clip_threshold: 梯度裁剪阈值

    返回:
        history: 训练历史记录
    """
    n_samples = X_train.shape[0]
    history = {
        'train_loss': [],
        'val_loss': [] if X_val is not None else None
    }

    for epoch in range(epochs):
        # 打乱数据
        indices = np.random.permutation(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        epoch_loss = 0
        n_batches = 0

        # 小批量训练
        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            # 前向传播
            y_pred = model.forward(X_batch)

            # 反向传播（带梯度裁剪）
            model.backward(y_batch, learning_rate, clip_threshold)

            # 计算损失
            batch_loss = mean_squared_error(y_batch, y_pred)
            epoch_loss += batch_loss
            n_batches += 1

        avg_epoch_loss = epoch_loss / n_batches
        history['train_loss'].append(avg_epoch_loss)

        # 验证集损失
        if X_val is not None:
            val_pred = model.predict(X_val)
            val_loss = mean_squared_error(y_val, val_pred)
            history['val_loss'].append(val_loss)

        # 打印进度
        if (epoch + 1) % print_every == 0:
            if X_val is not None:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_epoch_loss:.4f}")

    return history


def plot_loss_curve(history, save_path=None):
    """绘制损失曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
    if history['val_loss'] is not None:
        plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_predictions(y_true, y_pred, save_path=None):
    """绘制预测值与真实值对比图"""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('True Values', fontsize=12)
    plt.ylabel('Predictions', fontsize=12)
    plt.title('True vs Predicted Values', fontsize=14)
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_data_visualization(X, y, feature_names=None, save_path=None):
    """
    绘制数据可视化图
    由于特征维度很高，我们选择几个重要特征进行可视化
    """
    # 选择数值型特征进行可视化
    if X.shape[1] > 4:
        # 选择前4个特征进行可视化
        X_vis = X[:, :4]
        if feature_names is not None:
            feature_names = feature_names[:4]
    else:
        X_vis = X

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i in range(min(4, X_vis.shape[1])):
        axes[i].scatter(X_vis[:, i], y, alpha=0.5)
        if feature_names is not None:
            axes[i].set_xlabel(feature_names[i], fontsize=10)
        axes[i].set_ylabel('SalePrice', fontsize=10)
        axes[i].set_title(f'Feature {i+1} vs SalePrice', fontsize=12)
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def evaluate_model(model, X_test, y_test):
    """评估模型性能"""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    # 计算R²分数
    ss_res = np.sum((y_test - y_pred.flatten()) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    return {
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'predictions': y_pred.flatten()
    }
