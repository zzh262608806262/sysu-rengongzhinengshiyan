"""
购房预测分类任务 - MLP多层感知机实现

"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_preprocessing import load_data, preprocess_data, get_feature_names
from mlp_model import MLP
from train import train, plot_loss_curve, plot_predictions, plot_data_visualization, evaluate_model


# 设置随机种子
np.random.seed(42)

# 路径配置
DATA_DIR = r'e:\AI-Experiments-Repo\bp神经网络\house-prices-advanced-regression-techniques'
OUTPUT_DIR = r'e:\AI-Experiments-Repo\bp神经网络\code'
TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_PATH = os.path.join(DATA_DIR, 'test.csv')


def main():
    print("=" * 60)
    print("购房预测分类任务 - MLP多层感知机")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1] 加载数据...")
    train_df, test_df = load_data(TRAIN_PATH, TEST_PATH)
    print(f"    训练集大小: {train_df.shape}")
    print(f"    测试集大小: {test_df.shape}")

    # 2. 数据预处理
    print("\n[2] 数据预处理...")
    X_train, y_train, X_test, train_ids, test_ids, scaler = preprocess_data(train_df, test_df)
    print(f"    特征数量: {X_train.shape[1]}")
    print(f"    训练样本数: {X_train.shape[0]}")
    print(f"    测试样本数: {X_test.shape[0]}")

    # 3. 划分训练集和验证集
    print("\n[3] 划分训练集和验证集...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    print(f"    训练集: {X_tr.shape[0]} 样本")
    print(f"    验证集: {X_val.shape[0]} 样本")

    # 4. 对目标变量进行标准化（防止数值溢出）
    print("\n[4] 标准化目标变量...")
    y_scaler = StandardScaler()
    y_tr_scaled = y_scaler.fit_transform(y_tr.reshape(-1, 1)).flatten()
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).flatten()
    print(f"    目标变量均值: {y_scaler.mean_[0]:.2f}")
    print(f"    目标变量标准差: {y_scaler.scale_[0]:.2f}")

    # 5. 数据可视化
    print("\n[5] 生成数据可视化图...")
    feature_names = train_df.columns.tolist()
    feature_names = [col for col in feature_names if col not in ['Id', 'SalePrice']]
    plot_data_visualization(
        X_train, y_train, feature_names,
        save_path=os.path.join(OUTPUT_DIR, 'data_visualization.png')
    )
    print(f"    已保存: data_visualization.png")

    # 6. 创建MLP模型
    print("\n[6] 创建MLP模型...")
    input_size = X_train.shape[1]
    hidden_sizes = [128, 64, 32]  # 三层隐藏层
    model = MLP(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=1,
        activation='relu'
    )
    print(f"    输入层大小: {input_size}")
    print(f"    隐藏层大小: {hidden_sizes}")
    print(f"    输出层大小: 1")
    print(f"    激活函数: ReLU")

    # 打印模型结构
    print("\n    模型结构:")
    print(f"    Input({input_size}) -> ", end="")
    for i, hs in enumerate(hidden_sizes):
        print(f"Hidden{i+1}({hs}) -> ", end="")
    print("Output(1)")

    # 7. 训练模型（使用标准化的目标变量）
    print("\n[7] 训练模型...")
    print("    参数设置:")
    print(f"    - 训练轮数: 1000")
    print(f"    - 学习率: 0.001")
    print(f"    - 批量大小: 32")
    print(f"    - 梯度裁剪阈值: 1.0")

    history = train(
        model, X_tr, y_tr_scaled,  # 使用标准化的目标变量
        X_val=X_val, y_val=y_val_scaled,  # 使用标准化的目标变量
        epochs=1000,
        learning_rate=0.001,
        batch_size=32,
        print_every=100,
        clip_threshold=1.0
    )

    # 8. 绘制损失曲线
    print("\n[8] 生成损失曲线图...")
    plot_loss_curve(
        history,
        save_path=os.path.join(OUTPUT_DIR, 'loss_curve.png')
    )
    print(f"    已保存: loss_curve.png")

    # 9. 在验证集上评估（需要反标准化预测结果）
    print("\n[9] 模型评估...")
    val_pred_scaled = model.predict(X_val)
    # 反标准化预测结果
    val_pred = y_scaler.inverse_transform(val_pred_scaled.reshape(-1, 1)).flatten()
    metrics = evaluate_model(model, X_val, y_val_scaled)
    mse_original = np.mean((y_val - val_pred) ** 2)
    rmse_original = np.sqrt(mse_original)
    ss_res = np.sum((y_val - val_pred) ** 2)
    ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    print(f"    验证集 MSE (原始尺度):  {mse_original:.4f}")
    print(f"    验证集 RMSE (原始尺度): {rmse_original:.4f}")
    print(f"    验证集 R²:   {r2:.4f}")

    # 10. 绘制预测对比图（使用原始尺度）
    print("\n[10] 生成预测对比图...")
    plot_predictions(
        y_val, val_pred,
        save_path=os.path.join(OUTPUT_DIR, 'predictions.png')
    )
    print(f"    已保存: predictions.png")

    # 11. 在测试集上预测
    print("\n[11] 测试集预测...")
    test_pred_scaled = model.predict(X_test)
    test_predictions = y_scaler.inverse_transform(test_pred_scaled.reshape(-1, 1)).flatten()
    print(f"    测试集预测完成，共 {len(test_predictions)} 个样本")

    # 保存预测结果
    results_path = os.path.join(OUTPUT_DIR, 'test_predictions.npy')
    np.save(results_path, test_predictions)
    print(f"    已保存预测结果: test_predictions.npy")

    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print(f"\n生成的文件:")
    print(f"  - data_visualization.png (数据可视化)")
    print(f"  - loss_curve.png (损失曲线)")
    print(f"  - predictions.png (预测对比)")
    print(f"  - test_predictions.npy (测试集预测结果)")


if __name__ == '__main__':
    main()
