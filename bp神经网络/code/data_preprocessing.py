"""
数据预处理模块
负责加载和预处理房价预测数据
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_data(train_path, test_path):
    """加载训练集和测试集"""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def preprocess_data(train_df, test_df):
    """
    数据预处理：
    1. 分离特征和目标变量
    2. 处理缺失值
    3. 编码分类变量
    4. 标准化数值特征
    """
    # 保存Id和目标变量
    train_ids = train_df['Id'].values
    test_ids = test_df['Id'].values
    y_train = train_df['SalePrice'].values

    # 删除Id和目标变量
    train_df = train_df.drop(['Id', 'SalePrice'], axis=1)
    test_df = test_df.drop(['Id'], axis=1)

    # 合并数据进行统一预处理
    n_train = len(train_df)
    all_data = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    # 分离数值特征和分类特征
    numeric_cols = all_data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = all_data.select_dtypes(include=['object']).columns.tolist()

    # 处理缺失值
    # 数值特征：用均值填充
    for col in numeric_cols:
        all_data[col] = all_data[col].fillna(all_data[col].median())

    # 分类特征：用众数填充
    for col in categorical_cols:
        all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

    # 标签编码分类变量
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        all_data[col] = le.fit_transform(all_data[col].astype(str))
        label_encoders[col] = le

    # 分离回训练集和测试集
    X_train = all_data.iloc[:n_train].values
    X_test = all_data.iloc[n_train:].values

    # 标准化数值特征（包含所有特征）
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, train_ids, test_ids, scaler


def get_feature_names(train_path):
    """获取特征名称"""
    train_df = pd.read_csv(train_path)
    feature_names = [col for col in train_df.columns if col not in ['Id', 'SalePrice']]
    return feature_names
