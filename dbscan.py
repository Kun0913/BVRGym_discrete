import os
import joblib
import json
import random
import numpy as np
import pandas as pd
import argparse

from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from sklearn.cluster import DBSCAN
# 从skfuzzy中导入模糊规则库
import skfuzzy as fuzz
from sklearn.base import BaseEstimator, ClassifierMixin
# 优化算法
from pyswarm import pso

from fuzzy_model import FuzzyClassifier

def main():
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='Train fuzzy classifier with optional ICA')
    parser.add_argument('--use_ica', action='store_true', 
                       help='Use ICA for dimensionality reduction')
    parser.add_argument('--ica_components', type=int, default=6,
                       help='Number of ICA components (default: 6)')
    parser.add_argument('--data_file', type=str, 
                       default='all_win_state_action_pairs - 50000.txt',
                       help='Path to data file')
    parser.add_argument('--model_dir', type=str, default='dbscan_saved_model',
                       help='Directory to save model')
    parser.add_argument('--min_samples_per_label', type=int, default=133,
                       help='Minimum samples per label to keep (default: 133)')
    
    args = parser.parse_args()
    
    data_file_path = args.data_file
    model_save_dir = args.model_dir
    use_ica = args.use_ica
    ica_components = args.ica_components
    min_samples = args.min_samples_per_label
    
    print(f"Configuration:")
    print(f"  Data file: {data_file_path}")
    print(f"  Model directory: {model_save_dir}")
    print(f"  Use ICA: {use_ica}")
    if use_ica:
        print(f"  ICA components: {ica_components}")
    print(f"  Min samples per label: {min_samples}")
    print()
    
    data = pd.read_csv(data_file_path, header=None, delim_whitespace=True)

    # 消除重复值，并输出消除前后的数据条数
    print("消除重复值前数据条数:", len(data))
    data.drop_duplicates(inplace=True)
    print("消除重复值后数据条数:", len(data))

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print(X.shape)
    print("共有多少个不同的动作标签:")
    print(len(np.unique(y)))
    print("每个动作标签对应的样本数量:")
    print(pd.Series(y).value_counts())

    # 去除数据集中样本数量少于n的标签对应的数据
    n = min_samples
    # 创建一个掩码来过滤数据
    mask = np.ones(len(y), dtype=bool)
    for label in np.unique(y):
        if pd.Series(y).value_counts()[label] < n:
            # 更新掩码，将要过滤掉的样本标记为False
            mask[y == label] = False

    # 同时应用掩码到X和y
    X = X[mask]
    y = y[mask]

    # 验证X和y的长度是否匹配
    print(f"过滤后的样本数: {X.shape[0]}")

    # 对标签进行编码
    label_encoder = LabelEncoder()
    # 对标签进行编码并转换为一维
    y = label_encoder.fit_transform(y).ravel()
    label_map = np.array(label_encoder.classes_, dtype=int)  # 原始类别顺序
    
    # 创建模型保存目录
    os.makedirs(model_save_dir, exist_ok=True)
    joblib.dump(label_map, f'{model_save_dir}/label_map.pkl')
    
    # 标准化数据
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 可选的ICA降维
    dimensionality_reducer = None
    if use_ica:
        print(f"\n使用ICA进行降维，目标维度: {ica_components}")
        from sklearn.decomposition import FastICA
        dimensionality_reducer = FastICA(n_components=ica_components, random_state=13, max_iter=500)
        X = dimensionality_reducer.fit_transform(X)
        
        # 获取混合矩阵（形状为 [n_features, n_components]）
        mixing_matrix = dimensionality_reducer.mixing_

        # 敌机距离 敌机方位角(水平) 敌机方位角(垂直) 
        # 我方航向(正弦) 我方航向(余弦)
        # 红方导弹距离 红方导弹方位角(水平) 红方导弹方位角(垂直)
        feature_names = ['dis', 'azim_h', 'azim_v', 
                         'heading_sin', 'heading_cos', 
                         'aim_dis', 'aim_azim_h', 'aim_azim_v']

        print("混合矩阵（每列对应一个ICA成分的物理意义）:")
        for comp_idx in range(mixing_matrix.shape[1]):
            print(f"\nComponent {comp_idx}:")
            for feat_idx, weight in enumerate(mixing_matrix[:, comp_idx]):
                print(f"  {feature_names[feat_idx]:<6}: {weight:.3f}")
    else:
        print("\n不使用降维，直接使用原始特征")
        print(f"特征维度: {X.shape[1]}")

    # 使用交叉验证来评估分类器的准确率
    fuzzy_classifier = FuzzyClassifier(eps=0.5, min_samples=5, num_mfs=3)

    from sklearn.model_selection import train_test_split
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
    fuzzy_classifier.fit(X, y)

    # 在验证集上进行预测
    y_pred, _ = fuzzy_classifier.predict(X_val)

    # 输出分类报告
    print("\n分类报告:")
    print(classification_report(y_val, y_pred))

    # 保存训练好的模型
    joblib.dump(fuzzy_classifier, f'{model_save_dir}/fuzzy_classifier.pkl')
    # 同时保存预处理器
    if use_ica:
        joblib.dump(dimensionality_reducer, f'{model_save_dir}/pca.pkl')  # 保持文件名一致性
    else:
        # 如果不使用ICA，保存一个None对象或者恒等变换
        joblib.dump(None, f'{model_save_dir}/pca.pkl')
    
    joblib.dump(scaler, f'{model_save_dir}/scaler.pkl')
    joblib.dump(label_encoder, f'{model_save_dir}/label_encoder.pkl')
    
    # 保存配置信息
    config = {
        'use_ica': use_ica,
        'ica_components': ica_components if use_ica else None,
        'original_features': X.shape[1] if not use_ica else scaler.n_features_in_,
        'final_features': X.shape[1]
    }
    with open(f'{model_save_dir}/config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n模型和预处理器已保存到 {model_save_dir} 目录")
    print(f"配置信息已保存到 {model_save_dir}/config.json")

if __name__ == '__main__':
    main()