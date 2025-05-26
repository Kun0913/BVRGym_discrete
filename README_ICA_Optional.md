# 模糊分类器 - 可选ICA降维功能

本项目已修改为支持可选的ICA（独立成分分析）降维功能。您可以选择使用或不使用ICA来训练模糊分类器。

## 主要修改

### 1. dbscan.py 修改
- 添加了命令行参数支持
- 使ICA降维成为可选功能
- 保存配置信息到JSON文件
- 支持自定义参数

### 2. test.py 修改
- 自动检测模型是否使用了ICA
- 兼容有ICA和无ICA的模型
- 添加了json导入

## 使用方法

### 训练模型

#### 不使用ICA训练（推荐用于较小数据集）
```bash
python dbscan.py --model_dir models_no_ica
```

#### 使用ICA训练（推荐用于高维数据）
```bash
python dbscan.py --use_ica --ica_components 6 --model_dir models_with_ica
```

#### 自定义参数训练
```bash
python dbscan.py \
    --use_ica \
    --ica_components 4 \
    --model_dir my_model \
    --data_file my_data.txt \
    --min_samples_per_label 100
```

### 测试模型

测试脚本会自动检测模型配置：

```bash
# 测试不使用ICA的模型
python test.py --model_dir models_no_ica --num_episodes 50

# 测试使用ICA的模型  
python test.py --model_dir models_with_ica --num_episodes 50
```

### 运行示例

使用提供的示例脚本：

```bash
python run_examples.py
```

## 命令行参数

### dbscan.py 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--use_ica` | flag | False | 启用ICA降维 |
| `--ica_components` | int | 6 | ICA组件数量 |
| `--data_file` | str | 'all_win_state_action_pairs - 50000.txt' | 数据文件路径 |
| `--model_dir` | str | 'dbscan_saved_model' | 模型保存目录 |
| `--min_samples_per_label` | int | 133 | 每个标签的最小样本数 |

### test.py 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model_dir` | str | 'dbscan_saved_model' | 模型目录 |
| `--num_episodes` | int | 50 | 测试回合数 |
| `--visualize` | flag | False | 启用FlightGear可视化 |

## 配置文件

训练完成后，会在模型目录中生成 `config.json` 文件，包含以下信息：

```json
{
  "use_ica": true,
  "ica_components": 6,
  "original_features": 8,
  "final_features": 6
}
```

## ICA vs 无ICA 对比

### 使用ICA的优势
- **降维效果**: 减少特征维度，提高计算效率
- **特征解耦**: 分离独立的信号成分
- **噪声减少**: 可能提高模型的泛化能力
- **适用场景**: 高维数据，特征间存在混合关系

### 不使用ICA的优势
- **保留信息**: 保持所有原始特征信息
- **简单直接**: 无需额外的降维步骤
- **易于解释**: 特征含义更直观
- **适用场景**: 低维数据，特征已经很好分离

## 文件结构

```
project/
├── dbscan.py                    # 主训练脚本（已修改）
├── test.py                      # 测试脚本（已修改）
├── run_examples.py              # 使用示例脚本（新增）
├── README_ICA_Optional.md       # 本说明文件（新增）
├── fuzzy_model.py               # 模糊分类器实现
├── models_no_ica/               # 不使用ICA的模型目录
│   ├── fuzzy_classifier.pkl
│   ├── scaler.pkl
│   ├── pca.pkl                  # 存储None
│   ├── label_encoder.pkl
│   ├── label_map.pkl
│   └── config.json
└── models_with_ica/             # 使用ICA的模型目录
    ├── fuzzy_classifier.pkl
    ├── scaler.pkl
    ├── pca.pkl                  # 存储ICA对象
    ├── label_encoder.pkl
    ├── label_map.pkl
    └── config.json
```

## 注意事项

1. **向后兼容**: 修改后的代码兼容之前训练的模型
2. **配置检测**: 如果没有config.json文件，默认假设不使用ICA
3. **文件命名**: 为保持兼容性，ICA对象仍保存为'pca.pkl'
4. **内存使用**: 使用ICA可能会减少内存使用，特别是对于高维数据

## 性能建议

- **小数据集** (< 10000 样本): 建议不使用ICA
- **中等数据集** (10000-100000 样本): 可以尝试两种方式并比较
- **大数据集** (> 100000 样本): 建议使用ICA降维
- **高维特征** (> 20 维): 建议使用ICA
- **低维特征** (< 10 维): 建议不使用ICA

## 故障排除

### 常见问题

1. **ImportError**: 确保安装了所有依赖包
2. **FileNotFoundError**: 检查数据文件路径是否正确
3. **配置不匹配**: 删除旧的模型文件重新训练

### 依赖包
```bash
pip install scikit-learn numpy pandas joblib tqdm matplotlib skfuzzy pyswarm
``` 