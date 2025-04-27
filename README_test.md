# BVRGym离散动作PPO模型测试工具

这个测试工具用于评估在Dog场景中训练的离散动作PPO模型的性能。

## 功能

- 加载已训练的PPO模型参数
- 在Dog场景中运行多个测试回合
- 收集并计算性能指标：
  - 蓝方胜率
  - 红蓝双方获胜次数与超时次数
  - 平均奖励与回合长度
  - 导弹命中统计
  - 离散动作分布
- 生成可视化结果图表
- 比较不同训练阶段模型的性能

## 使用方法

### 1. 单个模型测试 (BVRGym_DiscreteTest.py)

#### 命令行参数

```bash
python BVRGym_DiscreteTest.py [-m MODEL_PATH] [-n NUM_EPISODES] [-v]
```

参数说明:
- `-m`, `--model`: 训练好的模型参数文件路径（默认: "jsb_gym/logs/RL/Dog/Dog1500.pth"）
- `-n`, `--num_episodes`: 测试回合数（默认: 100）
- `-v`, `--visualize`: 是否在FlightGear中可视化（默认: False）

#### 示例

1. 使用默认参数测试模型（Dog1500.pth，100回合）:

```bash
python BVRGym_DiscreteTest.py
```

2. 测试指定模型，运行50回合:

```bash
python BVRGym_DiscreteTest.py -m jsb_gym/logs/RL/Dog/Dog1000.pth -n 50
```

3. 使用FlightGear可视化（需要先启动FlightGear）:

```bash
python BVRGym_DiscreteTest.py -v
```

### 2. 多个模型比较 (compare_models.py)

此脚本可以比较多个训练阶段模型的性能，包括胜率、奖励、回合长度和导弹命中等指标。

#### 命令行参数

```bash
python compare_models.py -m MODEL_PATH1 MODEL_PATH2 ... [-n NUM_EPISODES]
```

参数说明:
- `-m`, `--models`: 要比较的模型参数文件路径列表（必需）
- `-n`, `--num_episodes`: 每个模型测试的回合数（默认: 50）

#### 示例

比较不同训练阶段的模型，每个模型测试30回合:

```bash
python compare_models.py -m jsb_gym/logs/RL/Dog/Dog500.pth jsb_gym/logs/RL/Dog/Dog1000.pth jsb_gym/logs/RL/Dog/Dog1500.pth -n 30
```

## 输出结果

### 单个模型测试输出

运行BVRGym_DiscreteTest.py后，脚本会生成以下输出:

1. 终端输出的性能指标摘要
2. `action_distribution.png`: 离散动作选择分布的热力图
3. `test_results_pie.png`: 测试结果分布饼图（蓝方胜利、红方胜利、超时）

### 多个模型比较输出

运行compare_models.py后，脚本会生成以下输出:

1. 终端输出的各模型性能指标汇总表格
2. `model_win_rate_comparison.png`: 不同模型胜率对比图
3. `model_reward_comparison.png`: 不同模型平均奖励对比图
4. `model_episode_length_comparison.png`: 不同模型平均回合长度对比图
5. `model_missile_hits_comparison.png`: 不同模型导弹命中次数对比图

## 注意事项

- 确保已安装所有必要的依赖包（numpy, torch, matplotlib, tqdm等）
- 模型测试与训练使用相同的环境配置和离散动作映射
- 使用可视化功能需要正确配置FlightGear
- 比较多个大型模型时，可能需要较长时间，建议适当减少每个模型的测试回合数 