import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import multiprocessing
import joblib
import os
import time
import threading
import json

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 使用系统中已有的中文字体
plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'SimSun']

# 导入必要的模块
from jsb_gym.environmets import bvrdog
from gym.spaces import Discrete
from jsb_gym.environmets.config import BVRGym_PPODog
from jsb_gym.TAU.config import aim_dog_BVRGym, f16_dog_BVRGym
from enum import Enum


from fuzzy_model import FuzzyClassifier

# 保持与训练文件一致的枚举
class Maneuvers(Enum):
    Evasive = 0
    Crank = 1

# 复用离散动作映射类
class DiscreteActionMapping:
    def __init__(self):
        # 定义49种离散动作组合（7x7网格：方向x高度）
        self.num_actions = 49
        self.action_space = Discrete(49)
        
        # 定义方向值（7个级别）
        self.heading_values = [-1.0, -0.67, -0.33, 0.0, 0.33, 0.67, 1.0]
        # 定义高度值（7个级别）
        self.altitude_values = [-1.0, -0.67, -0.33, 0.0, 0.33, 0.67, 1.0]
        
    def discrete_to_continuous(self, discrete_action, fixed_thrust=0.0):
        """Convert discrete action to continuous action [heading, altitude, thrust]"""
        # 计算行列索引
        row = discrete_action // 7  # 高度索引
        col = discrete_action % 7   # 方向索引
        
        heading = self.heading_values[col]
        altitude = self.altitude_values[6-row]  # 反转行索引使得上升为正向
        
        # 返回连续动作向量，固定推力
        return np.array([heading, altitude, fixed_thrust])

def test_fuzzy_classifier(model_dir, num_episodes=10, visualize=False):
    """
    Test decision making using fuzzy classifier in Dog scenario
    """
    # 设置随机种子
    np.random.seed(42)
    random.seed(42)
    
    # 加载模型和预处理器
    fuzzy_classifier = joblib.load(os.path.join(model_dir, 'fuzzy_classifier.pkl'))
    dimensionality_reducer = joblib.load(os.path.join(model_dir, 'pca.pkl'))  # 可能是ICA或None
    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
    
    # 加载配置信息
    config_path = os.path.join(model_dir, 'config.json')
    use_ica = False
    original_feature_names = None
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            use_ica = config.get('use_ica', False)
            original_feature_names = config.get('original_feature_names')
    
    # 设置预处理器信息到模糊分类器
    if original_feature_names is None:
        original_feature_names = ['dis', 'azim_h', 'azim_v', 
                                 'heading_sin', 'heading_cos', 
                                 'aim_dis', 'aim_azim_h', 'aim_azim_v']
    fuzzy_classifier.set_preprocessors(scaler, dimensionality_reducer, original_feature_names)
    
    print("Successfully loaded model and preprocessors")
    print(f"Using ICA: {use_ica}")
    
    # 设置环境参数
    args = {
        'track': 'Dog',
        'vizualize': visualize,
        'cpu_cores': 1,
        'discrete_actions': True
    }
    
    # 创建环境
    env = bvrdog.BVRDog(BVRGym_PPODog, args, aim_dog_BVRGym, f16_dog_BVRGym)
    env.sim_time_sec_max = 600  # 10分钟模拟时间上限
    
    # 创建动作映射器
    action_mapper = DiscreteActionMapping()
    action_dim = action_mapper.num_actions
    
    # 初始化性能指标
    rewards = []
    episode_lengths = []
    blue_wins = 0
    red_wins = 0
    timeouts = 0
    both_dead = 0
    
    # 记录导弹命中统计
    blue_missile_hit = {'aim1': 0, 'aim2': 0}
    red_missile_hit = {'aim1r': 0, 'aim2r': 0}
    
    # 记录动作分布
    action_counts = np.zeros(action_dim)
    
    # 测试循环
    maneuver = Maneuvers.Evasive
    
    for ep in tqdm(range(num_episodes), desc="Fuzzy classifier testing progress"):
        print(f"\nStarting episode {ep+1}/{num_episodes}")
        
        # 强制完全重置环境
        try:
            env.close()
        except:
            pass
        
        env = bvrdog.BVRDog(BVRGym_PPODog, args, aim_dog_BVRGym, f16_dog_BVRGym)
        env.sim_time_sec_max = 600
        
        # 初始化环境
        state = env.reset()
        env.f16_alive = True
        env.f16r_alive = True
        env.reward_max_time = False
        
        # 重置导弹命中状态
        for missile in ['aim1', 'aim2']:
            if missile in env.aim_block:
                env.aim_block[missile].target_hit = False
        for missile in ['aim1r', 'aim2r']:
            if missile in env.aimr_block:
                env.aimr_block[missile].target_hit = False
        
        action = np.zeros(3)
        action[2] = 0.0  # 固定推力为0以避免飞机坠毁
        
        done = False
        episode_reward = 0
        episode_length = 0
        stagnation_counter = 0
        last_state = None
        
        # 设置回合最大步数
        max_steps = 1000
        
        # 单个回合循环
        while not done and episode_length < max_steps:
            # 检测状态是否停滞
            if last_state is not None and np.array_equal(state, last_state):
                stagnation_counter += 1
                if stagnation_counter > 50:
                    print(f"Episode {ep+1}: Detected state stagnation, forcing termination")
                    done = True
                    timeouts += 1
                    break
            else:
                stagnation_counter = 0
            
            last_state = state.copy()
            
            # 使用模糊分类器预测动作
            # 1. 对状态进行预处理
            state_reshaped = state.reshape(1, -1)  # 确保是2D数组
            
            # 2. 应用与训练相同的预处理
            state_scaled = scaler.transform(state_reshaped)  # 标准化
            
            # 3. 可选的降维处理
            if use_ica and dimensionality_reducer is not None:
                state_transformed = dimensionality_reducer.transform(state_scaled)  # ICA降维
            else:
                state_transformed = state_scaled  # 不使用降维
            
            # 4. 使用模糊分类器预测动作
            label_map = joblib.load(os.path.join(model_dir, 'label_map.pkl'))
            discrete_action_encoded, _ = fuzzy_classifier.predict(state_transformed)
            discrete_action = label_map[discrete_action_encoded[0]]
            
            # 记录动作分布
            action_counts[discrete_action] += 1
            
            # 转换为连续动作
            action = action_mapper.discrete_to_continuous(discrete_action, fixed_thrust=0.0)
            
            # 在采取动作前检查飞机存活状态
            if not env.f16_alive and not env.f16r_alive:
                print(f"Episode {ep+1}: Detected both aircraft destroyed at step {episode_length}")
                done = True
                both_dead += 1
                break
            
            # 执行步骤
            try:
                next_state, reward, done, info = env.step(action, action_type=maneuver.value, blue_armed=True, red_armed=True)
                
                # 立即检查环境是否结束回合
                if not env.f16_alive or not env.f16r_alive or env.reward_max_time:
                    done = True
            except Exception as e:
                print(f"Episode {ep+1} error at step {episode_length}: {str(e)}")
                done = True
                timeouts += 1
                break
            
            # 更新状态和累积奖励
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            # 检查是否有飞机被击落
            if not env.f16_alive or not env.f16r_alive:
                print(f"Episode {ep+1} at step {episode_length}: Aircraft shot down: Blue alive={env.f16_alive}, Red alive={env.f16r_alive}")
                done = True
            
            # 如果回合持续太长，强制结束
            if episode_length >= max_steps:
                print(f"Episode {ep+1} reached maximum steps {max_steps}, forcing termination")
                done = True
                timeouts += 1
        
        # 记录回合结果
        rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # 分析结束原因
        outcome = "Unknown"
        if env.reward_max_time:
            timeouts += 1
            outcome = "Timeout"
        elif env.f16_alive and not env.f16r_alive:
            blue_wins += 1
            outcome = "Blue wins"
        elif not env.f16_alive and env.f16r_alive:
            red_wins += 1
            outcome = "Red wins"
        elif not env.f16_alive and not env.f16r_alive:
            both_dead += 1
            outcome = "Both dead"
        
        # 打印每回合的详细结果
        print(f"Episode {ep+1} result: {outcome}, length: {episode_length}, reward: {episode_reward:.2f}")
        print(f"Blue alive: {env.f16_alive}, Red alive: {env.f16r_alive}, timeout: {env.reward_max_time}")
        
        # 记录导弹命中情况
        if env.aim_block['aim1'].target_hit:
            blue_missile_hit['aim1'] += 1
        if env.aim_block['aim2'].target_hit:
            blue_missile_hit['aim2'] += 1
        if env.aimr_block['aim1r'].target_hit:
            red_missile_hit['aim1r'] += 1
        if env.aimr_block['aim2r'].target_hit:
            red_missile_hit['aim2r'] += 1
    
    # 打印测试结果摘要
    print(f"\nCompleted {num_episodes} episode test: Blue wins {blue_wins}, Red wins {red_wins}, Timeouts {timeouts}, Both dead {both_dead}")
    print(f"Blue win rate: {blue_wins/num_episodes*100:.2f}%, Red win rate: {red_wins/num_episodes*100:.2f}%")
    print(f"Average episode length: {np.mean(episode_lengths):.2f} steps, Average reward: {np.mean(rewards):.2f}")
    
    # 可视化结果
    visualize_fuzzy_results(blue_wins, red_wins, timeouts, both_dead, action_counts, action_mapper)
    
    # 保存更新后的规则使用统计
    rules_file = os.path.join(model_dir, 'fuzzy_rules_updated.json')
    fuzzy_classifier.save_rules_to_file(rules_file)
    
    # 生成测试后的规则使用报告
    test_report_file = os.path.join(model_dir, 'fuzzy_rules_test_report.txt')
    fuzzy_classifier.generate_rule_report(test_report_file)
    
    # 重新生成规则使用统计图
    test_plots_dir = os.path.join(model_dir, 'fuzzy_rules_test_plots')
    fuzzy_classifier._plot_rule_usage_statistics(test_plots_dir)
    
    # 打印规则使用摘要
    summary = fuzzy_classifier.get_rule_summary()
    print(f"\nFuzzy Rules Usage Summary:")
    print(f"  Total rules: {summary['total_rules']}")
    print(f"  Average usage count: {summary['average_usage']:.2f}")
    if summary['most_used_rule']:
        print(f"  Most used rule: Rule_{summary['most_used_rule'][0]}_{summary['most_used_rule'][1]}")
    if summary['least_used_rule']:
        print(f"  Least used rule: Rule_{summary['least_used_rule'][0]}_{summary['least_used_rule'][1]}")
    
    print(f"\nUpdated rule statistics saved to {rules_file}")
    print(f"Test rule report saved to {test_report_file}")
    
    return blue_wins, red_wins, timeouts, both_dead

def visualize_fuzzy_results(blue_wins, red_wins, timeouts, both_dead, action_counts, action_mapper):
    """Visualize fuzzy classifier test results"""
    # 可视化动作分布
    plt.figure(figsize=(10, 8))
    action_dist = action_counts / np.sum(action_counts) if np.sum(action_counts) > 0 else np.zeros_like(action_counts)
    action_dist = action_dist.reshape(7, 7)
    plt.imshow(action_dist, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Action selection frequency')
    plt.title('Fuzzy classifier discrete action distribution heatmap')
    plt.xlabel('Heading action')
    plt.ylabel('Altitude action')
    plt.xticks(range(7), [f"{val:.2f}" for val in action_mapper.heading_values])
    plt.yticks(range(7), [f"{val:.2f}" for val in action_mapper.altitude_values[::-1]])
    plt.savefig('fuzzy_action_distribution.png')
    plt.close()
    
    # 创建性能饼图
    labels = ['Blue wins', 'Red wins', 'Timeout']
    sizes = [blue_wins, red_wins, timeouts]
    colors = ['blue', 'red', 'gray']
    
    if both_dead > 0:
        labels.append('Both dead')
        sizes.append(both_dead)
        colors.append('purple')
    
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Fuzzy classifier test results distribution')
    plt.savefig('fuzzy_test_results_pie.png')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--model_dir", type=str, default="dbscan_saved_model",
                       help="Directory to save model and preprocessors")
    parser.add_argument("-n", "--num_episodes", type=int, default=5,
                       help="Number of test episodes")
    parser.add_argument("-v", "--visualize", action='store_true',
                       help="Enable FlightGear visualization")
    args = parser.parse_args()
    
    # 调用测试函数
    test_fuzzy_classifier(args.model_dir, args.num_episodes, args.visualize)
    
