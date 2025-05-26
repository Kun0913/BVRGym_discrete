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
from real_time_visualizer import init_visualizer, update_visualization, start_visualization
from battle_logger import BattleLogger, create_post_analysis_visualizer

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
        """将离散动作转换为连续动作 [heading, altitude, thrust]"""
        # 计算行列索引
        row = discrete_action // 7  # 高度索引
        col = discrete_action % 7   # 方向索引
        
        heading = self.heading_values[col]
        altitude = self.altitude_values[6-row]  # 反转行索引使得上升为正向
        
        # 返回连续动作向量，固定推力
        return np.array([heading, altitude, fixed_thrust])

def test_fuzzy_classifier(model_dir, num_episodes=10, visualize=False):
    """
    使用模糊分类器在Dog场景中进行决策测试
    """
    # 设置随机种子
    np.random.seed(42)
    random.seed(42)
    
    # 初始化日志记录器
    logger = BattleLogger(f"battle_log_{int(time.time())}.csv")
    
    # 加载模型和预处理器
    fuzzy_classifier = joblib.load(os.path.join(model_dir, 'fuzzy_classifier.pkl'))
    pca = joblib.load(os.path.join(model_dir, 'pca.pkl'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
    
    print("成功加载模型和预处理器")
    
    # 初始化可视化器
    if visualize:
        vis = init_visualizer()
        # 在单独线程中启动可视化
        vis_thread = threading.Thread(target=start_visualization, daemon=True)
        vis_thread.start()
        time.sleep(2)  # 等待可视化窗口启动
    
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
    
    for ep in tqdm(range(num_episodes), desc="模糊分类器测试进度"):
        print(f"\n开始回合 {ep+1}/{num_episodes}")
        
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
                    print(f"回合 {ep+1}: 检测到状态停滞，强制终止")
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
            state_transformed = pca.transform(state_scaled)  # 降维
            
            # 3. 使用模糊分类器预测动作
            label_map = joblib.load(os.path.join(model_dir, 'label_map.pkl'))
            discrete_action_encoded, _ = fuzzy_classifier.predict(state_transformed)
            discrete_action = label_map[discrete_action_encoded[0]]
            
            # 记录动作分布
            action_counts[discrete_action] += 1
            
            # 转换为连续动作
            action = action_mapper.discrete_to_continuous(discrete_action, fixed_thrust=0.0)
            
            # 在采取动作前检查飞机存活状态
            if not env.f16_alive and not env.f16r_alive:
                print(f"回合 {ep+1}: 在步骤 {episode_length} 检测到双方飞机阵亡")
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
                print(f"回合 {ep+1} 在步骤 {episode_length} 出错: {str(e)}")
                done = True
                timeouts += 1
                break
            
            # 更新状态和累积奖励
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            # 检查是否有飞机被击落
            if not env.f16_alive or not env.f16r_alive:
                print(f"回合 {ep+1} 在步骤 {episode_length}: 飞机被击落: 蓝方存活={env.f16_alive}, 红方存活={env.f16r_alive}")
                done = True
            
            # 如果回合持续太长，强制结束
            if episode_length >= max_steps:
                print(f"回合 {ep+1} 达到最大步数 {max_steps}，强制终止")
                done = True
                timeouts += 1
            
            # 更新可视化
            if visualize:
                # 准备导弹信息
                missiles = {}
                missiles.update(env.aim_block)
                missiles.update(env.aimr_block)
                
                # 准备状态信息
                info_text = f"回合: {ep+1}/{num_episodes}\n"
                info_text += f"步数: {episode_length}\n"
                info_text += f"蓝方存活: {env.f16_alive}\n"
                info_text += f"红方存活: {env.f16r_alive}\n"
                info_text += f"当前奖励: {episode_reward:.2f}"
                
                # 更新可视化
                update_visualization(env.f16, env.f16r, missiles, info_text)
                
                # 控制更新频率
                time.sleep(0.05)
            
            # 记录战斗数据（每5步记录一次，减少开销）
            if episode_length % 5 == 0:
                try:
                    missiles = {}
                    missiles.update(env.aim_block)
                    missiles.update(env.aimr_block)
                    
                    logger.log_step(
                        episode=ep+1,
                        step=episode_length,
                        blue_aircraft=env.f16,
                        red_aircraft=env.f16r,
                        missiles=missiles,
                        episode_reward=episode_reward,
                        blue_alive=env.f16_alive,
                        red_alive=env.f16r_alive,
                        info=f"动作:{discrete_action}"
                    )
                    
                    # 每50步打印一次摘要
                    if episode_length % 50 == 0:
                        logger.print_summary(
                            ep+1, episode_length, env.f16, env.f16r,
                            env.f16_alive, env.f16r_alive, episode_reward
                        )
                        
                except Exception as e:
                    print(f"日志记录失败: {e}")
        
        # 记录回合结果
        rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # 分析结束原因
        outcome = "未知"
        if env.reward_max_time:
            timeouts += 1
            outcome = "超时"
        elif env.f16_alive and not env.f16r_alive:
            blue_wins += 1
            outcome = "蓝方胜利"
        elif not env.f16_alive and env.f16r_alive:
            red_wins += 1
            outcome = "红方胜利"
        elif not env.f16_alive and not env.f16r_alive:
            both_dead += 1
            outcome = "双方阵亡"
        
        # 打印每回合的详细结果
        print(f"回合 {ep+1} 结果: {outcome}, 长度: {episode_length}, 奖励: {episode_reward:.2f}")
        print(f"蓝方存活: {env.f16_alive}, 红方存活: {env.f16r_alive}, 超时: {env.reward_max_time}")
        
        # 记录导弹命中情况
        if env.aim_block['aim1'].target_hit:
            blue_missile_hit['aim1'] += 1
        if env.aim_block['aim2'].target_hit:
            blue_missile_hit['aim2'] += 1
        if env.aimr_block['aim1r'].target_hit:
            red_missile_hit['aim1r'] += 1
        if env.aimr_block['aim2r'].target_hit:
            red_missile_hit['aim2r'] += 1
        
        # 记录回合结束状态
        try:
            missiles = {}
            missiles.update(env.aim_block)
            missiles.update(env.aimr_block)
            
            logger.log_step(
                episode=ep+1,
                step=episode_length,
                blue_aircraft=env.f16,
                red_aircraft=env.f16r,
                missiles=missiles,
                episode_reward=episode_reward,
                blue_alive=env.f16_alive,
                red_alive=env.f16r_alive,
                info=f"回合结束:{outcome}"
            )
        except:
            pass
        # time.sleep(10)
    
    # 打印测试结果摘要
    print(f"\n完成 {num_episodes} 回合测试: 蓝方胜 {blue_wins}, 红方胜 {red_wins}, 超时 {timeouts}, 双方阵亡 {both_dead}")
    print(f"蓝方胜率: {blue_wins/num_episodes*100:.2f}%, 红方胜率: {red_wins/num_episodes*100:.2f}%")
    print(f"平均回合长度: {np.mean(episode_lengths):.2f} 步, 平均奖励: {np.mean(rewards):.2f}")
    
    # 可视化结果
    visualize_fuzzy_results(blue_wins, red_wins, timeouts, both_dead, action_counts, action_mapper)
    
    # 生成事后分析
    if visualize:
        print("\n生成战斗分析图...")
        create_post_analysis_visualizer(logger.log_file)
    
    return blue_wins, red_wins, timeouts, both_dead

def visualize_fuzzy_results(blue_wins, red_wins, timeouts, both_dead, action_counts, action_mapper):
    """可视化模糊分类器的测试结果"""
    # 可视化动作分布
    plt.figure(figsize=(10, 8))
    action_dist = action_counts / np.sum(action_counts) if np.sum(action_counts) > 0 else np.zeros_like(action_counts)
    action_dist = action_dist.reshape(7, 7)
    plt.imshow(action_dist, cmap='hot', interpolation='nearest')
    plt.colorbar(label='动作选择频率')
    plt.title('模糊分类器离散动作分布热图')
    plt.xlabel('航向动作')
    plt.ylabel('高度动作')
    plt.xticks(range(7), [f"{val:.2f}" for val in action_mapper.heading_values])
    plt.yticks(range(7), [f"{val:.2f}" for val in action_mapper.altitude_values[::-1]])
    plt.savefig('fuzzy_action_distribution.png')
    plt.close()
    
    # 创建性能饼图
    labels = ['蓝方胜', '红方胜', '超时']
    sizes = [blue_wins, red_wins, timeouts]
    colors = ['blue', 'red', 'gray']
    
    if both_dead > 0:
        labels.append('双方阵亡')
        sizes.append(both_dead)
        colors.append('purple')
    
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('模糊分类器测试结果分布')
    plt.savefig('fuzzy_test_results_pie.png')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--model_dir", type=str, default="dbscan_saved_model",
                       help="保存模型和预处理器的目录")
    parser.add_argument("-n", "--num_episodes", type=int, default=50,
                       help="测试回合数")
    parser.add_argument("-v", "--visualize", action='store_true',
                       help="启用FlightGear可视化")
    args = parser.parse_args()
    
    # 调用测试函数
    test_fuzzy_classifier(args.model_dir, args.num_episodes, args.visualize)
    
