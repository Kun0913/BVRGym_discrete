import argparse
import numpy as np
import torch
from enum import Enum
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from numpy.random import seed
import multiprocessing
# 导入必要的模块
from jsb_gym.environmets import bvrdog
from jsb_gym.RL.discrete_actor import DiscretePPO
from jsb_gym.environmets.config import BVRGym_PPODog
from jsb_gym.TAU.config import aim_dog_BVRGym, f16_dog_BVRGym

# 保留与训练脚本一致的枚举和离散动作映射
class Maneuvers(Enum):
    Evasive = 0
    Crank = 1

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

def set_seeds(base_seed=42):
    worker_id = multiprocessing.current_process()._identity[0] if hasattr(multiprocessing.current_process(), '_identity') else 1
    worker_seed = base_seed + worker_id
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    random.seed(worker_seed)
    # 如果使用CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(worker_seed)
        torch.cuda.manual_seed_all(worker_seed)
        torch.backends.cudnn.deterministic = True
    # 然后生成环境的种子
    return worker_seed

# 从gym.spaces导入Discrete类
from gym.spaces import Discrete

def test_model(model_path, num_episodes=100, visualize=False):
    """
    测试训练好的模型在Dog场景中的性能
    
    参数:
        model_path: 训练好的模型权重路径
        num_episodes: 测试回合数
        visualize: 是否可视化（使用FlightGear）
    
    返回:
        metrics: 包含性能指标的字典
    """
    set_seeds(42)
    # 设置参数
    args = {
        'track': 'Dog',
        'vizualize': visualize,
        'cpu_cores': 10,
        'discrete_actions': True  # 使用离散动作空间
    }
    
    # 创建环境
    env = bvrdog.BVRDog(BVRGym_PPODog, args, aim_dog_BVRGym, f16_dog_BVRGym)
    # env.sim_time_sec_max = 60 * 16
    state_dim = env.observation_space
    
    # 创建动作映射器
    action_mapper = DiscreteActionMapping()
    action_dim = action_mapper.num_actions
    
    # 加载模型配置
    from jsb_gym.RL.config.ppo_evs_PPO_BVRDog import conf_ppo
    
    # 创建PPO模型
    ppo = DiscretePPO(state_dim, action_dim, conf_ppo, use_gpu=False)
    
    # 加载训练好的模型权重
    ppo.policy.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    ppo.policy_old.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    # 设置为评估模式
    ppo.policy.eval()
    ppo.policy_old.eval()
    
    # 初始化性能指标
    rewards = []
    episode_lengths = []
    blue_wins = 0
    red_wins = 0
    timeouts = 0
    
    # 记录导弹命中统计
    blue_missile_hit = {'aim1': 0, 'aim2': 0}
    red_missile_hit = {'aim1r': 0, 'aim2r': 0}
    
    # 记录动作分布
    action_counts = np.zeros(action_dim)
    
    # 创建文件来记录状态动作对
    all_sa_pairs_file = open('all_state_action_pairs.txt', 'w')
    win_sa_pairs_file = open('win_state_action_pairs.txt', 'w')
    
    # 测试循环
    maneuver = Maneuvers.Evasive
    
    for ep in tqdm(range(num_episodes), desc="测试进度"):
        action = np.zeros(3)
        
        # 初始化环境
        state = env.reset()
        action[2] = 0.0  # 固定推力为0以避免飞机坠毁
        
        done = False
        episode_reward = 0
        episode_length = 0
        
        # 记录当前回合的状态动作对
        episode_sa_pairs = []
        
        # 单个回合循环
        while not done:
            # 选择动作（不需要Memory，因为只是测试）
            state_tensor = torch.FloatTensor(state)
            with torch.no_grad():
                action_probs = ppo.policy_old(state_tensor)
                dist = torch.distributions.Categorical(action_probs)
                discrete_action = dist.sample().item()
            
            # 记录动作分布
            action_counts[discrete_action] += 1
            
            # 记录状态动作对
            sa_pair = {
                'state': state.tolist(),
                'action': discrete_action
            }
            episode_sa_pairs.append(sa_pair)
            
            # 记录到全局文件，只包含状态和动作，以空格分隔
            state_str = ' '.join(map(str, state.tolist()))
            all_sa_pairs_file.write(f"{state_str} {discrete_action}\n")
            
            # 转换为连续动作
            action = action_mapper.discrete_to_continuous(discrete_action, fixed_thrust=0.0)
            
            # 执行步骤
            next_state, reward, done, info = env.step(action, action_type=maneuver.value, blue_armed=True, red_armed=True)
            
            # 更新状态和累积奖励
            state = next_state
            episode_reward += reward
            episode_length += 1
        
        # 记录回合结果
        rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # 分析结束原因
        if env.reward_max_time:
            timeouts += 1
        elif env.f16_alive and not env.f16r_alive:
            blue_wins += 1
            # 蓝方获胜，记录这个回合的状态动作对到胜利文件
            for sa_pair in episode_sa_pairs:
                state_str = ' '.join(map(str, sa_pair['state']))
                win_sa_pairs_file.write(f"{state_str} {sa_pair['action']}\n")
        elif not env.f16_alive and env.f16r_alive:
            red_wins += 1
        
        # 记录导弹命中情况
        if env.aim_block['aim1'].target_hit:
            blue_missile_hit['aim1'] += 1
        if env.aim_block['aim2'].target_hit:
            blue_missile_hit['aim2'] += 1
        if env.aimr_block['aim1r'].target_hit:
            red_missile_hit['aim1r'] += 1
        if env.aimr_block['aim2r'].target_hit:
            red_missile_hit['aim2r'] += 1
        state = env.reset()
    
    # 关闭文件
    all_sa_pairs_file.close()
    win_sa_pairs_file.close()
    
    '''每行包含一个状态向量和对应的动作，以空格分隔'''
    print(f"所有状态动作对已保存到 all_state_action_pairs.txt")
    print(f"获胜状态动作对已保存到 win_state_action_pairs.txt")
       
    # 计算综合指标
    win_rate = blue_wins / num_episodes
    average_reward = np.mean(rewards)
    average_episode_length = np.mean(episode_lengths)
    
    # 总结统计
    metrics = {
        'win_rate': win_rate,
        'blue_wins': blue_wins,
        'red_wins': red_wins,
        'timeouts': timeouts,
        'average_reward': average_reward,
        'average_episode_length': average_episode_length,
        'blue_missile_hit': blue_missile_hit,
        'red_missile_hit': red_missile_hit,
        'action_distribution': action_counts / np.sum(action_counts),
        'rewards': rewards,
        'episode_lengths': episode_lengths
    }
    
    return metrics

def visualize_metrics(metrics):
    """可视化测试指标"""
    print("\n===== 模型性能指标 =====")
    print(f"蓝方胜率: {metrics['win_rate']*100:.2f}%")
    print(f"蓝方赢得回合: {metrics['blue_wins']}")
    print(f"红方赢得回合: {metrics['red_wins']}")
    print(f"超时回合: {metrics['timeouts']}")
    print(f"平均奖励: {metrics['average_reward']:.4f}")
    print(f"平均回合长度: {metrics['average_episode_length']:.2f}")
    print(f"蓝方导弹命中率: aim1={metrics['blue_missile_hit']['aim1']}, aim2={metrics['blue_missile_hit']['aim2']}")
    print(f"红方导弹命中率: aim1r={metrics['red_missile_hit']['aim1r']}, aim2r={metrics['red_missile_hit']['aim2r']}")
    
    # 可视化动作分布
    plt.figure(figsize=(10, 8))
    action_dist = metrics['action_distribution'].reshape(7, 7)
    plt.imshow(action_dist, cmap='hot', interpolation='nearest')
    plt.colorbar(label='action selection frequency')
    plt.title('discrete action distribution heatmap')
    plt.xlabel('heading action')
    plt.ylabel('altitude action')
    plt.xticks(range(7), [f"{val:.2f}" for val in DiscreteActionMapping().heading_values])
    plt.yticks(range(7), [f"{val:.2f}" for val in DiscreteActionMapping().altitude_values[::-1]])
    plt.savefig('action_distribution.png')
    plt.close()
    
    # 创建性能饼图
    labels = ['blue win', 'red win', 'timeout']
    sizes = [metrics['blue_wins'], metrics['red_wins'], metrics['timeouts']]
    colors = ['blue', 'red', 'gray']
    
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('test results distribution')
    plt.savefig('test_results_pie.png')
    plt.close()

def parallel_test_model(model_path, num_episodes=100, num_cores=4):
    # 确保num_cores至少为1
    num_cores = max(1, num_cores)
    pool = multiprocessing.Pool(processes=num_cores, initializer=set_seeds)
    
    # 分配测试任务
    episodes_per_core = num_episodes // num_cores
    input_data = [(model_path, episodes_per_core) for _ in range(num_cores)]
    
    # 并行执行测试
    results = pool.map(test_model_worker, input_data)
    
    # 整合结果
    combined_metrics = combine_metrics(results)
    return combined_metrics

def test_model_worker(args):
    model_path, episodes = args
    
    # 使用与训练相同的环境配置
    env_args = {
        'track': 'Dog',
        'vizualize': False,
        'cpu_cores': 1,
        'discrete_actions': True
    }
    
    # 设置独立随机种子
    set_seeds(np.random.randint(10000))
    
    # 调用test_model函数并返回其结果
    metrics = test_model(model_path, episodes, visualize=False)
    return metrics

def init_pool():
    seed()  # 随机初始化每个进程的种子

def combine_metrics(results):
    # 初始化组合后的指标
    combined = {
        'blue_wins': 0,
        'red_wins': 0,
        'timeouts': 0,
        'rewards': [],
        'episode_lengths': [],
        'blue_missile_hit': {'aim1': 0, 'aim2': 0},
        'red_missile_hit': {'aim1r': 0, 'aim2r': 0},
        'action_distribution': np.zeros(49)
    }
    
    # 累加各个进程的结果
    total_episodes = 0
    for res in results:
        combined['blue_wins'] += res['blue_wins']
        combined['red_wins'] += res['red_wins']
        combined['timeouts'] += res['timeouts']
        combined['rewards'].extend(res.get('rewards', []))
        combined['episode_lengths'].extend(res.get('episode_lengths', []))
        
        # 累加导弹命中数据
        combined['blue_missile_hit']['aim1'] += res['blue_missile_hit']['aim1']
        combined['blue_missile_hit']['aim2'] += res['blue_missile_hit']['aim2']
        combined['red_missile_hit']['aim1r'] += res['red_missile_hit']['aim1r']
        combined['red_missile_hit']['aim2r'] += res['red_missile_hit']['aim2r']
        
        # 累加动作分布
        combined['action_distribution'] += res['action_distribution'] * np.sum(res['action_distribution'])
        
        # 计算总回合数
        total_episodes += res['blue_wins'] + res['red_wins'] + res['timeouts']
    
    # 计算综合指标
    combined['win_rate'] = combined['blue_wins'] / total_episodes if total_episodes > 0 else 0
    combined['average_reward'] = np.mean(combined['rewards']) if combined['rewards'] else 0
    combined['average_episode_length'] = np.mean(combined['episode_lengths']) if combined['episode_lengths'] else 0
    
    # 归一化动作分布
    if np.sum(combined['action_distribution']) > 0:
        combined['action_distribution'] = combined['action_distribution'] / np.sum(combined['action_distribution'])
    
    return combined

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="jsb_gym/logs/RL/Dog/Dog500.pth", 
                        help="训练好的模型路径")
    parser.add_argument("-n", "--num_episodes", type=int, default=20, 
                        help="测试回合数")
    parser.add_argument("-v", "--visualize", action='store_true', 
                        help="是否在FlightGear中可视化")
    parser.add_argument("-c", "--cores", type=int, default=10,
                        help="并行测试的CPU核心数")
    args = parser.parse_args()
    
    # 使用正确的参数顺序调用
    metrics = parallel_test_model(args.model, args.num_episodes, args.cores)
    
    # 可视化结果
    visualize_metrics(metrics)
    
    print("\n测试完成! 结果已保存为图片文件。") 