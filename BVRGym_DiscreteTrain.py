import argparse, time
from jsb_gym.environmets import evasive, bvrdog
import numpy as np
from enum import Enum
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
import multiprocessing
from jsb_gym.RL.ppo import Memory, PPO
from jsb_gym.RL.discrete_actor import DiscretePPO  # 导入新添加的离散PPO类
from jsb_gym.environmets.config import BVRGym_PPO1, BVRGym_PPO2, BVRGym_PPODog
from numpy.random import seed
from jsb_gym.TAU.config import aim_evs_BVRGym, f16_evs_BVRGym, aim_dog_BVRGym, f16_dog_BVRGym

import datetime
from gym.spaces import Discrete

def init_pool():
    seed()

class Maneuvers(Enum):
    Evasive = 0
    Crank = 1

# 保留现有的离散动作映射类
class DiscreteActionMapping:
    def __init__(self):
        # 定义49种离散动作组合（7x7网格：方向x高度）
        self.num_actions = 49
        self.action_space = Discrete(self.num_actions)
        
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

def runPPO(args):
    if args['track'] == 'M1':
        from jsb_gym.RL.config.ppo_evs_PPO1 import conf_ppo
        
        env = evasive.Evasive(BVRGym_PPO1, args, aim_evs_BVRGym, f16_evs_BVRGym)
        torch_save = 'jsb_gym/logs/RL/M1.pth'
        state_scale = 1
        # 使用连续动作空间
        use_discrete = False
    elif args['track'] == 'M2':
        from jsb_gym.RL.config.ppo_evs_PPO2 import conf_ppo
        env = evasive.Evasive(BVRGym_PPO2, args, aim_evs_BVRGym, f16_evs_BVRGym)
        torch_save = 'jsb_gym/logs/RL/M2.pth'
        state_scale = 2
        # 使用连续动作空间
        use_discrete = False
    elif args['track'] in ['Dog', 'DogR']:
        from jsb_gym.RL.config.ppo_evs_PPO_BVRDog import conf_ppo
        if args['track'] == 'Dog':
            env = bvrdog.BVRDog(BVRGym_PPODog, args, aim_dog_BVRGym, f16_dog_BVRGym)
            torch_save = 'jsb_gym/logs/RL/Dog/'
        else:
            env = bvrdog.BVRDog(BVRGym_PPODog, args, aim_dog_BVRGym, f16_dog_BVRGym)
            torch_save = 'jsb_gym/logs/RL/DogR.pth'
        state_scale = 1
        # 使用离散动作空间（默认）
        use_discrete = args.get('discrete_actions', True)

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter('runs/' + current_time + '/' + args['track'] )
    state_dim = env.observation_space
    
    # 根据离散/连续动作选择不同设置
    if use_discrete and args['track'] in ['Dog', 'DogR']:
        # 创建离散动作映射器
        action_mapper = DiscreteActionMapping()
        action_dim = action_mapper.num_actions
        ppo = DiscretePPO(state_dim*state_scale, action_dim, conf_ppo, use_gpu=False)
    else:
        # 原始连续动作设置
        action_dim = env.action_space.shape[1]
        ppo = PPO(state_dim*state_scale, action_dim, conf_ppo, use_gpu=False)
    
    memory = Memory()
    pool = multiprocessing.Pool(processes=int(args['cpu_cores']), initializer=init_pool)
    
    for i_episode in range(1, args['Eps']+1):
        ppo_policy = ppo.policy.state_dict()    
        ppo_policy_old = ppo.policy_old.state_dict()
        input_data = [(args, ppo_policy, ppo_policy_old, conf_ppo, state_scale)for _ in range(args['cpu_cores'])]
        running_rewards = []
        tb_obs = []
        
        results = pool.map(train, input_data)
        for idx, tmp in enumerate(results):
            memory.actions.extend(tmp[0])
            memory.states.extend(tmp[1])
            memory.logprobs.extend(tmp[2])
            memory.rewards.extend(tmp[3])
            memory.is_terminals.extend(tmp[4])
            running_rewards.append(tmp[5])
            tb_obs.append(tmp[6])
            
        ppo.set_device(use_gpu=True)
        ppo.update(memory, to_tensor=True, use_gpu=True)
        memory.clear_memory()
        ppo.set_device(use_gpu=False)
        torch.cuda.empty_cache()
        
        writer.add_scalar("running_rewards", sum(running_rewards)/len(running_rewards), i_episode)
        tb_obs0 = None
        for i in tb_obs:
            if tb_obs0 == None:
                tb_obs0 = i
            else:
                for key in tb_obs0:
                    if key in i:
                        tb_obs0[key] += i[key]

        nr = len(tb_obs)
        for key in tb_obs0:
            tb_obs0[key] = tb_obs0[key]/nr
            writer.add_scalar(key, tb_obs0[key], i_episode)
        
        if i_episode % 500 == 0:
            # 保存模型
            torch.save(ppo.policy.state_dict(), torch_save + 'Dog'+str(i_episode) + '.pth')

    pool.close()
    pool.join()

def train(args):
    use_discrete = args[0].get('discrete_actions', True)
    
    if args[0]['track'] == 'M1':
        env = evasive.Evasive(BVRGym_PPO1, args[0], aim_evs_BVRGym, f16_evs_BVRGym)
        use_discrete = False
    elif args[0]['track'] == 'M2':
        env = evasive.Evasive(BVRGym_PPO2, args[0], aim_evs_BVRGym, f16_evs_BVRGym)
        use_discrete = False
    elif args[0]['track'] in ['Dog', 'DogR']:
        env = bvrdog.BVRDog(BVRGym_PPODog, args[0], aim_dog_BVRGym, f16_dog_BVRGym)
        # 使用离散动作空间用于Dog任务
        if use_discrete:
            action_mapper = DiscreteActionMapping()

    maneuver = Maneuvers.Evasive
    memory = Memory()
    state_dim = env.observation_space
    
    # 根据离散/连续动作选择不同设置
    if use_discrete and args[0]['track'] in ['Dog', 'DogR']:
        action_dim = action_mapper.num_actions
        ppo = DiscretePPO(state_dim*args[4], action_dim, args[3], use_gpu=False)
    else:
        action_dim = env.action_space.shape[1]
        ppo = PPO(state_dim*args[4], action_dim, args[3], use_gpu=False)

    ppo.policy.load_state_dict(args[1])
    ppo.policy_old.load_state_dict(args[2])

    ppo.policy.eval()
    ppo.policy_old.eval()
    running_reward = 0.0
    
    # 统计离散动作分布（仅用于Dog任务）
    if use_discrete and args[0]['track'] in ['Dog', 'DogR']:
        action_counts = np.zeros(action_mapper.num_actions)

    for i_episode in range(1, args[0]['eps']+1):
        action = np.zeros(3)
        # 根据任务初始化状态
        if args[0]['track'] == 'M1':
            state_block = env.reset(True, True)
            state = state_block['aim1']
            # 最大推力 
            action[2] = 1

        elif args[0]['track'] == 'M2':
            state_block = env.reset(True, True)
            state = np.concatenate((state_block['aim1'][0], state_block['aim2'][0]))
            # 最大推力
            action[2] = 1
        
        elif args[0]['track'] in ['Dog', 'DogR']:
            state = env.reset()
            # 固定推力0.0以避免飞机坠毁
            action[2] = 0.0
        
        done = False
        while not done:
            if use_discrete and args[0]['track'] in ['Dog', 'DogR']:
                # 离散动作选择
                discrete_action = ppo.select_action(state, memory)
                # 统计动作分布
                action_counts[discrete_action] += 1
                # 转换为连续动作
                action = action_mapper.discrete_to_continuous(discrete_action, fixed_thrust=0.0)
            else:
                # 原始连续动作选择
                act = ppo.select_action(state, memory)
                action[0] = act[0]
                action[1] = act[1]

            # 执行环境步进
            if args[0]['track'] == 'M1':
                state_block, reward, done, _ = env.step(action, action_type=maneuver.value)
                state = state_block['aim1']
            
            elif args[0]['track'] == 'M2':
                state_block, reward, done, _ = env.step(action, action_type=maneuver.value)
                state = np.concatenate((state_block['aim1'], state_block['aim2']))
            
            elif args[0]['track'] == 'Dog':
                state, reward, done, _ = env.step(action, action_type=maneuver.value, blue_armed=True, red_armed=True)

            elif args[0]['track'] == 'DogR':
                state, reward, done, _ = env.step(action, action_type=maneuver.value, blue_armed=False, red_armed=True)

            memory.rewards.append(reward)
            memory.is_terminals.append(done)

        running_reward += reward 
    
    running_reward = running_reward/args[0]['eps']
    
    # 准备Tensorboard观察数据
    if args[0]['track'] == 'M1':
        tb_obs = {}
    elif args[0]['track'] == 'M2':
        tb_obs = {}            
    elif args[0]['track'] in ['Dog', 'DogR']:
        tb_obs = get_tb_obs_dog(env)
        # 添加动作分布数据
        if use_discrete:
            total_actions = np.sum(action_counts)
            if total_actions > 0:
                action_dist = action_counts / total_actions
                # 创建7x7网格形式的动作分布
                action_grid = action_dist.reshape(7, 7)
                
                # 添加动作分布统计
                for i in range(7):
                    for j in range(7):
                        action_idx = i * 7 + j
                        tb_obs[f'action_{i}_{j}'] = action_dist[action_idx]
                
                # 计算动作熵
                action_entropy = -np.sum(action_dist * np.log(action_dist + 1e-10))
                tb_obs['action_entropy'] = action_entropy

    # # 返回训练结果
    # actions = [i.detach().numpy() if isinstance(i, torch.Tensor) else i for i in memory.actions]
    # states = [i.detach().numpy() if isinstance(i, torch.Tensor) else i for i in memory.states]
    # logprobs = [i.detach().numpy() if isinstance(i, torch.Tensor) else i for i in memory.logprobs]
    # rewards = [i for i in memory.rewards]
    # is_terminals = [i for i in memory.is_terminals]  
     # 返回训练结果时不转换为numpy数组
    actions = memory.actions  # 不要转换为numpy
    states = memory.states    # 不要转换为numpy
    logprobs = memory.logprobs  # 不要转换为numpy
    rewards = memory.rewards
    is_terminals = memory.is_terminals   
    return [actions, states, logprobs, rewards, is_terminals, running_reward, tb_obs]

# 保持原有函数不变
def get_tb_obs_dog(env):
    tb_obs = {}
    tb_obs['Blue_ground'] = env.reward_f16_hit_ground
    tb_obs['Red_ground'] = env.reward_f16r_hit_ground
    tb_obs['maxTime'] = env.reward_max_time

    tb_obs['Blue_alive'] = env.f16_alive
    tb_obs['Red_alive'] = env.f16r_alive


    tb_obs['aim1_active'] = env.aim_block['aim1'].active
    tb_obs['aim1_alive'] = env.aim_block['aim1'].alive
    tb_obs['aim1_target_lost'] = env.aim_block['aim1'].target_lost
    tb_obs['aim1_target_hit'] = env.aim_block['aim1'].target_hit


    tb_obs['aim2_active'] = env.aim_block['aim2'].active
    tb_obs['aim2_alive'] = env.aim_block['aim2'].alive
    tb_obs['aim2_target_lost'] = env.aim_block['aim2'].target_lost
    tb_obs['aim2_target_hit'] = env.aim_block['aim2'].target_hit

    tb_obs['aim1r_active'] = env.aimr_block['aim1r'].active
    tb_obs['aim1r_alive'] = env.aimr_block['aim1r'].alive
    tb_obs['aim1r_target_lost'] = env.aimr_block['aim1r'].target_lost
    tb_obs['aim1r_target_hit'] = env.aimr_block['aim1r'].target_hit

    tb_obs['aim2r_active'] = env.aimr_block['aim2r'].active
    tb_obs['aim2r_alive'] = env.aimr_block['aim2r'].alive
    tb_obs['aim2r_target_lost'] = env.aimr_block['aim2r'].target_lost
    tb_obs['aim2r_target_hit'] = env.aimr_block['aim2r'].target_hit


    if env.aim_block['aim1'].target_lost:
        tb_obs['aim1_MD'] = env.aim_block['aim1'].position_tgt_NED_norm
        
    if env.aim_block['aim2'].target_lost:
        tb_obs['aim2_lost'] = 1
        tb_obs['aim2_MD'] = env.aim_block['aim2'].position_tgt_NED_norm
    
    if env.aimr_block['aim1r'].target_lost:
        tb_obs['aim1r_lost'] = 1
        tb_obs['aim1r_MD'] = env.aimr_block['aim1r'].position_tgt_NED_norm
    
    if env.aimr_block['aim2r'].target_lost:
        tb_obs['aim2r_lost'] = 1
        tb_obs['aim2r_MD'] = env.aimr_block['aim2r'].position_tgt_NED_norm

    return tb_obs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vizualize", action='store_true', help="Render in FG")
    parser.add_argument("-track", "--track", type=str, help="Tracks: M1, M2, Dog, DogR", default=' ')
    parser.add_argument("-cpus", "--cpu_cores", type=int, help="Number of cores to use", default=None)
    parser.add_argument("-Eps", "--Eps", type=int, help="Number of episodes for training", default=int(1e3))
    parser.add_argument("-eps", "--eps", type=int, help="Number of episodes per process", default=5)
    parser.add_argument("-disc", "--discrete_actions", action='store_true', help="Use discrete action space")
    args = vars(parser.parse_args())
    
    # 默认参数
    args['track'] = 'Dog'
    args['cpu_cores'] = 10
    args['Eps'] = 10000
    args['eps'] = 1
    args['discrete_actions'] = True  # 默认使用离散动作
    
    runPPO(args)