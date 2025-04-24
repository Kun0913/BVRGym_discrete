# jsb_gym/RL/discrete_actor.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class DiscreteActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DiscreteActor, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        return self.actor(state)

class DiscreteCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super(DiscreteCritic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state):
        return self.critic(state)

class DiscretePPO:
    def __init__(self, state_dim, action_dim, conf_ppo, use_gpu=False):
        self.device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
        self.lr = conf_ppo['lr']
        self.betas = conf_ppo['betas']
        self.gamma = conf_ppo['gamma']
        self.eps_clip = conf_ppo['eps_clip']
        self.K_epochs = conf_ppo['K_epochs']
        
        self.policy = DiscreteActor(state_dim, action_dim).to(self.device)
        self.policy_old = DiscreteActor(state_dim, action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.critic = DiscreteCritic(state_dim).to(self.device)
        
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.parameters(), 'lr': self.lr},
            {'params': self.critic.parameters(), 'lr': self.lr}
        ], lr=self.lr, betas=self.betas)
    
    def set_device(self, use_gpu=False):
        self.device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
        self.policy = self.policy.to(self.device)
        self.policy_old = self.policy_old.to(self.device)
        self.critic = self.critic.to(self.device)
    
    def select_action(self, state, memory):
        state = torch.FloatTensor(state).to(self.device)
        
        with torch.no_grad():
            action_probs = self.policy_old(state)
            dist = Categorical(action_probs)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
        
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        
        return action.item()
    
    def update(self, memory, to_tensor=False, use_gpu=True):
        # 设置更新的设备
        old_device = self.device
        self.set_device(use_gpu)
        
        # 转换记忆到适当的张量
        if to_tensor:
            # 检查并转换为tensor
            converted_states = []
            for state in memory.states:
                if isinstance(state, np.ndarray):
                    converted_states.append(torch.FloatTensor(state))
                else:
                    converted_states.append(state)
            
            converted_actions = []
            for action in memory.actions:
                if isinstance(action, np.ndarray):
                    converted_actions.append(torch.LongTensor(action))
                else:
                    converted_actions.append(action)
            
            converted_logprobs = []
            for logprob in memory.logprobs:
                if isinstance(logprob, np.ndarray):
                    converted_logprobs.append(torch.FloatTensor(logprob))
                else:
                    converted_logprobs.append(logprob)
            
            old_states = torch.stack(converted_states).to(self.device).detach()
            old_actions = torch.stack(converted_actions).to(self.device).detach()
            old_logprobs = torch.stack(converted_logprobs).to(self.device).detach()
            rewards = torch.tensor(memory.rewards, dtype=torch.float32).to(self.device)
            is_terminals = torch.tensor(memory.is_terminals, dtype=torch.bool).to(self.device)
        else:
            old_states = memory.states
            old_actions = memory.actions
            old_logprobs = memory.logprobs
            rewards = torch.tensor(memory.rewards).to(self.device)
            is_terminals = memory.is_terminals
        
        # 计算折扣回报
        returns = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
        
        returns = torch.tensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        
        # PPO更新
        for _ in range(self.K_epochs):
            # 计算新的动作概率
            action_probs = self.policy(old_states)
            dist = Categorical(action_probs)
            new_logprobs = dist.log_prob(old_actions)
            entropy = dist.entropy()
            
            # 计算状态值函数
            state_values = self.critic(old_states).squeeze()
            
            # 计算比率和裁剪
            ratios = torch.exp(new_logprobs - old_logprobs)
            advantages = returns - state_values.detach()
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            # 最终损失
            loss = -torch.min(surr1, surr2) + 0.5 * F.mse_loss(state_values, returns) - 0.01 * entropy
            
            # 更新
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        # 更新旧策略
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # 恢复原来的设备
        self.set_device(old_device == torch.device("cuda:0"))
