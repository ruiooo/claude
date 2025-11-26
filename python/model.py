"""
model.py - DQN神经网络模型（支持CUDA加速）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Tuple


class DQN(nn.Module):
    """
    深度Q网络 (Deep Q-Network)
    使用全连接层处理状态并输出每个动作的Q值
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list):
        """
        初始化DQN网络

        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            hidden_dims: 隐藏层维度列表
        """
        super(DQN, self).__init__()

        # 构建网络层
        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))  # 防止过拟合
            input_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(input_dim, action_dim))

        self.network = nn.Sequential(*layers)

        # 初始化权重（Xavier初始化）
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """初始化网络权重"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            state: 状态张量 [batch_size, state_dim]

        Returns:
            Q值 [batch_size, action_dim]
        """
        return self.network(state)


class DQNAgent:
    """
    DQN智能体，负责训练和决策
    """

    def __init__(self, state_dim: int, action_dim: int, config: dict, device: str = 'cuda'):
        """
        初始化DQN智能体

        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            config: 配置字典
            device: 运行设备（cuda或cpu）
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 网络配置
        self.gamma = config['gamma']
        self.epsilon = config['epsilon_start']
        self.epsilon_end = config['epsilon_end']
        self.epsilon_decay = config['epsilon_decay']
        self.target_update_freq = config['target_update_freq']

        # 创建策略网络和目标网络
        hidden_dims = config['hidden_dims']
        self.policy_net = DQN(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_net = DQN(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # 目标网络不训练

        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(),
                                    lr=config['learning_rate'])

        # 学习率调度器（可选）
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                    step_size=1000,
                                                    gamma=0.95)

        # 训练计数器
        self.train_step = 0

        print(f"DQN模型初始化完成 - 设备: {self.device}")
        print(f"模型参数量: {sum(p.numel() for p in self.policy_net.parameters()):,}")

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        选择动作（epsilon-greedy策略）

        Args:
            state: 当前状态
            training: 是否在训练模式

        Returns:
            选择的动作索引
        """
        # epsilon-greedy探索
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim)

        # 利用策略网络选择动作
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            action = q_values.argmax(dim=1).item()

        return action

    def train(self, batch: Tuple) -> float:
        """
        训练网络（使用GPU加速）

        Args:
            batch: (states, actions, rewards, next_states, dones)

        Returns:
            损失值
        """
        states, actions, rewards, next_states, dones = batch

        # 转换为张量并移到GPU
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # 当前Q值
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # 使用目标网络计算下一个状态的最大Q值
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # 计算损失（Huber Loss更稳定）
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)

        # 反向传播和优化（充分利用GPU）
        self.optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)

        self.optimizer.step()
        self.scheduler.step()

        self.train_step += 1

        # 定期更新目标网络
        if self.train_step % self.target_update_freq == 0:
            self.update_target_network()

        return loss.item()

    def update_target_network(self):
        """更新目标网络"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_epsilon(self):
        """更新探索率"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path: str):
        """
        保存模型

        Args:
            path: 保存路径
        """
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_step': self.train_step,
        }, path)
        print(f"模型已保存到: {path}")

    def load(self, path: str):
        """
        加载模型（支持继续训练）

        Args:
            path: 模型路径
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.train_step = checkpoint['train_step']

        print(f"模型已从 {path} 加载")
        print(f"训练步数: {self.train_step}, Epsilon: {self.epsilon:.4f}")

    def get_model_version(self) -> int:
        """获取模型版本（基于训练步数）"""
        return self.train_step // 1000
