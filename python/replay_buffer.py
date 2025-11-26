"""
replay_buffer.py - 经验回放缓冲区（优化GPU批处理）
"""

import numpy as np
from collections import deque
import random


class ReplayBuffer:
    """
    经验回放缓冲区
    用于存储和采样经验以训练DQN
    """

    def __init__(self, capacity: int, state_dim: int):
        """
        初始化经验回放缓冲区

        Args:
            capacity: 缓冲区容量
            state_dim: 状态维度
        """
        self.capacity = capacity
        self.state_dim = state_dim

        # 使用numpy数组以提高效率（便于批处理到GPU）
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        self.position = 0
        self.size = 0

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        """
        添加经验到缓冲区

        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
        """
        idx = self.position

        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = 1.0 if done else 0.0

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        """
        随机采样一批经验

        Args:
            batch_size: 批大小

        Returns:
            (states, actions, rewards, next_states, dones)
        """
        # 随机采样索引
        indices = np.random.choice(self.size, batch_size, replace=False)

        # 返回批数据
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )

    def __len__(self):
        """返回缓冲区当前大小"""
        return self.size

    def is_ready(self, min_size: int) -> bool:
        """
        检查缓冲区是否准备好进行训练

        Args:
            min_size: 最小经验数量

        Returns:
            是否准备好
        """
        return self.size >= min_size

    def clear(self):
        """清空缓冲区"""
        self.position = 0
        self.size = 0


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    优先经验回放缓冲区（可选）
    根据TD误差优先采样重要的经验
    """

    def __init__(self, capacity: int, state_dim: int, alpha: float = 0.6, beta: float = 0.4):
        """
        初始化优先经验回放缓冲区

        Args:
            capacity: 缓冲区容量
            state_dim: 状态维度
            alpha: 优先级指数
            beta: 重要性采样指数
        """
        super().__init__(capacity, state_dim)

        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001

        # 优先级存储
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        """添加经验（使用最大优先级）"""
        super().push(state, action, reward, next_state, done)

        # 新经验使用最大优先级
        idx = (self.position - 1) % self.capacity
        self.priorities[idx] = self.max_priority

    def sample(self, batch_size: int):
        """
        根据优先级采样

        Args:
            batch_size: 批大小

        Returns:
            (states, actions, rewards, next_states, dones, indices, weights)
        """
        # 计算采样概率
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # 采样
        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)

        # 计算重要性采样权重
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        self.beta = min(1.0, self.beta + self.beta_increment)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            indices,
            weights
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """
        更新优先级

        Args:
            indices: 经验索引
            priorities: 新优先级
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
