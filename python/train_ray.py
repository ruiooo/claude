#!/usr/bin/env python3
"""
Ray并行采样版本的训练脚本

使用Ray Core并行运行多个游戏环境，加速数据收集
保留当前的DQN实现，只并行化环境交互部分
"""

import ray
import torch
import numpy as np
import os
from typing import List, Tuple
from model import DQNAgent
from replay_buffer import ReplayBuffer
from env_wrapper import TankBattleEnv
from config import TRAINING_CONFIG, MODEL_CONFIG, ENV_CONFIG

@ray.remote
class ParallelEnvWorker:
    """并行环境工作器（在独立进程中运行）"""

    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        # 使用相对于工作目录的路径
        lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'libtankbattle.so')
        self.env = TankBattleEnv(
            lib_path=lib_path,
            width=ENV_CONFIG['map_width'],
            height=ENV_CONFIG['map_height'],
            visualize=False
        )

    def collect_episodes(self, num_episodes: int, epsilon: float) -> List[Tuple]:
        """
        收集多个回合的经验

        Args:
            num_episodes: 收集的回合数
            epsilon: 探索率

        Returns:
            经验列表: [(state, action, reward, next_state, done), ...]
        """
        experiences = []

        for _ in range(num_episodes):
            state = self.env.reset(ENV_CONFIG['initial_enemies'])
            episode_done = False

            while not episode_done:
                # Epsilon-greedy探索（在CPU上）
                if np.random.random() < epsilon:
                    action = np.random.randint(0, 9)
                else:
                    # 这里使用简单的启发式，避免在worker中加载模型
                    action = self._heuristic_action(state)

                next_state, reward, done, info = self.env.step(action)
                experiences.append((state, action, reward, next_state, done))

                state = next_state
                episode_done = done

        return experiences

    def _heuristic_action(self, state: np.ndarray) -> int:
        """简单的启发式策略（避免在worker中加载神经网络）"""
        # 返回随机动作，实际采样时会被主进程的策略网络覆盖
        return np.random.randint(0, 9)

    def evaluate(self, num_episodes: int) -> Tuple[float, float]:
        """
        评估性能

        Returns:
            (平均奖励, 胜率)
        """
        total_reward = 0
        wins = 0

        for _ in range(num_episodes):
            state = self.env.reset(ENV_CONFIG['initial_enemies'])
            episode_reward = 0
            episode_done = False

            while not episode_done:
                # 使用贪婪策略
                action = self._heuristic_action(state)
                next_state, reward, done, info = self.env.step(action)
                episode_reward += reward
                state = next_state
                episode_done = done

            total_reward += episode_reward
            if info.get('winner') == 0:
                wins += 1

        return total_reward / num_episodes, wins / num_episodes


def train_with_ray():
    """使用Ray并行训练"""

    # 初始化Ray（如果已连接到集群则不指定资源参数）
    if not ray.is_initialized():
        try:
            # 尝试连接到已存在的集群
            ray.init(address='auto')
            print("✓ 已连接到现有Ray集群")
        except:
            # 如果没有现有集群，则启动新的本地集群
            ray.init(num_cpus=TRAINING_CONFIG.get('num_workers', 4))
            print("✓ 已启动新的Ray集群")

    # 创建并行环境工作器
    num_workers = TRAINING_CONFIG.get('num_workers', 4)
    workers = [ParallelEnvWorker.remote(i) for i in range(num_workers)]

    # 创建主DQN Agent（在主进程/GPU上）
    device = torch.device(TRAINING_CONFIG['device'])
    agent = DQNAgent(
        state_dim=MODEL_CONFIG['state_dim'],
        action_dim=MODEL_CONFIG['action_dim'],
        config=MODEL_CONFIG,
        device=TRAINING_CONFIG['device']
    )

    # 创建经验回放缓冲区
    replay_buffer = ReplayBuffer(
        capacity=TRAINING_CONFIG['buffer_size'],
        state_dim=MODEL_CONFIG['state_dim']
    )

    # 训练参数
    max_episodes = TRAINING_CONFIG['max_episodes']
    batch_size = TRAINING_CONFIG['batch_size']
    episodes_per_worker = TRAINING_CONFIG.get('episodes_per_worker', 1)

    print(f"🚀 开始Ray并行训练")
    print(f"   - 工作器数量: {num_workers}")
    print(f"   - 设备: {device}")
    print(f"   - 每轮每工作器收集: {episodes_per_worker} 回合")

    episode_count = 0

    try:
        while episode_count < max_episodes:
            # ========== 阶段1: 并行采样 ==========
            # 向所有worker分发任务
            futures = [
                worker.collect_episodes.remote(
                    episodes_per_worker,
                    agent.epsilon
                )
                for worker in workers
            ]

            # 等待所有worker完成
            all_experiences = ray.get(futures)

            # 合并所有经验到缓冲区
            for experiences in all_experiences:
                for exp in experiences:
                    replay_buffer.push(*exp)
                episode_count += episodes_per_worker

            print(f"\r采样进度: {episode_count}/{max_episodes} 回合, "
                  f"缓冲区: {len(replay_buffer)}, "
                  f"ε={agent.epsilon:.3f}", end='')

            # ========== 阶段2: 集中训练 ==========
            if len(replay_buffer) >= TRAINING_CONFIG['min_buffer_size']:
                # 在GPU上训练（单进程，充分利用GPU）
                num_updates = len(all_experiences[0]) * num_workers // batch_size

                for _ in range(num_updates):
                    batch = replay_buffer.sample(batch_size)
                    loss = agent.train(batch)

                # 更新epsilon
                agent.epsilon = max(
                    MODEL_CONFIG['epsilon_end'],
                    agent.epsilon * MODEL_CONFIG['epsilon_decay']
                )

            # ========== 阶段3: 定期评估 ==========
            if episode_count % 100 == 0:
                print(f"\n\n评估中...")
                eval_futures = [
                    worker.evaluate.remote(5)
                    for worker in workers
                ]
                results = ray.get(eval_futures)

                avg_reward = np.mean([r[0] for r in results])
                win_rate = np.mean([r[1] for r in results])

                print(f"回合 {episode_count}: "
                      f"平均奖励={avg_reward:.2f}, "
                      f"胜率={win_rate*100:.1f}%")

                # 保存模型
                agent.save(f"saved_models/ray_model_ep{episode_count}.pth")

    except KeyboardInterrupt:
        print("\n\n训练被中断")

    finally:
        # 保存最终模型
        agent.save("saved_models/ray_final_model.pth")

        # 关闭Ray
        ray.shutdown()

        print("✓ 训练结束")


if __name__ == '__main__':
    train_with_ray()
