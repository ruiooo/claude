#!/usr/bin/env python3
"""
Ray RLlib版本的训练脚本

使用Ray RLlib的DQN算法，获得最佳性能和可扩展性
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple
import ctypes
import os

from ray import tune
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.env.env_context import EnvContext


class TankBattleGymEnv(gym.Env):
    """
    Gymnasium标准接口的坦克大战环境

    符合RLlib要求的环境包装
    """

    def __init__(self, config: EnvContext):
        super().__init__()

        # 加载C共享库
        lib_path = os.path.join(os.path.dirname(__file__), '..', 'libtankbattle.so')
        self.lib = ctypes.CDLL(lib_path)

        # 定义C函数接口
        self.lib.ai_init_env.argtypes = []
        self.lib.ai_init_env.restype = ctypes.c_void_p

        self.lib.ai_reset_env.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.ai_reset_env.restype = None

        self.lib.ai_step.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_bool),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int)
        ]
        self.lib.ai_step.restype = None

        # 初始化游戏
        self.game_state = self.lib.ai_init_env()

        # 环境配置
        self.max_enemies = config.get('max_enemies', 3)
        self.current_enemies = config.get('initial_enemies', 2)

        # 定义观察和动作空间
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(43,),
            dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(9)

        # 统计信息
        self.episode_reward = 0
        self.episode_length = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """重置环境"""
        super().reset(seed=seed)

        # 重置C环境
        self.lib.ai_reset_env(self.game_state, self.current_enemies)

        # 获取初始观察
        observation = np.zeros(43, dtype=np.float32)

        # 重置统计
        self.episode_reward = 0
        self.episode_length = 0

        info = {
            'enemy_count': self.current_enemies
        }

        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """执行一步"""
        # 准备输出变量
        next_observation = np.zeros(43, dtype=np.float32)
        reward = ctypes.c_float(0.0)
        done = ctypes.c_bool(False)
        winner = ctypes.c_int(-1)
        enemies_killed = ctypes.c_int(0)

        # 调用C接口
        self.lib.ai_step(
            self.game_state,
            action,
            next_observation.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(reward),
            ctypes.byref(done),
            ctypes.byref(winner),
            ctypes.byref(enemies_killed)
        )

        # 更新统计
        self.episode_reward += reward.value
        self.episode_length += 1

        # 构建info
        info = {
            'winner': winner.value,
            'enemies_killed': enemies_killed.value,
            'episode_reward': self.episode_reward,
            'episode_length': self.episode_length,
        }

        # RLlib要求分开terminated和truncated
        terminated = done.value
        truncated = False

        return next_observation, reward.value, terminated, truncated, info

    def close(self):
        """清理资源"""
        # C库会自动清理
        pass


def train_with_rllib():
    """使用Ray RLlib训练"""

    # 配置DQN算法
    config = (
        DQNConfig()
        .environment(
            TankBattleGymEnv,
            env_config={
                'initial_enemies': 2,
                'max_enemies': 10,
            }
        )
        .framework("torch")
        .resources(
            num_gpus=1,  # 使用GPU训练
            num_cpus_per_worker=1,
        )
        .rollouts(
            num_rollout_workers=8,  # 8个并行环境
            num_envs_per_worker=1,
            rollout_fragment_length=200,
        )
        .training(
            # DQN超参数
            lr=0.0003,
            train_batch_size=256,
            gamma=0.99,
            target_network_update_freq=100,
            replay_buffer_config={
                "type": "MultiAgentPrioritizedReplayBuffer",
                "capacity": 100000,
                "prioritized_replay_alpha": 0.6,
                "prioritized_replay_beta": 0.4,
                "prioritized_replay_eps": 1e-6,
            },
            # 网络架构
            model={
                "fcnet_hiddens": [512, 512, 256, 128],
                "fcnet_activation": "relu",
            },
            # 探索
            exploration_config={
                "type": "EpsilonGreedy",
                "initial_epsilon": 1.0,
                "final_epsilon": 0.1,
                "epsilon_timesteps": 50000,
            },
        )
        .debugging(
            log_level="INFO",
        )
        .reporting(
            min_train_timesteps_per_iteration=1000,
            min_sample_timesteps_per_iteration=1000,
        )
    )

    # 使用Ray Tune进行训练
    tuner = tune.Tuner(
        "DQN",
        param_space=config.to_dict(),
        run_config=tune.RunConfig(
            stop={
                "training_iteration": 1000,  # 最大迭代次数
                "episode_reward_mean": 200,  # 达到目标奖励
            },
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_frequency=10,
                checkpoint_at_end=True,
            ),
            storage_path="./ray_results",
            name="tank_battle_dqn",
        ),
    )

    print("🚀 开始Ray RLlib训练")
    print("   - 算法: DQN with Prioritized Replay")
    print("   - 并行环境数: 8")
    print("   - GPU加速: 启用")
    print("   - 网络: [512, 512, 256, 128]")
    print()

    results = tuner.fit()

    # 获取最佳checkpoint
    best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
    print(f"\n✓ 训练完成")
    print(f"   - 最佳平均奖励: {best_result.metrics['episode_reward_mean']:.2f}")
    print(f"   - 最佳checkpoint: {best_result.checkpoint.path}")

    return best_result


if __name__ == '__main__':
    # 需要先 pip install ray[rllib]
    try:
        import ray
        from ray import tune
        from ray.rllib.algorithms.dqn import DQNConfig
    except ImportError:
        print("错误: 需要安装Ray RLlib")
        print("请运行: pip install 'ray[rllib]'")
        exit(1)

    train_with_rllib()
