"""
Ray分布式训练配置

包含Ray Core和Ray RLlib的配置参数
"""

# ========== Ray Core配置 ==========
RAY_CORE_CONFIG = {
    # 并行工作器数量
    'num_workers': 8,  # 建议设为CPU核心数

    # 每个工作器每轮收集的回合数
    'episodes_per_worker': 2,

    # 设备配置
    'device': 'cuda',  # 主训练进程使用GPU

    # 训练参数（继承自原config.py）
    'max_episodes': 10000,
    'batch_size': 256,
    'buffer_size': 100000,
    'min_buffer_size': 1000,

    # 评估频率
    'eval_frequency': 100,  # 每100回合评估一次
    'eval_episodes': 5,     # 每次评估5个回合
}

# ========== Ray RLlib配置 ==========
RAY_RLLIB_CONFIG = {
    # 环境配置
    'env_config': {
        'initial_enemies': 2,
        'max_enemies': 10,
    },

    # 资源配置
    'resources': {
        'num_gpus': 1,              # 训练GPU数量
        'num_cpus_per_worker': 1,   # 每个worker的CPU数
    },

    # 并行配置
    'rollouts': {
        'num_rollout_workers': 8,   # 并行环境数
        'num_envs_per_worker': 1,   # 每个worker的环境数
        'rollout_fragment_length': 200,  # 每次rollout的步数
    },

    # DQN训练参数
    'training': {
        'lr': 0.0003,               # 学习率
        'train_batch_size': 256,    # 训练批次大小
        'gamma': 0.99,              # 折扣因子
        'target_network_update_freq': 100,  # 目标网络更新频率

        # 优先经验回放
        'replay_buffer_config': {
            'type': 'MultiAgentPrioritizedReplayBuffer',
            'capacity': 100000,
            'prioritized_replay_alpha': 0.6,
            'prioritized_replay_beta': 0.4,
            'prioritized_replay_eps': 1e-6,
        },

        # 网络架构
        'model': {
            'fcnet_hiddens': [512, 512, 256, 128],
            'fcnet_activation': 'relu',
        },

        # 探索策略
        'exploration_config': {
            'type': 'EpsilonGreedy',
            'initial_epsilon': 1.0,
            'final_epsilon': 0.1,
            'epsilon_timesteps': 50000,
        },
    },

    # 停止条件
    'stop': {
        'training_iteration': 1000,     # 最大迭代次数
        'episode_reward_mean': 200,     # 目标平均奖励
        'timesteps_total': 1000000,     # 最大总步数
    },

    # Checkpoint配置
    'checkpoint': {
        'checkpoint_frequency': 10,     # 每10次迭代保存
        'checkpoint_at_end': True,      # 结束时保存
    },
}

# ========== Ray Tune超参数搜索配置 ==========
RAY_TUNE_CONFIG = {
    # 搜索空间
    'search_space': {
        'lr': tune.loguniform(1e-5, 1e-3),
        'gamma': tune.choice([0.95, 0.99, 0.999]),
        'train_batch_size': tune.choice([64, 128, 256]),
        'model': {
            'fcnet_hiddens': tune.choice([
                [256, 256, 128],
                [512, 512, 256, 128],
                [1024, 512, 256, 128],
            ]),
        },
    },

    # 搜索算法
    'search_alg': 'hyperopt',  # 使用HyperOpt搜索

    # 调度器
    'scheduler': 'asha',  # 使用ASHA提前停止

    # 试验数量
    'num_samples': 10,  # 尝试10组超参数
}

# ========== 性能预估 ==========
PERFORMANCE_ESTIMATES = {
    'single_process': {
        'description': '当前单进程训练',
        'episodes_per_hour': 2000,
        'time_to_10k_episodes': '5小时',
        'gpu_utilization': '80%',
        'cpu_utilization': '25%',
    },

    'ray_core': {
        'description': 'Ray Core并行采样',
        'episodes_per_hour': 8000,  # 4倍加速（4 workers）
        'time_to_10k_episodes': '1.25小时',
        'gpu_utilization': '95%',   # 更充分利用GPU
        'cpu_utilization': '80%',   # 充分利用多核CPU
        'speedup': '4x',
    },

    'ray_rllib': {
        'description': 'Ray RLlib完整优化',
        'episodes_per_hour': 16000,  # 8倍加速（8 workers + 优化）
        'time_to_10k_episodes': '0.6小时',
        'gpu_utilization': '98%',
        'cpu_utilization': '90%',
        'speedup': '8x',
        'extra_features': [
            '优先经验回放',
            '分布式训练',
            '自动超参数优化',
            'TensorBoard集成',
        ],
    },
}

# ========== 硬件需求 ==========
HARDWARE_REQUIREMENTS = {
    'minimum': {
        'cpu_cores': 4,
        'ram_gb': 8,
        'gpu_vram_gb': 4,  # GTX 1660或更高
        'description': '可以运行，但速度较慢',
    },

    'recommended': {
        'cpu_cores': 8,
        'ram_gb': 16,
        'gpu_vram_gb': 6,  # RTX 3060或更高
        'description': '最佳性价比配置',
    },

    'optimal': {
        'cpu_cores': 16,
        'ram_gb': 32,
        'gpu_vram_gb': 8,  # RTX 3070或更高
        'description': '充分利用并行能力',
    },
}

# ========== 成本分析 ==========
COST_ANALYSIS = {
    'development': {
        'integration_time': '2-4小时',  # Ray Core集成
        'testing_time': '1-2小时',
        'total_effort': '3-6小时',
    },

    'training': {
        'electricity_cost_per_hour': 0.5,  # 假设0.5元/小时（GPU功耗）
        'time_saved_per_10k_episodes': 3.75,  # 5小时 -> 1.25小时
        'cost_saved_per_10k_episodes': 1.88,  # 节省电费
    },

    'cloud': {
        'aws_p3_2xlarge_per_hour': 3.06,  # 美元/小时
        'gcp_n1_highmem_8_v100_per_hour': 2.48,
        'recommendation': '本地训练更经济（如果有GPU）',
    },
}


def print_comparison():
    """打印性能对比"""
    print("=" * 60)
    print("Ray分布式训练 vs 单进程训练性能对比")
    print("=" * 60)

    for key, config in PERFORMANCE_ESTIMATES.items():
        print(f"\n【{config['description']}】")
        print(f"  训练速度: {config['episodes_per_hour']} 回合/小时")
        print(f"  10k回合耗时: {config['time_to_10k_episodes']}")
        print(f"  GPU利用率: {config['gpu_utilization']}")
        print(f"  CPU利用率: {config['cpu_utilization']}")
        if 'speedup' in config:
            print(f"  加速比: {config['speedup']}")
        if 'extra_features' in config:
            print(f"  额外功能: {', '.join(config['extra_features'])}")

    print("\n" + "=" * 60)
    print("硬件需求")
    print("=" * 60)

    for level, req in HARDWARE_REQUIREMENTS.items():
        print(f"\n【{level.upper()}】 - {req['description']}")
        print(f"  CPU: {req['cpu_cores']} 核")
        print(f"  内存: {req['ram_gb']} GB")
        print(f"  显存: {req['gpu_vram_gb']} GB")


if __name__ == '__main__':
    from ray import tune  # 动态导入
    print_comparison()
