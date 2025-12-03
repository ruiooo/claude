"""
config.py - 训练配置文件
"""

# 环境配置
ENV_CONFIG = {
    'map_width': 800,
    'map_height': 600,
    'initial_enemies': 2,  # 初始敌人数量
}

# DQN模型配置
MODEL_CONFIG = {
    'state_dim': 43,  # 状态维度（6 + 25 + 12）
    'action_dim': 9,  # 动作维度（idle + 4移动 + 4射击）
    'hidden_dims': [256, 256, 128],  # 隐藏层维度
    'learning_rate': 0.0001,
    'gamma': 0.99,  # 折扣因子
    'epsilon_start': 1.0,  # 初始探索率
    'epsilon_end': 0.05,  # 最终探索率
    'epsilon_decay': 0.995,  # 探索率衰减
    'target_update_freq': 100,  # 目标网络更新频率
}

# 训练配置
TRAINING_CONFIG = {
    'batch_size': 128,  # 批大小
    'buffer_size': 100000,  # 经验回放缓冲区大小
    'min_buffer_size': 1000,  # 开始训练的最小经验数
    'max_episodes': 100000,  # 最大训练回合数
    'max_steps_per_episode': 3600,  # 每回合最大步数（60秒 @ 60fps）
    'save_interval': 100,  # 模型保存间隔（回合）
    'log_interval': 10,  # 日志打印间隔
    'visualize': False,  # 是否可视化（默认关闭）
    'device': 'cuda',  # 使用设备（cuda或cpu）
}

# 自我对弈配置
SELF_PLAY_CONFIG = {
    'enabled': True,  # 是否启用自我对弈
    'history_interval': 200,  # 每N回合保存历史版本
    'self_play_prob': 0.3,  # 使用历史版本作为敌人的概率
    'max_history_versions': 10,  # 最多保存的历史版本数
}

# 动态难度配置
DIFFICULTY_CONFIG = {
    'enabled': True,  # 是否启用动态难度
    'win_threshold': 5,  # 连续胜利N次增加一个敌人
    'max_enemies': 10,  # 最大敌人数量
    'enemy_type_probs': {
        'tracking': 0.7,  # 追踪型敌人概率
        'self_play': 0.3,  # 历史版本敌人概率
    }
}

# 人类数据学习配置（模仿学习）
HUMAN_LEARNING_CONFIG = {
    'enabled': True,  # 是否使用人类经验数据
    'human_data_dir': 'human_data',  # 人类数据目录
    'preload': True,  # 是否在训练开始时预加载人类数据
    'filter_quality': True,  # 是否过滤低质量数据
    'min_reward': -50.0,  # 过滤阈值：低于此奖励的经验将被过滤
    'sampling_weight': 1.0,  # 人类数据采样权重（相对于AI自己的经验）
}

# 模型保存路径
PATHS = {
    'models': 'saved_models',
    'checkpoints': 'saved_models/checkpoints',
    'history': 'saved_models/history',
    'logs': 'logs',
    'human_data': 'human_data',  # 人类经验数据目录
}
