"""
config_fixed.py - 修复后的训练配置文件

主要修复:
1. 延长探索期 - epsilon衰减更慢
2. 降低动态难度增长速度
3. 降低自我对弈概率
4. 增加训练稳定性参数
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
    'learning_rate': 0.0001,  # 保持不变，已经比较合理
    'gamma': 0.99,  # 折扣因子

    # ✅ 修复1: 延长探索期
    'epsilon_start': 1.0,  # 初始探索率
    'epsilon_end': 0.1,  # 🔧 提高最终探索率: 0.05 -> 0.1 (保持10%探索)
    'epsilon_decay': 0.9995,  # 🔧 减慢衰减: 0.995 -> 0.9995 (衰减慢10倍)
    # 新衰减速度: 1000回合 epsilon≈0.606, 5000回合 epsilon≈0.082

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
    'self_play_prob': 0.1,  # 🔧 降低历史版本概率: 0.3 -> 0.1
    'max_history_versions': 10,  # 最多保存的历史版本数
}

# 动态难度配置
DIFFICULTY_CONFIG = {
    'enabled': True,  # 是否启用动态难度
    'win_threshold': 10,  # 🔧 提高连胜要求: 5 -> 10 (更慢增加难度)
    'max_enemies': 10,  # 最大敌人数量
    'enemy_type_probs': {
        'tracking': 0.9,  # 🔧 提高追踪型敌人概率: 0.7 -> 0.9
        'self_play': 0.1,  # 🔧 降低历史版本概率: 0.3 -> 0.1
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

"""
修复说明:

1. **Epsilon 衰减修复** (最重要):
   - epsilon_decay: 0.995 -> 0.9995
   - epsilon_end: 0.05 -> 0.1
   - 效果: 保持更长时间的探索能力，防止过早固化策略
   - 对比:
     * 旧配置: 500回合 ε≈0.08, 1000回合 ε≈0.007
     * 新配置: 1000回合 ε≈0.606, 5000回合 ε≈0.082

2. **动态难度修复**:
   - win_threshold: 5 -> 10
   - 效果: 给AI更多时间适应当前难度，再增加敌人

3. **自我对弈修复**:
   - self_play_prob: 0.3 -> 0.1
   - tracking: 0.7 -> 0.9
   - 效果: 减少强历史版本的干扰，让AI稳定学习基础策略

使用方法:
1. 备份原配置: cp python/config.py python/config_backup.py
2. 应用修复: cp python/config_fixed.py python/config.py
3. 继续训练: make train (选择latest_model.pth继续训练)
"""
