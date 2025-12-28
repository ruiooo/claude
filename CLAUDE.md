# Tank Battle AI Training System - Project Rules

## 项目概述

基于C语言+SDL2游戏引擎和Python3+PyTorch+CUDA深度强化学习的坦克大战AI训练系统。

## 核心架构

### 状态空间 (47维)

```
AI坦克状态 (6维): x, y, vx, vy, health, shoot_cooldown
敌人信息 (25维): 5个敌人 × (dx, dy, vx, vy, health)
子弹信息 (12维): 3个子弹 × (dx, dy, vx, vy)
战略信息 (4维): 敌人数量, 最近敌人距离, 水平墙距, 垂直墙距
```

**关键文件**:
- `src/game.c:game_get_observation()` - 训练时状态生成
- `src/model_ai.c:game_state_to_observation()` - 推理时状态生成
- `python/config.py:MODEL_CONFIG['state_dim']` - 配置 (47)
- `python/env_wrapper.py` - Python接口 (state_dim=47)

### 动作空间 (9维)

```c
ACTION_IDLE = 0,      // 静止
ACTION_MOVE_UP,       // 上移
ACTION_MOVE_DOWN,     // 下移
ACTION_MOVE_LEFT,     // 左移
ACTION_MOVE_RIGHT,    // 右移
ACTION_SHOOT_UP,      // 上射
ACTION_SHOOT_DOWN,    // 下射
ACTION_SHOOT_LEFT,    // 左射
ACTION_SHOOT_RIGHT    // 右射
```

## 代码结构

### C代码 (src/)

```
tank.{c,h}          - 坦克实体
bullet.{c,h}        - 子弹实体
game.{c,h}          - 游戏核心逻辑
collision.{c,h}     - 碰撞检测
enemy_ai.{c,h}      - 追踪型敌人AI (预判射击)
model_ai.{c,h}      - DQN模型推理接口
ai_interface.{c,h}  - Python训练接口 (ctypes)
rendering.{c,h}     - SDL2渲染
network.{c,h}       - 多人对战网络
main.c              - 玩家对战入口
multiplayer_main.c  - 多人对战入口
```

### Python代码 (python/)

```
config.py           - 训练配置
model.py            - DQN网络和Agent
replay_buffer.py    - 经验回放缓冲区
env_wrapper.py      - 游戏环境包装 (ctypes)
human_data_loader.py - 人类数据加载器
train_ray.py        - Ray并行训练脚本
evaluate_human_data.py - 人类数据评估工具
```

## 奖励系统 (v5.1)

位置: `src/ai_interface.c`

### 基础奖励

| 类型 | 值 | 说明 |
|------|-----|------|
| 击杀 | +80 | 每击杀一个敌人 |
| 胜利 | +50 | 消灭所有敌人 |
| 死亡 | -50 | AI坦克死亡 |
| 平局 | -30 | 超时平局 (鼓励进攻) |
| 受伤 | -15 | 每次受伤 |
| IDLE | -0.3 | 每帧静止惩罚 (解决被动问题) |

### 行为塑形奖励

| 类型 | 范围 | 说明 |
|------|------|------|
| 生存奖励 | +0.01/帧 | 基础存活奖励 (降低50%) |
| 距离控制 | 0~1.0 | 保持最佳战斗距离 |
| 瞄准奖励 | 0~0.8 | 朝向敌人方向 |
| 射击时机 | 0~2.0 | 对准时射击 |
| 躲避奖励 | ±1.5 | 避开危险子弹 |

### v5.1更新 (2025-12-25)

**问题**: 训练停滞在23.8%胜率，AI表现被动
**原因**: 模型学会了"不动=存活"策略

**修复**:
1. 添加IDLE惩罚 (-0.3/帧) - 鼓励主动行动
2. 添加平局惩罚 (-30) - 阻止消极拖延
3. 降低存活奖励 (0.02→0.01) - 减少被动生存激励
4. 玩家模式强制行动 - 连续5帧IDLE后自动向玩家移动

## 课程学习系统 (v6.0)

**问题**: 训练3万回合后胜率仍停滞在24.5%
**根本原因**: 敌人AI太强（预判射击+躲避+定位），DQN在随机探索时几乎无法获得正反馈

### 解决方案：渐进式难度训练

从简单敌人开始训练，逐步提升难度，让DQN有机会学习基础技能。

### 难度级别（5级系统 v6.1）

| 级别 | 名称 | 敌人行为 | 升级条件 |
|------|------|---------|----------|
| 0 | 假人 | 静止不动，不射击 | 70%胜率 (100局) |
| 1 | 慢移动 | 缓慢随机移动，不射击 | 60%胜率 (100局) |
| 2 | 移动+射击 | 随机移动，20%概率射击 | 50%胜率 (100局) |
| 3 | 追踪 | 追踪AI，简单射击 | 40%胜率 (100局) |
| 4 | 完整AI | 预判射击+躲避+定位 | 最终目标 |

**v6.1更新**：增加"慢移动"过渡难度，解决难度跳跃太大的问题

### 使用方法

课程学习已集成到训练流程，自动启用：

```bash
make train-ray
# 选择 [0] 创建新模型

# 训练输出示例:
# 回合 500/30000 | 难度:0[假人] | ε=0.990 | 胜率:75% | 本难度:75%(100局)
# 🎉 难度升级！假人 → 简单
```

### 关键文件

- `src/ai_interface.c`: `ai_set_difficulty()`, `ai_get_difficulty()`
- `src/enemy_ai.c`: 难度检查逻辑 (行455-481)
- `python/env_wrapper.py`: `set_difficulty()`, `get_difficulty()`
- `python/train_ray.py`: 课程学习主循环 (CURRICULUM_CONFIG)

### 预期效果

| 阶段 | 回合数 | 难度 | 预期胜率 |
|------|--------|------|---------|
| 学习射击 | 0-500 | 假人 | 70%+ |
| 学习追踪 | 500-2000 | 慢移动 | 60%+ |
| 学习躲避 | 2000-5000 | 移动+射击 | 50%+ |
| 学习战术 | 5000-10000 | 追踪 | 40%+ |
| 精通对战 | 10000+ | 完整AI | 30%+ |

## 训练配置

位置: `python/config.py`

```python
MODEL_CONFIG = {
    'state_dim': 47,
    'action_dim': 9,
    'hidden_dims': [256, 256, 128],  # 约11万参数
    'learning_rate': 0.0001,
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_end': 0.15,
    'epsilon_decay': 0.9999,
}

TRAINING_CONFIG = {
    'batch_size': 256,      # 可通过GPU配置自动调整
    'buffer_size': 35000,   # 可通过GPU配置自动调整
    'max_episodes': 30000,
    'device': 'cuda',
}
```

### GPU性能自适应配置 (v7.0 - GPU密集训练优化)

训练时会根据GPU自动推荐配置，也可手动选择：

| 配置级别 | 适用GPU | Batch | Workers | 每轮训练 | 网络参数 | 每轮样本 | 预期GPU利用率 |
|---------|---------|-------|---------|---------|---------|---------|--------------|
| **入门级** | GTX 1660, RTX 2060, RTX 3050 | 256 | 8 | 100次 | 11万 | 2.5万 | 40-60% |
| **中端级** | RTX 3060, RTX 3070, RTX 4060 | 512 | 16 | 200次 | 40万 | 10万 | 60-80% |
| **高端级** | RTX 3080, RTX 3090, RTX 4070 Ti | 1024 | 24 | 300次 | 70万 | 31万 | 70-90% |
| **旗舰级** | RTX 4090, RTX 5090, A100 | 4096 | 32 | 500次 | 100万 | **204万** | **80-95%** |
| **自定义** | 任意 | 自定义 | 自定义 | 自定义 | 自定义 | 自定义 | 根据配置而定 |

**v7.0关键优化（解决GPU利用率<5%问题）**：
1. **更大网络**：从11万参数增加到100万参数（5090配置），GPU计算量增加9倍
2. **固定训练次数**：每轮固定训练500次（5090），不依赖CPU采样量
3. **更大batch**：从2048增加到4096，充分利用GPU并行计算
4. **GPU密集训练**：每轮处理204万样本，GPU持续工作不等待CPU

**关键参数说明**：
- **Batch Size**: 每次训练采样的经验数，越大GPU利用率越高
- **Workers**: 并行采样的进程数，越多采样越快
- **每轮训练次数**: 固定训练次数，让GPU持续工作（不等待CPU采样）
- **网络参数**: 更大网络=更多GPU计算，旗舰配置100万参数
- **每轮样本**: batch_size × 训练次数，5090每轮处理204万样本

**使用方法**：
```bash
make train
# 启动时会自动检测GPU并推荐配置
# 选择对应数字即可
```

**提示**：
- 5090用户强烈推荐选择[4]旗舰级配置，可将GPU利用率从1%提升到80-95%
- 训练倍数越高，每条经验被学习的次数越多，GPU忙碌时间越长
- 如遇显存不足（OOM），可选择低一级配置或自定义降低batch_size

## 常用命令

```bash
make all              # 编译
make train            # 开始训练 (Ray并行)
make run-player       # 玩家对战
make eval-human-data  # 评估人类数据
make help             # 查看帮助
```

## 模型管理

### 保存位置

- `saved_models/latest_model.pth` - 最新模型 (每100回合更新)
- `saved_models/final_model.pth` - 最终模型 (训练结束时)
- `saved_models/checkpoints/checkpoint_ep{N}.pth` - 定期检查点

### 模型内容

```python
{
    'policy_net': policy_net.state_dict(),
    'target_net': target_net.state_dict(),
    'optimizer': optimizer.state_dict(),
    'epsilon': epsilon,
    'train_step': train_step,
}
```

## 人类数据学习

### 数据收集

```bash
make run-player
# 启用经验记录: y
```

数据保存到: `human_data/experience_<timestamp>.dat`

### 配置

```python
HUMAN_LEARNING_CONFIG = {
    'enabled': True,
    'human_data_dir': 'human_data',
    'preload': True,
    'filter_quality': True,
    'min_reward': -50.0,
    'sampling_weight': 10.0,
}
```

## 多人对战

```bash
# 服务器
make run-multiplayer-server

# 客户端 (另一个终端)
./tank_battle_multiplayer client 127.0.0.1
```

控制: `W/A/S/D` 移动, `空格` 射击

## 常见问题

### 状态维度不匹配

确保以下三处维度一致 (当前为47):
1. `src/game.c:game_get_observation()`
2. `src/model_ai.c:game_state_to_observation()`
3. `python/config.py:MODEL_CONFIG['state_dim']`
4. `python/env_wrapper.py:self.state_dim`

### CUDA内存不足

```python
# 减小配置
TRAINING_CONFIG['batch_size'] = 128
TRAINING_CONFIG['buffer_size'] = 20000
```

### Ray训练问题

```bash
# 停止Ray进程
ray stop
pkill -9 ray
```

## 编码规范

### C代码

- 结构体: `PascalCase` (GameState, Tank)
- 函数: `module_snake_case` (game_init, tank_update)
- 常量: `UPPER_SNAKE_CASE` (TANK_SPEED, MAX_BULLETS)

### Python代码

- 类: `PascalCase` (DQNAgent, ReplayBuffer)
- 函数: `snake_case` (select_action, train)
- 常量: `UPPER_SNAKE_CASE` (MODEL_CONFIG, ENV_CONFIG)

## Git工作流

```
修复/添加/优化 <简短描述>

问题：
- <问题描述>

修复：
- <解决方案>

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

---

**最后更新**: 2025-12-25
**维护者**: Claude Code
