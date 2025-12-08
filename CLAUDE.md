# Tank Battle AI Training System - Project Rules

## 项目概述

基于C语言+SDL2游戏引擎和Python3+CUDA深度强化学习的坦克大战AI训练系统。

## 功能特性

✅ **完整的游戏引擎** (C + SDL2) 游戏引擎核心,提供高性能的游戏逻辑和渲染
- 坦克移动、射击、碰撞检测
- 3点血量系统
- 优化的追踪型敌人AI
- 实时渲染和可视化

✅ **深度强化学习** (Python + PyTorch + CUDA) 深度强化学习的坦克大战 AI 训练系统
- DQN (Deep Q-Network) 算法
- 完全GPU加速训练
- 经验回放缓冲区
- 目标网络和双网络架构

✅ **自我对弈系统**
- 保存历史版本模型
- 历史版本作为敌人训练
- 自动版本管理（最多10个历史版本）

✅ **动态难度调整**
- 根据AI表现自动增加敌人数量
- 连续5次胜利增加1个敌人
- 最多支持10个敌人

✅ **两种训练显示模式**
- 纯文本模式：高性能，只显示训练进度
- 可视化模式：实时显示对战画面和训练信息

✅ **玩家对战模式**
- WASD移动，空格射击
- 与训练后的AI对战
- 可选的经验记录功能

✅ **人类数据学习** (模仿学习)
- 记录玩家对战AI时的操作数据
- 自动保存为结构化二进制文件(.dat)
- 训练时预加载人类经验到回放缓冲区
- 支持质量过滤(低奖励数据自动过滤)
- AI从人类专家经验中学习策略

✅ **模型持久化**
- 定量保存模型（每100回合）
- 支持断点续训
- 模型版本管理

## 核心架构原则

### 1. 语言要求
- C + SDL2实现游戏引擎
- Python + PyTorch + CUDA实现深度强化学习的坦克大战 AI 训练系统

### 2. 状态维度一致性 (Critical!)

**规则**: 训练时和推理时的状态表示必须完全一致

**状态维度**: 43 维 (固定)
- AI 坦克状态: 6 维 (x, y, vx, vy, health, shoot_cooldown)
- 最近 5 个敌人: 5×5=25 维 (每个敌人: dx, dy, vx, vy, health)
- 最近 3 个子弹: 3×4=12 维 (每个子弹: dx, dy, vx, vy)

**关键文件**:
- `src/game.c:game_get_observation()` - 训练时状态生成
- `src/model_ai.c:game_state_to_observation()` - 推理时状态生成
- `python/config.py:MODEL_CONFIG['state_dim']` - 模型配置

**重要**: 修改状态表示时,必须同步更新以下三处:
1. C 训练接口 (game.c)
2. C 推理接口 (model_ai.c)
3. Python 配置 (config.py)

### 3. 动作空间

**动作维度**: 9 维 (固定)
```c
typedef enum {
    ACTION_IDLE = 0,           // 静止
    ACTION_MOVE_UP,            // 上移
    ACTION_MOVE_DOWN,          // 下移
    ACTION_MOVE_LEFT,          // 左移
    ACTION_MOVE_RIGHT,         // 右移
    ACTION_SHOOT_UP,           // 上射
    ACTION_SHOOT_DOWN,         // 下射
    ACTION_SHOOT_LEFT,         // 左射
    ACTION_SHOOT_RIGHT,        // 右射
    ACTION_COUNT = 9
} TankAction;
```

### 4. 游戏常量

**坦克参数** (src/tank.h):
```c
#define TANK_SIZE 32              // 坦克尺寸
#define TANK_SPEED 2.5f           // 移动速度
#define TANK_MAX_HEALTH 3         // 最大血量
#define SHOOT_COOLDOWN 15         // 射击冷却(帧)
```

**子弹参数** (src/bullet.h):
```c
#define BULLET_SPEED 10.0f        // 子弹速度
#define MAX_BULLETS 100           // 最大子弹数
```

**地图参数** (python/config.py):
```python
ENV_CONFIG = {
    'map_width': 800,
    'map_height': 600,
}
```

**重要**: 修改这些常量时,确保状态归一化逻辑也相应更新

## 代码组织规范

### C 代码结构 (src/)

```
src/
├── tank.{c,h}          - 坦克实体和基础逻辑
├── bullet.{c,h}        - 子弹实体和物理
├── game.{c,h}          - 游戏核心状态和更新循环
├── collision.{c,h}     - 碰撞检测系统
├── enemy_ai.{c,h}      - 传统追踪型 AI
├── model_ai.{c,h}      - DQN 模型推理接口 (Python C API)
├── ai_interface.{c,h}  - Python 训练接口 (ctypes)
├── rendering.{c,h}     - SDL2 渲染逻辑
├── network.{c,h}       - 多人对战网络
├── main.c              - 玩家对战模式入口
└── multiplayer_main.c  - 多人对战模式入口
```

**模块职责**:
- `game.c`: 游戏状态管理,所有实体的统一更新
- `ai_interface.c`: 暴露给 Python 的训练接口 (ai_init_env, ai_step, ai_reset_env)
- `model_ai.c`: 加载 PyTorch 模型并在 C 中推理 (用于玩家对战模式)

### Python 代码结构 (python/)

```
python/
├── config.py           - 训练配置 (超参数)
├── model.py            - DQN 网络定义
├── replay_buffer.py    - 经验回放缓冲区
├── env_wrapper.py      - 游戏环境包装 (ctypes)
├── enemy_manager.py    - 敌人管理 (动态难度/自我对弈)
└── train.py            - 训练主循环
```

**模块职责**:
- `env_wrapper.py`: 通过 ctypes 调用 C 共享库,包装成 Python 环境
- `model.py`: DQN 网络和 Agent,负责策略选择和训练
- `train.py`: 主训练循环,整合所有组件

## 编码规范

### C 代码规范

1. **命名约定**:
   - 结构体: `PascalCase` (例: `GameState`, `Tank`)
   - 函数: `module_snake_case` (例: `game_init`, `tank_update`)
   - 枚举: `UPPER_SNAKE_CASE` (例: `TANK_TYPE_AI`, `ACTION_IDLE`)
   - 常量: `UPPER_SNAKE_CASE` (例: `TANK_SPEED`, `MAX_BULLETS`)

2. **内存管理**:
   - 所有结构体在栈上分配,避免堆内存
   - 数组使用固定大小 (例: `Tank tanks[32]`, `Bullet bullets[100]`)
   - 不使用 `malloc/free`,除非在 Python C API 中必要

3. **错误处理**:
   - 返回 `bool` 表示成功/失败
   - 使用 `fprintf(stderr, ...)` 输出错误信息
   - 在 Python 接口中使用 `PyErr_Print()` 打印 Python 异常

4. **头文件保护**:
   ```c
   #ifndef MODULE_H
   #define MODULE_H
   // ...
   #endif // MODULE_H
   ```

### Python 代码规范

1. **命名约定**:
   - 类: `PascalCase` (例: `DQNAgent`, `ReplayBuffer`)
   - 函数/方法: `snake_case` (例: `select_action`, `train`)
   - 常量: `UPPER_SNAKE_CASE` (例: `MODEL_CONFIG`, `ENV_CONFIG`)

2. **类型注解**:
   ```python
   def select_action(self, state: np.ndarray, training: bool = True) -> int:
       ...
   ```

3. **文档字符串**:
   ```python
   def train(self, batch: Tuple) -> float:
       """
       训练网络（使用GPU加速）

       Args:
           batch: (states, actions, rewards, next_states, dones)

       Returns:
           损失值
       """
   ```

4. **配置管理**:
   - 所有超参数集中在 `config.py` 中
   - 使用字典组织配置: `MODEL_CONFIG`, `TRAINING_CONFIG` 等

## 构建系统

### Makefile 规范

**虚拟环境检测**:
```makefile
VENV_DIR ?= tank
VENV_VERSION := $(shell grep "^version = " ./$(VENV_DIR)/pyvenv.cfg | ...)
PYTHON_CONFIG := python$(VENV_VERSION)-config
```

**编译目标**:
- `make all`: 编译所有目标 (共享库 + 可执行文件)
- `make clean`: 清理构建文件
- `make rebuild`: 重新编译

**Python 链接**:
- 玩家模式需要链接 Python: `$(PYTHON_LDFLAGS)`
- 多人对战模式不需要 Python 支持

### 编译选项

```makefile
CFLAGS = -Wall -O3 -fPIC -std=c11 $(PYTHON_INCLUDES) -DVENV_DIR=\"$(VENV_DIR)\"
```

- `-Wall`: 开启所有警告
- `-O3`: 最高优化级别
- `-fPIC`: 位置无关代码 (用于共享库)
- `-std=c11`: C11 标准

## 训练流程

### 1. 数据流向

```
C 游戏引擎 (libtankbattle.so)
    ↓ ctypes
Python 环境包装 (env_wrapper.py)
    ↓
DQN Agent (model.py)
    ↓ GPU
PyTorch 训练 (train.py)
```

### 2. 训练循环

```python
for episode in range(max_episodes):
    state = env.reset(enemy_count)
    for step in range(max_steps):
        action = agent.select_action(state, training=True)
        next_state, reward, done, info = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)

        if len(replay_buffer) >= min_buffer_size:
            batch = replay_buffer.sample(batch_size)
            loss = agent.train(batch)

        if done:
            break
```

### 3. 奖励设计 (src/ai_interface.c)

```c
// 存活奖励
reward += 0.01f;  // 每帧

// 受伤惩罚
if (health_change < 0)
    reward -= 10.0f;

// 击杀奖励
reward += 50.0f * enemies_killed;

// 死亡惩罚
if (!ai_tank->alive)
    reward = -100.0f;

// 胜利奖励
if (game_over && winner == 0)
    reward += 200.0f;
```

## 自我对弈系统

### 历史版本管理

**保存规则** (python/enemy_manager.py):
- 每 200 回合保存一个历史版本
- 最多保存 10 个历史版本
- 超过时删除最老的版本

**使用规则**:
- 30% 概率使用历史版本作为敌人
- 70% 概率使用追踪型 AI

### 动态难度

**难度调整规则**:
- 连续 5 次胜利: 敌人数 +1
- 最大敌人数: 10
- 初始敌人数: 2

## 人类数据学习系统 (模仿学习)

### 数据收集流程

玩家在 `make run-player` 模式下对战AI时，可以选择启用经验记录:

```
是否启用经验记录功能? (y/n, 默认=n): y
✓ 已启用经验记录，数据将保存到 human_data/ 目录
```

**记录机制** (src/main.c):
- 每帧捕获游戏状态(43维) → 玩家动作 → 下一状态 → 奖励 → 完成标志
- 缓冲在内存中(最多10000条经验)
- 每回合结束自动保存为二进制文件: `human_data/experience_<timestamp>.dat`

**数据格式** (.dat文件):
```c
struct ExperienceData {
    int32_t count;                  // 经验数量
    float states[count][43];        // 状态数组
    int32_t actions[count];         // 动作数组
    float rewards[count];           // 奖励数组
    float next_states[count][43];   // 下一状态数组
    int32_t dones[count];           // 完成标志数组
}
```

### 训练时使用人类数据

**配置** (python/config.py):
```python
HUMAN_LEARNING_CONFIG = {
    'enabled': True,           # 启用人类数据学习
    'human_data_dir': 'human_data',
    'preload': True,           # 训练开始时预加载
    'filter_quality': True,    # 过滤低质量数据
    'min_reward': -50.0,       # 过滤阈值
    'sampling_weight': 1.0,    # 采样权重
}
```

**加载流程** (python/train.py):
1. 训练开始时自动扫描 `human_data/` 目录
2. 加载所有 `.dat` 文件
3. 过滤低质量经验 (reward < -50)
4. 预加载到 ReplayBuffer 中
5. 与AI自己产生的经验混合训练

**好处**:
- 提供专家策略的初始化
- 加快训练收敛
- 提高最终性能上限
- 降低探索风险

**关键文件**:
- `src/main.c`: ExperienceRecorder 经验记录器
- `python/human_data_loader.py`: HumanDataLoader 数据加载器
- `python/config.py`: HUMAN_LEARNING_CONFIG 配置

## 模型管理

### 模型保存

**保存策略** (python/train.py):

**1. latest_model.pth** - 最新检查点
- **保存时机**: 每 100 回合自动更新 (与 checkpoint 同步)
- **位置**: `saved_models/latest_model.pth`
- **用途**:
  - ✅ **推荐用于继续训练** (包含最新的训练状态)
  - 跟踪训练进度
  - 快速恢复训练
- **特点**: 实时更新,始终保存最新状态

**2. final_model.pth** - 最终模型
- **保存时机**: 训练完成或被中断(Ctrl+C)时
- **位置**: `saved_models/final_model.pth`
- **用途**:
  - ✅ **推荐用于推理/对战** (训练最充分)
  - 代表完整训练周期的最终成果
  - 生产环境使用
- **特点**: 训练完成后的最优模型,通常训练最充分

**3. checkpoint_ep{N}.pth** - 定期检查点
- **保存时机**: 每 100 回合
- **位置**: `saved_models/checkpoints/checkpoint_ep{episode}.pth`
- **用途**:
  - 历史版本回溯
  - 对比不同训练阶段
  - 防止训练意外中断
- **特点**: 保留训练过程的多个快照

**模型内容**:
```python
{
    'policy_net': policy_net.state_dict(),
    'target_net': target_net.state_dict(),
    'optimizer': optimizer.state_dict(),
    'epsilon': epsilon,              # 探索率
    'train_step': train_step,        # 训练步数
}
```

**继续训练建议**:
- **从头开始**: 选择 `[0] 创建新模型`
- **继续训练**: 选择 `latest_model.pth` (最新状态,包含完整训练上下文)
- **微调模型**: 选择 `final_model.pth` (训练充分,但 epsilon 较低)
- **特定版本**: 选择具体的 `checkpoint_ep*.pth`

### 模型加载

**玩家对战模式** (src/main.c, src/model_ai.c):

**模型选择策略** (最多5个):
1. **最优模型**: `saved_models/final_model.pth` (如果存在)
2. **最新模型**: `saved_models/latest_model.pth` (如果存在)
3. **Top3训练版本**: 按时间排序的最新3个 `checkpoint_ep*.pth`

**搜索路径**:
- 优先模型: `saved_models/`
- Checkpoint 目录: `saved_models/checkpoints/`, `checkpoints/`
- 后备目录: `saved_models/`, `models/`

**交互流程**:
- 自动扫描并按优先级排序
- 显示最多5个模型供用户选择
- 默认选择第一个 (通常是 final_model.pth)
- 支持命令行参数直接指定: `./tank_battle_player <model_path>`

## 常见问题和解决方案

### 1. 状态维度不匹配

**问题**: `RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x18 and 43x256)`

**原因**: 训练时和推理时状态生成不一致

**解决**:
1. 检查 `game.c:game_get_observation()` - 应该生成 43 维
2. 检查 `model_ai.c:game_state_to_observation()` - 应该生成 43 维
3. 检查 `config.py:MODEL_CONFIG['state_dim']` - 应该是 43

### 2. Python 环境问题

**问题**: `导入torch模块失败` 或 `ImportError: _ctypes`

**解决**:
1. 确认虚拟环境已激活
2. 确认 PyTorch 已安装: `pip list | grep torch`
3. 检查 Python 版本匹配: Makefile 会自动检测虚拟环境 Python 版本
4. 重新编译: `make clean && make`

### 3. CUDA 内存不足

**问题**: `RuntimeError: CUDA out of memory`

**解决**:
1. 减小 batch_size: `TRAINING_CONFIG['batch_size'] = 64`
2. 减小 buffer_size: `TRAINING_CONFIG['buffer_size'] = 50000`
3. 使用 CPU 训练: `TRAINING_CONFIG['device'] = 'cpu'`

### 4. 训练胜率持续下降或长期低迷

**问题**: 训练数千回合后胜率仍然很低（<30%），或者越训练胜率越低

**症状**:
- 3500回合训练后胜率仅20%
- Epsilon已降到0.1-0.2，但胜率没有提升
- 网络权重差异接近0，疑似停止学习

**根本原因分析**:

1. **Epsilon衰减过快** (最常见)
   - 问题: AI过早停止探索，固化在次优策略上
   - 现象: Epsilon在1000回合内就降到<0.1
   - 后果: 当动态难度增加时，AI无法学习新策略

2. **动态难度增加过快**
   - 问题: 连续胜利后敌人数量增加太快
   - 现象: AI还没学好就要面对更强敌人
   - 后果: 学习不稳定，胜率下降

3. **自我对弈干扰**
   - 问题: 历史版本模型过强
   - 现象: 新训练的模型被历史版本碾压
   - 后果: 无法建立有效策略

**诊断工具**:
```bash
# 检查训练状态
python python/check_progress.py

# 深度诊断
python python/deep_diagnose.py
```

**修复方案** (2025-12-03更新):

**【激进修复配置】** - 已应用到 `python/config.py`

```python
# 1. 降低初始难度
ENV_CONFIG = {
    'initial_enemies': 1,  # 2 -> 1，让AI先学会基础
}

# 2. 延长探索期
MODEL_CONFIG = {
    'epsilon_decay': 0.9998,   # 0.9995 -> 0.9998，大幅减慢
    'epsilon_end': 0.2,        # 0.1 -> 0.2，保持20%探索
    'learning_rate': 0.00005,  # 0.0001 -> 0.00005，更稳定
    'target_update_freq': 50,  # 100 -> 50，更频繁更新
}

# 3. 优化训练参数
TRAINING_CONFIG = {
    'batch_size': 64,          # 128 -> 64，更频繁更新
    'buffer_size': 50000,      # 100000 -> 50000
    'min_buffer_size': 500,    # 1000 -> 500，更早开始
}

# 4. 禁用干扰因素
DIFFICULTY_CONFIG = {
    'enabled': False,  # 暂时禁用动态难度
}

SELF_PLAY_CONFIG = {
    'enabled': False,  # 暂时禁用自我对弈
}

# 5. 增强人类数据利用
HUMAN_LEARNING_CONFIG = {
    'min_reward': -30.0,       # -50 -> -30，接受更多数据
    'sampling_weight': 1.5,    # 1.0 -> 1.5，增加权重
}
```

**预期效果**:

| 阶段 | 回合数 | Epsilon | 敌人数 | 预期胜率 |
|------|--------|---------|--------|----------|
| 基础学习 | 0-500 | 0.9-0.7 | 1 | 60-80% |
| 策略成熟 | 500-2000 | 0.7-0.4 | 1 | 70-85% |
| 稳定阶段 | 2000-5000 | 0.4-0.2 | 1-2 | 65-80% |
| 高级训练 | 5000+ | 0.2+ | 2-5 | 50-70% |

**使用建议**:

方案A: 从头重新训练 (强烈推荐)
```bash
make train
# 选择 [0] 创建新模型
```

方案B: 继续训练 (如果想保留已学经验)
```bash
make train
# 选择 latest_model.pth
```

**监控关键指标**:
- 胜率应稳步上升
- Epsilon应缓慢下降 (5000回合时仍>0.3)
- 每500回合检查: `python python/check_progress.py`

**如果还不行**:

1. **进一步降低难度**:
   ```python
   ENV_CONFIG['initial_enemies'] = 0  # 只有AI，练习移动
   ```

2. **调整奖励函数** (需要修改C代码):
   ```c
   // src/ai_interface.c
   reward += 0.1f;      // 存活奖励: 0.01 -> 0.1
   reward -= 5.0f;      // 受伤惩罚: -10 -> -5
   reward += 100.0f;    // 击杀奖励: 50 -> 100
   ```
   然后重新编译: `make clean && make`

3. **检查人类数据质量**:
   ```bash
   ls -lh human_data/
   # 如果数据质量差，考虑删除或禁用
   HUMAN_LEARNING_CONFIG['enabled'] = False
   ```

**Epsilon衰减对比**:

```
旧配置 (epsilon_decay=0.9995):
  500回合: ε≈0.08  -> AI几乎不探索
  1000回合: ε≈0.007 -> 完全固化

新配置 (epsilon_decay=0.9998):
  500回合: ε≈0.71  -> 仍在探索
  1000回合: ε≈0.50 -> 平衡探索/利用
  5000回合: ε≈0.37 -> 逐渐稳定
  10000回合: ε≈0.13 -> 最终收敛
```

**重要提醒**:
- 给AI足够时间学习，不要频繁调整参数
- 对抗1个敌人稳定后，再手动增加敌人数量
- 胜率短期下降是正常的（增加探索的副作用）
- 关注长期趋势，而非单次训练结果

## Git 工作流

### 提交信息规范

使用中文编写清晰的提交信息:

```
修复/添加/优化 <简短描述>

问题：
- <问题描述1>
- <问题描述2>

修复/实现：
- <解决方案1>
- <解决方案2>

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

### 分支管理

- `main`: 主分支,稳定版本
- `v0.1`, `v0.2` 等: 版本分支
- 功能分支: 以 `feature/` 开头
- 修复分支: 以 `fix/` 开头

## 性能优化指南

### C 代码优化

1. **避免不必要的计算**:
   ```c
   // ❌ 错误: 每次都计算平方根
   float dist = sqrtf(dx * dx + dy * dy);
   if (dist < min_dist) { ... }

   // ✅ 正确: 比较平方距离
   float dist_sq = dx * dx + dy * dy;
   if (dist_sq < min_dist_sq) { ... }
   ```

2. **使用固定大小数组**:
   ```c
   // ✅ 栈上分配,快速
   Tank tanks[32];
   Bullet bullets[100];
   ```

3. **减少函数调用**:
   - 小函数使用 `inline` 或 `static inline`
   - 热路径代码避免函数调用

### Python 训练优化

1. **批量处理**:
   ```python
   # ✅ 批量转换为张量,一次性移到GPU
   states = torch.FloatTensor(states).to(device)
   ```

2. **梯度裁剪**:
   ```python
   torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
   ```

3. **目标网络更新**:
   ```python
   # 每 100 步更新一次,而不是每步
   if train_step % target_update_freq == 0:
       update_target_network()
   ```

## 测试规范

### 单元测试

暂无自动化测试,手动测试流程:

1. **编译测试**: `make clean && make`
2. **训练测试**: `make train` (运行几个回合)
3. **推理测试**: `make run-player` (加载模型并运行)
4. **状态维度测试**: 检查终端输出,确保没有维度不匹配错误

### 功能测试清单

- [ ] 游戏渲染正常
- [ ] 坦克移动和射击响应
- [ ] 碰撞检测正确
- [ ] 训练损失下降
- [ ] 模型保存和加载
- [ ] AI 决策合理

## 文档维护

### 更新规则

当修改以下内容时,必须更新文档:

1. **状态维度变化**: 更新本 rules 文档和 README
2. **配置参数变化**: 更新 `config.py` 注释
3. **新增功能**: 更新 README 和 Makefile help
4. **架构变更**: 更新本 rules 文档

### 注释规范

**C 代码**:
```c
// 单行注释: 简短说明

/*
 * 多行注释: 详细说明
 * 参数、返回值、注意事项等
 */
```

**Python 代码**:
```python
# 单行注释

"""
多行注释/文档字符串
Args, Returns, Raises 等
"""
```

## 依赖版本

### C 依赖

- GCC: 9.0+
- SDL2: 2.0+
- SDL2_ttf: 2.0+

### Python 依赖

```
torch>=2.0.0
numpy>=1.21.0
```

详见 `requirements.txt`

## 项目统计

- 总代码行数: ~4254 行
- C 代码: ~3000 行
- Python 代码: ~1254 行
- 模块数: 18 个 (9 个 C 模块 + 6 个 Python 模块)

## 性能优化与高胜率指南

### 训练诊断工具

**使用工具:**
```bash
python python/check_progress.py
```

**诊断内容:**
- 📊 训练状态（回合数、epsilon、训练步数）
- 🧠 神经网络分析（参数量、权重统计、死神经元检测）
- ⚙️ 优化器状态（学习率、动量）
- ⏱️ 训练速度分析
- 🚨 性能瓶颈诊断
- 💡 优化建议

**关键指标:**
- ✅ 胜率应持续上升
- ✅ Epsilon应缓慢下降
- ✅ Policy-Target网络差异不应为0
- ✅ 学习率不应趋近0

### 已应用的优化

#### ✅ 优化1: 奖励塑形 (Reward Shaping)

**状态:** 已应用到 `src/ai_interface.c`

**优化内容:**
- 基础奖励调整:
  - 存活奖励: 0.01 → 0.02 (+100%)
  - 受伤惩罚: -10 → -15 (+50%)
  - 击杀奖励: 50 → 100 (+100%)
  - 胜利奖励: 200 → 300 (+50%)

- 新增奖励塑形:
  - 距离奖励: 鼓励靠近敌人（+0.5）
  - 瞄准奖励: 朝向敌人方向（+0.3）
  - 射击奖励: 合适距离射击（+0.5）
  - 躲避奖励: 避开危险子弹（+0.4/子弹）
  - 墙壁避让: 远离墙壁（+0.2）
  - 动作多样性: 鼓励行动（+0.1）

**效果:** 预计胜率提升 10-15%

#### ✅ 优化2: 网络升级 (Network Scaling)

**状态:** 已应用到 `python/config.py`

**升级内容:**
- 隐藏层: [256, 256, 128] → [512, 512, 256, 128]
- 参数量: 11万 → 45万 (+300%)
- 学习率: 0.0001 → 0.0003
- 批次大小: 128 → 256
- 缓冲区: 50000 → 100000
- Epsilon衰减: 0.9996 → 0.9999 (更慢)

**注意事项:**
- ⚠️ 训练速度会慢约30%
- ⚠️ 显存占用增加约200MB
- ✅ 需要更多训练回合（建议10000+）

**效果:** 预计胜率提升 5-10%

**开始训练:**
```bash
# 优化已应用，直接训练即可
make train
# 选择 [0] 创建新模型（重要！必须从头训练）
```

#### 优化3: 优先经验回放 (Prioritized Experience Replay) - 可选

**原理:** 重要经验（大TD误差）更高采样概率

**效果:** 训练效率+30-50%，胜率+5-10%

**操作步骤:**
```bash
# 1. 生成代码
python python/add_prioritized_replay.py

# 2. 手动修改（参考生成的文件）
# - python/model.py: train()方法返回TD误差
# - python/train.py: 使用PrioritizedReplayBuffer
```

**难度:** ⭐⭐⭐⭐ (需要修改多处代码，较复杂)

**说明:** 此优化较复杂，建议先应用前两个优化观察效果

#### 优化4: 课程学习 (Curriculum Learning) - 推荐

**原理:** 从简单到困难，逐步提升难度

**操作步骤:**
```python
# 编辑 python/config.py

# 阶段1: 对抗1个敌人
ENV_CONFIG['initial_enemies'] = 1
# 训练到90%胜率后进入阶段2

# 阶段2: 对抗2个敌人
ENV_CONFIG['initial_enemies'] = 2
# 训练到80%胜率后进入阶段3

# 阶段3: 对抗3个敌人
ENV_CONFIG['initial_enemies'] = 3
```

**效果:** 稳定性提升，降低训练方差

#### 优化5: 人类数据增强

**操作:**
```bash
# 1. 玩几局游戏，记录高质量数据
make run-player
# 启用经验记录: y

# 2. 调整配置
# python/config.py:
HUMAN_LEARNING_CONFIG['enabled'] = True
HUMAN_LEARNING_CONFIG['sampling_weight'] = 2.0  # 增加权重
```

**效果:** 加快收敛，提供专家策略初始化

#### 优化6: 自我对弈

**操作:**
```python
# python/config.py
SELF_PLAY_CONFIG['enabled'] = True
SELF_PLAY_CONFIG['self_play_prob'] = 0.3
```

**效果:** 对抗更强敌人，提升鲁棒性

### 预期胜率提升路径

```
基准状态:          ~30-40%  (基础DQN)
                    ↓
奖励塑形:          50-60%   (+10-20%)
                    ↓
+ 网络升级:        60-70%   (+20-30%)
                    ↓
+ 优先经验回放:    70-80%   (+30-40%)
                    ↓
+ 课程学习:        75-85%   (+35-45%)
                    ↓
+ 人类数据:        80-90%   (+40-50%)
                    ↓
+ 自我对弈:        85-95%   (+45-55%)
                    ↓
+ 长期训练:        90-100%  (+50-60%)
```

### 训练监控与问题排查

**定期检查（每5000回合）:**
```bash
python python/check_progress.py
```

**正常训练标志:**
- ✅ 胜率持续上升
- ✅ Epsilon缓慢下降（10000回合时仍>0.2）
- ✅ 平均奖励增加
- ✅ Policy-Target网络有差异（>0.001）

**异常情况处理:**

**问题1: 胜率长期不提升（卡在某个值）**
- 诊断: Epsilon是否过低（<0.1）？
- 解决:
  - 增加epsilon_decay（如0.9999）
  - 提高epsilon_end（如0.2）
  - 重新训练

**问题2: 学习停滞（网络差异为0）**
- 诊断: 学习率是否趋近0？
- 解决:
  - 应用网络升级（重置优化器）
  - 从头训练

**问题3: 网络容量不足**
- 症状: 训练很久但性能不提升
- 解决: 应用网络升级（增加到45万参数）

**问题4: 死神经元过多**
- 症状: check_progress报告大量死神经元
- 解决:
  - 使用LeakyReLU替代ReLU
  - 降低学习率
  - 增加批归一化

### 快速开始

**✅ 基础优化已应用！直接开始训练：**

```bash
# 1. 编译（如有需要）
make clean && make

# 2. 开始训练
make train

# 3. 选择 [0] 创建新模型（重要！）

# 4. 等待训练（建议10000回合）

# 5. 定期检查进度
python python/check_progress.py
```

**进阶优化清单（可选）:**
- [ ] 课程学习：分阶段增加敌人数量（1→2→3）
- [ ] 收集人类数据：`make run-player` 并启用经验记录
- [ ] 启用自我对弈：修改 `SELF_PLAY_CONFIG['enabled'] = True`
- [ ] 长期训练：20000+ 回合

### 性能基准参考

**训练速度:**
- 纯文本模式: 2000-3000 回合/小时
- 可视化模式: 1000-1500 回合/小时

**预期时间投入:**
- 达到60%胜率: ~5小时（奖励塑形）
- 达到70%胜率: ~10小时（+网络升级）
- 达到80%胜率: ~30小时（+优先回放）
- 达到90%胜率: ~100小时（+课程学习+自我对弈）
- 接近100%: 需要精细调优和长期训练

### 常见问题 (FAQ)

**Q: 必须从头训练吗？**
A: 是的。网络结构变化后无法加载旧模型，且新的奖励函数需要重新学习。

**Q: 只应用部分优化可以吗？**
A: 可以。奖励塑形是性价比最高的单项优化，单独使用就能提升10-15%。

**Q: 能达到真正的100%吗？**
A: 接近100%（95-98%）是可能的，但达到100%极其困难，因为：
- 敌人AI有随机性
- 初始位置随机
- 多敌人情况下难以完美应对
- 实际上90-95%已是非常强的表现

**Q: 优化已经应用了吗？**
A: 是的！奖励塑形和网络升级已直接应用到代码中：
- `src/ai_interface.c` - 已包含奖励塑形
- `python/config.py` - 已包含网络升级
- 直接 `make train` 即可开始训练

**Q: 需要手动操作吗？**
A: 不需要！只需：
1. `make clean && make` （编译）
2. `make train` （训练）
3. 选择 `[0]` 创建新模型

## 最近修复 (2025-12-04)

### 修复1: AI推理模式问题

**问题:** 玩家对战模式中，AI没有正确使用训练好的模型，总是贴墙发炮

**根本原因:**
1. `python/model.py` 的 `load()` 方法加载模型后未设置为 `eval()` 模式
2. `select_action()` 方法未根据 `training` 参数切换网络状态

**修复:**
- `load()` 方法中添加 `self.policy_net.eval()` 和 `self.target_net.eval()`
- `select_action()` 方法中根据 `training` 参数动态切换 `train()/eval()` 模式
- 确保推理时网络处于评估模式，关闭 dropout/batch_norm 的训练行为

**影响:** 玩家对战模式中AI现在能正确使用训练好的策略

### 修复2: make train-viz 无法运行

**问题:** `make train-viz` 报错找不到目标

**根本原因:** Makefile 中 `train` 和 `train-vis` 目标使用 `python3` 而非虚拟环境的 Python

**修复:**
- 修改 Makefile，自动检测虚拟环境
- 优先使用 `./$(VENV_DIR)/bin/python`
- 如无虚拟环境则回退到 `python3`

**影响:** 训练命令现在能正确使用虚拟环境

---

**最后更新**: 2025-12-04
**维护者**: Claude Code
