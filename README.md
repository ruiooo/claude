# 坦克大战 AI 训练系统

基于C语言+SDL2游戏引擎和Python3+PyTorch+CUDA深度强化学习的坦克大战AI训练系统。

## 功能特性

- **完整的游戏引擎** (C + SDL2) - 坦克移动、射击、碰撞检测、3点血量系统
- **深度强化学习** (Python + PyTorch + CUDA) - DQN算法、GPU加速、经验回放
- **Ray并行训练** - 4-6倍训练加速
- **人类数据学习** - 模仿学习，从玩家经验中学习
- **动态难度调整** - 根据AI表现自动增加敌人数量
- **玩家对战模式** - WASD移动，空格射击
- **多人联机对战** - TCP网络通信，实时对战

## 系统要求

- **操作系统**: Linux (Ubuntu 20.04+)
- **C编译器**: GCC 9.0+
- **Python**: Python 3.8+
- **GPU**: NVIDIA GPU (推荐GTX 1060+)
- **依赖**: SDL2, SDL2_ttf, PyTorch, Ray

## 快速开始

### 1. 安装依赖

```bash
# 安装C依赖
make install-deps

# 创建虚拟环境并安装Python依赖
make venv
source tank/bin/activate
make install-python-deps
make install-ray
```

### 2. 编译

```bash
make all
```

### 3. 开始训练

```bash
make train
```

### 4. 玩家对战

```bash
make run-player
```

控制: `W/A/S/D` 移动, `空格` 射击, `ESC` 退出

## 项目结构

```
├── src/                    # C语言游戏引擎
│   ├── game.c/h           # 游戏核心逻辑
│   ├── tank.c/h           # 坦克实体
│   ├── bullet.c/h         # 子弹实体
│   ├── enemy_ai.c/h       # 追踪型敌人AI
│   ├── ai_interface.c/h   # Python训练接口
│   ├── model_ai.c/h       # 模型推理接口
│   └── rendering.c/h      # SDL2渲染
├── python/                 # Python训练代码
│   ├── train_ray.py       # Ray并行训练脚本
│   ├── model.py           # DQN模型
│   ├── env_wrapper.py     # 环境包装器
│   ├── replay_buffer.py   # 经验回放
│   └── config.py          # 配置文件
├── saved_models/           # 保存的模型
├── human_data/             # 人类经验数据
└── Makefile               # 编译脚本
```

## 常用命令

```bash
make all                  # 编译所有目标
make train                # 开始训练
make run-player           # 玩家对战模式
make eval-human-data      # 评估人类数据质量
make help                 # 查看所有命令
```

## 配置说明

所有训练配置在 `python/config.py`:

- **状态维度**: 47维 (坦克状态 + 敌人信息 + 子弹信息 + 战略信息)
- **动作维度**: 9个动作 (静止、4方向移动、4方向射击)
- **网络架构**: [256, 256, 128] 隐藏层
- **学习率**: 0.0001
- **批次大小**: 256

## 许可证

MIT License
