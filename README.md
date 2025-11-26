# 坦克大战 AI 训练系统

基于C语言+SDL2游戏引擎和Python3+CUDA深度强化学习的坦克大战AI训练系统。

## 功能特性

✅ **完整的游戏引擎** (C + SDL2)
- 坦克移动、射击、碰撞检测
- 3点血量系统
- 优化的追踪型敌人AI
- 实时渲染和可视化

✅ **深度强化学习** (Python + PyTorch + CUDA)
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
- WASD移动，方向键射击
- 与训练后的AI对战

✅ **模型持久化**
- 定量保存模型（每100回合）
- 支持断点续训
- 模型版本管理

## 系统要求

### 软件依赖
- **操作系统**: Linux (Ubuntu 20.04+ 推荐)
- **C编译器**: GCC 9.0+
- **Python**: Python 3.8+
- **CUDA**: CUDA 12.8 (用于GPU加速)
- **SDL2**: libsdl2-dev, libsdl2-ttf-dev

### 硬件要求
- **GPU**: NVIDIA GPU (推荐GTX 1060或更好)
- **显存**: 至少2GB
- **内存**: 至少4GB

## 快速开始

### 1. 克隆项目

```bash
git clone <repository-url>
cd tank-battle-ai
```

### 2. 安装依赖

#### 安装C依赖 (Ubuntu/Debian)
```bash
make install-deps
```

或手动安装:
```bash
sudo apt-get update
sudo apt-get install -y libsdl2-dev libsdl2-ttf-dev gcc make
```

#### 安装Python依赖

首先安装PyTorch (CUDA 12.8版本):
```bash
# 访问 https://pytorch.org 获取对应CUDA版本的安装命令
# 示例 (CUDA 12.8):
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

然后安装其他依赖:
```bash
make install-python-deps
```

### 3. 编译

```bash
make all
```

这将创建:
- `libtankbattle.so` - 共享库（用于Python训练）
- `tank_battle_player` - 玩家对战程序

### 4. 开始训练

#### 纯文本模式（推荐用于长时间训练）
```bash
make train
# 或
python3 python/train.py --no-visualize
```

#### 可视化模式（用于观察训练过程）
```bash
make train-vis
# 或
python3 python/train.py --visualize
```

#### 继续训练（从已有模型）
```bash
make train-continue
# 或
python3 python/train.py --continue saved_models/latest_model.pth
```

### 5. 玩家对战模式

```bash
make run-player
# 或
./tank_battle_player
```

**控制方式:**
- `W/A/S/D` - 移动
- `方向键` - 射击
- `ESC` - 退出
- `ENTER` - 重新开始

## 项目结构

```
tank-battle-ai/
├── src/                    # C语言游戏引擎源码
│   ├── tank.c/h           # 坦克实体
│   ├── bullet.c/h         # 子弹实体
│   ├── collision.c/h      # 碰撞检测
│   ├── game.c/h           # 游戏核心逻辑
│   ├── enemy_ai.c/h       # 追踪型敌人AI
│   ├── rendering.c/h      # SDL2渲染
│   ├── ai_interface.c/h   # Python接口
│   └── main.c             # 玩家模式入口
├── python/                 # Python训练代码
│   ├── train.py           # 主训练脚本
│   ├── model.py           # DQN模型
│   ├── replay_buffer.py   # 经验回放
│   ├── env_wrapper.py     # 环境包装器
│   ├── enemy_manager.py   # 敌人管理
│   └── config.py          # 配置文件
├── saved_models/           # 保存的模型
│   ├── checkpoints/       # 训练检查点
│   └── history/           # 历史版本
├── Makefile               # 编译脚本
├── requirements.txt       # Python依赖
└── README.md              # 本文件
```

## 配置说明

所有训练配置在 `python/config.py` 中:

### 环境配置
```python
ENV_CONFIG = {
    'map_width': 800,      # 地图宽度
    'map_height': 600,     # 地图高度
    'initial_enemies': 2,  # 初始敌人数量
}
```

### 模型配置
```python
MODEL_CONFIG = {
    'state_dim': 43,                    # 状态维度
    'action_dim': 9,                    # 动作维度
    'hidden_dims': [256, 256, 128],     # 隐藏层
    'learning_rate': 0.0001,            # 学习率
    'gamma': 0.99,                      # 折扣因子
    'epsilon_start': 1.0,               # 初始探索率
    'epsilon_end': 0.05,                # 最终探索率
}
```

### 训练配置
```python
TRAINING_CONFIG = {
    'batch_size': 128,              # 批大小
    'buffer_size': 100000,          # 缓冲区大小
    'max_episodes': 100000,         # 最大回合数
    'save_interval': 100,           # 保存间隔
    'device': 'cuda',               # 设备（cuda/cpu）
}
```

## 游戏机制

### 坦克类型
- **蓝色坦克**: AI训练坦克
- **红色坦克**: 追踪型敌人
- **红白坦克**: 历史版本AI敌人
- **绿色坦克**: 玩家坦克

### 动作空间
- `0` - 待机
- `1-4` - 上下左右移动
- `5-8` - 上下左右射击

### 奖励设计
- **存活**: +0.01/帧
- **受伤**: -10
- **击杀**: +50
- **死亡**: -100
- **胜利**: +200

## 训练技巧

### 1. GPU加速
确保PyTorch正确使用GPU:
```python
import torch
print(torch.cuda.is_available())  # 应该返回True
print(torch.cuda.get_device_name(0))  # 显示GPU名称
```

### 2. 调整超参数
- 增加 `hidden_dims` 提高模型容量
- 调整 `learning_rate` 控制学习速度
- 修改 `epsilon_decay` 调整探索策略

### 3. 监控训练
观察以下指标:
- **胜率**: 应该逐渐提升
- **平均奖励**: 应该逐渐增加
- **探索率**: 应该逐渐降低
- **损失**: 应该逐渐收敛

### 4. 断点续训
定期保存检查点，出现问题可以继续训练:
```bash
python3 python/train.py --continue saved_models/checkpoint_ep1000.pth
```

## 常见问题

### Q: 编译失败，提示找不到SDL2
A: 确保安装了SDL2开发包:
```bash
sudo apt-get install libsdl2-dev libsdl2-ttf-dev
```

### Q: PyTorch提示CUDA不可用
A: 检查CUDA版本并安装对应的PyTorch:
```bash
nvcc --version  # 查看CUDA版本
# 访问 https://pytorch.org 安装对应版本
```

### Q: 训练速度太慢
A:
1. 使用纯文本模式（`--no-visualize`）
2. 确保使用GPU（`device='cuda'`）
3. 增加batch_size（如果显存足够）
4. 减少网络层数

### Q: AI一直输
A:
1. 降低初始敌人数量（`initial_enemies=1`）
2. 增加训练时间
3. 调整奖励函数
4. 检查epsilon衰减是否太快

### Q: 如何使用历史版本作为敌人
A: 训练会自动保存历史版本（每200回合）并用于自我对弈。可在 `config.py` 中调整:
```python
SELF_PLAY_CONFIG = {
    'enabled': True,
    'history_interval': 200,
    'self_play_prob': 0.3,
}
```

## 性能优化

### GPU充分利用
- 使用批处理训练（batch_size=128）
- 数据预加载到GPU
- 混合精度训练（可选）

### 代码优化
- 经验回放使用numpy数组
- C代码使用O3优化编译
- SDL2硬件加速渲染

## 贡献

欢迎提交Issue和Pull Request!

## 许可证

MIT License

## 作者

Tank Battle AI Training System

## 致谢

- SDL2 - 图形库
- PyTorch - 深度学习框架
- CUDA - GPU加速
