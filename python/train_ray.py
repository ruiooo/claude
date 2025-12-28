#!/usr/bin/env python3
"""
Ray并行采样版本的训练脚本

使用Ray Core并行运行多个游戏环境，加速数据收集
保留当前的DQN实现，只并行化环境交互部分

支持两种模式:
- 默认模式: Ray并行训练（无可视化，高性能）
- 可视化模式: 单进程训练（显示游戏画面，用于观察学习过程）
  使用方法: python train_ray.py --visualize
"""

# ============== 导入依赖库 ==============
import ray                          # Ray分布式计算框架，用于并行化
import torch                        # PyTorch深度学习框架
import numpy as np                  # 数值计算库
import os                           # 操作系统接口，用于路径操作
import argparse                     # 命令行参数解析
import time                         # 时间控制（可视化模式帧率）
import pickle                       # 轨迹数据序列化
import glob                         # 文件模式匹配
import subprocess                   # 用于调用nvidia-smi
from datetime import datetime       # 时间戳
from typing import List, Tuple, Dict     # 类型提示，提高代码可读性
from model import DQNAgent, DQN          # 自定义的DQN智能体类和网络类
from replay_buffer import ReplayBuffer  # ⚠️ 临时回退：先测试标准回放
from env_wrapper import TankBattleEnv   # 坦克大战游戏环境包装器
from human_data_loader import HumanDataLoader  # 人类数据加载器
from config import TRAINING_CONFIG, MODEL_CONFIG, ENV_CONFIG, PATHS, HUMAN_LEARNING_CONFIG, TRAJECTORY_CONFIG, GPU_PRESETS  # 配置文件


# ============================================================
# GPU利用率监控工具
# ============================================================

def get_gpu_utilization():
    """
    获取GPU利用率和显存使用情况

    Returns:
        dict: {'utilization': GPU利用率%, 'memory_used': 已用显存GB, 'memory_total': 总显存GB}
        或 None（如果无法获取）
    """
    try:
        # 使用nvidia-smi查询GPU状态
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=1
        )

        if result.returncode == 0:
            # 解析输出: "利用率, 已用显存MB, 总显存MB"
            parts = result.stdout.strip().split(',')
            if len(parts) >= 3:
                util = int(parts[0].strip())
                mem_used = float(parts[1].strip()) / 1024  # MB -> GB
                mem_total = float(parts[2].strip()) / 1024  # MB -> GB
                return {
                    'utilization': util,
                    'memory_used': mem_used,
                    'memory_total': mem_total
                }
    except Exception:
        pass

    return None


# ============================================================
# 轨迹保存和回放功能
# ============================================================

# ============================================================
# 轨迹管理系统 - 每个保存点存储10局精选对战
# ============================================================

class TrajectoryBuffer:
    """
    轨迹缓冲区 - 收集训练过程中的轨迹，定期保存精选集合

    保存策略：
    - 2局最快胜利（步数最短的2局胜利）
    - 8局随机抽取（不限胜负，展示训练多样性）
    """

    def __init__(self, max_buffer_size: int = 500):
        self.wins = []      # 胜利轨迹
        self.losses = []    # 失败+平局轨迹
        self.max_buffer_size = max_buffer_size

    def add(self, trajectory: dict):
        """添加轨迹到缓冲区"""
        if trajectory['winner'] == 0:  # 胜利
            self.wins.append(trajectory)
            # 保持缓冲区大小
            if len(self.wins) > self.max_buffer_size:
                self.wins = self.wins[-self.max_buffer_size:]
        else:  # 失败或平局
            self.losses.append(trajectory)
            if len(self.losses) > self.max_buffer_size:
                self.losses = self.losses[-self.max_buffer_size:]

    def select_best(self) -> dict:
        """
        选择最佳的10局轨迹

        策略：
        - 2局：最快胜利（步数最短的2局）
        - 8局：随机抽取（从所有轨迹中随机选择）

        Returns:
            包含wins和losses的字典（wins最多2局，剩余为随机）
        """
        import random

        selected_wins = []
        selected_random = []

        # 1. 选择2局最快胜利（如果有的话）
        if len(self.wins) >= 2:
            # 按步数排序，选择最短的2局
            sorted_wins = sorted(self.wins, key=lambda x: x['num_steps'])
            selected_wins = sorted_wins[:2]
        elif len(self.wins) > 0:
            # 不足2局，全选
            selected_wins = self.wins[:]

        # 2. 从所有轨迹中随机选择8局（不包括已选的最快胜利）
        # 合并所有轨迹（排除已选的2局最快胜利）
        all_trajectories = []

        # 添加剩余的胜利轨迹（排除已选的2局）
        if len(self.wins) >= 2:
            all_trajectories.extend(self.wins[2:])

        # 添加所有失败/平局轨迹
        all_trajectories.extend(self.losses)

        # 随机选择8局（如果轨迹不足8局，全选）
        num_random = min(8, len(all_trajectories))
        if num_random > 0:
            selected_random = random.sample(all_trajectories, num_random)

        return {
            'wins': selected_wins,  # 最快胜利（最多2局）
            'losses': selected_random,  # 随机抽取（8局或更少）
        }

    def clear(self):
        """清空缓冲区"""
        self.wins = []
        self.losses = []

    def stats(self) -> str:
        """返回缓冲区状态"""
        return f"胜:{len(self.wins)} 负/平:{len(self.losses)}"


def collect_trajectory_from_result(experiences: list, winner: int, episode: int,
                                    difficulty: int, epsilon: float, seed: int = None) -> dict:
    """
    从训练结果中构建轨迹（不需要额外采样）

    Args:
        experiences: 经验列表 [(state, action, reward, next_state, done), ...]
        winner: 胜负结果
        episode: 回合号
        difficulty: 难度
        epsilon: 探索率
        seed: 随机种子（用于回放时还原环境初始状态）

    Returns:
        轨迹字典
    """
    steps = []
    total_reward = 0.0

    for exp in experiences:
        state, action, reward, next_state, done = exp
        steps.append({
            'state': state.copy() if hasattr(state, 'copy') else state,
            'action': action,
            'reward': reward,
            'next_state': next_state.copy() if hasattr(next_state, 'copy') else next_state,
            'done': done,
        })
        total_reward += reward

    return {
        'episode': episode,
        'difficulty': difficulty,
        'epsilon': epsilon,
        'seed': seed,  # 保存随机种子
        'timestamp': datetime.now().isoformat(),
        'steps': steps,
        'winner': winner,
        'total_reward': total_reward,
        'num_steps': len(steps),
    }


def save_trajectory_batch(batch: dict, episode: int, trajectory_dir: str) -> str:
    """
    保存轨迹批次到文件

    Args:
        batch: {'wins': [...], 'losses': [...]}
        episode: 回合号
        trajectory_dir: 保存目录

    Returns:
        保存的文件路径
    """
    os.makedirs(trajectory_dir, exist_ok=True)

    # 文件名格式：ep{回合号}_batch.pkl
    filename = f"ep{episode:06d}_batch.pkl"
    filepath = os.path.join(trajectory_dir, filename)

    # 添加元信息
    batch_data = {
        'episode': episode,
        'timestamp': datetime.now().isoformat(),
        'wins': batch['wins'],
        'losses': batch['losses'],
        'total_count': len(batch['wins']) + len(batch['losses']),
    }

    with open(filepath, 'wb') as f:
        pickle.dump(batch_data, f)

    # 清理旧批次
    cleanup_old_batches(trajectory_dir, TRAJECTORY_CONFIG['max_trajectories'])

    return filepath


def cleanup_old_batches(trajectory_dir: str, max_count: int):
    """清理旧轨迹批次，保持最大数量限制"""
    files = sorted(glob.glob(os.path.join(trajectory_dir, '*_batch.pkl')))
    if len(files) > max_count:
        for f in files[:-max_count]:
            os.remove(f)


def list_trajectory_batches(trajectory_dir: str) -> List[dict]:
    """列出所有可用的轨迹批次"""
    files = sorted(glob.glob(os.path.join(trajectory_dir, '*_batch.pkl')), reverse=True)
    summaries = []

    for filepath in files:
        try:
            with open(filepath, 'rb') as f:
                batch = pickle.load(f)

            wins = batch.get('wins', [])
            losses = batch.get('losses', [])

            # 计算统计信息
            win_steps = [t['num_steps'] for t in wins] if wins else [0]
            loss_steps = [t['num_steps'] for t in losses] if losses else [0]

            summaries.append({
                'filepath': filepath,
                'filename': os.path.basename(filepath),
                'episode': batch.get('episode', 0),
                'timestamp': batch.get('timestamp', ''),
                'win_count': len(wins),
                'loss_count': len(losses),
                'total_count': len(wins) + len(losses),
                'avg_win_steps': sum(win_steps) / len(win_steps) if win_steps else 0,
                'avg_loss_steps': sum(loss_steps) / len(loss_steps) if loss_steps else 0,
                # 获取第一个轨迹的难度
                'difficulty': wins[0]['difficulty'] if wins else (losses[0]['difficulty'] if losses else 0),
            })
        except Exception as e:
            print(f"⚠️ 无法读取轨迹批次 {filepath}: {e}")

    return summaries


def load_trajectory_batch(filepath: str) -> dict:
    """加载轨迹批次"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

@ray.remote  # Ray装饰器：将类标记为远程可执行对象，实例会运行在独立的进程中
class ParallelEnvWorker:
    """
    并行环境工作器（在独立进程中运行）

    每个Worker负责运行一个游戏环境实例，独立收集经验数据
    通过Ray框架实现多进程并行，加速数据采样效率
    """

    def __init__(self, worker_id: int, hidden_dims: list = None):
        """
        初始化工作器

        Args:
            worker_id: 工作器编号，用于调试和日志记录
            hidden_dims: 神经网络隐藏层维度列表（必须与主进程一致）
        """
        self.worker_id = worker_id  # 保存工作器ID

        # 使用传入的hidden_dims，或默认值
        if hidden_dims is None:
            hidden_dims = MODEL_CONFIG['hidden_dims']

        # 构建C库文件的绝对路径（libtankbattle.so是游戏引擎的共享库）
        # 在Ray分布式环境中，worker进程可能在不同的工作目录下运行
        # 因此需要使用绝对路径而不是相对路径
        # 获取项目根目录（python/目录的父目录）
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # 构建共享库的绝对路径
        lib_path = os.path.join(project_root, 'libtankbattle.so')

        # 调试信息：显示共享库路径（用于排查路径问题）
        print(f"  [Worker {worker_id}] 共享库路径: {lib_path}")
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"找不到共享库: {lib_path}")

        # 创建游戏环境实例
        self.env = TankBattleEnv(
            lib_path=lib_path,                # C库路径
            width=ENV_CONFIG['map_width'],    # 地图宽度（从配置文件读取）
            height=ENV_CONFIG['map_height'],  # 地图高度（从配置文件读取）
            visualize=False                   # 不显示游戏画面（训练时不需要可视化）
        )

        # ✅ 修复1：创建本地策略网络副本
        # 每个Worker维护自己的网络，用于选择动作（使用训练好的策略）
        device = 'cpu'  # Worker在CPU上运行（避免GPU显存竞争）
        self.device = torch.device(device)

        # ✅ 关键修复：使用传入的hidden_dims创建网络（与主进程架构一致）
        self.policy_net = DQN(
            state_dim=MODEL_CONFIG['state_dim'],
            action_dim=MODEL_CONFIG['action_dim'],
            hidden_dims=hidden_dims  # 使用传入的hidden_dims
        ).to(self.device)

        # 设为评估模式（推理时不需要Dropout）
        self.policy_net.eval()

    def collect_episodes(self, num_episodes: int, epsilon: float) -> List[Tuple[List[Tuple], int, int]]:
        """
        收集多个回合的经验（这个方法会在远程进程中执行）

        Args:
            num_episodes: 需要收集的游戏回合数
            epsilon: ε-greedy策略的探索率（0-1之间，越大越随机）

        Returns:
            回合数据列表: [(episode_experiences, winner, seed), ...]
                - episode_experiences: 该回合的经验列表 [(state, action, reward, next_state, done), ...]
                - winner: 该回合的胜负 (0=AI胜, 1=敌人胜, -1=平局)
                - seed: 该回合的随机种子（用于轨迹回放）
        """
        episodes_data = []  # 按回合分组的经验数据

        # 循环执行指定数量的游戏回合
        for _ in range(num_episodes):
            episode_experiences = []  # 本回合的经验列表

            # 生成随机种子（用于轨迹回放）
            seed = np.random.randint(0, 2**31 - 1)

            # 重置环境，使用固定种子开始新的一局游戏
            state = self.env.reset(ENV_CONFIG['initial_enemies'], seed=seed)
            episode_done = False

            # 单个回合的主循环，直到游戏结束
            while not episode_done:
                # ========== ε-greedy 探索策略 ==========
                if np.random.random() < epsilon:
                    action = np.random.randint(0, 9)
                else:
                    action = self._select_action_with_network(state)

                # 执行动作，获取环境反馈
                next_state, reward, done, info = self.env.step(action)

                # 将这一步的经验保存到本回合列表中
                episode_experiences.append((state, action, reward, next_state, done))

                state = next_state
                episode_done = done

            # 记录本回合的数据（经验列表 + 胜负 + 种子）
            winner = info.get('winner', -1)
            episodes_data.append((episode_experiences, winner, seed))

        return episodes_data

    def update_network_params(self, state_dict: Dict):
        """
        ✅ 新增方法：更新Worker的网络参数

        主进程训练后会调用此方法，将最新参数同步给Worker
        这是关键修复！确保Worker使用最新训练的策略

        Args:
            state_dict: 策略网络的参数字典（从主进程传来）
        """
        self.policy_net.load_state_dict(state_dict)
        self.policy_net.eval()

    def _select_action_with_network(self, state: np.ndarray) -> int:
        """
        ✅ 修复2：使用神经网络选择动作

        这是最关键的修复！之前只返回随机动作，现在使用训练好的网络

        Args:
            state: 当前游戏状态（43维向量）

        Returns:
            Q值最大的动作索引 (0-8)
        """
        with torch.no_grad():
            # 转换为张量
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            # 前向传播计算Q值
            q_values = self.policy_net(state_tensor)

            # 选择Q值最大的动作（贪婪策略）
            action = q_values.argmax(dim=1).item()

        return action

    # evaluate方法已删除 - 不再需要评估

    def set_difficulty(self, level: int):
        """
        设置敌人AI难度级别（课程学习用）

        Args:
            level: 难度级别 (0-3)
                0 = 假人模式（静止靶，练习射击）
                1 = 简单模式（随机移动，练习追踪）
                2 = 中等模式（追踪但不躲避，练习战术）
                3 = 困难模式（完整AI，最终挑战）
        """
        self.env.set_difficulty(level)


def train_with_ray():
    """
    使用Ray框架进行分布式并行训练的主函数

    训练流程：
    1. 初始化Ray集群
    2. 创建多个并行环境工作器（用于采样）
    3. 创建主DQN智能体（用于训练）
    4. 循环执行：并行采样 -> 集中训练 -> 定期评估
    """

    # ========== 步骤0: GPU配置选择 ==========
    print("\n" + "="*80)
    print("🎮 GPU性能配置选择")
    print("="*80)

    # 检测GPU信息
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✓ 检测到GPU: {gpu_name}")
        print(f"✓ 显存容量: {gpu_memory:.1f} GB")
    else:
        print("⚠️  未检测到GPU，将使用CPU训练（速度较慢）")
        gpu_name = "CPU"
        gpu_memory = 0

    print(f"\n请选择训练配置:")
    print(f"  [1] 入门级配置 (GTX 1660, RTX 2060, RTX 3050)")
    print(f"      Batch: 256, Workers: 8, 每轮训练: 100次, 网络: 11万参数")
    print(f"      预期GPU利用率: 40-60%")
    print(f"  ")
    print(f"  [2] 中端级配置 (RTX 3060, RTX 3070, RTX 4060)")
    print(f"      Batch: 512, Workers: 16, 每轮训练: 200次, 网络: 40万参数")
    print(f"      预期GPU利用率: 60-80%")
    print(f"  ")
    print(f"  [3] 高端级配置 (RTX 3080, RTX 3090, RTX 4070 Ti)")
    print(f"      Batch: 1024, Workers: 24, 每轮训练: 300次, 网络: 70万参数")
    print(f"      预期GPU利用率: 70-90%")
    print(f"  ")
    print(f"  [4] 旗舰级配置 (RTX 4090, RTX 5090, A100) ⚡⚡⚡ 推荐5090")
    print(f"      Batch: 4096, Workers: 32, 每轮训练: 500次, 网络: 100万参数")
    print(f"      每轮处理: 204万样本，GPU密集计算优化")
    print(f"      预期GPU利用率: 80-95%")
    print(f"  ")
    print(f"  [5] 自定义配置")
    print(f"      手动设置所有参数（批次、网络、训练次数）")

    # 智能推荐
    if "5090" in gpu_name or "4090" in gpu_name or "A100" in gpu_name:
        recommended = 4
        print(f"\n💡 根据您的GPU ({gpu_name})，推荐选择 [4] 旗舰级配置")
    elif "3090" in gpu_name or "3080" in gpu_name or "4070" in gpu_name or "4080" in gpu_name:
        recommended = 3
        print(f"\n💡 根据您的GPU ({gpu_name})，推荐选择 [3] 高端级配置")
    elif "3060" in gpu_name or "3070" in gpu_name or "4060" in gpu_name:
        recommended = 2
        print(f"\n💡 根据您的GPU ({gpu_name})，推荐选择 [2] 中端级配置")
    else:
        recommended = 1
        print(f"\n💡 根据您的GPU ({gpu_name})，推荐选择 [1] 入门级配置")

    choice = input(f"\n请输入选项 [1-5] (默认={recommended}): ").strip()
    if not choice:
        choice = str(recommended)

    # 应用配置
    preset_map = {
        '1': 'entry',
        '2': 'mid',
        '3': 'high',
        '4': 'flagship',
        '5': 'custom',
    }

    preset_key = preset_map.get(choice, 'entry')

    if preset_key == 'custom':
        print("\n🔧 自定义配置:")
        batch_size = int(input(f"  批次大小 (默认=256): ") or "256")
        num_workers = int(input(f"  并行worker数 (默认=8): ") or "8")
        buffer_size = int(input(f"  经验缓冲区大小 (默认=35000): ") or "35000")
        train_iterations = int(input(f"  每轮训练次数 (默认=100): ") or "100")
        print(f"  网络架构 (例如: 256,256,128 或 512,512,256,128):")
        hidden_str = input(f"    (默认=256,256,128): ").strip() or "256,256,128"
        hidden_dims = [int(x.strip()) for x in hidden_str.split(',')]
        gpu_config = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'buffer_size': buffer_size,
            'train_multiplier': 4,  # 保留但不使用
            'train_iterations': train_iterations,
            'hidden_dims': hidden_dims,
            'description': '自定义配置',
            'expected_gpu_util': '根据配置而定',
        }
    else:
        gpu_config = GPU_PRESETS[preset_key]

    print(f"\n✅ 已选择: {gpu_config['description']}")
    print(f"   - 批次大小: {gpu_config['batch_size']}")
    print(f"   - 并行Worker: {gpu_config['num_workers']}")
    print(f"   - 经验缓冲区: {gpu_config['buffer_size']:,}")
    print(f"   - 每轮训练次数: {gpu_config.get('train_iterations', 100)}次")
    print(f"   - 每轮处理样本: {gpu_config['batch_size'] * gpu_config.get('train_iterations', 100):,}个")
    print(f"   - 网络架构: {' → '.join(map(str, gpu_config.get('hidden_dims', [256, 256, 128])))}")
    print(f"   - 预期GPU利用率: {gpu_config['expected_gpu_util']}")
    print(f"\n💡 提示: 每轮训练{gpu_config.get('train_iterations', 100)}次，让GPU持续工作不等待CPU采样")
    print("="*80 + "\n")

    # ========== 步骤0.5: 创建必要的目录 ==========
    # 确保所有保存路径存在（模型、checkpoint、日志等）
    for path in PATHS.values():
        os.makedirs(path, exist_ok=True)

    # ========== 步骤0.5: 创建轨迹缓冲区（用于回放功能）==========
    trajectory_buffer = None
    if TRAJECTORY_CONFIG['enabled']:
        trajectory_buffer = TrajectoryBuffer(max_buffer_size=500)
        print(f"✓ 轨迹缓冲区已创建")
        print(f"   - 保存间隔: 每{TRAJECTORY_CONFIG['save_interval']}回合保存10局精选")
        print(f"   - 保存策略: 2局最快胜利 + 8局随机抽取")

    # ========== 步骤1: 初始化Ray集群 ==========
    # 检查Ray是否已经初始化（避免重复初始化）
    if not ray.is_initialized():
        try:
            # 尝试连接到已经运行的Ray集群
            # address='auto': 自动发现本地Ray集群
            ray.init(address='auto')
            print("✓ 已连接到现有Ray集群")
        except:
            # 如果没有现有集群，启动新的本地Ray集群
            # num_cpus: 指定可用的CPU核心数（默认4个）
            ray.init(num_cpus=TRAINING_CONFIG.get('num_workers', 4))
            print("✓ 已启动新的Ray集群")

    # ========== 步骤2: 准备网络配置 ==========
    # ✅ 首先获取网络架构配置（Worker和主进程都需要）
    ray_model_config = MODEL_CONFIG.copy()
    ray_model_config['epsilon_start'] = 1.0  # 初始探索率（100%随机）
    # ✅ 使用GPU配置中的网络架构
    ray_model_config['hidden_dims'] = gpu_config.get('hidden_dims', MODEL_CONFIG['hidden_dims'])

    # ========== 步骤3: 创建并行环境工作器 ==========
    # 使用GPU配置中的worker数量
    num_workers = gpu_config['num_workers']

    # 创建多个远程工作器实例
    # .remote(i): Ray的远程调用语法，在独立进程中创建对象
    # 每个worker会在不同的CPU核心上运行，实现并行
    # ✅ 关键修复：传递hidden_dims给Worker，确保网络架构一致
    workers = [ParallelEnvWorker.remote(i, ray_model_config['hidden_dims']) for i in range(num_workers)]

    # ========== 步骤4: 创建主DQN智能体（在主进程中） ==========
    # 创建PyTorch设备对象（CPU或CUDA GPU）
    device = torch.device(TRAINING_CONFIG['device'])

    # ✅ Epsilon参数配置
    # epsilon_end 和 epsilon_decay 直接使用 MODEL_CONFIG 中的值

    # ✅ 重要：epsilon在每个回合结束后衰减一次（按回合数衰减）
    # 使用 MODEL_CONFIG 中的超慢衰减配置（epsilon_decay=0.99999）
    # 衰减公式: ε_new = max(ε_end, ε_old × decay)
    #
    # 衰减速度示例（decay=0.99999）：
    # - 1000回合:   ε ≈ 0.990
    # - 10000回合:  ε ≈ 0.905
    # - 100000回合: ε ≈ 0.368
    #
    # 这确保AI有充足的探索时间，避免过早收敛
    # 每个worker完成一个回合后，epsilon立即衰减一次

    # 计算网络参数量
    def count_params(hidden_dims, state_dim=47, action_dim=9):
        total = 0
        prev = state_dim
        for h in hidden_dims:
            total += prev * h + h
            prev = h
        total += prev * action_dim + action_dim
        return total

    param_count = count_params(ray_model_config['hidden_dims'])
    print(f"\n🧠 神经网络配置:")
    print(f"   - 架构: {MODEL_CONFIG['state_dim']} → {' → '.join(map(str, ray_model_config['hidden_dims']))} → {MODEL_CONFIG['action_dim']}")
    print(f"   - 参数量: {param_count:,} ({param_count/1e6:.2f}M)")

    # 实例化DQN智能体（包含策略网络和目标网络）
    agent = DQNAgent(
        state_dim=MODEL_CONFIG['state_dim'],      # 状态空间维度（观察向量的长度）
        action_dim=MODEL_CONFIG['action_dim'],    # 动作空间维度（可选动作的数量）
        config=ray_model_config,                  # 使用修复后的配置
        device=TRAINING_CONFIG['device']          # 训练设备（'cpu' 或 'cuda'）
    )

    # ========== 步骤3.5: 检查并加载已有模型 ==========
    start_episode = 0  # 起始回合数
    loaded_difficulty = 0  # 加载的课程进度（仅完全继续模式使用）
    latest_model_path = os.path.join(PATHS['models'], 'latest_model.pth')

    if os.path.exists(latest_model_path):
        print(f"\n📦 发现已有模型: {latest_model_path}")

        # 读取模型信息
        try:
            checkpoint = torch.load(latest_model_path, map_location=device)
            saved_episode = checkpoint.get('episode', 0)
            saved_epsilon = checkpoint.get('epsilon', 1.0)
            saved_train_step = checkpoint.get('train_step', 0)
            saved_difficulty = checkpoint.get('difficulty', 0)  # 保存的课程进度

            print(f"   回合: {saved_episode}, Epsilon: {saved_epsilon:.4f}, 训练步数: {saved_train_step:,}")
            print(f"   课程难度: {saved_difficulty}")
            print(f"\n请选择训练模式:")
            print(f"  [1] 叠加训练（推荐）- 继承网络权重，重新探索 ε=1.0，课程从0开始")
            print(f"  [2] 完全继续 - 保留所有状态（权重+ε+optimizer+课程进度）")
            print(f"  [3] 从头开始 - 丢弃旧模型，全新训练")

            choice = input("请输入选项 [1/2/3] (默认=1): ").strip()

            if choice == '3':
                # 完全重新开始
                print(f"\n✨ 从头开始训练（丢弃旧模型）")
                print(f"   - Epsilon: {ray_model_config['epsilon_start']:.2f}")
                print(f"   - 网络权重: 随机初始化")
                print(f"   - 课程进度: 从难度0开始\n")
                start_episode = 0
                loaded_difficulty = 0

            elif choice == '2':
                # 完全继续（保留所有状态，包括课程进度）
                agent.load(latest_model_path)
                start_episode = saved_episode
                loaded_difficulty = saved_difficulty  # ✅ 继承课程进度
                print(f"\n✅ 完全继续训练")
                print(f"   - 回合: 从第 {start_episode} 回合继续")
                print(f"   - Epsilon: {agent.epsilon:.4f} (保留)")
                print(f"   - Optimizer: 保留状态")
                print(f"   - 课程进度: 难度 {loaded_difficulty} (保留)\n")

            else:
                # 叠加训练（默认）- 只加载网络权重，课程从0开始
                # 只加载policy_net和target_net的权重
                agent.policy_net.load_state_dict(checkpoint['policy_net'])
                agent.target_net.load_state_dict(checkpoint['target_net'])
                # 不加载optimizer、epsilon、train_step、difficulty等
                # epsilon保持初始值1.0，optimizer保持新初始化状态

                print(f"\n🔄 叠加训练模式（继承策略，重新探索）")
                print(f"   - 网络权重: 已加载（继承已学策略）")
                print(f"   - Epsilon: {agent.epsilon:.2f} (重新开始探索)")
                print(f"   - Optimizer: 新初始化")
                print(f"   - 课程进度: 从难度0开始（重新挑战每个难度）")
                print(f"   - 效果: 多次训练结果叠加，持续改进\n")
                start_episode = 0
                loaded_difficulty = 0

        except Exception as e:
            print(f"⚠️ 加载模型失败: {e}")
            print(f"   将从头开始训练\n")
            start_episode = 0
            loaded_difficulty = 0
    else:
        print(f"ℹ️ 未找到已有模型，从头开始训练\n")

    # ========== 步骤4: 创建经验回放缓冲区 ==========
    # 使用GPU配置中的buffer_size
    replay_buffer = ReplayBuffer(
        capacity=gpu_config['buffer_size'],  # 缓冲区最大容量（存储多少条经验）
        state_dim=MODEL_CONFIG['state_dim']   # 状态维度（用于预分配内存）
    )
    print(f"✓ 经验回放缓冲区已创建 (容量: {gpu_config['buffer_size']:,})")

    # ========== 步骤4.5: 加载人类数据（模仿学习）==========
    if HUMAN_LEARNING_CONFIG['enabled']:
        print(f"\n📚 加载人类数据（模仿学习）...")
        try:
            human_loader = HumanDataLoader(
                data_dir=HUMAN_LEARNING_CONFIG['human_data_dir']
            )

            # 使用preload_to_buffer方法（内置质量过滤，默认min_reward=-50）
            loaded_count = human_loader.preload_to_buffer(
                replay_buffer,
                filter_quality=HUMAN_LEARNING_CONFIG['filter_quality']
            )

            if loaded_count > 0:
                print(f"✓ 已预加载 {loaded_count:,} 条人类经验")
                print(f"   - 缓冲区使用率: {len(replay_buffer):,} / {gpu_config['buffer_size']:,} ({len(replay_buffer)/gpu_config['buffer_size']*100:.1f}%)")
                print(f"   - AI将从人类专家策略中学习 🎓\n")
            else:
                print(f"⚠️  未找到人类数据文件\n")
        except Exception as e:
            print(f"⚠️  加载人类数据失败: {e}\n")

    # ========== 步骤5: 准备训练参数 ==========
    # 使用GPU配置中的训练参数
    max_episodes = TRAINING_CONFIG['max_episodes']  # 最大训练回合数
    batch_size = gpu_config['batch_size']           # 使用GPU配置中的批次大小
    train_multiplier = gpu_config['train_multiplier']  # 训练倍数
    episodes_per_worker = TRAINING_CONFIG.get('episodes_per_worker', 1)  # 每个worker每轮收集的回合数

    # ========== 步骤5.5: 课程学习配置（5级难度系统 v6.5）==========
    # v6.3: 敌人血量递增，让早期学习更容易
    # - 难度0-1：敌人1点血（一击必杀）
    # - 难度2：敌人2点血
    # - 难度3-4：敌人3点血（正常）
    #
    # v6.5 优化：大幅增加最小训练回合数（40倍），确保每个难度充分学习
    CURRICULUM_CONFIG = {
        'enabled': True,
        'initial_difficulty': 0,
        'difficulty_levels': [
            # (难度级别, 升级所需胜率, 评估窗口大小, 最小训练回合数)
            (0, 0.75, 200, 20000),   # 假人(1血): 需75%胜率 + 至少2万回合
            (1, 0.70, 200, 20000),   # 慢移动(1血): 需70%胜率 + 至少2万回合
            (2, 0.65, 200, 40000),   # 正常移动(2血): 需65%胜率 + 至少4万回合
            (3, 0.55, 200, 80000),   # 移动+射击: 需55%胜率 + 至少8万回合
            (4, None, None, None),   # 完整AI: 最终难度，不升级
        ],
        'difficulty_names': ['假人(1血)', '慢移动(1血)', '正常移动(2血)', '移动+射击', '完整AI'],
    }

    # 课程学习状态变量
    # 使用加载的难度（完全继续模式）或从0开始（叠加训练/从头开始）
    current_difficulty = loaded_difficulty
    difficulty_start_episode = 0  # 当前难度开始的回合数（用于最小回合数检查）
    # 使用滑动窗口记录最近N局的胜负（解决统计累积污染问题）
    from collections import deque
    HISTORY_WINDOW_SIZE = 200  # 滑动窗口大小
    difficulty_history = deque(maxlen=HISTORY_WINDOW_SIZE)  # 最近N局的胜负记录 (1=胜, 0=负)
    reward_history = deque(maxlen=HISTORY_WINDOW_SIZE)  # 最近N局的奖励记录（用于p5/p95统计）

    # 打印训练配置信息
    print(f"🚀 开始Ray并行训练（v7.0 - GPU密集训练优化）")
    print(f"   - GPU配置: {gpu_config['description']}")
    print(f"   - GPU型号: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    if torch.cuda.is_available():
        print(f"   - GPU显存: {gpu_memory:.1f} GB")
    print(f"   - 网络参数: {param_count:,} ({param_count/1e6:.2f}M)")
    print(f"   - 工作器数量: {num_workers}")
    print(f"   - 批次大小: {batch_size}")
    print(f"   - 每轮训练次数: {gpu_config.get('train_iterations', 100)}次（固定）")
    print(f"   - 每轮处理样本: {batch_size * gpu_config.get('train_iterations', 100):,}个")
    print(f"   - 经验缓冲区: {gpu_config['buffer_size']:,}")
    print(f"   - 预期GPU利用率: {gpu_config['expected_gpu_util']}")
    print(f"   - 主设备: {device}")
    print(f"   - Worker设备: CPU")
    print(f"   - 每轮每工作器收集: {episodes_per_worker} 回合")
    print(f"   - 缓冲区初始大小: {len(replay_buffer):,} / {gpu_config['buffer_size']:,} ({len(replay_buffer)/gpu_config['buffer_size']*100:.1f}%)")
    if len(replay_buffer) > 0:
        print(f"   🎓 包含人类数据: {len(replay_buffer):,} 条经验（模仿学习）")
    print(f"   ✅ Worker使用训练好的神经网络策略（非随机）")
    print(f"   ✅ 每轮采样前同步最新网络参数")
    print(f"   ✅ 固定训练次数，GPU持续工作不等待CPU")
    print(f"   ✅ 模型保存到标准路径（玩家模式可用）")
    print(f"   ✅ Epsilon按回合数衰减（decay={MODEL_CONFIG['epsilon_decay']}，每回合衰减一次）")

    # 课程学习信息
    if CURRICULUM_CONFIG['enabled']:
        print(f"\n📚 课程学习已启用（5级难度系统 v6.2）:")
        print(f"   难度0 [假人]: 静止靶 → 50%胜率升级")
        print(f"   难度1 [慢移动]: 只移动不射击 → 45%胜率升级")
        print(f"   难度2 [移动+射击]: 随机移动+20%射击 → 40%胜率升级")
        print(f"   难度3 [追踪]: 追踪AI+简单射击 → 35%胜率升级")
        print(f"   难度4 [完整AI]: 预判+躲避 ← 最终目标")
        print(f"   当前难度: {current_difficulty} [{CURRICULUM_CONFIG['difficulty_names'][current_difficulty]}]")

    print()  # 空行

    # ========== 步骤5.6: 初始化所有Worker的难度级别 ==========
    if CURRICULUM_CONFIG['enabled']:
        difficulty_futures = [
            worker.set_difficulty.remote(current_difficulty)
            for worker in workers
        ]
        ray.get(difficulty_futures)
        print(f"✅ 所有Worker已设置为难度 {current_difficulty} [{CURRICULUM_CONFIG['difficulty_names'][current_difficulty]}]\n")

    episode_count = start_episode  # 已完成的训练回合计数器（从加载的模型继续）

    # 统计变量
    total_wins = 0      # 总胜利次数
    total_losses = 0    # 总失败次数
    total_draws = 0     # 总平局次数
    all_rewards = []    # 所有回合的奖励列表

    # ========== 步骤6: 主训练循环 ==========
    try:
        # 持续训练直到达到最大回合数
        while episode_count < max_episodes:

            # ✅ 每轮采样前，同步网络参数给所有Worker
            # 这确保Worker使用最新训练的策略进行采样
            policy_state_dict = agent.policy_net.state_dict()

            # 将参数移到CPU（Worker在CPU上运行）
            cpu_state_dict = {k: v.cpu() for k, v in policy_state_dict.items()}

            # 异步更新所有Worker的网络参数
            update_futures = [
                worker.update_network_params.remote(cpu_state_dict)
                for worker in workers
            ]

            # 等待所有Worker更新完成
            ray.get(update_futures)

            # ==================== 阶段1: 并行采样（多进程） ====================
            # 这个阶段利用多个CPU核心并行收集游戏经验，提高采样效率

            # 向所有worker分发采样任务（异步调用）
            # .remote(): Ray的远程方法调用，立即返回Future对象而不阻塞
            futures = [
                worker.collect_episodes.remote(  # 调用远程worker的方法
                    episodes_per_worker,         # 每个worker收集的回合数
                    agent.epsilon                # 当前的探索率（从主智能体获取）
                )
                for worker in workers  # 对每个worker都发起调用
            ]

            # 等待所有worker完成采样任务（阻塞等待）
            # ray.get(): 获取Future的实际结果，会阻塞直到所有任务完成
            # all_results: List[List[Tuple[episode_experiences, winner]]]
            all_results = ray.get(futures)

            # 合并所有worker收集的经验到主进程的回放缓冲区
            episode_rewards_this_round = []
            for worker_episodes in all_results:  # 遍历每个worker的结果
                for episode_experiences, winner, seed in worker_episodes:  # 遍历每个回合（现在包含seed）
                    # 计算本回合奖励并将经验存入缓冲区
                    episode_reward = 0
                    for exp in episode_experiences:
                        # 显式转换为numpy数组，防止Ray序列化导致的内存问题
                        state, action, reward, next_state, done = exp
                        state = np.asarray(state, dtype=np.float32)
                        next_state = np.asarray(next_state, dtype=np.float32)
                        replay_buffer.push(state, action, reward, next_state, done)
                        episode_reward += reward

                    # 记录本回合奖励
                    all_rewards.append(episode_reward)
                    episode_rewards_this_round.append(episode_reward)
                    reward_history.append(episode_reward)

                    # ========== 收集轨迹到缓冲区 ==========
                    if trajectory_buffer is not None:
                        trajectory = collect_trajectory_from_result(
                            episode_experiences, winner, episode_count,
                            current_difficulty, agent.epsilon, seed=seed  # 传递seed
                        )
                        trajectory_buffer.add(trajectory)
                    # 统计胜负
                    if winner == 0:
                        total_wins += 1
                    elif winner == 1:
                        total_losses += 1
                    else:
                        total_draws += 1

                    # ========== 课程学习：滑动窗口统计胜率 ==========
                    if CURRICULUM_CONFIG['enabled']:
                        # 记录本局胜负到滑动窗口
                        difficulty_history.append(1 if winner == 0 else 0)

                        # 获取当前难度的升级条件
                        level, required_winrate, window_size, min_episodes = CURRICULUM_CONFIG['difficulty_levels'][current_difficulty]

                        # 计算在当前难度训练的回合数
                        episodes_at_difficulty = episode_count - difficulty_start_episode

                        # 窗口满了才检查升级条件
                        if required_winrate is not None and len(difficulty_history) >= window_size:
                            # 计算最近N局的胜率（滑动窗口）
                            current_winrate = sum(difficulty_history) / len(difficulty_history)

                            # 同时满足：胜率达标 + 最小回合数达标
                            if current_winrate >= required_winrate and episodes_at_difficulty >= min_episodes:
                                # 升级难度！
                                old_difficulty = current_difficulty
                                current_difficulty = min(current_difficulty + 1, 4)

                                # 难度升级时部分重置epsilon（给AI探索新策略的空间）
                                old_epsilon = agent.epsilon
                                epsilon_reset_threshold = 0.5  # 重置到的目标值
                                if agent.epsilon < epsilon_reset_threshold:
                                    agent.epsilon = epsilon_reset_threshold

                                print(f"\n\n🎉 难度升级！")
                                print(f"   {CURRICULUM_CONFIG['difficulty_names'][old_difficulty]} → {CURRICULUM_CONFIG['difficulty_names'][current_difficulty]}")
                                print(f"   最近{window_size}局胜率: {current_winrate:.1%} (目标: {required_winrate:.0%})")
                                print(f"   当前难度训练回合: {episodes_at_difficulty} (最小要求: {min_episodes})")
                                print(f"   总回合: {episode_count}")
                                if old_epsilon < epsilon_reset_threshold:
                                    print(f"   Epsilon重置: {old_epsilon:.3f} → {agent.epsilon:.3f} (增加探索)")
                                print()

                                # 更新所有Worker的难度
                                difficulty_futures = [
                                    worker.set_difficulty.remote(current_difficulty)
                                    for worker in workers
                                ]
                                ray.get(difficulty_futures)

                                # 清空滑动窗口，重新开始统计新难度
                                difficulty_history.clear()
                                difficulty_start_episode = episode_count  # 记录新难度开始时间

                    # ✅ 按回合数衰减epsilon（每完成一个回合衰减一次）
                    agent.epsilon = max(
                        MODEL_CONFIG['epsilon_end'],
                        agent.epsilon * MODEL_CONFIG['epsilon_decay']
                    )

                    # 更新回合计数
                    episode_count += 1

            # 打印采样进度（\r使光标回到行首，实现原地更新）
            avg_reward_this_round = np.mean(episode_rewards_this_round) if episode_rewards_this_round else 0

            # 计算奖励统计（p5/p95百分位数）
            if len(reward_history) >= 10:
                reward_p5 = np.percentile(reward_history, 5)
                reward_p95 = np.percentile(reward_history, 95)
                reward_stats = f"[p5:{reward_p5:.0f} p95:{reward_p95:.0f}]"
            else:
                reward_stats = ""

            # 计算总胜率
            total_games = total_wins + total_losses + total_draws
            win_rate = (total_wins / total_games * 100) if total_games > 0 else 0

            # 计算当前难度下的滑动窗口胜率
            difficulty_winrate = (sum(difficulty_history) / len(difficulty_history) * 100) if len(difficulty_history) > 0 else 0
            difficulty_games = len(difficulty_history)

            # 显示训练进度（包含课程学习信息和GPU监控）
            difficulty_name = CURRICULUM_CONFIG['difficulty_names'][current_difficulty] if CURRICULUM_CONFIG['enabled'] else ''
            progress_str = (f"\r回合 {episode_count}/{max_episodes} | "
                  f"难度:{current_difficulty}[{difficulty_name}] | "  # 当前难度
                  f"ε={agent.epsilon:.3f} | "  # 探索率
                  f"奖励:{avg_reward_this_round:.1f}{reward_stats} | "
                  f"胜率:{win_rate:.1f}%({total_wins}W/{total_losses}L/{total_draws}D) | "
                  f"近{HISTORY_WINDOW_SIZE}局:{difficulty_winrate:.0f}%({difficulty_games}/{HISTORY_WINDOW_SIZE})")

            # 显示损失值（如果正在训练）
            if len(replay_buffer) >= TRAINING_CONFIG['min_buffer_size'] and 'avg_loss' in locals():
                progress_str += f" | Loss: {avg_loss:.4f}"

            # 每100回合显示一次GPU利用率
            if episode_count % 100 == 0 and torch.cuda.is_available():
                gpu_stats = get_gpu_utilization()
                if gpu_stats:
                    progress_str += f" | GPU: {gpu_stats['utilization']}%({gpu_stats['memory_used']:.1f}GB/{gpu_stats['memory_total']:.1f}GB)"

            # 显示数据来源（简洁版）
            if len(replay_buffer) > 0:
                source_stats = replay_buffer.get_source_stats()
                human_count = source_stats.get('human_count', 0)
                ai_count = source_stats.get('ai_count', 0)
                total_count = human_count + ai_count

                if human_count > 0:
                    human_ratio = human_count / total_count if total_count > 0 else 0
                    progress_str += f" | 数据: 👤{human_count}({human_ratio:.0%}) + 🤖{ai_count}"
                else:
                    progress_str += f" | 数据: 🤖{ai_count}"

            print(progress_str, end='')

            # ==================== 阶段2: 集中训练（GPU加速） ====================
            # 这个阶段在主进程的GPU上训练神经网络，利用GPU并行计算优势

            # 检查缓冲区是否有足够的经验开始训练
            # min_buffer_size: 最小缓冲区大小，避免初期数据过少导致训练不稳定
            if len(replay_buffer) >= TRAINING_CONFIG['min_buffer_size']:

                # ✅ GPU优化：使用固定训练次数，持续利用GPU
                # 不再依赖采样量，而是固定每轮训练N次（让GPU持续工作）
                # 5090配置：每轮训练500次 × batch_size 4096 = 每轮处理204万样本
                num_updates = gpu_config.get('train_iterations', 100)

                # ✅ 始终使用人类数据（如果有的话）
                if hasattr(replay_buffer, 'include_human_data'):
                    replay_buffer.include_human_data = True

                # 执行多次梯度更新（GPU密集计算）
                total_loss = 0.0
                for _ in range(num_updates):
                    # ⚠️ 临时回退：使用标准回放（不使用优先级）
                    batch = replay_buffer.sample(batch_size)

                    # ✅ 只使用Double DQN（不使用优先回放）
                    loss = agent.train(batch, use_prioritized=False)
                    total_loss += loss

                # 记录平均损失（用于监控）
                avg_loss = total_loss / num_updates if num_updates > 0 else 0.0

                # 注意: epsilon衰减已经在每个回合完成时执行（见第463-468行）
                # 不再需要在这里按采样轮次衰减

            # ==================== 阶段3: 定期保存模型 ====================
            # 每10000个回合保存一次模型

            if episode_count % 10000 == 0 and episode_count > 0:
                # 构造包含回合数的checkpoint
                checkpoint = {
                    'policy_net': agent.policy_net.state_dict(),
                    'target_net': agent.target_net.state_dict(),
                    'optimizer': agent.optimizer.state_dict(),
                    'epsilon': agent.epsilon,
                    'train_step': agent.train_step,
                    'episode': episode_count,
                    'difficulty': current_difficulty,  # ✅ 保存课程进度
                }

                # 保存checkpoint
                checkpoint_path = os.path.join(
                    PATHS['checkpoints'],
                    f'checkpoint_ep{episode_count}.pth'
                )
                torch.save(checkpoint, checkpoint_path)

                # 同时更新latest_model（玩家模式会优先加载这个，也用于断点续训）
                latest_path = os.path.join(PATHS['models'], 'latest_model.pth')
                torch.save(checkpoint, latest_path)

                print(f"✓ 已保存到: {checkpoint_path} (回合 {episode_count}, ε={agent.epsilon:.4f})")

            # ==================== 阶段3.5: 定期保存轨迹批次（10局精选）====================
            if TRAJECTORY_CONFIG['enabled'] and trajectory_buffer is not None:
                if episode_count % TRAJECTORY_CONFIG['save_interval'] == 0 and episode_count > 0:
                    # 选择最佳10局轨迹
                    # - 2局最快胜利（步数最短）
                    # - 8局随机抽取（不限胜负）
                    best_trajectories = trajectory_buffer.select_best()
                    fastest_wins = len(best_trajectories['wins'])  # 最快胜利（最多2局）
                    random_games = len(best_trajectories['losses'])  # 随机抽取（8局）

                    if fastest_wins + random_games > 0:
                        # 保存轨迹批次
                        batch_path = save_trajectory_batch(
                            best_trajectories, episode_count, PATHS['trajectories']
                        )
                        print(f"📹 轨迹批次已保存: ep{episode_count} [{fastest_wins}局最快胜利 + {random_games}局随机] 缓冲:{trajectory_buffer.stats()}")

                        # 清空缓冲区（开始收集下一批）
                        trajectory_buffer.clear()

    except KeyboardInterrupt:
        # 捕获Ctrl+C中断信号，允许用户手动停止训练
        print("\n\n训练被中断")

    finally:
        # 无论训练正常结束还是被中断，都执行清理工作

        # ✅ 修复7：保存到final_model.pth（玩家模式会加载这个）
        final_checkpoint = {
            'policy_net': agent.policy_net.state_dict(),
            'target_net': agent.target_net.state_dict(),
            'optimizer': agent.optimizer.state_dict(),
            'epsilon': agent.epsilon,
            'train_step': agent.train_step,
            'episode': episode_count,
            'difficulty': current_difficulty,  # ✅ 保存课程进度
        }
        final_path = os.path.join(PATHS['models'], 'final_model.pth')
        torch.save(final_checkpoint, final_path)

        # 打印最终统计
        print("\n" + "="*80)
        print("训练完成!")
        print("="*80)
        print(f"总回合数: {episode_count}")
        if all_rewards:
            print(f"平均奖励: {np.mean(all_rewards):.2f}")
            print(f"最佳奖励: {np.max(all_rewards):.2f}")

        # 打印胜率统计
        total_games = total_wins + total_losses + total_draws
        if total_games > 0:
            final_win_rate = total_wins / total_games * 100
            print(f"胜/负/平: {total_wins} / {total_losses} / {total_draws}")
            print(f"最终胜率: {final_win_rate:.2f}%")

        # 打印课程学习结果
        if CURRICULUM_CONFIG['enabled']:
            print(f"\n📚 课程学习结果:")
            print(f"   最终难度: {current_difficulty} [{CURRICULUM_CONFIG['difficulty_names'][current_difficulty]}]")
            if len(difficulty_history) > 0:
                final_diff_winrate = sum(difficulty_history) / len(difficulty_history) * 100
                print(f"   近{len(difficulty_history)}局胜率: {final_diff_winrate:.1f}%")
            if current_difficulty == 3:
                print(f"   🎉 恭喜！已达到最高难度！")
            else:
                print(f"   💡 提示：继续训练可提升到更高难度")

        print(f"\n✓ 最终模型已保存到: {final_path}")
        print("  玩家对战模式将自动加载此模型")

        # 打印轨迹保存信息
        if TRAJECTORY_CONFIG['enabled']:
            traj_files = glob.glob(os.path.join(PATHS['trajectories'], '*.pkl'))
            print(f"\n📹 已保存 {len(traj_files)} 条训练轨迹")
            print(f"   使用 'make demo' 可视化回放训练过程")

        print("="*80 + "\n")

        # 关闭Ray集群，释放资源
        ray.shutdown()


def replay_trajectory():
    """
    轨迹回放模式 - 可视化还原训练过程中的对战

    功能：
    1. 列出所有保存的轨迹批次（每批10局）
    2. 用户选择要回放的批次
    3. 依次回放批次中的所有轨迹
    4. 按空格键继续下一局
    5. 按 q 或 Ctrl+C 退出

    存储结构：
    - 每1000回合保存一个批次（10局精选对战）
    - 2局：最快胜利（步数最短的2局胜利）
    - 8局：随机抽取（不限胜负，展示训练多样性）
    """

    # 难度名称映射
    DIFFICULTY_NAMES = ['假人(1血)', '慢移动(1血)', '正常移动(2血)', '移动+射击', '完整AI']
    ACTION_NAMES = ['静止', '上移', '下移', '左移', '右移', '上射', '下射', '左射', '右射']

    # ========== 查找可用轨迹批次 ==========
    trajectory_dir = PATHS.get('trajectories', 'saved_models/trajectories')

    if not os.path.exists(trajectory_dir):
        print("❌ 轨迹目录不存在！请先运行训练。")
        print(f"   目录: {trajectory_dir}")
        print("   运行: make train")
        return

    batches = list_trajectory_batches(trajectory_dir)

    if not batches:
        print("❌ 未找到任何轨迹批次！")
        print(f"   请确保训练已运行超过 {TRAJECTORY_CONFIG['save_interval']} 回合")
        print("   运行: make train")
        return

    # ========== 创建可视化环境 ==========
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lib_path = os.path.join(project_root, 'libtankbattle.so')

    # 回放模式禁用VSYNC，支持任意倍速播放
    env = TankBattleEnv(
        lib_path=lib_path,
        width=ENV_CONFIG['map_width'],
        height=ENV_CONFIG['map_height'],
        visualize=True,
        disable_vsync=True  # 禁用垂直同步，突破60fps限制
    )

    def wait_for_space():
        """等待用户按空格键继续"""
        print("\n>>> 按 Enter 继续下一局，输入 q 退出 <<<")
        user_input = input().strip().lower()
        return user_input != 'q'

    def play_single_trajectory(traj, traj_num, total_num, frame_delay):
        """
        播放单个轨迹（优化帧率控制）

        Returns:
            'completed': 正常播放完成
            'skipped': 用户按空格跳过
            'quit': 用户按ESC或关闭窗口退出
        """
        steps = traj['steps']
        result_text = '🏆胜' if traj['winner'] == 0 else ('💀负' if traj['winner'] == 1 else '🤝平')
        diff_name = DIFFICULTY_NAMES[traj['difficulty']] if traj['difficulty'] < len(DIFFICULTY_NAMES) else '?'

        print(f"\n{'='*70}")
        print(f"📹 第 {traj_num}/{total_num} 局 | {result_text} | 难度:{traj['difficulty']}[{diff_name}] | {len(steps)}步")
        print(f"   奖励: {traj['total_reward']:.1f} | ε: {traj['epsilon']:.3f}")
        print(f"{'='*70}")

        # 设置难度
        env.set_difficulty(traj['difficulty'])

        # 重置环境，使用保存的随机种子（确保环境初始状态一致）
        seed = traj.get('seed', None)
        if seed is None:
            print(f"  ⚠️  轨迹未保存种子，环境状态可能不一致")
        env.reset(ENV_CONFIG['initial_enemies'], seed=seed)

        # 使用精确的时间戳控制帧率
        start_time = time.time()
        actual_steps = 0  # 实际执行的步数
        skipped = False

        for i, step in enumerate(steps):
            # 检测SDL按键事件
            key = env.poll_key()
            if key == 32:  # 空格键 - 跳过当前局
                print(f"  ⏭️ 跳过（空格）")
                skipped = True
                break
            elif key == 27 or key == 113 or key == -1:  # ESC/Q键或窗口关闭 - 退出
                print(f"  ⏹️ 退出")
                return 'quit'

            # 计算当前帧应该到达的时间
            target_time = start_time + (i * frame_delay)

            env.render()
            action = step['action']
            next_state, reward, done, info = env.step(action)
            actual_steps += 1

            # 每50步显示一次进度
            if i % 50 == 0:
                action_name = ACTION_NAMES[action] if action < len(ACTION_NAMES) else '?'
                elapsed = time.time() - start_time
                real_fps = (i + 1) / elapsed if elapsed > 0 else 0
                # 显示敌人数量（诊断用）
                enemy_count = info.get('enemy_count', '?')
                print(f"  步骤 {i+1:4d}/{len(steps)} | {action_name} | 敌人:{enemy_count} | 帧率:{real_fps:.0f}fps | 空格=跳过")

            # 精确延迟到目标时间
            current_time = time.time()
            sleep_time = target_time - current_time
            if sleep_time > 0:
                time.sleep(sleep_time)

            if done:
                # 游戏提前结束，显示诊断信息
                if actual_steps < len(steps):
                    winner_text = "AI胜" if info.get('winner') == 0 else ("敌人胜" if info.get('winner') == 1 else "平局")
                    print(f"  ⚠️  游戏在第 {actual_steps}/{len(steps)} 步提前结束 | 结果:{winner_text}")
                break

        if skipped:
            return 'skipped'

        # 显示结果和性能统计
        total_time = time.time() - start_time
        avg_fps = actual_steps / total_time if total_time > 0 else 0

        # 区分记录步数和实际执行步数
        steps_info = f"{actual_steps}步"
        if actual_steps != len(steps):
            steps_info = f"{actual_steps}/{len(steps)}步（提前结束）"

        print(f"  ✅ 完成! {result_text} | {steps_info} | 奖励:{traj['total_reward']:.1f} | 平均帧率: {avg_fps:.0f} fps")

        return 'completed'

    try:
        # ========== 主循环 ==========
        while True:
            # 显示批次列表
            print("\n📹 轨迹回放模式 - 批次选择")
            print("=" * 70)
            print("\n💡 说明：每1000回合保存一个批次（10局精选对战）")
            print("   - 2局：最快胜利（步数最短的2局）")
            print("   - 8局：随机抽取（不限胜负，展示多样性）\n")
            print("可用批次:")

            display_count = min(20, len(batches))
            for i, batch in enumerate(batches[:display_count]):
                diff_name = DIFFICULTY_NAMES[batch['difficulty']] if batch['difficulty'] < len(DIFFICULTY_NAMES) else '?'
                fastest_count = batch['win_count']  # 最快胜利
                random_count = batch['loss_count']   # 随机抽取
                print(f"  [{i:2d}] 回合 {batch['episode']:6d} | "
                      f"难度:{batch['difficulty']}[{diff_name}] | "
                      f"{fastest_count}局最快+{random_count}局随机 | "
                      f"平均步数: {batch['avg_win_steps']:.0f}/{batch['avg_loss_steps']:.0f}")

            if len(batches) > display_count:
                print(f"  ... 还有 {len(batches) - display_count} 个批次")

            # 用户选择
            print("\n输入 q 退出")
            choice = input(f"请选择批次 [0-{display_count-1}] (默认=0): ").strip()

            if choice.lower() == 'q':
                print("\n退出回放模式")
                break

            try:
                idx = int(choice) if choice else 0
                idx = max(0, min(idx, display_count - 1))
            except:
                idx = 0

            selected = batches[idx]
            print(f"\n已选择: 回合 {selected['episode']} 批次")

            # 选择回放速度
            print("\n回放速度:")
            print("  [0] 0.5x (慢放)")
            print("  [1] 1x   (实时)")
            print("  [2] 2x   (快速)")
            print("  [3] 4x   (极速)")
            speed_choice = input("选择速度 [0-3] (默认=2): ").strip()
            speed_map = {'0': 0.5, '1': 1.0, '2': 2.0, '3': 4.0}
            playback_speed = speed_map.get(speed_choice, 2.0)
            frame_delay = 0.016 / playback_speed

            # 加载批次
            batch_data = load_trajectory_batch(selected['filepath'])
            wins = batch_data.get('wins', [])
            losses = batch_data.get('losses', [])
            all_trajectories = wins + losses
            total_count = len(all_trajectories)

            print(f"\n🎬 开始连续回放批次 ep{selected['episode']}")
            print(f"   共 {total_count} 局（{len(wins)}局最快胜利 + {len(losses)}局随机）")
            print(f"   回放速度: {playback_speed}x")
            print(f"\n   💡 空格=跳过当前局 | ESC/Q=退出回放")
            print("-" * 70)

            # 连续播放所有10局
            quit_requested = False
            for idx, traj in enumerate(all_trajectories):
                # 显示当前是最快胜利还是随机抽取
                if idx < len(wins):
                    category = "🏆最快胜利"
                else:
                    category = "🎲随机"

                # 显示进度
                print(f"\n[{idx+1}/{total_count}] {category}")

                result = play_single_trajectory(traj, idx+1, total_count, frame_delay)

                if result == 'quit':
                    quit_requested = True
                    break
                elif result == 'skipped':
                    # 跳过后继续下一局
                    continue
                else:
                    # 正常完成，短暂暂停后继续
                    if idx < total_count - 1:
                        time.sleep(0.5)

            if quit_requested:
                print("\n退出回放模式")
                break

            # 批次回放结束
            print("\n" + "=" * 70)
            print(f"✅ 批次 ep{selected['episode']} 回放完成!")
            print(f"   最快胜利: {len(wins)}局 | 随机抽取: {len(losses)}局")
            print("=" * 70)

            # 询问是否继续
            cont = input("\n继续观看其他批次? [Y/n]: ").strip().lower()
            if cont == 'n':
                print("\n退出回放模式")
                break

            # 刷新批次列表
            batches = list_trajectory_batches(trajectory_dir)

    except KeyboardInterrupt:
        print("\n\n回放被中断")

    finally:
        env.close()


if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='Tank Battle AI 训练与回放脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python train_ray.py                    # 启动Ray并行训练
  python train_ray.py --demo             # 轨迹回放模式（还原训练过程）
"""
    )
    parser.add_argument('--demo', '-v', action='store_true',
                        help='轨迹回放模式（可视化还原训练过程中的对战）')
    args = parser.parse_args()

    if args.demo:
        # 轨迹回放模式：可视化还原训练过程
        replay_trajectory()
    else:
        # 默认模式：Ray并行训练，高性能
        train_with_ray()
