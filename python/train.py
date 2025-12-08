"""
train.py - DQN强化学习主训练脚本
================================

【核心功能】
这是坦克大战AI的训练入口程序，实现了完整的DQN训练循环。

【训练流程】
1. 初始化环境（C游戏引擎 + Python训练系统）
2. 加载/创建DQN模型
3. 运行训练循环：
   - 环境交互（state → action → reward）
   - 经验存储（experience replay buffer）
   - 模型训练（批量梯度下降）
   - 模型保存（checkpoints）
4. 输出训练统计

【使用方法】
    python train.py --visualize          # 可视化训练（慢但直观）
    python train.py --no-visualize       # 纯文本训练（快，推荐）

【主要特性】
- GPU加速训练（自动检测CUDA）
- 经验回放缓冲区（提高样本效率）
- 目标网络（稳定训练）
- Epsilon-greedy探索（平衡探索/利用）
- 动态难度调整（自适应挑战）
- 自我对弈系统（与历史版本对战）
- 人类数据学习（模仿学习）
- 实时训练监控（胜率、奖励、损失）

【关键参数】（见config.py）
- state_dim: 43维状态（6 AI状态 + 25 敌人 + 12 子弹）
- action_dim: 9维动作（idle + 4移动 + 4射击）
- hidden_dims: [512, 512, 256, 128] 网络结构
- epsilon_decay: 0.9999 探索衰减
- batch_size: 256 批次大小
- buffer_size: 100000 经验池容量
"""

# ==================== 依赖导入 ====================

import argparse  # 命令行参数解析
import os  # 文件系统操作
import time  # 时间测量
import numpy as np  # 数值计算
import torch  # PyTorch深度学习框架
from datetime import datetime  # 日期时间处理
import glob  # 文件模式匹配

# 导入自定义模块
from config import *  # 训练配置（超参数、路径等）
from model import DQNAgent  # DQN智能体（网络+优化器）
from replay_buffer import ReplayBuffer  # 经验回放缓冲区
from env_wrapper import TankBattleEnv  # 游戏环境包装器（C库接口）
from enemy_manager import EnemyManager  # 敌人管理器（难度+自我对弈）
from human_data_loader import HumanDataLoader  # 人类经验数据加载器


def select_model_interactively():
    """
    交互式模型选择界面

    【功能】
    让用户选择是创建新模型还是继续训练已有模型

    【选项】
    - [0] 创建新模型 - 从头开始训练
    - [1-5] 继续训练 - 从已有checkpoint恢复

    【优先级】
    1. saved_models/final_model.pth (最终模型)
    2. saved_models/latest_model.pth (最新模型)
    3. saved_models/checkpoints/*.pth (定期检查点)

    【返回值】
    str: 选定的模型路径，如果创建新模型则返回None

    【注意】
    - 只显示最近修改的5个模型（避免列表过长）
    - 按修改时间倒序排列（最新的在前）
    """
    print("\n" + "="*60)
    print("模型选择")
    print("="*60)

    # 扫描saved_models目录下的所有.pth文件
    model_files = []
    if os.path.exists(PATHS['models']):  # 检查目录是否存在
        # glob.glob: 文件模式匹配，查找所有.pth文件
        model_files = glob.glob(os.path.join(PATHS['models'], '*.pth'))
        # 按修改时间降序排序（最新的在前）
        # key=os.path.getmtime: 使用文件修改时间作为排序键
        model_files.sort(key=os.path.getmtime, reverse=True)

    # 扫描checkpoints子目录
    checkpoint_files = []
    if os.path.exists(PATHS['checkpoints']):
        checkpoint_files = glob.glob(os.path.join(PATHS['checkpoints'], '*.pth'))
        checkpoint_files.sort(key=os.path.getmtime, reverse=True)

    # 合并两个列表：优先显示主目录的模型（final/latest）
    all_files = model_files + checkpoint_files

    # 如果没有找到任何模型，直接创建新模型
    if not all_files:
        print("未找到已有模型，将创建新模型")
        return None

    # 只显示最近的5个模型（用户体验优化）
    all_files = all_files[:5]

    # 显示选项列表
    print("\n请选择:")
    print("  [0] 创建新模型")
    print("\n可用的模型 (最近5个):")

    # enumerate(list, start=1): 从1开始编号
    for i, model_path in enumerate(all_files, 1):
        # 获取文件大小（转换为MB）
        file_size = os.path.getsize(model_path) / (1024 * 1024)
        # 获取修改时间并格式化
        mod_time = datetime.fromtimestamp(os.path.getmtime(model_path))
        # 显示文件信息：序号、文件名、大小、修改时间
        print(f"  [{i}] {os.path.basename(model_path)}")
        print(f"      大小: {file_size:.2f} MB | 修改时间: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")

    print("\n" + "="*60)
    print("提示: 按 Ctrl+C 取消并创建新模型")

    # 用户输入循环（持续等待有效输入）
    while True:
        try:
            # input(): 等待用户输入字符串
            choice = input("请输入选项 (0-%d): " % len(all_files))
            choice = int(choice)  # 转换为整数（可能抛出ValueError）

            if choice == 0:
                # 用户选择创建新模型
                print("✓ 将创建新模型")
                return None  # 返回None表示创建新模型
            elif 1 <= choice <= len(all_files):
                # 用户选择已有模型（索引从1开始，需要-1）
                selected_model = all_files[choice - 1]
                print(f"✓ 将从 {os.path.basename(selected_model)} 继续训练")
                return selected_model  # 返回模型路径
            else:
                # 无效输入（超出范围）
                print("⚠ 无效选项，请重新输入")

        except (ValueError, KeyboardInterrupt):
            # ValueError: 输入不是数字
            # KeyboardInterrupt: 用户按Ctrl+C
            print("\n⚠ 已取消，将创建新模型")
            return None
        except EOFError:
            # EOFError: 输入流结束（例如管道输入）
            print("\n⚠ 已取消，将创建新模型")
            return None


class Trainer:
    """
    DQN训练器类
    ===========

    【职责】
    管理整个强化学习训练流程，包括：
    - 环境初始化
    - 模型创建/加载
    - 训练循环执行
    - 数据收集与存储
    - 模型保存与评估
    - 训练监控与日志

    【核心组件】
    1. env: TankBattleEnv - C游戏引擎包装器
    2. agent: DQNAgent - DQN智能体（网络+算法）
    3. replay_buffer: ReplayBuffer - 经验回放缓冲区
    4. enemy_manager: EnemyManager - 敌人管理（难度+自我对弈）

    【训练流程】
    1. run_episode(): 运行一个完整回合
       - reset环境 → 循环(select action → step → store) → 统计
    2. train_step(): 从缓冲区采样并训练网络
    3. _save_checkpoint(): 定期保存模型
    4. _print_progress(): 输出训练统计

    【关键概念】
    - Episode（回合）: 从环境reset到game_over的一次完整游戏
    - Step（步）: 一次action执行（约1/60秒）
    - Batch（批次）: 从回放缓冲区采样的一组经验用于训练
    """

    def __init__(self, visualize: bool = False, continue_from: str = None):
        """
        初始化训练器

        【参数】
        visualize: bool - 是否启用SDL2可视化
            - True: 实时显示游戏画面（慢，约1000回合/小时）
            - False: 纯文本模式（快，约3000回合/小时）
        continue_from: str - 要加载的模型路径
            - None: 创建新模型，随机初始化权重
            - "path/to/model.pth": 加载已有模型继续训练

        【初始化流程】
        1. 创建必要目录（models, checkpoints, logs等）
        2. 初始化C游戏引擎（通过ctypes）
        3. 创建DQN智能体（策略网络+目标网络）
        4. 初始化经验回放缓冲区
        5. 加载人类经验数据（如果启用）
        6. 初始化敌人管理器（难度+自我对弈）
        7. 初始化统计变量
        """
        # 保存可视化设置
        self.visualize = visualize

        # 创建所有必要的目录（如果不存在）
        # PATHS: 配置文件中定义的路径字典
        # exist_ok=True: 目录已存在时不报错
        for path in PATHS.values():
            os.makedirs(path, exist_ok=True)

        # ========== 初始化游戏环境 ==========

        # C共享库路径（游戏引擎编译产物）
        lib_path = './libtankbattle.so'

        # 创建环境包装器
        # TankBattleEnv: 封装C库的Python接口
        # - map_width/height: 游戏地图尺寸（像素）
        # - visualize: 是否启用SDL2渲染
        self.env = TankBattleEnv(
            lib_path,
            ENV_CONFIG['map_width'],    # 800像素宽
            ENV_CONFIG['map_height'],   # 600像素高
            visualize
        )

        # ========== 初始化DQN智能体 ==========

        # DQNAgent: 封装了策略网络、目标网络、优化器
        # state_dim: 43维（6 AI状态 + 25 敌人 + 12 子弹）
        # action_dim: 9维（idle + 4移动 + 4射击）
        # config: 网络配置（hidden_dims, learning_rate等）
        # device: 'cuda'或'cpu'（自动检测GPU）
        self.agent = DQNAgent(
            MODEL_CONFIG['state_dim'],     # 43
            MODEL_CONFIG['action_dim'],    # 9
            MODEL_CONFIG,                  # 网络超参数
            TRAINING_CONFIG['device']      # 'cuda' 或 'cpu'
        )

        # ========== 加载已有模型（如果提供）==========

        if continue_from and os.path.exists(continue_from):
            # agent.load(): 加载模型权重、优化器状态、epsilon值
            self.agent.load(continue_from)
            print(f"从 {continue_from} 继续训练")
            # 注意：继续训练会保留epsilon值，确保探索策略的连续性

        # ========== 初始化经验回放缓冲区 ==========

        # ReplayBuffer: 存储(s,a,r,s',done)五元组
        # buffer_size: 最多存储10万条经验
        # state_dim: 每条经验的状态维度（43维）
        self.replay_buffer = ReplayBuffer(
            TRAINING_CONFIG['buffer_size'],  # 100000条经验
            MODEL_CONFIG['state_dim']        # 43维状态
        )

        # ========== 加载人类经验数据（模仿学习）==========

        # 如果启用人类数据学习且配置为预加载
        if HUMAN_LEARNING_CONFIG['enabled'] and HUMAN_LEARNING_CONFIG['preload']:
            print("\n正在加载人类经验数据...")

            # HumanDataLoader: 加载玩家对战时记录的经验数据
            human_loader = HumanDataLoader(HUMAN_LEARNING_CONFIG['human_data_dir'])

            # 预加载到回放缓冲区
            # filter_quality: 是否过滤低奖励经验（<-30的数据）
            loaded_count = human_loader.preload_to_buffer(
                self.replay_buffer,
                filter_quality=HUMAN_LEARNING_CONFIG['filter_quality']
            )

            if loaded_count > 0:
                print(f"✓ 已预加载 {loaded_count} 条人类经验到回放缓冲区")
                # 显示人类数据统计信息
                stats = human_loader.get_stats()
                if stats['total'] > 0:
                    print(f"  人类数据统计:")
                    print(f"    - 总经验数: {stats['total']}")
                    print(f"    - 数据文件: {stats['files']}")
                    print(f"    - 平均奖励: {stats['avg_reward']:.2f}")
                    print(f"    - 胜率: {stats['win_rate']:.1f}%")
            else:
                print("未找到人类经验数据或数据质量过低")
            print()

        # ========== 初始化敌人管理器 ==========

        # 将所有配置打包传递给敌人管理器
        config_dict = {
            'SELF_PLAY_CONFIG': SELF_PLAY_CONFIG,      # 自我对弈配置
            'DIFFICULTY_CONFIG': DIFFICULTY_CONFIG,    # 动态难度配置
            'ENV_CONFIG': ENV_CONFIG,                  # 环境配置
            'PATHS': PATHS                             # 路径配置
        }

        # EnemyManager: 管理敌人数量和类型
        # - 动态难度: 根据胜率自动增加敌人
        # - 自我对弈: 保存历史模型作为敌人
        self.enemy_manager = EnemyManager(config_dict)

        # ========== 初始化统计变量 ==========

        # 回合计数
        self.episode = 0  # 当前回合数
        self.total_steps = 0  # 总步数（所有回合累计）

        # 胜负统计
        self.wins = 0  # 胜利次数（AI击败所有敌人）
        self.losses = 0  # 失败次数（AI被击败）
        self.draws = 0  # 平局次数（超时）

        # 奖励统计
        self.best_reward = -float('inf')  # 历史最佳回合奖励

        # 训练日志（用于绘图和分析）
        self.episode_rewards = []  # 每回合总奖励
        self.episode_lengths = []  # 每回合步数
        self.training_losses = []  # 训练损失值

        # ========== 打印初始化信息 ==========

        print("\n" + "="*60)
        print("坦克大战 AI 训练系统")
        print("="*60)
        print(f"设备: {self.agent.device}")  # 显示使用CPU还是GPU
        print(f"可视化: {visualize}")
        print(f"GPU加速: {torch.cuda.is_available()}")

        # 如果有GPU，显示GPU信息
        if torch.cuda.is_available():
            # get_device_name(): 获取GPU名称（如"NVIDIA RTX 3090"）
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            # 显示显存容量（转换为GB）
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"显存: {total_memory:.2f} GB")
        print("="*60 + "\n")

    def train_step(self):
        """
        执行一次训练步骤（网络参数更新）

        【DQN训练算法】
        1. 从回放缓冲区随机采样一个batch
        2. 计算当前Q值: Q(s,a)
        3. 计算目标Q值: r + γ * max Q'(s',a')
        4. 计算TD误差损失: L = (Q - target)²
        5. 反向传播更新网络参数

        【为什么用回放缓冲区？】
        - 打破数据相关性（连续经验高度相关）
        - 提高样本效率（每条经验可复用多次）
        - 稳定训练（平滑梯度波动）

        【返回值】
        float: 训练损失值，如果缓冲区未准备好则返回None

        【调用频率】
        每个step调用一次（约60次/秒），但只有缓冲区>=min_size时才真正训练
        """
        # 检查缓冲区是否有足够的经验
        # min_buffer_size: 最少1000条经验才开始训练（确保多样性）
        if not self.replay_buffer.is_ready(TRAINING_CONFIG['min_buffer_size']):
            return None  # 经验不足，跳过训练

        # 从缓冲区随机采样一个batch
        # batch_size: 256条经验
        # 返回: (states, actions, rewards, next_states, dones)
        batch = self.replay_buffer.sample(TRAINING_CONFIG['batch_size'])

        # 调用agent的训练方法
        # 内部会计算loss并更新网络参数
        # 利用GPU加速矩阵运算
        loss = self.agent.train(batch)

        return loss  # 返回损失值用于监控训练进度

    def run_episode(self):
        """
        运行一个完整的训练回合（episode）

        【回合流程】
        1. 重置环境 → 初始状态s₀
        2. 循环执行：
           a. 选择动作: a = ε-greedy(s)
           b. 执行动作: s', r, done = env.step(a)
           c. 存储经验: buffer.push(s, a, r, s', done)
           d. 训练网络: agent.train(batch)
           e. 更新状态: s = s'
        3. 回合结束 → 统计胜负、保存模型

        【关键概念】
        - 回合: 从reset到game_over的完整游戏过程
        - 步数: 一个回合通常100-3600步（1.7-60秒）
        - 奖励: 累积奖励，反映AI表现好坏

        【训练技巧】
        - 边交互边训练（online learning）
        - 使用ε-greedy平衡探索/利用
        - 定期更新目标网络（agent内部处理）
        - 定期保存检查点（防止训练中断）

        【返回值】
        float: 本回合的总累积奖励
        """
        # 回合计数器+1
        self.episode += 1

        # ========== 重置环境 ==========

        # 获取当前敌人数量（可能随难度变化）
        enemy_count = self.enemy_manager.current_enemy_count

        # 重置环境并获取初始状态
        # reset(): 清空游戏状态，生成新的敌人位置
        # 返回: 43维numpy数组
        state = self.env.reset(enemy_count)

        # 初始化回合统计
        episode_reward = 0  # 本回合累积奖励
        episode_steps = 0  # 本回合执行的步数
        done = False  # 游戏是否结束

        # 记录回合开始时间（用于计算速度）
        start_time = time.time()

        # ========== 主训练循环 ==========

        # 循环直到游戏结束或达到最大步数
        # max_steps_per_episode: 3600步（60秒@60fps）
        while not done and episode_steps < TRAINING_CONFIG['max_steps_per_episode']:

            # --- Step 1: 选择动作 ---
            # select_action(): ε-greedy策略
            # training=True: 使用探索（随机动作）
            # 返回: 0-8的整数（动作索引）
            action = self.agent.select_action(state, training=True)

            # --- Step 2: 执行动作并获取反馈 ---
            # step(): 将动作发送给C引擎执行
            # 返回:
            #   next_state: 新状态（43维）
            #   reward: 即时奖励（标量）
            #   done: 是否结束（布尔）
            #   info: 额外信息（字典，包含winner等）
            next_state, reward, done, info = self.env.step(action)

            # --- Step 3: 存储经验到回放缓冲区 ---
            # push(): 存储(s, a, r, s', done)五元组
            # 这些经验稍后会被随机采样用于训练
            self.replay_buffer.push(state, action, reward, next_state, done)

            # --- Step 4: 训练网络 ---
            # train_step(): 从缓冲区采样并更新参数
            # 返回loss值（None表示缓冲区未准备好）
            loss = self.train_step()
            if loss is not None:
                # 记录损失值用于监控
                self.training_losses.append(loss)

            # --- Step 5: 渲染（如果启用）---
            # 可视化模式下实时显示游戏画面
            # 会显著降低训练速度（约慢3倍）
            if self.visualize:
                self.env.render()  # 调用SDL2渲染

            # --- Step 6: 更新状态 ---
            state = next_state  # 当前状态变为下一状态
            episode_reward += reward  # 累加奖励
            episode_steps += 1  # 步数+1
            self.total_steps += 1  # 总步数+1（跨回合累计）

        # ========== 回合后处理 ==========

        # 更新epsilon值（逐渐减少探索）
        # epsilon = epsilon * decay
        # 例如: 1.0 → 0.9999 → 0.9998 → ... → 0.1
        self.agent.update_epsilon()

        # 计算回合耗时
        elapsed_time = time.time() - start_time

        # ========== 统计胜负 ==========

        # 从info中获取胜者信息
        # winner: 0=AI胜, 1=敌人胜, -1=平局
        winner = info.get('winner', -1)

        if winner == 0:
            # AI获胜（击败所有敌人）
            self.wins += 1
            win = True
        elif winner == 1:
            # AI失败（被击败）
            self.losses += 1
            win = False
        else:
            # 平局（超时或其他情况）
            self.draws += 1
            win = False

        # ========== 动态难度调整 ==========

        # 根据胜负更新敌人数量
        # update_difficulty(): 连胜则增加敌人，连败不变
        # 返回新的敌人数量
        new_enemy_count = self.enemy_manager.update_difficulty(win)

        # ========== 记录统计信息 ==========

        # 记录本回合数据（用于计算移动平均）
        self.episode_rewards.append(episode_reward)  # 总奖励
        self.episode_lengths.append(episode_steps)  # 总步数

        # 更新历史最佳奖励
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward

        # ========== 打印进度 ==========

        # 纯文本模式：每回合打印
        # 可视化模式：每10回合打印（避免刷屏）
        if not self.visualize or self.episode % TRAINING_CONFIG['log_interval'] == 0:
            self._print_progress(episode_reward, episode_steps, elapsed_time, winner)

        # ========== 保存模型 ==========

        # 每100回合保存一次checkpoint
        if self.episode % TRAINING_CONFIG['save_interval'] == 0:
            self._save_checkpoint()

        # ========== 自我对弈系统 ==========

        # 根据配置判断是否保存历史版本
        # should_save_history_version(): 检查回合数是否满足间隔
        if self.enemy_manager.should_save_history_version(self.episode):
            # save_history_version(): 保存当前模型作为未来的对手
            self.enemy_manager.save_history_version(self.agent, self.episode)

        return episode_reward  # 返回本回合奖励

    def _print_progress(self, reward: float, steps: int, elapsed_time: float, winner: int):
        """
        打印训练进度（实时监控）

        【显示内容】
        - 回合信息: 回合号、胜负、奖励、步数、耗时
        - 平均指标: 100回合平均奖励、损失
        - 学习参数: Epsilon、训练步数、缓冲区大小
        - 胜率统计: 胜/负/平、总胜率
        - 难度信息: 敌人数量、连胜次数、历史版本数

        【参数】
        reward: float - 本回合总奖励
        steps: int - 本回合步数
        elapsed_time: float - 回合耗时（秒）
        winner: int - 胜者（0=AI, 1=敌人, -1=平局）

        【颜色编码】
        - 绿色: 胜利
        - 红色: 失败
        - 黄色: 平局
        """
        # ========== 计算平均值 ==========

        # 计算最近100回合的平均奖励
        # episode_rewards[-100:]: 取最后100个元素
        # np.mean(): 计算平均值，用于平滑波动
        avg_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0

        # 计算最近100步的平均损失
        avg_loss = np.mean(self.training_losses[-100:]) if self.training_losses else 0

        # ========== 计算胜率 ==========

        # 总对局数 = 胜+负+平
        total_games = self.wins + self.losses + self.draws

        # 胜率 = 胜场数 / 总场数 * 100%
        # 使用三元运算避免除零错误
        win_rate = self.wins / total_games * 100 if total_games > 0 else 0

        # ========== 获取敌人管理器统计 ==========

        # get_stats(): 返回字典
        # - current_enemy_count: 当前敌人数
        # - consecutive_wins: 连胜次数
        # - history_versions: 历史模型数量
        enemy_stats = self.enemy_manager.get_stats()

        # ========== 格式化胜负标记 ==========

        # 根据winner确定结果字符串和颜色
        result_str = "WIN" if winner == 0 else "LOSS" if winner == 1 else "DRAW"

        # ANSI颜色代码
        # \033[92m: 绿色, \033[91m: 红色, \033[93m: 黄色
        result_color = "\033[92m" if winner == 0 else "\033[91m" if winner == 1 else "\033[93m"
        reset_color = "\033[0m"  # 重置颜色

        # ========== 打印格式化输出 ==========

        # 分隔线
        print(f"\n{'='*80}")

        # 第一行：回合号、结果、奖励、步数、耗时
        # :5d: 右对齐，宽度5，整数
        # :7.2f: 右对齐，宽度7，2位小数
        print(f"回合 {self.episode:5d} | {result_color}{result_str:4s}{reset_color} | "
              f"奖励: {reward:7.2f} | 步数: {steps:4d} | 耗时: {elapsed_time:.2f}s")

        print(f"{'='*80}")

        # 平均指标
        print(f"  平均奖励 (100回合): {avg_reward:7.2f} | 最佳奖励: {self.best_reward:7.2f}")
        print(f"  平均损失 (100步):   {avg_loss:7.4f}")

        # 学习参数
        print(f"  探索率 (Epsilon):   {self.agent.epsilon:.4f}")  # epsilon值（0-1）
        print(f"  训练步数:           {self.total_steps:7d}")  # 总step数

        # 缓冲区状态
        # len(replay_buffer): 当前存储的经验数
        print(f"  经验缓冲区:         {len(self.replay_buffer):7d} / {TRAINING_CONFIG['buffer_size']}")

        # 胜负统计
        # :4d: 右对齐，宽度4
        print(f"  胜/负/平:           {self.wins:4d} / {self.losses:4d} / {self.draws:4d} "
              f"(胜率: {win_rate:.1f}%)")

        # 敌人信息
        print(f"  当前敌人数量:       {enemy_stats['current_enemy_count']}")
        print(f"  连胜次数:           {enemy_stats['consecutive_wins']}")
        print(f"  历史版本数:         {enemy_stats['history_versions']}")

        # 结束分隔线
        print(f"{'='*80}\n")

    def _save_checkpoint(self):
        """
        保存训练检查点

        【保存策略】
        1. 定期检查点: checkpoint_ep{回合数}.pth
           - 每100回合保存一次
           - 保存在 checkpoints/ 子目录
           - 用于历史回顾和版本对比

        2. 最新模型: latest_model.pth
           - 同时更新（与checkpoint同步）
           - 保存在主目录
           - 用于断点续训（推荐继续训练时使用）

        【检查点内容】
        - 策略网络权重
        - 目标网络权重
        - 优化器状态（动量、学习率等）
        - Epsilon值（探索率）
        - 训练步数

        【注意】
        final_model.pth 只在训练完全结束时保存（见train()方法）
        """
        # 构造checkpoint文件名
        # 格式: checkpoint_ep{回合数}.pth
        # 例如: checkpoint_ep100.pth, checkpoint_ep200.pth
        checkpoint_path = os.path.join(
            PATHS['checkpoints'],  # checkpoints/子目录
            f'checkpoint_ep{self.episode}.pth'
        )

        # 保存模型到checkpoint
        # agent.save(): 保存网络权重+优化器状态+训练参数
        self.agent.save(checkpoint_path)

        # 同时更新latest_model（覆盖旧文件）
        # 这样用户可以方便地加载最新模型继续训练
        latest_path = os.path.join(PATHS['models'], 'latest_model.pth')
        self.agent.save(latest_path)

    def train(self):
        """
        主训练循环

        【训练流程】
        1. 打印训练配置
        2. 循环执行回合（episode）
        3. 捕获中断（Ctrl+C）优雅退出
        4. 保存最终模型
        5. 打印训练统计
        6. 清理资源

        【异常处理】
        - KeyboardInterrupt: 用户按Ctrl+C中断
          → 保存final_model.pth后退出
        - finally: 无论如何都执行清理
          → 确保模型被保存，资源被释放

        【训练终止条件】
        - 达到最大回合数（max_episodes）
        - 用户手动中断（Ctrl+C）

        【输出文件】
        - final_model.pth: 最终训练完成的模型
        - latest_model.pth: 最后一次checkpoint（每100回合更新）
        - checkpoint_ep*.pth: 定期保存的检查点
        """
        # ========== 打印训练配置 ==========

        print("开始训练...")
        print(f"最大回合数: {TRAINING_CONFIG['max_episodes']}")
        # 三元运算：visualize为True则显示"可视化"，否则"纯文本"
        print(f"显示模式: {'可视化' if self.visualize else '纯文本'}\n")

        try:
            # ========== 主训练循环 ==========

            # range(max_episodes): 生成0到max_episodes-1的序列
            # 例如: range(100000)生成[0, 1, 2, ..., 99999]
            for episode in range(TRAINING_CONFIG['max_episodes']):
                # 运行一个完整回合
                # run_episode(): 返回本回合的总奖励
                self.run_episode()
                # 注意：回合计数在run_episode()内部完成

        except KeyboardInterrupt:
            # ========== 用户中断处理 ==========

            # 捕获Ctrl+C信号（SIGINT）
            # 这样可以优雅退出，而不是强制终止
            print("\n\n训练被用户中断")
            # 继续执行finally块保存模型

        finally:
            # ========== 清理和保存 ==========

            # finally块：无论是否异常都会执行
            # 确保模型一定会被保存

            # 保存最终模型（训练完成或被中断）
            final_path = os.path.join(PATHS['models'], 'final_model.pth')
            self.agent.save(final_path)
            # final_model.pth: 代表最终训练成果
            # 与latest_model.pth区别：
            # - final: 训练结束时的模型（训练最充分）
            # - latest: 最后一次checkpoint（可能提前结束）

            # 打印训练总结统计
            self._print_final_stats()

            # 清理环境资源
            # close(): 关闭SDL窗口、释放C库内存
            self.env.close()

    def _print_final_stats(self):
        """
        打印最终训练统计

        【统计内容】
        - 总回合数: 训练了多少个episode
        - 总步数: 所有回合累计的step数
        - 胜负平: 各结果的次数
        - 最终胜率: wins / total * 100%
        - 最佳奖励: 历史最高单回合奖励
        - 平均奖励: 所有回合的平均奖励

        【用途】
        - 评估训练效果
        - 决定是否需要继续训练
        - 对比不同配置的性能
        """
        print("\n" + "="*80)
        print("训练完成!")
        print("="*80)

        # 总回合数
        print(f"总回合数:     {self.episode}")

        # 总步数（所有回合累计）
        print(f"总步数:       {self.total_steps}")

        # 胜负平统计
        print(f"胜/负/平:     {self.wins} / {self.losses} / {self.draws}")

        # 计算最终胜率
        total = self.wins + self.losses + self.draws
        final_win_rate = self.wins / total * 100 if total > 0 else 0
        print(f"最终胜率:     {final_win_rate:.2f}%")

        # 奖励统计
        print(f"最佳奖励:     {self.best_reward:.2f}")  # 历史最高
        print(f"平均奖励:     {np.mean(self.episode_rewards):.2f}")  # 所有回合平均

        print("="*80 + "\n")


def main():
    """
    主函数 - 程序入口

    【功能】
    1. 解析命令行参数（--visualize / --no-visualize）
    2. 交互式选择模型（新建 / 继续训练）
    3. 创建训练器并启动训练

    【命令行用法】
    python train.py                 # 默认纯文本模式
    python train.py --visualize     # 启用可视化（慢）
    python train.py --no-visualize  # 明确指定纯文本（同默认）

    【交互流程】
    1. 显示可用模型列表
    2. 用户选择：
       - [0] 创建新模型 → 随机初始化
       - [1-5] 继续训练 → 加载已有权重
    3. 启动训练循环

    【典型使用场景】
    - 初次训练: python train.py → 选择[0]
    - 继续训练: python train.py → 选择latest_model.pth
    - 微调模型: python train.py → 选择final_model.pth
    """
    # ========== 解析命令行参数 ==========

    # ArgumentParser: 标准库argparse，用于处理命令行参数
    parser = argparse.ArgumentParser(description='坦克大战AI训练')

    # 添加--visualize参数
    # action='store_true': 如果指定该参数，值为True，否则False
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='启用可视化训练模式'
    )

    # 添加--no-visualize参数
    parser.add_argument(
        '--no-visualize',
        action='store_true',
        help='使用纯文本训练模式（默认）'
    )

    # 解析参数
    # args: Namespace对象，包含所有参数
    # 例如: Namespace(visualize=True, no_visualize=False)
    args = parser.parse_args()

    # ========== 确定是否可视化 ==========

    # 逻辑: 只有明确指定--visualize且没有--no-visualize时才可视化
    # - python train.py → False (默认)
    # - python train.py --visualize → True
    # - python train.py --no-visualize → False
    # - python train.py --visualize --no-visualize → False (no覆盖yes)
    visualize = args.visualize and not args.no_visualize

    # ========== 交互式选择模型 ==========

    # select_model_interactively(): 显示模型列表供用户选择
    # 返回:
    #   - None: 用户选择创建新模型
    #   - "path/to/model.pth": 用户选择继续训练的模型路径
    continue_from = select_model_interactively()

    # ========== 创建训练器并开始训练 ==========

    # 创建Trainer实例
    # visualize: 是否启用SDL2渲染
    # continue_from: 要加载的模型路径（None表示新建）
    trainer = Trainer(visualize=visualize, continue_from=continue_from)

    # 启动训练循环
    # train(): 执行主训练流程直到完成或中断
    trainer.train()

    # 训练完成后自动退出
    # 模型已保存在 saved_models/final_model.pth


# ==================== 程序入口 ====================

if __name__ == '__main__':
    # 当直接运行此脚本时执行main()
    # 如果被其他脚本import，则不执行
    # 这是Python的标准做法
    main()
