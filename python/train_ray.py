#!/usr/bin/env python3
"""
Ray并行采样版本的训练脚本

使用Ray Core并行运行多个游戏环境，加速数据收集
保留当前的DQN实现，只并行化环境交互部分
"""

# ============== 导入依赖库 ==============
import ray                          # Ray分布式计算框架，用于并行化
import torch                        # PyTorch深度学习框架
import numpy as np                  # 数值计算库
import os                           # 操作系统接口，用于路径操作
from typing import List, Tuple, Dict     # 类型提示，提高代码可读性
from model import DQNAgent, DQN          # 自定义的DQN智能体类和网络类
from replay_buffer import ReplayBuffer  # 经验回放缓冲区
from env_wrapper import TankBattleEnv   # 坦克大战游戏环境包装器
from human_data_loader import HumanDataLoader  # 人类数据加载器
from config import TRAINING_CONFIG, MODEL_CONFIG, ENV_CONFIG, PATHS, HUMAN_LEARNING_CONFIG, STAGED_TRAINING_CONFIG, EARLY_STOPPING_CONFIG  # 配置文件
from training_stages import TrainingStageManager  # AlphaGo风格的分阶段训练管理器
from early_stopping import EarlyStopping, ConvergenceDetector, TrainingRecommendation  # 早停和收敛监控

@ray.remote  # Ray装饰器：将类标记为远程可执行对象，实例会运行在独立的进程中
class ParallelEnvWorker:
    """
    并行环境工作器（在独立进程中运行）

    每个Worker负责运行一个游戏环境实例，独立收集经验数据
    通过Ray框架实现多进程并行，加速数据采样效率
    """

    def __init__(self, worker_id: int):
        """
        初始化工作器

        Args:
            worker_id: 工作器编号，用于调试和日志记录
        """
        self.worker_id = worker_id  # 保存工作器ID

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

        # 创建本地DQN网络（与主进程架构相同）
        self.policy_net = DQN(
            state_dim=MODEL_CONFIG['state_dim'],
            action_dim=MODEL_CONFIG['action_dim'],
            hidden_dims=MODEL_CONFIG['hidden_dims']
        ).to(self.device)

        # 设为评估模式（推理时不需要Dropout）
        self.policy_net.eval()

    def collect_episodes(self, num_episodes: int, epsilon: float) -> Tuple[List[Tuple], List[int]]:
        """
        收集多个回合的经验（这个方法会在远程进程中执行）

        Args:
            num_episodes: 需要收集的游戏回合数
            epsilon: ε-greedy策略的探索率（0-1之间，越大越随机）

        Returns:
            (经验列表, 胜负列表):
                - 经验列表: [(state, action, reward, next_state, done), ...]
                - 胜负列表: [winner, ...] (0=AI胜, 1=敌人胜, -1=平局)
        """
        experiences = []  # 用于存储所有收集到的经验
        winners = []  # 用于存储每个回合的胜负结果

        # 循环执行指定数量的游戏回合
        for _ in range(num_episodes):
            # 重置环境，开始新的一局游戏
            # ENV_CONFIG['initial_enemies']: 初始敌人数量
            state = self.env.reset(ENV_CONFIG['initial_enemies'])
            episode_done = False  # 回合是否结束的标志

            # 单个回合的主循环，直到游戏结束
            while not episode_done:
                # ========== ε-greedy 探索策略 ==========
                # 以epsilon的概率进行随机探索，以(1-epsilon)的概率利用当前策略
                if np.random.random() < epsilon:  # 生成[0,1)之间的随机数
                    # 随机选择一个动作（探索）
                    # 动作空间: 0-8 共9个动作（上下左右、射击等）
                    action = np.random.randint(0, 9)
                else:
                    # ✅ 修复3：使用训练好的神经网络选择动作（利用）
                    # 这是关键！之前只是随机，现在真正使用策略网络
                    action = self._select_action_with_network(state)

                # 执行动作，获取环境反馈
                # next_state: 执行动作后的新状态
                # reward: 即时奖励（如击杀敌人+分，被击中-分）
                # done: 回合是否结束（玩家死亡或敌人全灭）
                # info: 额外信息（如击杀数、胜负等）
                next_state, reward, done, info = self.env.step(action)

                # 将这一步的经验保存到列表中
                # 这些经验稍后会被送入经验回放缓冲区用于训练
                experiences.append((state, action, reward, next_state, done))

                # 更新状态，准备下一步
                state = next_state
                episode_done = done  # 更新回合结束标志

            # 记录本回合的胜负结果
            winner = info.get('winner', -1)
            winners.append(winner)

        return experiences, winners  # 返回经验和胜负列表

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

    def evaluate(self, num_episodes: int) -> Tuple[float, float]:
        """
        评估当前策略的性能（在远程进程中执行）

        Args:
            num_episodes: 评估使用的游戏回合数

        Returns:
            (平均奖励, 胜率) 的元组
            - 平均奖励: 所有回合的平均总奖励
            - 胜率: 玩家获胜的回合占比（0-1之间）
        """
        total_reward = 0  # 累计总奖励
        wins = 0          # 累计获胜次数

        # 执行多个评估回合
        for _ in range(num_episodes):
            # 重置环境，开始新回合
            state = self.env.reset(ENV_CONFIG['initial_enemies'])
            episode_reward = 0  # 当前回合的累计奖励
            episode_done = False  # 回合结束标志

            # 单回合评估循环
            while not episode_done:
                # ✅ 修复4：评估时使用真实网络（贪婪策略，无探索）
                action = self._select_action_with_network(state)

                # 执行动作
                next_state, reward, done, info = self.env.step(action)

                # 累计奖励
                episode_reward += reward

                # 更新状态
                state = next_state
                episode_done = done

            # 记录本回合结果
            total_reward += episode_reward

            # 检查是否获胜（winner == 0 表示玩家获胜）
            if info.get('winner') == 0:
                wins += 1

        # 返回平均值
        return total_reward / num_episodes, wins / num_episodes


def train_with_ray():
    """
    使用Ray框架进行分布式并行训练的主函数

    训练流程：
    1. 初始化Ray集群
    2. 创建多个并行环境工作器（用于采样）
    3. 创建主DQN智能体（用于训练）
    4. 循环执行：并行采样 -> 集中训练 -> 定期评估
    """

    # ========== 步骤0: 创建必要的目录 ==========
    # 确保所有保存路径存在（模型、checkpoint、日志等）
    for path in PATHS.values():
        os.makedirs(path, exist_ok=True)

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

    # ========== 步骤2: 创建并行环境工作器 ==========
    # 从配置文件读取工作器数量（默认4个）
    num_workers = TRAINING_CONFIG.get('num_workers', 4)

    # 创建多个远程工作器实例
    # .remote(i): Ray的远程调用语法，在独立进程中创建对象
    # 每个worker会在不同的CPU核心上运行，实现并行
    workers = [ParallelEnvWorker.remote(i) for i in range(num_workers)]

    # ========== 步骤3: 创建主DQN智能体（在主进程中） ==========
    # 创建PyTorch设备对象（CPU或CUDA GPU）
    device = torch.device(TRAINING_CONFIG['device'])

    # ✅ Epsilon参数配置
    # 使用config.py中的配置（不要硬编码！）
    ray_model_config = MODEL_CONFIG.copy()
    ray_model_config['epsilon_start'] = 1.0  # 初始探索率（100%随机）
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

    # 实例化DQN智能体（包含策略网络和目标网络）
    agent = DQNAgent(
        state_dim=MODEL_CONFIG['state_dim'],      # 状态空间维度（观察向量的长度）
        action_dim=MODEL_CONFIG['action_dim'],    # 动作空间维度（可选动作的数量）
        config=ray_model_config,                  # 使用修复后的配置
        device=TRAINING_CONFIG['device']          # 训练设备（'cpu' 或 'cuda'）
    )

    # ========== 步骤3.5: 检查并加载已有模型（断点续训） ==========
    start_episode = 0  # 起始回合数
    latest_model_path = os.path.join(PATHS['models'], 'latest_model.pth')

    if os.path.exists(latest_model_path):
        print(f"\n📦 发现已有模型: {latest_model_path}")

        # 读取模型信息
        try:
            checkpoint = torch.load(latest_model_path, map_location=device)
            saved_episode = checkpoint.get('episode', 0)
            saved_epsilon = checkpoint.get('epsilon', 1.0)
            saved_train_step = checkpoint.get('train_step', 0)

            print(f"   回合: {saved_episode}, Epsilon: {saved_epsilon:.4f}, 训练步数: {saved_train_step:,}")
            print(f"\n请选择训练模式:")
            print(f"  [1] 继续训练（保留网络权重和epsilon）")
            print(f"  [2] 重新开始（使用新配置，epsilon={ray_model_config['epsilon_start']:.2f}）")

            choice = input("请输入选项 [1/2] (默认=1): ").strip()

            if choice == '2':
                # 完全重新开始
                print(f"\n✨ 使用新配置从头开始训练")
                print(f"   - Epsilon: {ray_model_config['epsilon_start']:.2f} (新配置)")
                print(f"   - 学习率: {ray_model_config['learning_rate']:.6f} (新配置)")
                print(f"   - Epsilon衰减: {ray_model_config['epsilon_decay']} (新配置)\n")
                start_episode = 0
            else:
                # 继续训练（默认）
                agent.load(latest_model_path)
                start_episode = saved_episode
                old_lr = agent.optimizer.param_groups[0]['lr']

                # 应用新配置的学习率和epsilon_decay
                new_lr = ray_model_config['learning_rate']
                if new_lr != old_lr:
                    for param_group in agent.optimizer.param_groups:
                        param_group['lr'] = new_lr
                    print(f"\n✅ 继续训练（应用部分新配置）")
                    print(f"   - 回合: 从第 {start_episode} 回合继续")
                    print(f"   - 网络权重: 已恢复 (保留)")
                    print(f"   - Epsilon: {agent.epsilon:.4f} (保留旧值)")
                    print(f"   - 学习率: {old_lr:.6f} → {new_lr:.6f} (应用新配置)")
                    print(f"   - Epsilon衰减: {ray_model_config['epsilon_decay']} (应用新配置)\n")
                else:
                    print(f"\n✅ 继续训练")
                    print(f"   - 回合: 从第 {start_episode} 回合继续")
                    print(f"   - Epsilon: {agent.epsilon:.4f}")
                    print(f"   - 学习率: {new_lr:.6f}\n")

        except Exception as e:
            print(f"⚠️ 加载模型失败: {e}")
            print(f"   将从头开始训练\n")
            start_episode = 0
    else:
        print(f"ℹ️ 未找到已有模型，从头开始训练\n")

    # ========== 步骤4: 创建经验回放缓冲区 ==========
    # 经验回放是DQN的核心技术，用于打破数据相关性
    replay_buffer = ReplayBuffer(
        capacity=TRAINING_CONFIG['buffer_size'],  # 缓冲区最大容量（存储多少条经验）
        state_dim=MODEL_CONFIG['state_dim']       # 状态维度（用于预分配内存）
    )

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
                print(f"   - 缓冲区使用率: {len(replay_buffer):,} / {TRAINING_CONFIG['buffer_size']:,} ({len(replay_buffer)/TRAINING_CONFIG['buffer_size']*100:.1f}%)")
                print(f"   - AI将从人类专家策略中学习 🎓\n")
            else:
                print(f"⚠️  未找到人类数据文件\n")
        except Exception as e:
            print(f"⚠️  加载人类数据失败: {e}\n")

    # ========== 步骤5: 准备训练参数 ==========
    # 从配置文件读取训练超参数
    max_episodes = TRAINING_CONFIG['max_episodes']  # 最大训练回合数
    batch_size = TRAINING_CONFIG['batch_size']      # 每次训练的批次大小
    episodes_per_worker = TRAINING_CONFIG.get('episodes_per_worker', 1)  # 每个worker每轮收集的回合数

    # ========== 步骤5.5: 初始化AlphaGo风格的分阶段训练管理器 ==========
    stage_manager = None
    if STAGED_TRAINING_CONFIG['enabled']:
        # 创建阶段管理器
        stage_manager = TrainingStageManager(STAGED_TRAINING_CONFIG['stages'])

        # 打印训练阶段规划
        print(stage_manager.get_all_stages_summary())
        print()

    # ========== 步骤5.6: 初始化早停和收敛监控 ==========
    early_stopping = None
    convergence_detector = None

    if EARLY_STOPPING_CONFIG['enabled']:
        # 创建早停监控器
        early_stopping = EarlyStopping(
            patience=EARLY_STOPPING_CONFIG['patience'],
            min_delta=EARLY_STOPPING_CONFIG['min_delta'],
            metric=EARLY_STOPPING_CONFIG['metric'],
            mode='max',  # 胜率越大越好
            baseline=EARLY_STOPPING_CONFIG.get('baseline')
        )

        # 创建收敛检测器
        convergence_detector = ConvergenceDetector(
            window_size=EARLY_STOPPING_CONFIG.get('convergence_window', 100),
            win_rate_threshold=EARLY_STOPPING_CONFIG.get('convergence_win_rate', 0.75),
            stability_threshold=EARLY_STOPPING_CONFIG.get('convergence_stability', 0.05)
        )

        min_episodes = EARLY_STOPPING_CONFIG.get('min_episodes', 0)
        print(f"✅ 早停监控已配置")
        print(f"   - 最小训练回合: {min_episodes:,} 回合（达到后启用早停）")
        print(f"   - 容忍次数: {EARLY_STOPPING_CONFIG['patience']} 次评估（{EARLY_STOPPING_CONFIG['patience']*100} 回合）")
        print(f"   - 最小改进: {EARLY_STOPPING_CONFIG['min_delta']*100:.0f}%")
        print(f"   - 基线胜率: {EARLY_STOPPING_CONFIG.get('baseline', 0)*100:.0f}%\n")

    # 打印训练配置信息
    training_mode = "AlphaGo风格分阶段" if STAGED_TRAINING_CONFIG['enabled'] else "传统单一阶段"
    print(f"🚀 开始Ray并行训练（v4.0 - {training_mode}）")
    print(f"   - 工作器数量: {num_workers}")
    print(f"   - 主设备: {device}")
    print(f"   - Worker设备: CPU")
    print(f"   - 每轮每工作器收集: {episodes_per_worker} 回合")
    print(f"   - GPU型号: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"   - 缓冲区初始大小: {len(replay_buffer):,} / {TRAINING_CONFIG['buffer_size']:,} ({len(replay_buffer)/TRAINING_CONFIG['buffer_size']*100:.1f}%)")
    if len(replay_buffer) > 0:
        print(f"   🎓 包含人类数据: {len(replay_buffer):,} 条经验（模仿学习）")
    print(f"   ✅ Worker使用训练好的神经网络策略（非随机）")
    print(f"   ✅ 每轮采样前同步最新网络参数")
    print(f"   ✅ 模型保存到标准路径（玩家模式可用）")
    print(f"   ✅ Epsilon按回合数衰减（decay={MODEL_CONFIG['epsilon_decay']}，每回合衰减一次）")
    print(f"   ✅ 增强GPU训练频率（每条经验训练4次）\n")

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

            # ========== AlphaGo风格分阶段训练：更新当前阶段 ==========
            if stage_manager is not None:
                stage_changed, stage_info = stage_manager.update(episode_count)

                # 如果阶段切换，打印详细信息并重置epsilon
                if stage_changed:
                    if STAGED_TRAINING_CONFIG['verbose']:
                        stage_manager.print_stage_info(episode_count)

                    # 阶段切换时，重置epsilon为新阶段的起始值
                    agent.epsilon = stage_info['epsilon']
                    print(f"   ⚙️  Epsilon已重置为: {agent.epsilon:.3f}")

                # 动态调整人类数据采样权重
                replay_buffer.human_data_weight = stage_info['human_data_weight']

                # 根据阶段调整epsilon
                if stage_info['index'] == 0:
                    # 阶段1：如果启用override，每轮都覆盖epsilon（防止衰减）
                    if STAGED_TRAINING_CONFIG['override_epsilon_in_stage1']:
                        agent.epsilon = stage_info['epsilon']
                else:
                    # 阶段2/3：使用阶段配置的epsilon（线性插值）
                    # 这样可以让epsilon按照阶段配置自然增长
                    agent.epsilon = stage_info['epsilon']

            # ✅ 修复5：每轮采样前，同步网络参数给所有Worker
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
            # all_results: List[Tuple[List[经验], List[胜负]]]
            all_results = ray.get(futures)

            # 合并所有worker收集的经验到主进程的回放缓冲区
            episode_rewards_this_round = []
            for experiences, winners in all_results:  # 遍历每个worker的结果
                episode_reward = 0
                for exp in experiences:  # 遍历该worker的每条经验
                    # 将经验存入缓冲区
                    # exp: (state, action, reward, next_state, done)
                    replay_buffer.push(*exp)  # *exp 解包元组作为参数
                    episode_reward += exp[2]  # reward是第3个元素

                # 记录本回合奖励
                all_rewards.append(episode_reward)
                episode_rewards_this_round.append(episode_reward)

                # 统计胜负并在每个回合完成后衰减epsilon
                # 每个worker收集了episodes_per_worker个回合
                for winner in winners:
                    # 统计胜负
                    if winner == 0:
                        total_wins += 1
                    elif winner == 1:
                        total_losses += 1
                    else:
                        total_draws += 1

                    # ✅ 按回合数衰减epsilon（每完成一个回合衰减一次）
                    # 注意：如果启用分阶段训练，epsilon由阶段管理器控制，不使用这里的衰减
                    # 只在未启用分阶段训练时才使用epsilon_decay
                    if stage_manager is None:
                        agent.epsilon = max(
                            MODEL_CONFIG['epsilon_end'],
                            agent.epsilon * MODEL_CONFIG['epsilon_decay']
                        )

                    # 更新回合计数
                    episode_count += 1

            # 打印采样进度（\r使光标回到行首，实现原地更新）
            avg_reward_this_round = np.mean(episode_rewards_this_round) if episode_rewards_this_round else 0

            # 计算总胜率
            total_games = total_wins + total_losses + total_draws
            win_rate = (total_wins / total_games * 100) if total_games > 0 else 0

            # 显示训练进度（简洁版）
            progress_str = (f"\r回合 {episode_count}/{max_episodes} | "
                  f"ε={agent.epsilon:.3f} | "  # 探索率
                  f"本轮奖励: {avg_reward_this_round:.2f} | "
                  f"胜率: {win_rate:.1f}% ({total_wins}胜/{total_losses}负/{total_draws}平)")

            # 显示损失值（如果正在训练）
            if len(replay_buffer) >= TRAINING_CONFIG['min_buffer_size'] and 'avg_loss' in locals():
                progress_str += f" | Loss: {avg_loss:.4f}"

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

                # ✅ 增加训练频率：确保GPU充分利用
                # 原逻辑：训练次数 = 新收集经验数 / batch_size
                # 新逻辑：训练次数 = 新收集经验数 * 训练倍数 / batch_size
                # 训练倍数：每条经验平均训练多少次（提高样本效率）
                train_multiplier = 4  # 每条经验训练4次（增加GPU利用率）
                num_updates = (len(all_results[0][0]) * num_workers * train_multiplier) // batch_size

                # ✅ 人类数据使用策略
                # 优先级：分阶段训练 > 周期控制 > 默认启用
                if stage_manager is not None:
                    # 启用分阶段训练时，始终开启人类数据，通过权重控制使用量
                    # 阶段1权重100.0 → 大量使用，阶段4权重1.0 → 少量使用
                    replay_buffer.include_human_data = True
                elif HUMAN_LEARNING_CONFIG.get('periodic_usage', False):
                    # 周期性控制（仅在未启用分阶段训练时生效）
                    interval = HUMAN_LEARNING_CONFIG.get('usage_interval', 200)
                    duration = HUMAN_LEARNING_CONFIG.get('usage_duration', 1)

                    cycle_position = episode_count % interval
                    should_use_human = cycle_position < duration

                    prev_state = getattr(replay_buffer, '_prev_human_state', None)
                    replay_buffer.include_human_data = should_use_human

                    # 只在状态变化时输出
                    if prev_state != should_use_human:
                        status = "✅ 启用人类数据" if should_use_human else "⏸️  仅用AI数据"
                        print(f"\n[数据源切换] {status} (周期 {cycle_position}/{interval})")
                        replay_buffer._prev_human_state = should_use_human
                else:
                    # 默认：始终使用人类数据
                    replay_buffer.include_human_data = True

                # 执行多次梯度更新
                total_loss = 0.0
                for _ in range(num_updates):
                    # 从回放缓冲区随机采样一个批次
                    # 随机采样可以打破数据之间的时间相关性
                    batch = replay_buffer.sample(batch_size)

                    # 使用采样的批次训练DQN网络
                    # 返回TD-error损失值
                    loss = agent.train(batch)
                    total_loss += loss

                # 记录平均损失（用于监控）
                avg_loss = total_loss / num_updates if num_updates > 0 else 0.0

                # 注意: epsilon衰减已经在每个回合完成时执行（见第463-468行）
                # 不再需要在这里按采样轮次衰减

            # ==================== 阶段3: 定期评估 ====================
            # 每100个回合评估一次模型性能，监控训练进度

            # 检查是否到达评估时机
            if episode_count % 100 == 0 and episode_count > 0:
                print(f"\n\n📊 评估中...")  # 换行输出评估信息

                # ✅ 先同步最新参数给Worker（确保评估使用最新策略）
                policy_state_dict = agent.policy_net.state_dict()
                cpu_state_dict = {k: v.cpu() for k, v in policy_state_dict.items()}
                update_futures = [
                    worker.update_network_params.remote(cpu_state_dict)
                    for worker in workers
                ]
                ray.get(update_futures)

                # 向所有worker分发评估任务（并行评估）
                # 每个worker独立运行5个回合来评估当前策略
                eval_futures = [
                    worker.evaluate.remote(5)  # 远程调用evaluate方法，每个worker评估5回合
                    for worker in workers
                ]

                # 等待所有worker完成评估，获取结果
                # results: List[Tuple[float, float]]
                # 每个元组包含：(平均奖励, 胜率)
                results = ray.get(eval_futures)

                # 计算所有worker评估结果的平均值
                # r[0]: 第r个worker的平均奖励
                avg_reward = np.mean([r[0] for r in results])
                # r[1]: 第r个worker的胜率
                win_rate = np.mean([r[1] for r in results])

                # 打印评估结果
                print(f"回合 {episode_count}: "
                      f"平均奖励={avg_reward:.2f}, "       # 保留2位小数
                      f"胜率={win_rate*100:.1f}%")        # 转换为百分比，保留1位小数

                # ========== 早停和收敛检测 ==========
                if early_stopping is not None and convergence_detector is not None:
                    # 更新收敛检测器
                    convergence_detector.update(win_rate, avg_reward, 500)  # 假设平均长度500

                    # 检查是否达到最小训练回合数
                    min_episodes = EARLY_STOPPING_CONFIG.get('min_episodes', 0)

                    # 只有达到最小回合数后才启用早停检查
                    if episode_count >= min_episodes:
                        # 更新早停监控器
                        should_stop = early_stopping.step(win_rate, episode_count)
                    else:
                        # 未达到最小回合数，不启用早停
                        should_stop = False
                        if episode_count % 500 == 0:
                            remaining = min_episodes - episode_count
                            print(f"\n⏳ 早停尚未启用（需要 {min_episodes:,} 回合，剩余 {remaining:,} 回合）")

                    # 每5次评估（500回合）打印一次状态
                    if episode_count % 500 == 0:
                        early_stopping.print_status()
                        convergence_detector.print_metrics()

                        # 打印训练建议
                        recommendation = TrainingRecommendation.analyze(
                            episode_count, win_rate, avg_reward, agent.epsilon,
                            stage_info if stage_manager else None
                        )
                        print(f"\n💡 训练建议: {recommendation['status']}")
                        for rec in recommendation['recommendations']:
                            print(f"   {rec}")
                        print()

                    # 如果应该早停
                    if should_stop:
                        print(f"\n{'='*80}")
                        print(f"⛔ 早停触发！训练已收敛")
                        print(f"{'='*80}")
                        print(f"   最佳胜率: {early_stopping.best_value*100:.1f}% (回合 {early_stopping.best_episode})")
                        print(f"   当前胜率: {win_rate*100:.1f}%")
                        print(f"   未改进次数: {early_stopping.counter}/{early_stopping.patience}")
                        print(f"\n   建议: 使用回合 {early_stopping.best_episode} 的模型作为最终模型")
                        print(f"{'='*80}\n")

                        # 保存最终模型
                        final_checkpoint = {
                            'policy_net': agent.policy_net.state_dict(),
                            'target_net': agent.target_net.state_dict(),
                            'optimizer': agent.optimizer.state_dict(),
                            'epsilon': agent.epsilon,
                            'train_step': agent.train_step,
                            'episode': episode_count,
                            'early_stopped': True,
                            'best_episode': early_stopping.best_episode,
                            'best_win_rate': early_stopping.best_value,
                        }
                        final_path = os.path.join(PATHS['models'], 'final_model.pth')
                        torch.save(final_checkpoint, final_path)
                        print(f"✓ 最终模型已保存: {final_path}\n")

                        # 退出训练循环
                        break

                # ✅ 修复6：保存到标准路径（与玩家模式一致）
                # 构造包含回合数的checkpoint
                checkpoint = {
                    'policy_net': agent.policy_net.state_dict(),
                    'target_net': agent.target_net.state_dict(),
                    'optimizer': agent.optimizer.state_dict(),
                    'epsilon': agent.epsilon,
                    'train_step': agent.train_step,
                    'episode': episode_count,  # ✅ 新增：保存回合数用于断点续训
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
            'episode': episode_count,  # ✅ 包含回合数
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

        print(f"✓ 最终模型已保存到: {final_path}")
        print("  玩家对战模式将自动加载此模型")
        print("="*80 + "\n")

        # 关闭Ray集群，释放资源
        ray.shutdown()


if __name__ == '__main__':
    # 当脚本作为主程序运行时（而非被导入时）执行训练
    # 这是Python脚本的标准入口点写法
    train_with_ray()
