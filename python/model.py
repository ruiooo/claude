"""
model.py - DQN深度Q网络模型（支持CUDA加速）
================================================

【核心功能】
实现Deep Q-Network (DQN)算法的神经网络架构和训练逻辑。

【DQN算法简介】
DQN是深度强化学习的经典算法，将Q-learning与深度神经网络结合：
- Q-learning: 学习状态-动作价值函数 Q(s, a)
- 深度网络: 用神经网络近似Q函数（状态→Q值）
- 经验回放: 打破数据相关性，提高样本利用率
- 目标网络: 稳定训练，避免目标值震荡

【网络架构】
输入层: 状态向量（43维）
  ↓
隐藏层: 全连接 + ReLU + Dropout（防过拟合）
  ↓
输出层: Q值向量（9维，每个动作一个Q值）

【训练流程】
1. 从环境获取状态 s
2. 用策略网络计算 Q(s, a) 选择动作 a
3. 执行动作得到奖励 r 和下一状态 s'
4. 用目标网络计算目标值: y = r + γ * max Q(s', a')
5. 最小化损失: L = (Q(s,a) - y)²
6. 定期更新目标网络参数

【关键技术】
- Xavier初始化: 保证梯度稳定传播
- Dropout: 防止过拟合
- Huber Loss: 对异常值鲁棒
- 梯度裁剪: 防止梯度爆炸
- 学习率调度: 逐渐降低学习率
"""

import torch  # PyTorch深度学习框架
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 激活函数、损失函数等
import torch.optim as optim  # 优化器（Adam、SGD等）
import numpy as np  # 数值计算库
from typing import Tuple  # 类型注解


class DQN(nn.Module):
    """
    深度Q网络 (Deep Q-Network)

    功能: 输入状态向量，输出每个动作的Q值

    网络结构:
        Input(43) → Linear(43→512) → ReLU → Dropout(0.1)
                  → Linear(512→512) → ReLU → Dropout(0.1)
                  → Linear(512→256) → ReLU → Dropout(0.1)
                  → Linear(256→128) → ReLU → Dropout(0.1)
                  → Linear(128→9) → Output(9)

    为什么用全连接层?
    - 状态是特征向量（非图像），全连接层最适合
    - 如果是图像输入，应使用CNN（卷积神经网络）
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list):
        """
        初始化DQN网络

        构建过程:
        1. 创建多层全连接网络（根据hidden_dims配置）
        2. 每层后添加ReLU激活函数（引入非线性）
        3. 添加Dropout层（防止过拟合）
        4. 初始化权重（Xavier初始化）

        Args:
            state_dim: 状态空间维度（43维）
                - 6维AI状态（位置、速度、血量、冷却）
                - 25维敌人信息（5个敌人×5维）
                - 12维子弹信息（3个子弹×4维）
            action_dim: 动作空间维度（9维）
                - 0:静止 1-4:移动 5-8:射击
            hidden_dims: 隐藏层维度列表
                - 例: [512, 512, 256, 128]
                - 越大容量越强，但训练越慢
        """
        # 调用父类初始化（必须）
        super(DQN, self).__init__()

        # 构建网络层列表
        layers = []
        input_dim = state_dim  # 当前层的输入维度

        # 遍历隐藏层配置，逐层构建网络
        for hidden_dim in hidden_dims:
            # 1. 全连接层: input_dim → hidden_dim
            # 参数量: input_dim * hidden_dim + hidden_dim (权重+偏置)
            layers.append(nn.Linear(input_dim, hidden_dim))

            # 2. ReLU激活函数: max(0, x)
            # 作用: 引入非线性，使网络能学习复杂函数
            # 为什么用ReLU? 计算快、梯度稳定、不饱和
            layers.append(nn.ReLU())

            # 3. Dropout层: 训练时随机丢弃10%神经元
            # 作用: 防止过拟合，提高泛化能力
            # 原理: 强迫网络不依赖特定神经元
            layers.append(nn.Dropout(0.1))

            # 更新下一层的输入维度
            input_dim = hidden_dim

        # 输出层: 最后一个隐藏层 → action_dim（Q值）
        # 注意: 输出层不加激活函数，因为Q值可以是任意实数
        layers.append(nn.Linear(input_dim, action_dim))

        # 将所有层组合成Sequential模块
        # Sequential: 按顺序执行各层，前一层输出 = 后一层输入
        self.network = nn.Sequential(*layers)

        # 初始化所有层的权重
        # 好的初始化能加快训练、防止梯度消失/爆炸
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        初始化网络权重（Xavier均匀初始化）

        Xavier初始化原理:
        - 保证前向传播和反向传播时方差一致
        - 权重从均匀分布采样: U(-√(6/(in+out)), √(6/(in+out)))
        - 其中 in=输入维度, out=输出维度

        为什么要初始化?
        - 全0初始化: 所有神经元学习相同特征（对称性）
        - 随机太大: 梯度爆炸
        - 随机太小: 梯度消失
        - Xavier: 刚好保持梯度稳定

        Args:
            module: 网络中的某一层
        """
        # 只初始化全连接层（Linear）
        if isinstance(module, nn.Linear):
            # Xavier均匀初始化权重矩阵
            nn.init.xavier_uniform_(module.weight)

            # 偏置初始化为0（标准做法）
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播: 计算Q值

        流程:
        1. 输入状态向量 [batch_size, 43]
        2. 依次通过各层网络
        3. 输出Q值向量 [batch_size, 9]

        每个Q值的含义:
        Q[i] = 在当前状态下，执行动作i的预期累积奖励

        选择动作:
        - 训练时: epsilon-greedy（探索vs利用）
        - 测试时: argmax Q（贪婪策略）

        Args:
            state: 状态张量
                - shape: [batch_size, state_dim] 或 [state_dim]
                - dtype: float32
                - 已归一化到合理范围

        Returns:
            Q值张量
                - shape: [batch_size, action_dim]
                - Q[b, a] = 在状态b下执行动作a的Q值
        """
        # 通过Sequential网络，自动执行所有层
        # 等价于: x = layer1(state); x = layer2(x); ... return layerN(x)
        return self.network(state)


class DQNAgent:
    """
    DQN智能体，负责训练和决策

    核心组件:
    1. 策略网络 (policy_net): 用于选择动作
    2. 目标网络 (target_net): 用于计算目标Q值（稳定训练）
    3. 优化器 (optimizer): Adam优化算法
    4. Epsilon (ε): 探索率，控制探索vs利用的平衡

    工作流程:
    - 选择动作: epsilon-greedy策略
    - 训练网络: 最小化TD误差
    - 更新epsilon: 逐渐降低探索率
    - 更新目标网络: 定期同步参数
    """

    def __init__(self, state_dim: int, action_dim: int, config: dict, device: str = 'cuda'):
        """
        初始化DQN智能体

        初始化步骤:
        1. 创建策略网络和目标网络（架构相同）
        2. 目标网络参数初始化为策略网络的拷贝
        3. 创建Adam优化器
        4. 创建学习率调度器（可选）
        5. 初始化超参数（gamma, epsilon等）

        Args:
            state_dim: 状态空间维度（43）
            action_dim: 动作空间维度（9）
            config: 配置字典，包含:
                - gamma: 折扣因子（0.99）
                - epsilon_start: 初始探索率（1.0）
                - epsilon_end: 最终探索率（0.1）
                - epsilon_decay: 衰减率（0.9999）
                - target_update_freq: 目标网络更新频率（100步）
                - hidden_dims: 网络架构（[512, 512, 256, 128]）
                - learning_rate: 学习率（0.0003）
            device: 运行设备（'cuda'或'cpu'）
                - CUDA: GPU加速，训练快10-100倍
                - CPU: 通用但慢，适合调试
        """
        # 保存配置
        self.state_dim = state_dim
        self.action_dim = action_dim

        # 设置设备: 如果有GPU且可用则用GPU，否则用CPU
        # torch.cuda.is_available(): 检测CUDA是否可用
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # ========== 超参数设置 ==========

        # gamma (γ): 折扣因子，范围[0, 1]
        # 作用: 控制对未来奖励的重视程度
        # - γ=0: 只看当前奖励（短视）
        # - γ=1: 未来奖励和当前奖励同等重要（远视）
        # - γ=0.99: 标准值，平衡当前和未来
        self.gamma = config['gamma']

        # epsilon (ε): 探索率，范围[0, 1]
        # 作用: epsilon-greedy策略的探索概率
        # - ε=1.0: 完全随机探索（训练初期）
        # - ε=0.1: 10%探索，90%利用（训练后期）
        self.epsilon = config['epsilon_start']
        self.epsilon_end = config['epsilon_end']
        self.epsilon_decay = config['epsilon_decay']

        # 目标网络更新频率: 每N步更新一次
        # 为什么不是每步更新? 避免目标值震荡，稳定训练
        self.target_update_freq = config['target_update_freq']

        # ========== 创建神经网络 ==========

        # 网络架构配置: [512, 512, 256, 128]
        # 参数量计算:
        # - Layer1: 43→512 = 43*512 + 512 = 22,528
        # - Layer2: 512→512 = 512*512 + 512 = 262,656
        # - Layer3: 512→256 = 512*256 + 256 = 131,328
        # - Layer4: 256→128 = 256*128 + 128 = 32,896
        # - Output: 128→9 = 128*9 + 9 = 1,161
        # 总计: ≈450,000 参数
        hidden_dims = config['hidden_dims']

        # 策略网络: 用于选择动作和训练
        # .to(device): 将网络参数移到GPU/CPU
        self.policy_net = DQN(state_dim, action_dim, hidden_dims).to(self.device)

        # 目标网络: 用于计算目标Q值
        # 参数初始化为策略网络的拷贝
        self.target_net = DQN(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # 目标网络设为评估模式（不训练）
        # eval()作用: 关闭Dropout、BatchNorm等训练专用层
        self.target_net.eval()

        # ========== 创建优化器 ==========

        # Adam优化器: 自适应学习率优化算法
        # 原理: 结合动量(Momentum)和RMSProp
        # 优点:
        # - 自适应学习率（每个参数不同）
        # - 对超参数不敏感
        # - 收敛快、效果好
        #
        # 学习率(lr): 0.0003
        # - 太大: 训练不稳定，震荡
        # - 太小: 收敛慢
        # - 0.0003: 适中值，适合大多数DQN任务
        self.optimizer = optim.Adam(self.policy_net.parameters(),
                                    lr=config['learning_rate'])

        # ========== 学习率调度器（可选）==========

        # StepLR调度器: 每step_size步，学习率乘以gamma
        # 例: lr=0.0003, 每1000步后 lr*=0.95
        # - 0步:     lr=0.0003
        # - 1000步:  lr=0.000285
        # - 2000步:  lr=0.000271
        # - ...
        # 作用: 训练后期降低学习率，精细调整参数
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                    step_size=1000,
                                                    gamma=0.95)

        # 训练步数计数器: 用于决定何时更新目标网络
        self.train_step = 0

        # 打印初始化信息
        print(f"DQN模型初始化完成 - 设备: {self.device}")

        # 计算并打印总参数量
        # p.numel(): 返回张量p的元素个数
        # sum(...): 累加所有参数的元素个数
        total_params = sum(p.numel() for p in self.policy_net.parameters())
        print(f"模型参数量: {total_params:,}")  # :, 表示千位分隔符

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        选择动作（epsilon-greedy策略）

        Epsilon-Greedy策略:
        - 以概率ε: 随机选择动作（探索）
        - 以概率1-ε: 选择Q值最大的动作（利用）

        为什么要探索?
        - 只利用: AI只会重复已知的策略，无法发现更好策略
        - 只探索: AI完全随机，学不到任何东西
        - 平衡: 早期多探索（ε=1.0），后期多利用（ε=0.1）

        训练模式 vs 推理模式:
        - 训练: epsilon-greedy + Dropout开启
        - 推理: 完全利用（ε=0）+ Dropout关闭

        Args:
            state: 当前状态
                - shape: [state_dim] = [43]
                - dtype: float32 numpy数组
            training: 是否在训练模式
                - True: 使用epsilon-greedy
                - False: 完全利用（测试/对战时）

        Returns:
            选择的动作索引 (0-8)
                - 0: 静止
                - 1-4: 上下左右移动
                - 5-8: 上下左右射击
        """
        # 训练模式处理
        if training:
            # 将策略网络设为训练模式
            # train()作用: 启用Dropout、BatchNorm等
            self.policy_net.train()

            # epsilon-greedy探索
            # np.random.random(): 生成[0,1)之间的随机数
            if np.random.random() < self.epsilon:
                # 随机探索: 从0到action_dim-1随机选一个整数
                return np.random.randint(0, self.action_dim)
        else:
            # 推理模式: 将策略网络设为评估模式
            # eval()作用: 关闭Dropout（测试时不随机丢弃神经元）
            self.policy_net.eval()

        # 利用策略网络选择动作
        # torch.no_grad(): 禁用梯度计算
        # 作用: 节省内存，加快推理速度（不需要反向传播）
        with torch.no_grad():
            # 1. 将numpy数组转为torch张量
            # FloatTensor: 创建float32类型的张量
            # unsqueeze(0): 添加batch维度 [43] → [1, 43]
            # .to(device): 移到GPU/CPU
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            # 2. 前向传播计算Q值
            # q_values shape: [1, 9]
            # q_values[0, i] = 执行动作i的预期累积奖励
            q_values = self.policy_net(state_tensor)

            # 3. 选择Q值最大的动作（贪婪策略）
            # argmax(dim=1): 在动作维度（dim=1）找最大值的索引
            # item(): 将单元素张量转为Python int
            action = q_values.argmax(dim=1).item()

        return action

    def train(self, batch: Tuple, use_prioritized: bool = False):
        """
        训练网络（DQN算法核心）- 支持 Double DQN 和优先经验回放

        【算法升级】
        ✅ Double DQN: 用策略网络选择动作，目标网络评估（减少Q值过估计）
        ✅ 优先经验回放: 支持重要性采样权重（重要经验优先学习）

        DQN训练算法（Q-learning + 深度网络）:

        1. 从经验回放缓冲区采样一批数据:
           (s, a, r, s', done) [+ indices, weights (如果使用优先回放)]

        2. 计算当前Q值（策略网络）:
           Q_current = policy_net(s)[a]

        3. 计算目标Q值（Double DQN改进）:
           ❌ 标准DQN: Q_target = r + γ * max(target_net(s'))
           ✅ Double DQN:
              - 用策略网络选择最佳动作: a* = argmax policy_net(s')
              - 用目标网络评估该动作: Q_target = r + γ * target_net(s')[a*]
           原因: 减少Q值过估计（max操作的系统性高估）

        4. 计算TD误差（Temporal Difference Error）:
           TD_error = Q_current - Q_target

        5. 最小化损失函数:
           ❌ 标准: Loss = Huber(Q_current, Q_target)
           ✅ 优先回放: Loss = weights * Huber(Q_current, Q_target)
           原因: 重要性采样权重补偿优先采样的偏差

        6. 反向传播 + 梯度下降更新参数

        为什么用目标网络?
        - 如果用策略网络计算目标: y = r + γ*max Q_policy(s')
        - 问题: 策略网络一直在更新，目标值不稳定
        - 解决: 用固定的目标网络，每N步才更新一次

        为什么用Huber Loss?
        - MSE Loss: (x-y)² 对异常值敏感（梯度大）
        - MAE Loss: |x-y| 梯度恒定，收敛慢
        - Huber: 结合两者优点，对异常值鲁棒且收敛快

        Args:
            batch: 一批经验数据
                标准buffer: (states, actions, rewards, next_states, dones)
                优先buffer: (states, actions, rewards, next_states, dones, indices, weights)
                - states: [batch_size, 43]
                - actions: [batch_size]
                - rewards: [batch_size]
                - next_states: [batch_size, 43]
                - dones: [batch_size] (1=结束, 0=未结束)
                - indices: [batch_size] 采样索引（优先回放用）
                - weights: [batch_size] 重要性采样权重（优先回放用）

            use_prioritized: 是否使用优先经验回放
                - False: 标准DQN训练
                - True: 使用重要性采样权重，返回TD误差

        Returns:
            如果 use_prioritized=False:
                loss (float): 损失值
            如果 use_prioritized=True:
                (loss, td_errors): (损失值, TD误差数组)
                - loss: float，用于监控
                - td_errors: ndarray，用于更新优先级
        """
        # ========== 解包批数据 ==========

        if use_prioritized:
            # 优先经验回放: 包含权重和索引
            states, actions, rewards, next_states, dones, indices, weights = batch
            # 转换权重为PyTorch张量
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            # 标准经验回放
            states, actions, rewards, next_states, dones = batch
            weights = None  # 不使用权重

        # ========== 数据转换: numpy数组 → PyTorch张量 ==========

        # 将所有numpy数组转为张量并移到GPU
        # FloatTensor: float32类型（适合神经网络）
        # LongTensor: int64类型（适合索引）
        # .to(device): 移到GPU加速计算
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # ========== 计算当前Q值 ==========

        # 1. 前向传播: states → Q值矩阵
        # policy_net(states) shape: [batch_size, 9]
        # 每一行是一个状态对应的9个动作的Q值
        q_values = self.policy_net(states)

        # 2. 提取实际执行的动作对应的Q值
        # gather(dim, index): 沿指定维度收集值
        # - dim=1: 沿动作维度
        # - actions.unsqueeze(1): [batch_size] → [batch_size, 1]
        # 例: actions=[2,5,1], Q=[[q0,q1,q2,...], ...]
        #     → current_q = [Q[0,2], Q[1,5], Q[2,1]]
        # shape: [batch_size, 1]
        current_q_values = q_values.gather(1, actions.unsqueeze(1))

        # ========== 计算目标Q值（Double DQN）==========

        # 目标Q值计算需要用目标网络（不是策略网络）
        # 且不需要梯度（因为目标网络不训练）
        with torch.no_grad():
            # ✅ Double DQN改进: 解耦动作选择和动作评估
            # 原理: 标准DQN用同一个网络选择和评估，会系统性高估Q值
            # 改进: 用策略网络选择动作，用目标网络评估该动作

            # 步骤1: 用策略网络选择下一状态的最佳动作
            # policy_net(next_states): [batch_size, 9]
            # argmax(1): 沿动作维度找最大Q值的索引
            # shape: [batch_size]
            next_actions = self.policy_net(next_states).argmax(1)

            # 步骤2: 用目标网络计算下一状态的Q值
            # target_net(next_states): [batch_size, 9]
            next_q_values = self.target_net(next_states)

            # 步骤3: 提取策略网络选择的动作对应的Q值
            # gather(1, ...): 沿动作维度收集值
            # next_actions.unsqueeze(1): [batch_size] → [batch_size, 1]
            # gather(...).squeeze(): [batch_size, 1] → [batch_size]
            #
            # 对比标准DQN:
            # ❌ 标准DQN: max_next_q = next_q_values.max(1)[0]
            #    （同一个网络选择和评估，导致过估计）
            # ✅ Double DQN: max_next_q = next_q_values[next_actions]
            #    （分离选择和评估，减少过估计）
            max_next_q = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze()

            # 步骤4: 贝尔曼方程计算目标Q值
            # Q*(s,a) = r + γ * Q_target(s', a*_policy)  (如果未结束)
            # Q*(s,a) = r                                (如果已结束)
            #
            # (1 - dones): 如果done=1则系数为0，否则为1
            # 作用: 结束状态的目标Q值 = 当前奖励（无未来奖励）
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q

        # ========== 计算TD误差（用于优先经验回放）==========

        # TD误差 = 当前Q值 - 目标Q值
        # 用途:
        # 1. 衡量预测误差大小
        # 2. 优先经验回放: 根据|TD误差|设置优先级
        # 3. 监控训练进度
        #
        # detach(): 从计算图中分离，避免梯度传播
        # cpu(): 移到CPU（numpy需要）
        # numpy(): 转为numpy数组
        td_errors = (current_q_values.squeeze() - target_q_values).detach().cpu().numpy()

        # ========== 计算损失函数（支持优先经验回放）==========

        # Huber Loss (Smooth L1 Loss):
        # - |x-y| < 1: loss = 0.5 * (x-y)²    (类似MSE)
        # - |x-y| ≥ 1: loss = |x-y| - 0.5     (类似MAE)
        #
        # 优点:
        # - 对小误差敏感（二次项），收敛快
        # - 对大误差鲁棒（线性项），避免梯度爆炸

        if weights is not None:
            # ✅ 优先经验回放: 使用重要性采样权重
            # reduction='none': 不自动求平均，返回每个样本的损失
            # shape: [batch_size]
            element_wise_loss = F.smooth_l1_loss(
                current_q_values.squeeze(),
                target_q_values,
                reduction='none'  # ← 关键：保留每个样本的损失
            )

            # 应用重要性采样权重
            # 原理: 高优先级样本被过度采样，需要降低权重
            #      低优先级样本被欠采样，需要提高权重
            # weighted_loss shape: [batch_size]
            weighted_loss = element_wise_loss * weights

            # 对所有样本求平均
            loss = weighted_loss.mean()
        else:
            # ❌ 标准DQN: 均匀权重
            # reduction='mean': 自动对所有样本求平均
            # squeeze(): 移除大小为1的维度
            # current_q_values shape: [batch_size, 1] → [batch_size]
            loss = F.smooth_l1_loss(
                current_q_values.squeeze(),
                target_q_values,
                reduction='mean'
            )

        # ========== 反向传播和参数更新 ==========

        # 1. 清零梯度
        # 为什么要清零? PyTorch默认累积梯度，不清零会叠加
        self.optimizer.zero_grad()

        # 2. 反向传播: 计算梯度
        # loss.backward(): 自动计算所有参数的梯度 dL/dw
        # 原理: 链式法则从输出层反向传播到输入层
        loss.backward()

        # 3. 梯度裁剪: 防止梯度爆炸
        # 原理: 如果梯度范数 > 1.0，则缩放到1.0
        # 梯度爆炸后果: 参数更新太大，导致发散
        #
        # clip_grad_norm_: 裁剪所有参数的梯度范数
        # 1.0: 最大梯度范数
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)

        # 4. 参数更新: w = w - lr * dL/dw
        # optimizer.step(): Adam优化器执行参数更新
        # Adam更新公式（简化版）:
        #   m_t = β1*m_(t-1) + (1-β1)*grad        (一阶动量)
        #   v_t = β2*v_(t-1) + (1-β2)*grad²       (二阶动量)
        #   w_t = w_(t-1) - lr * m_t / (√v_t + ε) (参数更新)
        self.optimizer.step()

        # 5. 学习率调度: 每步后调用
        # 作用: 按调度器策略调整学习率（每1000步衰减）
        self.scheduler.step()

        # 训练步数+1
        self.train_step += 1

        # ========== 定期更新目标网络 ==========

        # 每target_update_freq步，同步目标网络参数
        # 例: target_update_freq=100，每100步更新一次
        if self.train_step % self.target_update_freq == 0:
            self.update_target_network()

        # ========== 返回结果 ==========

        # 根据是否使用优先经验回放返回不同结果
        if use_prioritized:
            # 优先经验回放: 返回 (损失值, TD误差)
            # TD误差用于更新经验的优先级
            # item(): 将单元素张量转为Python float
            return loss.item(), td_errors
        else:
            # 标准DQN: 只返回损失值
            return loss.item()

    def update_target_network(self):
        """
        更新目标网络参数

        操作: 将策略网络的参数完全拷贝到目标网络

        为什么不每步都更新?
        - 目标值会不断变化，训练不稳定
        - 类似"移动的靶子"，难以收敛

        为什么要定期更新?
        - 策略网络已经学到了新知识
        - 目标网络需要跟上，否则目标值过时

        更新频率的选择:
        - 太频繁(每步): 训练不稳定
        - 太稀疏(1000步): 目标值过时
        - 适中(100步): 平衡稳定性和时效性
        """
        # load_state_dict: 加载参数字典
        # state_dict(): 获取策略网络的所有参数
        # 效果: target_net参数 = policy_net参数（深拷贝）
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_epsilon(self):
        """
        更新探索率（epsilon衰减）

        Epsilon衰减策略:
        ε_(t+1) = max(ε_end, ε_t * decay)

        例: ε_start=1.0, ε_end=0.1, decay=0.9999
        - 第0回合:    ε=1.0000 (完全探索)
        - 第1000回合: ε=0.9048
        - 第5000回合: ε=0.6065
        - 第10000回合: ε=0.3679
        - 第20000回合: ε=0.1353
        - 第30000回合: ε=0.1000 (达到下限)

        为什么要衰减?
        - 训练初期: AI什么都不懂，需要多探索
        - 训练后期: AI已学到好策略，应多利用

        衰减太快的问题:
        - 策略过早固化，卡在局部最优
        - 无法适应动态难度（敌人数量增加）

        衰减太慢的问题:
        - 一直随机探索，学不到稳定策略
        - 训练效率低
        """
        # max: 保证epsilon不低于最小值
        # self.epsilon * self.epsilon_decay: 指数衰减
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path: str):
        """
        保存模型到文件

        保存内容:
        1. 策略网络参数 (policy_net)
        2. 目标网络参数 (target_net)
        3. 优化器状态 (optimizer)
        4. 当前epsilon值
        5. 训练步数

        为什么保存这么多?
        - 网络参数: 恢复模型用于推理/继续训练
        - 优化器状态: 包含动量等信息，继续训练时需要
        - epsilon: 继续训练需要知道当前探索率
        - train_step: 用于目标网络更新计数

        用途:
        - 定期保存: 防止训练中断丢失进度
        - 最终保存: 训练完成后用于部署
        - 历史版本: 用于自我对弈

        Args:
            path: 保存路径
                - 例: "saved_models/model_ep1000.pth"
                - .pth: PyTorch模型文件扩展名
        """
        # torch.save: 将Python对象序列化保存
        # state_dict(): 获取模型的参数字典
        #
        # 字典包含:
        # - 'policy_net': 策略网络的所有权重和偏置
        # - 'target_net': 目标网络的所有权重和偏置
        # - 'optimizer': Adam优化器的状态（动量、学习率等）
        # - 'epsilon': 当前探索率
        # - 'train_step': 训练了多少步
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_step': self.train_step,
        }, path)

        print(f"模型已保存到: {path}")

    def load(self, path: str):
        """
        从文件加载模型（支持继续训练和推理）

        加载步骤:
        1. 读取checkpoint文件
        2. 恢复策略网络参数
        3. 恢复目标网络参数
        4. 恢复优化器状态（继续训练需要）
        5. 恢复epsilon和train_step
        6. 设置为评估模式（推理时）

        使用场景:
        - 断点续训: 从上次保存的位置继续训练
        - 模型评估: 加载训练好的模型进行测试
        - 玩家对战: 加载模型让玩家对战AI
        - 自我对弈: 加载历史版本作为对手

        Args:
            path: 模型文件路径
                - 例: "saved_models/final_model.pth"
        """
        # torch.load: 反序列化加载模型
        # map_location: 指定加载到哪个设备
        # - 训练在GPU，推理在CPU: map_location='cpu'
        # - 自动适配当前设备: map_location=self.device
        checkpoint = torch.load(path, map_location=self.device)

        # 恢复策略网络参数
        # load_state_dict: 将参数字典加载到网络
        self.policy_net.load_state_dict(checkpoint['policy_net'])

        # 恢复目标网络参数
        self.target_net.load_state_dict(checkpoint['target_net'])

        # 恢复优化器状态（包含动量等信息）
        # 如果只用于推理，可以跳过这步
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        # 恢复epsilon（继续训练需要）
        self.epsilon = checkpoint['epsilon']

        # 恢复训练步数（用于目标网络更新）
        self.train_step = checkpoint['train_step']

        # 设置为评估模式
        # eval()作用:
        # - 关闭Dropout（推理时不随机丢弃神经元）
        # - 冻结BatchNorm统计量
        # 训练时会自动切换回train()模式
        self.policy_net.eval()
        self.target_net.eval()

        # 打印加载信息
        print(f"模型已从 {path} 加载")
        print(f"训练步数: {self.train_step}, Epsilon: {self.epsilon:.4f}")

    def get_model_version(self) -> int:
        """
        获取模型版本号（基于训练步数）

        版本计算: version = train_step // 1000

        例:
        - train_step=0:     version=0
        - train_step=999:   version=0
        - train_step=1000:  version=1
        - train_step=5432:  version=5
        - train_step=10000: version=10

        用途:
        - 自我对弈: 根据版本号管理历史模型
        - 模型追踪: 记录训练进度
        - 版本对比: 比较不同阶段的模型性能

        Returns:
            模型版本号（整数）
        """
        return self.train_step // 1000
