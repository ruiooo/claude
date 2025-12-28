"""
replay_buffer.py - 经验回放缓冲区（优化GPU批处理）
========================================================

【核心功能】
实现经验回放(Experience Replay)机制，这是DQN算法的关键组件之一。

【经验回放原理】
强化学习的一个核心挑战是数据相关性：
- 连续的经验高度相关（下一状态依赖当前状态）
- 直接用于训练会导致梯度估计偏差
- 神经网络容易过拟合到最近的经验

经验回放解决方案：
1. 将经验存储在缓冲区中
2. 训练时随机采样批次数据
3. 打破时间相关性，提高样本利用率

【DQN算法中的作用】
训练循环:
    for step in episode:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)

        # 存储经验
        buffer.push(state, action, reward, next_state, done)

        # 随机采样训练
        batch = buffer.sample(batch_size)
        loss = agent.train(batch)

【优化设计】
1. 使用numpy数组而非Python list（更快）
2. 预分配内存（避免动态扩容）
3. 循环覆盖（自动删除旧数据）
4. 批量操作（利用numpy向量化）
"""

import numpy as np  # 高效的数值数组库
from collections import deque  # 双端队列（PrioritizedReplayBuffer中使用）
import random  # 随机采样


class ReplayBuffer:
    """
    经验回放缓冲区（标准版）

    数据结构: 循环缓冲区（Ring Buffer）
    - 固定容量
    - 写满后从头开始覆盖
    - O(1) 插入和随机访问

    存储内容: (s, a, r, s', done)
    - s: 当前状态
    - a: 执行的动作
    - r: 获得的奖励
    - s': 下一个状态
    - done: 是否结束

    使用示例:
        buffer = ReplayBuffer(capacity=100000, state_dim=43)
        buffer.push(state, action, reward, next_state, done)

        if len(buffer) >= 1000:
            batch = buffer.sample(256)
            agent.train(batch)
    """

    def __init__(self, capacity: int, state_dim: int):
        """
        初始化经验回放缓冲区

        设计决策:
        1. 预分配所有内存（避免动态扩容开销）
        2. 使用numpy数组（比Python list快10-100倍）
        3. 分离存储（states, actions, rewards...）而非存储tuple
           原因: 批量采样时可以直接切片，无需解包

        Args:
            capacity: 缓冲区容量（最多存储多少条经验）
                - 典型值: 50,000 - 1,000,000
                - 太小: 数据不够多样，过拟合
                - 太大: 内存占用高，旧数据过时
            state_dim: 状态维度（43维）
                - 必须与环境的状态空间匹配
                - 用于预分配状态数组大小
        """
        # 保存配置
        self.capacity = capacity
        self.state_dim = state_dim

        # ========== 预分配内存（所有数组） ==========

        # 状态数组: [capacity, state_dim]
        # dtype=float32: 与PyTorch默认类型一致，节省内存
        # 每条经验的状态占用: 43 * 4 = 172 bytes
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)

        # 动作数组: [capacity]
        # dtype=int32: 动作是整数索引（0-8）
        # 每条经验的动作占用: 4 bytes
        self.actions = np.zeros(capacity, dtype=np.int32)

        # 奖励数组: [capacity]
        # dtype=float32: 奖励是浮点数
        # 每条经验的奖励占用: 4 bytes
        self.rewards = np.zeros(capacity, dtype=np.float32)

        # 下一状态数组: [capacity, state_dim]
        # 与states结构相同
        # 每条经验的下一状态占用: 43 * 4 = 172 bytes
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)

        # 完成标志数组: [capacity]
        # dtype=float32: 虽然是0/1，但用float方便GPU计算
        # 每条经验的done标志占用: 4 bytes
        # done=1.0: 该回合已结束
        # done=0.0: 该回合未结束
        self.dones = np.zeros(capacity, dtype=np.float32)

        # ✅ 数据来源标记: [capacity]
        # 用于区分人类数据和AI数据
        # 0: AI生成的数据
        # 1: 人类专家数据
        self.sources = np.zeros(capacity, dtype=np.int8)

        # 总内存占用估算:
        # (172 + 4 + 4 + 172 + 4 + 1) * capacity = 357 * capacity bytes
        # 100,000条经验 ≈ 34 MB (完全可以接受)

        # ========== 循环缓冲区指针 ==========

        # 当前写入位置（0 到 capacity-1）
        # 每次push后自动前进，到达capacity后回到0
        self.position = 0

        # 当前缓冲区实际大小（0 到 capacity）
        # push时递增，最多到capacity
        # 用于判断是否有足够数据进行训练
        self.size = 0

        # ========== 周期性人类数据控制 ==========
        # 控制是否在当前采样中包含人类数据
        # True: 采样包含人类数据，False: 只采样AI数据
        self.include_human_data = True

        # ✅ 性能优化：缓存AI数据索引，避免每次采样都扫描
        # 维护一个AI数据索引列表，只在添加数据时更新
        self._ai_indices_cache = []
        self._cache_valid = False  # 缓存是否有效

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool, source: int = 0):
        """
        添加一条经验到缓冲区

        循环覆盖机制:
        - 未满时: position递增, size递增
        - 已满时: position循环, size保持capacity
        - 旧数据自动被新数据覆盖（FIFO策略）

        为什么不用append?
        - append需要动态扩容（慢）
        - 预分配+索引赋值（快）

        Args:
            state: 当前状态
                - shape: [state_dim]
                - dtype: float32
                - 例: [x, y, vx, vy, ...]
            action: 执行的动作
                - 范围: 0-8
                - 例: 1 (向上移动)
            reward: 获得的奖励
                - 范围: -100 到 +300
                - 例: 0.02 (存活), 100.0 (击杀)
            next_state: 下一个状态
                - shape: [state_dim]
                - 执行action后的新状态
            done: 是否结束
                - True: 游戏结束（胜利/失败）
                - False: 游戏继续
            source: 数据来源
                - 0: AI生成的数据（默认）
                - 1: 人类专家数据

        示例:
            # AI数据
            buffer.push(
                state=[0.5, 0.3, ...],
                action=1,
                reward=0.02,
                next_state=[0.5, 0.35, ...],
                done=False,
                source=0
            )
            # 人类数据
            buffer.push(..., source=1)
        """
        # ========== 防御性检查（调试段错误） ==========
        # 确保 state 是有效的 numpy 数组
        if state is None:
            raise ValueError("push(): state is None")
        if not isinstance(state, np.ndarray):
            state = np.asarray(state, dtype=np.float32)
        if state.shape != (self.state_dim,):
            raise ValueError(f"push(): state shape mismatch: expected ({self.state_dim},), got {state.shape}")

        if next_state is None:
            raise ValueError("push(): next_state is None")
        if not isinstance(next_state, np.ndarray):
            next_state = np.asarray(next_state, dtype=np.float32)
        if next_state.shape != (self.state_dim,):
            raise ValueError(f"push(): next_state shape mismatch: expected ({self.state_dim},), got {next_state.shape}")

        # 获取当前写入位置（简化逻辑，移除永久保留）
        # 优化：允许覆盖人类数据，通过高采样权重保证利用率
        idx = self.position

        # 将数据写入对应位置
        # 直接赋值，不需要append（O(1)操作）
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state

        # 将bool转为float（1.0或0.0）
        # 原因: GPU计算时float更高效
        self.dones[idx] = 1.0 if done else 0.0

        # ✅ 记录数据来源
        self.sources[idx] = source

        # 更新写入位置（循环）
        # (position + 1) % capacity 实现循环:
        # - position=0,1,2,...,capacity-2,capacity-1,0,1,2,...
        # 例: capacity=5, position序列: 0,1,2,3,4,0,1,2,3,4,...
        self.position = (self.position + 1) % self.capacity

        # 更新实际大小
        # min确保size不超过capacity
        # 前capacity次push: size递增
        # 之后: size保持capacity
        self.size = min(self.size + 1, self.capacity)

        # ✅ 性能优化：标记缓存失效（有新数据添加）
        self._cache_valid = False

    def _update_ai_indices_cache(self):
        """
        更新AI数据索引缓存

        性能优化：避免每次采样都扫描整个缓冲区
        - 只在缓存失效时调用一次
        - 缓存结果供多次采样使用
        """
        if not self._cache_valid:
            # 找出所有AI数据（source=0）的索引
            self._ai_indices_cache = np.where(self.sources[:self.size] == 0)[0]
            self._cache_valid = True

    def sample(self, batch_size: int):
        """
        随机采样一批经验

        随机采样的好处:
        1. 打破时间相关性（连续的经验通常相似）
        2. 每个经验被使用多次（提高样本利用率）
        3. 梯度估计更准确（样本更多样化）

        采样策略: 均匀随机采样
        - 每条经验被采样的概率相同
        - 简单高效
        - 对比: 优先经验回放（根据TD误差加权采样）

        Args:
            batch_size: 批次大小
                - 典型值: 32, 64, 128, 256
                - 越大: 梯度估计越准确，但内存占用越高
                - 越小: 更新频率高，但梯度噪声大

        Returns:
            tuple: (states, actions, rewards, next_states, dones)
                - states: [batch_size, state_dim] float32
                - actions: [batch_size] int32
                - rewards: [batch_size] float32
                - next_states: [batch_size, state_dim] float32
                - dones: [batch_size] float32

        示例:
            batch = buffer.sample(256)
            states, actions, rewards, next_states, dones = batch

            # states.shape = [256, 43]
            # actions.shape = [256]
            # rewards.shape = [256]
        """
        # ✅ 周期性人类数据控制：根据标志决定采样范围
        if not self.include_human_data:
            # 只采样AI数据（source=0）
            # ✅ 性能优化：使用缓存的AI索引，避免重复扫描
            self._update_ai_indices_cache()
            ai_indices = self._ai_indices_cache

            # 如果AI数据不足batch_size，使用有放回采样
            if len(ai_indices) < batch_size:
                indices = np.random.choice(ai_indices, batch_size, replace=True)
            else:
                indices = np.random.choice(ai_indices, batch_size, replace=False)
        else:
            # 正常采样（包含所有数据）
            # np.random.choice: 从[0, size)中无放回随机抽取
            # - size: 缓冲区当前大小
            # - batch_size: 抽取数量
            # - replace=False: 不放回抽样（每个经验最多被抽一次）
            #
            # 为什么用size而不是capacity?
            # - 缓冲区未满时，只有前size个位置有有效数据
            # - 从未填充的位置采样会得到全0数据（错误）
            indices = np.random.choice(self.size, batch_size, replace=False)

        # 使用fancy indexing批量提取数据
        # numpy的高级索引非常高效（向量化操作）
        #
        # 例: indices = [3, 7, 15, 42]
        # self.states[indices] 返回:
        # [[states[3]], [states[7]], [states[15]], [states[42]]]
        #
        # 返回格式: tuple of arrays（方便解包）
        return (
            self.states[indices],       # [batch_size, state_dim]
            self.actions[indices],      # [batch_size]
            self.rewards[indices],      # [batch_size]
            self.next_states[indices],  # [batch_size, state_dim]
            self.dones[indices]         # [batch_size]
        )

    def __len__(self):
        """
        返回缓冲区当前大小

        魔术方法: 允许使用len(buffer)获取大小

        用途:
        - 判断是否有足够数据开始训练
        - 显示训练进度

        Returns:
            当前缓冲区中的经验数量（0 到 capacity）

        示例:
            if len(buffer) >= min_buffer_size:
                batch = buffer.sample(batch_size)
                agent.train(batch)
        """
        return self.size

    def is_ready(self, min_size: int) -> bool:
        """
        检查缓冲区是否准备好进行训练

        训练前提: 需要足够的经验数据
        - 太少: 梯度估计不准确
        - 适量: 提供多样化的初始数据

        Args:
            min_size: 最小经验数量
                - 典型值: batch_size 到 10 * batch_size
                - 例: batch_size=256, min_size=1000

        Returns:
            bool: True=可以开始训练, False=需要继续收集

        示例:
            # 收集初始经验
            while not buffer.is_ready(1000):
                action = random.randint(0, 8)
                next_state, reward, done = env.step(action)
                buffer.push(state, action, reward, next_state, done)
                if done:
                    state = env.reset()

            # 开始训练
            for episode in range(max_episodes):
                ...
        """
        # 简单比较: 当前大小是否达到要求
        return self.size >= min_size

    def clear(self):
        """
        清空缓冲区

        操作: 重置指针，不实际删除数据
        - position重置为0
        - size重置为0
        - 数组内容保留（会被新数据覆盖）

        为什么不清零数组?
        - 清零耗时（需要遍历整个数组）
        - 没必要（覆盖即可）
        - 只重置指针效率更高

        使用场景:
        - 切换任务时清空旧经验
        - 调试时重新开始
        - 通常不需要调用（循环覆盖自动管理）

        示例:
            buffer.clear()
            # 缓冲区现在为空
            assert len(buffer) == 0
        """
        # 重置写入位置到起点
        self.position = 0

        # 重置实际大小为0
        self.size = 0

    def get_source_stats(self):
        """
        获取缓冲区中数据来源的统计信息

        Returns:
            dict: {
                'human_count': 人类数据数量,
                'ai_count': AI数据数量,
                'human_ratio': 人类数据比例,
                'total': 总数据量
            }

        示例:
            stats = buffer.get_source_stats()
            print(f"人类数据: {stats['human_ratio']:.1%}")
        """
        if self.size == 0:
            return {
                'human_count': 0,
                'ai_count': 0,
                'human_ratio': 0.0,
                'total': 0
            }

        # 统计人类数据数量（source=1）
        human_count = int(np.sum(self.sources[:self.size] == 1))
        ai_count = self.size - human_count

        return {
            'human_count': human_count,
            'ai_count': ai_count,
            'human_ratio': human_count / self.size if self.size > 0 else 0.0,
            'total': self.size
        }


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    优先经验回放缓冲区（进阶版）

    核心思想: 重要的经验应该被更频繁地采样
    - 根据TD误差确定优先级
    - TD误差大 = 预测不准 = 更值得学习
    - 优先采样高TD误差的经验

    优先级计算:
        priority = |TD_error| + ε
        TD_error = Q(s,a) - (r + γ*max Q(s',a'))

    采样概率:
        P(i) = priority(i)^α / Σ priority(j)^α
        α: 优先级指数（0=均匀采样，1=完全按优先级）

    重要性采样权重:
        w(i) = (N * P(i))^(-β)
        β: 重要性采样指数（补偿优先级采样的偏差）

    优点:
    - 收敛更快（专注于难样本）
    - 样本利用率更高

    缺点:
    - 实现复杂
    - 计算开销大
    - 需要额外内存（优先级数组）

    论文: Prioritized Experience Replay (Schaul et al., 2016)
    """

    def __init__(self, capacity: int, state_dim: int, alpha: float = 0.6, beta: float = 0.4):
        """
        初始化优先经验回放缓冲区

        Args:
            capacity: 缓冲区容量
            state_dim: 状态维度
            alpha: 优先级指数（0-1）
                - 0: 退化为均匀采样
                - 1: 完全按优先级采样
                - 0.6: 推荐值（平衡探索和利用）
            beta: 重要性采样指数（0-1）
                - 0: 不补偿偏差
                - 1: 完全补偿偏差
                - 0.4: 初始值（训练过程中逐渐增加到1）
        """
        # 调用父类初始化（创建基础缓冲区）
        super().__init__(capacity, state_dim)

        # 优先级参数
        self.alpha = alpha  # 优先级指数
        self.beta = beta    # 重要性采样指数

        # Beta退火: 训练过程中逐渐增加
        # 原因: 初期优先级不准确，后期需要更多补偿
        self.beta_increment = 0.001

        # ========== 优先级存储 ==========

        # 优先级数组: [capacity]
        # 每条经验的优先级值
        # 初始化为0，push时设置为max_priority
        self.priorities = np.zeros(capacity, dtype=np.float32)

        # 最大优先级: 新经验使用此值
        # 原因: 新经验尚未训练，不知道TD误差，给予最高优先级
        # 训练后根据实际TD误差更新
        self.max_priority = 1.0

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool, source: int = 0):
        """
        添加经验（使用最大优先级）

        优先级初始化策略:
        - 新经验: priority = max_priority
        - 原因: 尚未训练，不知道实际重要性
        - 保证新经验至少被采样一次

        Args:
            参数同ReplayBuffer.push（包括source）
        """
        # 调用父类push方法存储数据
        super().push(state, action, reward, next_state, done, source)

        # 获取刚刚写入的位置
        # position已经前进了，所以-1
        idx = (self.position - 1) % self.capacity

        # 设置优先级为当前最大值
        # 确保新经验会被采样到
        self.priorities[idx] = self.max_priority

    def sample(self, batch_size: int):
        """
        根据优先级采样

        采样算法:
        1. 计算每条经验的采样概率
           P(i) = priority(i)^α / Σ priority(j)^α
        2. 根据概率分布采样索引
        3. 计算重要性采样权重
           w(i) = (N * P(i))^(-β)
        4. 归一化权重（除以最大权重）

        Args:
            batch_size: 批次大小

        Returns:
            tuple: (states, actions, rewards, next_states, dones, indices, weights)
                - 前5个同ReplayBuffer
                - indices: 采样的索引（用于更新优先级）
                - weights: 重要性采样权重（用于加权损失）
        """
        # ========== 计算采样概率 ==========

        # 只使用有效数据的优先级
        # priorities[:size]: 前size个元素
        priorities = self.priorities[:self.size]

        # 计算优先级的α次方
        # alpha控制优先级的影响程度:
        # - alpha=0: probs均匀分布（所有经验概率相同）
        # - alpha=1: probs完全按优先级（高优先级高概率）
        # - alpha=0.6: 折中（推荐）
        probs = priorities ** self.alpha

        # 归一化为概率分布（和为1）
        # probs.sum(): 所有优先级的总和
        # probs / probs.sum(): 归一化到[0,1]，和为1
        probs /= probs.sum()

        # ========== 优先级采样 ==========

        # 根据概率分布采样索引
        # np.random.choice:
        # - self.size: 从[0, size)采样
        # - batch_size: 采样数量
        # - p=probs: 采样概率（每个索引的概率）
        # - replace=False: 不放回（每个经验最多采样一次）
        #
        # 高优先级的经验更容易被采样
        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)

        # ========== 计算重要性采样权重 ==========

        # 为什么需要重要性采样权重?
        # - 优先级采样改变了数据分布（有偏采样）
        # - 会导致梯度估计偏差
        # - 需要用权重补偿偏差

        # 计算公式: w(i) = (N * P(i))^(-β)
        # - N: 总样本数（self.size）
        # - P(i): 采样概率（probs[indices]）
        # - β: 重要性采样指数（self.beta）
        #
        # 直觉:
        # - 高概率采样的样本权重低（被过度采样）
        # - 低概率采样的样本权重高（被欠采样）
        weights = (self.size * probs[indices]) ** (-self.beta)

        # 归一化权重（除以最大权重）
        # 原因: 保证权重在[0,1]范围，避免梯度爆炸
        # weights.max(): 所有权重的最大值
        weights /= weights.max()

        # Beta退火: 逐渐增加到1.0
        # 原因: 训练初期优先级不准，后期需要更强的补偿
        self.beta = min(1.0, self.beta + self.beta_increment)

        # 返回数据 + 索引 + 权重
        # 索引: 用于训练后更新优先级
        # 权重: 用于加权损失函数
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            indices,    # ← 额外返回
            weights     # ← 额外返回
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """
        更新经验的优先级（训练后调用）

        更新时机: 每次训练后
        - 计算每个样本的TD误差
        - TD误差 = |Q(s,a) - target|
        - 用TD误差更新优先级

        Args:
            indices: 采样的索引
                - 来自sample()的返回值
                - shape: [batch_size]
            priorities: 新的优先级值
                - 通常是 |TD_error| + ε
                - shape: [batch_size]
                - ε: 小常数（防止优先级为0）

        示例:
            # 训练循环
            batch = buffer.sample(256)
            states, actions, ..., indices, weights = batch

            # 计算TD误差
            td_errors = compute_td_errors(states, actions, ...)

            # 更新优先级
            new_priorities = np.abs(td_errors) + 1e-6
            buffer.update_priorities(indices, new_priorities)
        """
        # 遍历每个索引和对应的新优先级
        for idx, priority in zip(indices, priorities):
            # 更新优先级数组
            self.priorities[idx] = priority

            # 更新最大优先级（用于新经验）
            # max: 保持max_priority为当前最大值
            self.max_priority = max(self.max_priority, priority)
