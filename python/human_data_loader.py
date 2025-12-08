"""
human_data_loader.py - 加载人类对战经验数据（模仿学习）
==============================================================

【核心功能】
从人类玩家的游戏记录中加载经验数据，用于模仿学习(Imitation Learning)。

【模仿学习原理】
传统强化学习: AI通过试错学习（探索成本高）
模仿学习: AI从专家演示中学习（快速获得基础策略）

学习流程:
1. 人类玩家对战AI，系统记录操作
2. 数据保存为二进制文件(.dat)
3. 训练时加载人类经验
4. 混入回放缓冲区
5. AI从人类专家经验中学习

【优势】
- 提供高质量初始策略
- 加快训练收敛速度
- 避免探索危险区域
- 学习人类高级技巧

【数据格式】
二进制文件结构:
    int32: count (经验数量)
    float32[count][43]: states (状态数组)
    int32[count]: actions (动作数组)
    float32[count]: rewards (奖励数组)
    float32[count][43]: next_states (下一状态数组)
    int32[count]: dones (完成标志数组)

【经典应用】
- AlphaGo: 从人类棋谱预训练
- 自动驾驶: 从人类驾驶数据学习
- 机器人: 从人类演示学习操作
"""

import os  # 文件系统操作
import struct  # 二进制数据解包
import numpy as np  # 数值数组操作
from typing import Tuple, Optional  # 类型注解


class HumanDataLoader:
    """
    人类数据加载器

    功能:
    1. 扫描人类经验数据目录
    2. 加载二进制数据文件
    3. 解析为numpy数组
    4. 质量过滤（去除低质量数据）
    5. 预加载到回放缓冲区

    使用示例:
        # 初始化加载器
        loader = HumanDataLoader('human_data')

        # 加载所有数据
        data = loader.load_all()
        if data:
            states, actions, rewards, next_states, dones = data
            print(f"加载了 {len(states)} 条人类经验")

        # 预加载到缓冲区
        loaded_count = loader.preload_to_buffer(replay_buffer)
    """

    def __init__(self, data_dir: str = 'human_data'):
        """
        初始化数据加载器

        Args:
            data_dir: 人类经验数据目录
                - 默认: 'human_data'
                - 存储玩家对战时的经验数据
                - 文件命名: experience_<timestamp>.dat
        """
        # 保存数据目录路径
        self.data_dir = data_dir

        # 创建目录（如果不存在）
        # exist_ok=True: 目录存在时不报错
        os.makedirs(data_dir, exist_ok=True)

    def load_file(self, filepath: str) -> Optional[Tuple[np.ndarray, ...]]:
        """
        加载单个数据文件

        文件格式（二进制）:
        1. int32: count (经验数量)
        2. float32[count][43]: states
        3. int32[count]: actions
        4. float32[count]: rewards
        5. float32[count][43]: next_states
        6. int32[count]: dones

        读取流程:
        1. 打开二进制文件
        2. 读取count（经验数量）
        3. 按顺序读取各数组
        4. 转换为numpy数组
        5. 返回元组

        Args:
            filepath: 数据文件路径
                - 例: 'human_data/experience_20231201_120530.dat'
                - 必须是完整路径

        Returns:
            tuple: (states, actions, rewards, next_states, dones)
                或 None (如果加载失败)

        示例:
            data = loader.load_file('human_data/experience_001.dat')
            if data:
                states, actions, rewards, next_states, dones = data
        """
        try:
            # ========== 打开二进制文件 ==========

            # 以只读二进制模式打开
            # 'rb': read binary (读取二进制)
            with open(filepath, 'rb') as f:

                # ========== 读取经验数量 ==========

                # 读取前4个字节（int32大小）
                # f.read(4): 读取4字节
                count_bytes = f.read(4)

                # 检查是否读取成功
                # 文件太小或损坏时可能读不到4字节
                if len(count_bytes) < 4:
                    return None

                # 解包为整数
                # struct.unpack: 将字节序列解析为Python类型
                # 'i': int32类型
                # [0]: 取第一个元素（unpack返回元组）
                count = struct.unpack('i', count_bytes)[0]

                # ========== 安全检查 ==========

                # 检查数量是否合理
                # count <= 0: 无效数据
                # count > 100000: 可能是文件损坏（异常大）
                if count <= 0 or count > 100000:
                    print(f"⚠ 无效的数据量: {count} (文件: {filepath})")
                    return None

                # ========== 读取数据数组 ==========

                # 读取states数组
                # count * 43 * 4: 总字节数
                # - count: 经验数量
                # - 43: 状态维度
                # - 4: float32字节数
                # np.frombuffer: 从字节缓冲区创建数组
                # dtype=float32: 指定数据类型
                # reshape: 重塑为二维数组 [count, 43]
                states = np.frombuffer(f.read(count * 43 * 4), dtype=np.float32).reshape(count, 43)

                # 读取actions数组
                # count * 4: int32数组字节数
                # dtype=int32: 整数类型
                # shape: [count]
                actions = np.frombuffer(f.read(count * 4), dtype=np.int32)

                # 读取rewards数组
                # count * 4: float32数组字节数
                # dtype=float32: 浮点类型
                # shape: [count]
                rewards = np.frombuffer(f.read(count * 4), dtype=np.float32)

                # 读取next_states数组
                # 与states相同的格式
                # shape: [count, 43]
                next_states = np.frombuffer(f.read(count * 43 * 4), dtype=np.float32).reshape(count, 43)

                # 读取dones数组
                # count * 4: int32数组字节数
                # astype(float32): 转换为float（方便GPU计算）
                # shape: [count]
                dones = np.frombuffer(f.read(count * 4), dtype=np.int32).astype(np.float32)

                # 返回所有数据（元组形式）
                return states, actions, rewards, next_states, dones

        except Exception as e:
            # 捕获所有异常（文件不存在、权限问题、格式错误等）
            # 打印警告但不中断程序
            print(f"⚠ 加载文件失败 {filepath}: {e}")
            return None

    def load_all(self) -> Optional[Tuple[np.ndarray, ...]]:
        """
        加载所有人类经验数据

        加载流程:
        1. 扫描数据目录
        2. 找到所有.dat文件
        3. 逐个加载文件
        4. 合并所有数据
        5. 返回合并结果

        合并策略:
        - states: vstack (垂直堆叠)
        - actions: hstack (水平拼接)
        - rewards: hstack
        - next_states: vstack
        - dones: hstack

        Returns:
            tuple: (states, actions, rewards, next_states, dones)
                - states: [total_count, 43]
                - actions: [total_count]
                - rewards: [total_count]
                - next_states: [total_count, 43]
                - dones: [total_count]
            或 None (如果没有数据)

        示例:
            data = loader.load_all()
            if data:
                states, actions, rewards, next_states, dones = data
                print(f"总共加载 {len(states)} 条经验")
            else:
                print("没有找到人类经验数据")
        """
        # 检查数据目录是否存在
        if not os.path.exists(self.data_dir):
            return None

        # ========== 初始化数据列表 ==========

        # 用于收集所有文件的数据
        # 每个列表存储一个文件的对应数组
        all_states = []
        all_actions = []
        all_rewards = []
        all_next_states = []
        all_dones = []

        # ========== 扫描数据文件 ==========

        # 获取所有.dat文件
        # os.listdir: 列出目录中的所有文件
        # f.endswith('.dat'): 过滤出.dat文件
        data_files = [f for f in os.listdir(self.data_dir) if f.endswith('.dat')]

        # 检查是否有数据文件
        if not data_files:
            return None

        # 打印加载开始信息
        print(f"\n正在加载人类经验数据...")

        # 加载成功的文件计数
        loaded_count = 0

        # ========== 逐个加载文件 ==========

        # sorted: 按文件名排序（保证加载顺序一致）
        for filename in sorted(data_files):
            # 构建完整文件路径
            # os.path.join: 跨平台路径拼接
            filepath = os.path.join(self.data_dir, filename)

            # 加载单个文件
            data = self.load_file(filepath)

            # 检查是否加载成功
            if data is not None:
                # 解包数据
                states, actions, rewards, next_states, dones = data

                # 添加到对应列表
                all_states.append(states)
                all_actions.append(actions)
                all_rewards.append(rewards)
                all_next_states.append(next_states)
                all_dones.append(dones)

                # 成功计数+1
                loaded_count += 1

                # 打印文件加载信息
                # len(states): 该文件的经验数量
                print(f"  ✓ {filename}: {len(states)} 条经验")

        # ========== 检查是否有有效数据 ==========

        if not all_states:
            return None

        # ========== 合并所有数据 ==========

        # vstack: 垂直堆叠（按行拼接）
        # 例: [10,43] + [20,43] → [30,43]
        states = np.vstack(all_states)

        # hstack: 水平拼接（一维数组拼接）
        # 例: [10] + [20] → [30]
        actions = np.hstack(all_actions)
        rewards = np.hstack(all_rewards)

        # vstack: 垂直堆叠
        next_states = np.vstack(all_next_states)

        # hstack: 水平拼接
        dones = np.hstack(all_dones)

        # 打印总结信息
        print(f"✓ 总共加载了 {len(states)} 条人类经验数据 (来自 {loaded_count} 个文件)\n")

        # 返回合并后的数据
        return states, actions, rewards, next_states, dones

    def filter_quality(self, data: Tuple[np.ndarray, ...],
                      min_reward: float = -50.0) -> Tuple[np.ndarray, ...]:
        """
        过滤低质量的经验数据

        质量评估:
        - 高质量: reward > min_reward (保留)
        - 低质量: reward <= min_reward (丢弃)

        为什么要过滤?
        1. 人类也会犯错（走位失误、被击中）
        2. 低质量经验会误导AI
        3. 只学习成功策略更高效

        过滤策略:
        - 基于奖励阈值
        - 简单高效
        - 可调整阈值

        Args:
            data: 原始数据元组
                - (states, actions, rewards, next_states, dones)
            min_reward: 最小奖励阈值
                - 典型值: -50.0, -30.0
                - 受伤惩罚: -15
                - 死亡惩罚: -100
                - -50: 允许受伤，拒绝死亡

        Returns:
            tuple: 过滤后的数据
                - 格式同输入
                - 大小可能减小

        示例:
            # 原始数据
            data = loader.load_all()

            # 过滤低质量经验
            filtered = loader.filter_quality(data, min_reward=-30.0)

            # 统计过滤结果
            original_count = len(data[0])
            filtered_count = len(filtered[0])
            removed = original_count - filtered_count
            print(f"过滤了 {removed} 条低质量经验")
        """
        # 解包数据
        states, actions, rewards, next_states, dones = data

        # ========== 创建过滤掩码 ==========

        # 布尔数组: True=保留, False=丢弃
        # rewards > min_reward: 逐元素比较
        # 例: rewards=[-100, 0.02, -20, 10]
        #     min_reward=-50
        #     mask=[False, True, True, True]
        mask = rewards > min_reward

        # ========== 统计过滤结果 ==========

        # 计算被过滤的数量
        # ~mask: 逻辑非（取反）
        # np.sum: 求和（True=1, False=0）
        filtered_count = np.sum(~mask)

        # 打印过滤信息（如果有数据被过滤）
        if filtered_count > 0:
            print(f"  过滤了 {filtered_count} 条低质量经验 (奖励 < {min_reward})")

        # ========== 应用掩码过滤 ==========

        # 使用布尔索引过滤数组
        # array[mask]: 只保留mask为True的元素
        return (states[mask], actions[mask], rewards[mask],
                next_states[mask], dones[mask])

    def preload_to_buffer(self, replay_buffer, filter_quality: bool = True):
        """
        将人类数据预加载到经验回放缓冲区

        预加载流程:
        1. 加载所有人类数据
        2. 可选：质量过滤
        3. 逐条添加到缓冲区
        4. 返回加载数量

        预加载优势:
        - 提供初始高质量经验
        - 加快训练收敛
        - 避免完全随机探索

        混合策略:
        - 人类经验: 提供基础策略
        - AI经验: 探索改进策略
        - 两者混合训练

        Args:
            replay_buffer: ReplayBuffer实例
                - 必须有push()方法
                - 必须支持__len__()
            filter_quality: 是否过滤低质量数据
                - True: 只加载高质量经验（推荐）
                - False: 加载所有经验

        Returns:
            int: 加载的数据数量
                - 0: 没有数据或加载失败
                - >0: 成功加载的经验数量

        示例:
            # 创建缓冲区
            buffer = ReplayBuffer(capacity=100000, state_dim=43)

            # 预加载人类数据
            loader = HumanDataLoader('human_data')
            count = loader.preload_to_buffer(buffer, filter_quality=True)

            print(f"预加载了 {count} 条人类经验")
            print(f"缓冲区大小: {len(buffer)}")
        """
        # ========== 加载数据 ==========

        # 加载所有人类经验文件
        data = self.load_all()

        # 检查是否成功加载
        if data is None:
            print("未找到人类经验数据,跳过预加载")
            return 0

        # ========== 质量过滤 ==========

        # 如果启用质量过滤
        if filter_quality:
            # 调用filter_quality方法过滤低质量数据
            # 使用默认阈值（在filter_quality中定义）
            data = self.filter_quality(data)

        # 解包数据
        states, actions, rewards, next_states, dones = data

        # ========== 检查过滤后是否有数据 ==========

        if len(states) == 0:
            print("过滤后没有可用的人类经验数据")
            return 0

        # ========== 添加到缓冲区 ==========

        # 打印开始信息
        print("正在将人类经验添加到回放缓冲区...")

        # 逐条添加经验到缓冲区
        # range(len(states)): 0, 1, 2, ..., len-1
        for i in range(len(states)):
            # 调用缓冲区的push方法
            # 将第i条经验添加到缓冲区
            replay_buffer.push(states[i], actions[i], rewards[i],
                             next_states[i], dones[i])

        # ========== 打印完成信息 ==========

        print(f"✓ 已将 {len(states)} 条人类经验添加到缓冲区")
        print(f"  缓冲区当前大小: {len(replay_buffer)}\n")

        # 返回加载的数量
        return len(states)

    def get_stats(self) -> dict:
        """
        获取人类数据统计信息

        统计内容:
        - 总经验数
        - 文件数
        - 平均奖励
        - 最大/最小奖励
        - 胜率（奖励>0的比例）

        用途:
        - 评估数据质量
        - 决定是否使用
        - 调整过滤阈值

        Returns:
            dict: 统计信息字典
                - total: 总经验数
                - files: 文件数
                - avg_reward: 平均奖励
                - max_reward: 最大奖励
                - min_reward: 最小奖励
                - win_rate: 胜率（%）

        示例:
            stats = loader.get_stats()
            print(f"总经验数: {stats['total']}")
            print(f"平均奖励: {stats['avg_reward']:.2f}")
            print(f"胜率: {stats['win_rate']:.1f}%")
        """
        # 加载所有数据
        data = self.load_all()

        # 如果没有数据，返回空统计
        if data is None:
            return {'total': 0, 'files': 0}

        # 解包数据
        states, actions, rewards, next_states, dones = data

        # 计算统计信息
        return {
            # 总经验数
            'total': len(states),

            # 文件数
            # 列出所有.dat文件并计数
            'files': len([f for f in os.listdir(self.data_dir) if f.endswith('.dat')]),

            # 平均奖励
            # np.mean: 计算均值
            'avg_reward': np.mean(rewards),

            # 最大奖励
            # np.max: 找最大值
            'max_reward': np.max(rewards),

            # 最小奖励
            # np.min: 找最小值
            'min_reward': np.min(rewards),

            # 胜率（奖励>0的比例）
            # np.sum(rewards > 0): 奖励为正的数量
            # / len(rewards): 除以总数得到比例
            # * 100: 转换为百分比
            # if len(rewards) > 0: 防止除以0
            'win_rate': np.sum(rewards > 0) / len(rewards) * 100 if len(rewards) > 0 else 0
        }
