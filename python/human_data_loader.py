"""
human_data_loader.py - 加载人类对战经验数据
"""

import os
import struct
import numpy as np
from typing import Tuple, Optional


class HumanDataLoader:
    """加载人类玩家对战产生的经验数据"""

    def __init__(self, data_dir: str = 'human_data'):
        """
        初始化数据加载器

        Args:
            data_dir: 人类经验数据目录
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def load_file(self, filepath: str) -> Optional[Tuple[np.ndarray, ...]]:
        """
        加载单个数据文件

        Args:
            filepath: 数据文件路径

        Returns:
            (states, actions, rewards, next_states, dones) 或 None
        """
        try:
            with open(filepath, 'rb') as f:
                # 读取数量
                count_bytes = f.read(4)
                if len(count_bytes) < 4:
                    return None
                count = struct.unpack('i', count_bytes)[0]

                if count <= 0 or count > 100000:  # 安全检查
                    print(f"⚠ 无效的数据量: {count} (文件: {filepath})")
                    return None

                # 读取数据
                states = np.frombuffer(f.read(count * 43 * 4), dtype=np.float32).reshape(count, 43)
                actions = np.frombuffer(f.read(count * 4), dtype=np.int32)
                rewards = np.frombuffer(f.read(count * 4), dtype=np.float32)
                next_states = np.frombuffer(f.read(count * 43 * 4), dtype=np.float32).reshape(count, 43)
                dones = np.frombuffer(f.read(count * 4), dtype=np.int32).astype(np.float32)

                return states, actions, rewards, next_states, dones

        except Exception as e:
            print(f"⚠ 加载文件失败 {filepath}: {e}")
            return None

    def load_all(self) -> Optional[Tuple[np.ndarray, ...]]:
        """
        加载所有人类经验数据

        Returns:
            (states, actions, rewards, next_states, dones) 或 None
        """
        if not os.path.exists(self.data_dir):
            return None

        all_states = []
        all_actions = []
        all_rewards = []
        all_next_states = []
        all_dones = []

        # 获取所有数据文件
        data_files = [f for f in os.listdir(self.data_dir) if f.endswith('.dat')]

        if not data_files:
            return None

        print(f"\n正在加载人类经验数据...")
        loaded_count = 0

        for filename in sorted(data_files):
            filepath = os.path.join(self.data_dir, filename)
            data = self.load_file(filepath)

            if data is not None:
                states, actions, rewards, next_states, dones = data

                all_states.append(states)
                all_actions.append(actions)
                all_rewards.append(rewards)
                all_next_states.append(next_states)
                all_dones.append(dones)

                loaded_count += 1
                print(f"  ✓ {filename}: {len(states)} 条经验")

        if not all_states:
            return None

        # 合并所有数据
        states = np.vstack(all_states)
        actions = np.hstack(all_actions)
        rewards = np.hstack(all_rewards)
        next_states = np.vstack(all_next_states)
        dones = np.hstack(all_dones)

        print(f"✓ 总共加载了 {len(states)} 条人类经验数据 (来自 {loaded_count} 个文件)\n")
        return states, actions, rewards, next_states, dones

    def filter_quality(self, data: Tuple[np.ndarray, ...],
                      min_reward: float = -50.0) -> Tuple[np.ndarray, ...]:
        """
        过滤低质量的经验数据

        Args:
            data: (states, actions, rewards, next_states, dones)
            min_reward: 最小奖励阈值

        Returns:
            过滤后的数据
        """
        states, actions, rewards, next_states, dones = data

        # 过滤掉奖励过低的经验
        mask = rewards > min_reward

        filtered_count = np.sum(~mask)
        if filtered_count > 0:
            print(f"  过滤了 {filtered_count} 条低质量经验 (奖励 < {min_reward})")

        return (states[mask], actions[mask], rewards[mask],
                next_states[mask], dones[mask])

    def preload_to_buffer(self, replay_buffer, filter_quality: bool = True):
        """
        将人类数据预加载到经验回放缓冲区

        Args:
            replay_buffer: ReplayBuffer 实例
            filter_quality: 是否过滤低质量数据

        Returns:
            加载的数据数量
        """
        data = self.load_all()
        if data is None:
            print("未找到人类经验数据,跳过预加载")
            return 0

        # 过滤质量
        if filter_quality:
            data = self.filter_quality(data)

        states, actions, rewards, next_states, dones = data

        if len(states) == 0:
            print("过滤后没有可用的人类经验数据")
            return 0

        # 逐条添加到缓冲区
        print("正在将人类经验添加到回放缓冲区...")
        for i in range(len(states)):
            replay_buffer.push(states[i], actions[i], rewards[i],
                             next_states[i], dones[i])

        print(f"✓ 已将 {len(states)} 条人类经验添加到缓冲区")
        print(f"  缓冲区当前大小: {len(replay_buffer)}\n")
        return len(states)

    def get_stats(self) -> dict:
        """
        获取人类数据统计信息

        Returns:
            统计信息字典
        """
        data = self.load_all()
        if data is None:
            return {'total': 0, 'files': 0}

        states, actions, rewards, next_states, dones = data

        return {
            'total': len(states),
            'files': len([f for f in os.listdir(self.data_dir) if f.endswith('.dat')]),
            'avg_reward': np.mean(rewards),
            'max_reward': np.max(rewards),
            'min_reward': np.min(rewards),
            'win_rate': np.sum(rewards > 0) / len(rewards) * 100 if len(rewards) > 0 else 0
        }
