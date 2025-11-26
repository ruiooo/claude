"""
enemy_manager.py - 敌人管理系统（自我对弈、历史版本）
"""

import os
import numpy as np
from typing import List, Dict
import shutil


class EnemyManager:
    """
    管理敌人生成和历史版本
    实现自我对弈和动态难度调整
    """

    def __init__(self, config: dict):
        """
        初始化敌人管理器

        Args:
            config: 配置字典
        """
        self.self_play_config = config['SELF_PLAY_CONFIG']
        self.difficulty_config = config['DIFFICULTY_CONFIG']
        self.paths = config['PATHS']

        # 历史模型版本
        self.history_versions = []

        # 统计数据
        self.consecutive_wins = 0
        self.current_enemy_count = config['ENV_CONFIG']['initial_enemies']

        # 创建历史模型目录
        os.makedirs(self.paths['history'], exist_ok=True)

    def should_save_history_version(self, episode: int) -> bool:
        """
        判断是否应该保存历史版本

        Args:
            episode: 当前回合数

        Returns:
            是否保存
        """
        if not self.self_play_config['enabled']:
            return False

        return episode % self.self_play_config['history_interval'] == 0

    def save_history_version(self, agent, episode: int):
        """
        保存历史版本模型

        Args:
            agent: DQN智能体
            episode: 当前回合数
        """
        version = len(self.history_versions) + 1

        # 保存模型
        model_path = os.path.join(self.paths['history'], f'model_v{version}_ep{episode}.pth')
        agent.save(model_path)

        # 记录版本信息
        self.history_versions.append({
            'version': version,
            'episode': episode,
            'path': model_path,
            'epsilon': agent.epsilon,
            'train_step': agent.train_step
        })

        print(f"保存历史版本 v{version} (回合 {episode})")

        # 限制历史版本数量
        max_versions = self.self_play_config['max_history_versions']
        if len(self.history_versions) > max_versions:
            # 删除最旧的版本
            oldest = self.history_versions.pop(0)
            if os.path.exists(oldest['path']):
                os.remove(oldest['path'])
            print(f"删除旧版本 v{oldest['version']}")

    def get_random_history_version(self) -> Dict:
        """
        获取随机历史版本

        Returns:
            历史版本信息字典
        """
        if not self.history_versions:
            return None

        return np.random.choice(self.history_versions)

    def update_difficulty(self, win: bool) -> int:
        """
        根据胜负更新难度

        Args:
            win: 是否获胜

        Returns:
            新的敌人数量
        """
        if not self.difficulty_config['enabled']:
            return self.current_enemy_count

        if win:
            self.consecutive_wins += 1

            # 连续胜利则增加敌人
            if self.consecutive_wins >= self.difficulty_config['win_threshold']:
                max_enemies = self.difficulty_config['max_enemies']
                if self.current_enemy_count < max_enemies:
                    self.current_enemy_count += 1
                    print(f"难度提升! 敌人数量: {self.current_enemy_count}")
                self.consecutive_wins = 0
        else:
            # 失败则重置连胜计数
            self.consecutive_wins = 0

        return self.current_enemy_count

    def generate_enemies(self, env) -> List[Dict]:
        """
        生成敌人配置

        Args:
            env: 游戏环境

        Returns:
            敌人配置列表
        """
        enemies = []
        probs = self.difficulty_config['enemy_type_probs']

        for i in range(self.current_enemy_count):
            # 决定敌人类型
            if self.history_versions and np.random.random() < probs['self_play']:
                # 使用历史版本
                history = self.get_random_history_version()
                enemy = {
                    'type': 2,  # TANK_TYPE_SELF_PLAY
                    'model_version': history['version']
                }
                enemies.append(enemy)
            else:
                # 使用追踪型敌人
                enemy = {
                    'type': 1,  # TANK_TYPE_ENEMY
                    'model_version': 0
                }
                enemies.append(enemy)

        return enemies

    def get_stats(self) -> Dict:
        """
        获取统计信息

        Returns:
            统计字典
        """
        return {
            'consecutive_wins': self.consecutive_wins,
            'current_enemy_count': self.current_enemy_count,
            'history_versions': len(self.history_versions),
            'latest_version': self.history_versions[-1]['version'] if self.history_versions else 0
        }
