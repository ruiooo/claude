"""
enemy_manager.py - 敌人管理系统（自我对弈、历史版本）
==========================================================

【核心功能】
管理敌人生成策略，实现动态难度调整和自我对弈机制。

【自我对弈(Self-Play)】
概念: 让AI与自己的历史版本对战
- 保存训练过程中的模型快照
- 将历史版本作为对手
- 避免过拟合到固定敌人AI

优点:
1. 策略多样化: 不同版本有不同策略
2. 持续挑战: 随着训练，对手也在进步
3. 避免遗忘: 保持对历史策略的应对能力

经典应用:
- AlphaGo: 完全自我对弈训练
- OpenAI Five (Dota2): 混合自我对弈
- AlphaStar (星际争霸): 联盟对抗训练

【动态难度调整】
根据AI表现自动调整挑战难度:
- 连续胜利 → 增加敌人数量
- 连续失败 → 保持难度
- 课程学习: 从简单到困难逐步训练

【管理策略】
历史版本:
- 定期保存 (每N回合)
- 限制数量 (最多M个)
- FIFO删除 (先进先出)

敌人生成:
- 混合策略 (传统AI + 历史版本)
- 概率控制 (10% 自我对弈, 90% 传统AI)
"""

import os  # 文件系统操作
import numpy as np  # 数值计算
from typing import List, Dict  # 类型注解
import shutil  # 文件操作（删除等）


class EnemyManager:
    """
    敌人管理器

    职责:
    1. 管理历史版本模型
    2. 实现自我对弈机制
    3. 动态调整难度
    4. 生成敌人配置

    工作流程:
        # 初始化
        manager = EnemyManager(config)

        # 训练循环
        for episode in range(max_episodes):
            # 保存历史版本
            if manager.should_save_history_version(episode):
                manager.save_history_version(agent, episode)

            # 生成敌人
            enemies = manager.generate_enemies(env)
            for enemy in enemies:
                env.add_enemy(enemy['type'], enemy['model_version'])

            # 更新难度
            manager.update_difficulty(win)
    """

    def __init__(self, config: dict):
        """
        初始化敌人管理器

        Args:
            config: 配置字典，包含:
                - SELF_PLAY_CONFIG: 自我对弈配置
                - DIFFICULTY_CONFIG: 动态难度配置
                - ENV_CONFIG: 环境配置（初始敌人数）
                - PATHS: 路径配置（历史模型保存路径）
        """
        # 保存配置
        self.self_play_config = config['SELF_PLAY_CONFIG']
        self.difficulty_config = config['DIFFICULTY_CONFIG']
        self.paths = config['PATHS']

        # ========== 历史版本管理 ==========

        # 历史模型版本列表
        # 每个元素是一个字典:
        # {
        #     'version': 版本号（1, 2, 3...）
        #     'episode': 保存时的回合数
        #     'path': 模型文件路径
        #     'epsilon': 保存时的epsilon值
        #     'train_step': 训练步数
        # }
        self.history_versions = []

        # ========== 难度管理 ==========

        # 连续胜利计数器
        # 用于判断是否增加难度
        # 每次胜利+1，失败重置为0
        self.consecutive_wins = 0

        # 当前敌人数量
        # 从配置文件读取初始值（通常是1或2）
        # 随训练动态调整
        self.current_enemy_count = config['ENV_CONFIG']['initial_enemies']

        # ========== 创建目录 ==========

        # 创建历史模型保存目录
        # exist_ok=True: 目录已存在时不报错
        os.makedirs(self.paths['history'], exist_ok=True)

    def should_save_history_version(self, episode: int) -> bool:
        """
        判断是否应该保存历史版本

        保存策略:
        - 定期保存（每N回合）
        - 仅在启用自我对弈时保存

        为什么定期保存?
        - 太频繁: 版本差异小，浪费空间
        - 太稀疏: 版本差异大，缺少中间难度
        - 适中间隔: 平衡多样性和效率

        Args:
            episode: 当前回合数
                - 从0开始递增
                - 例: 0, 1, 2, ..., 10000

        Returns:
            bool: True=应该保存, False=不保存

        示例:
            # history_interval = 200
            # episode=0:    False (0 % 200 != 0)
            # episode=100:  False
            # episode=200:  True (200 % 200 == 0)
            # episode=400:  True
        """
        # 检查自我对弈是否启用
        if not self.self_play_config['enabled']:
            return False

        # 检查是否到达保存间隔
        # episode % interval == 0: 整除判断
        # 例: interval=200, episode=200时返回True
        return episode % self.self_play_config['history_interval'] == 0

    def save_history_version(self, agent, episode: int):
        """
        保存历史版本模型

        保存流程:
        1. 生成版本号（递增）
        2. 保存模型到文件
        3. 记录版本信息
        4. 限制版本数量（删除旧版本）

        版本号管理:
        - 从1开始递增
        - version = len(history_versions) + 1

        文件命名:
        - 格式: model_v{version}_ep{episode}.pth
        - 例: model_v1_ep200.pth, model_v2_ep400.pth

        Args:
            agent: DQN智能体
                - 包含策略网络、目标网络等
                - 调用agent.save()保存模型
            episode: 当前回合数
                - 用于文件命名
                - 记录保存时的训练进度
        """
        # 生成版本号（从1开始）
        # len(history_versions) = 0, 1, 2, ...
        # version = 1, 2, 3, ...
        version = len(self.history_versions) + 1

        # 构建模型保存路径
        # os.path.join: 跨平台路径拼接
        # 例: "saved_models/history/model_v1_ep200.pth"
        model_path = os.path.join(self.paths['history'], f'model_v{version}_ep{episode}.pth')

        # 保存模型到文件
        # agent.save(): 调用DQNAgent的保存方法
        # 保存内容: policy_net, target_net, optimizer, epsilon, train_step
        agent.save(model_path)

        # 记录版本信息到列表
        # 包含版本号、回合数、路径、训练状态等
        self.history_versions.append({
            'version': version,          # 版本号（1, 2, 3...）
            'episode': episode,          # 保存时的回合数
            'path': model_path,          # 模型文件路径
            'epsilon': agent.epsilon,    # 探索率（反映训练阶段）
            'train_step': agent.train_step  # 训练步数（反映训练量）
        })

        # 打印保存信息
        print(f"保存历史版本 v{version} (回合 {episode})")

        # ========== 限制版本数量 ==========

        # 获取最大版本数配置
        # 典型值: 10（保留最近10个版本）
        max_versions = self.self_play_config['max_history_versions']

        # 检查是否超过限制
        if len(self.history_versions) > max_versions:
            # 删除最旧的版本（FIFO策略）
            # pop(0): 删除列表第一个元素（最旧的）
            oldest = self.history_versions.pop(0)

            # 删除磁盘上的模型文件
            # os.path.exists: 检查文件是否存在
            # os.remove: 删除文件
            if os.path.exists(oldest['path']):
                os.remove(oldest['path'])

            # 打印删除信息
            print(f"删除旧版本 v{oldest['version']}")

    def get_random_history_version(self) -> Dict:
        """
        获取随机历史版本

        随机选择策略:
        - 从所有历史版本中均匀随机选择
        - 每个版本被选中的概率相同

        为什么随机?
        - 提供多样化的对手
        - 避免过拟合到特定版本
        - 保持对各阶段策略的适应性

        Returns:
            dict: 历史版本信息
                - version: 版本号
                - episode: 保存时的回合数
                - path: 模型文件路径
                - epsilon: 探索率
                - train_step: 训练步数
            或 None (如果没有历史版本)

        示例:
            version = manager.get_random_history_version()
            if version:
                print(f"使用历史版本 v{version['version']}")
                env.add_enemy(type=2, model_version=version['version'])
        """
        # 检查是否有历史版本
        if not self.history_versions:
            return None

        # 随机选择一个版本
        # np.random.choice: 从数组中随机选择一个元素
        # 返回值: 字典（版本信息）
        return np.random.choice(self.history_versions)

    def update_difficulty(self, win: bool) -> int:
        """
        根据胜负更新难度

        难度调整策略:
        - 胜利: 连胜计数+1
        - 失败: 连胜计数重置为0
        - 达到阈值: 敌人数+1，重置计数

        为什么这样设计?
        - 避免难度波动: 需要连续胜利才增加
        - 适应性强: AI变弱时难度不降
        - 课程学习: 逐步增加挑战

        难度上限:
        - max_enemies: 防止过于困难
        - 状态表示限制: 最多支持5个敌人

        Args:
            win: 是否获胜
                - True: AI获胜
                - False: AI失败

        Returns:
            int: 新的敌人数量
                - 可能与之前相同（未达阈值）
                - 可能+1（达到阈值且未到上限）

        示例:
            # 训练循环
            for episode in range(max_episodes):
                state = env.reset(enemy_count)
                # ... 游戏进行 ...
                if done:
                    win = (info['winner'] == 0)
                    enemy_count = manager.update_difficulty(win)
        """
        # 检查动态难度是否启用
        if not self.difficulty_config['enabled']:
            # 未启用: 返回当前敌人数（不变）
            return self.current_enemy_count

        # 处理胜利情况
        if win:
            # 连胜计数+1
            self.consecutive_wins += 1

            # 检查是否达到难度提升阈值
            # 例: win_threshold=20, 连续20次胜利后提升
            if self.consecutive_wins >= self.difficulty_config['win_threshold']:
                # 获取最大敌人数限制
                max_enemies = self.difficulty_config['max_enemies']

                # 检查是否已达上限
                if self.current_enemy_count < max_enemies:
                    # 敌人数+1
                    self.current_enemy_count += 1

                    # 打印难度提升信息
                    print(f"难度提升! 敌人数量: {self.current_enemy_count}")

                # 重置连胜计数
                # 下次提升需要再次连续胜利
                self.consecutive_wins = 0
        else:
            # 失败: 重置连胜计数
            # 下次提升需要重新开始连胜
            self.consecutive_wins = 0

        # 返回新的敌人数量
        return self.current_enemy_count

    def generate_enemies(self, env) -> List[Dict]:
        """
        生成敌人配置

        生成策略:
        - 根据当前敌人数量生成
        - 混合传统AI和历史版本
        - 随机概率决定类型

        敌人类型:
        1. TANK_TYPE_ENEMY (1): 传统追踪AI
           - 简单追击逻辑
           - 难度固定
           - 作为基准对手
        2. TANK_TYPE_SELF_PLAY (2): 历史版本AI
           - 使用训练过的模型
           - 难度动态变化
           - 策略多样化

        混合比例:
        - 典型配置: 90% 传统AI, 10% 历史版本
        - 原因: 传统AI稳定，历史版本多样

        Args:
            env: 游戏环境
                - 实际未使用，保留用于扩展
                - 可能用于根据环境状态调整策略

        Returns:
            list: 敌人配置列表
                每个元素是字典:
                {
                    'type': 敌人类型（1或2）
                    'model_version': 模型版本号（仅type=2时有效）
                }

        示例:
            enemies = manager.generate_enemies(env)
            # enemies = [
            #     {'type': 1, 'model_version': 0},  # 传统AI
            #     {'type': 2, 'model_version': 3},  # 历史版本v3
            # ]

            for enemy in enemies:
                env.add_enemy(enemy['type'], enemy['model_version'])
        """
        # 敌人配置列表
        enemies = []

        # 获取敌人类型概率配置
        # 例: {'tracking': 0.9, 'self_play': 0.1}
        probs = self.difficulty_config['enemy_type_probs']

        # 生成指定数量的敌人
        # range(current_enemy_count): 0, 1, 2, ..., count-1
        for i in range(self.current_enemy_count):
            # 决定敌人类型（随机选择）
            # 检查两个条件:
            # 1. 是否有历史版本可用
            # 2. 随机数是否小于自我对弈概率
            if self.history_versions and np.random.random() < probs['self_play']:
                # ========== 使用历史版本AI ==========

                # 随机选择一个历史版本
                history = self.get_random_history_version()

                # 创建敌人配置
                enemy = {
                    'type': 2,  # TANK_TYPE_SELF_PLAY
                    'model_version': history['version']  # 版本号（1, 2, 3...）
                }

                # 添加到列表
                enemies.append(enemy)

            else:
                # ========== 使用传统追踪AI ==========

                # 创建敌人配置
                enemy = {
                    'type': 1,  # TANK_TYPE_ENEMY
                    'model_version': 0  # 不使用模型（0表示传统AI）
                }

                # 添加到列表
                enemies.append(enemy)

        # 返回敌人配置列表
        return enemies

    def get_stats(self) -> Dict:
        """
        获取统计信息

        统计内容:
        - 连续胜利次数
        - 当前敌人数量
        - 历史版本数量
        - 最新版本号

        用途:
        - 监控训练进度
        - 调试难度系统
        - 显示训练统计

        Returns:
            dict: 统计信息字典
                - consecutive_wins: 连续胜利次数（0-N）
                - current_enemy_count: 当前敌人数量（1-max）
                - history_versions: 历史版本总数（0-max）
                - latest_version: 最新版本号（0或1-N）

        示例:
            stats = manager.get_stats()
            print(f"连胜: {stats['consecutive_wins']}")
            print(f"敌人数: {stats['current_enemy_count']}")
            print(f"历史版本: {stats['history_versions']}")
            print(f"最新版本: v{stats['latest_version']}")
        """
        # 构建统计字典
        return {
            # 连续胜利次数
            # 用于判断距离难度提升还有多远
            'consecutive_wins': self.consecutive_wins,

            # 当前敌人数量
            # 反映当前难度级别
            'current_enemy_count': self.current_enemy_count,

            # 历史版本总数
            # len(list): 列表长度
            'history_versions': len(self.history_versions),

            # 最新版本号
            # 三元表达式: 有版本时返回最后一个版本号，否则返回0
            # self.history_versions[-1]: 列表最后一个元素
            'latest_version': self.history_versions[-1]['version'] if self.history_versions else 0
        }
