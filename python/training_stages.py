"""
training_stages.py - AlphaGo风格的分阶段训练管理器
====================================================

灵感来源：AlphaGo的训练流程
1. 监督学习阶段 (Supervised Learning): 从人类棋谱学习
2. 强化学习阶段 (Reinforcement Learning): 自我对弈优化
3. 混合阶段: 逐渐过渡

训练阶段设计：
阶段1: 行为克隆 (Behavior Cloning)
  - 目标: 快速学习基础策略
  - 数据: 100% 人类数据
  - 探索: 极少 (epsilon=0.1)
  - 时长: 0-500回合

阶段2: 混合训练 (Hybrid Training)
  - 目标: 平衡模仿与探索
  - 数据: 50% 人类 + 50% AI
  - 探索: 渐进增加 (epsilon: 0.3→0.7)
  - 时长: 500-2000回合

阶段3: 强化学习主导 (RL-Dominant)
  - 目标: 自主优化策略
  - 数据: 10% 人类 + 90% AI
  - 探索: 正常 (epsilon: 0.7→0.99+)
  - 时长: 2000+回合
"""

from typing import Dict, Tuple


class TrainingStage:
    """训练阶段定义"""

    def __init__(self, name: str, start_episode: int, end_episode: int,
                 human_data_weight: float, epsilon_start: float,
                 epsilon_end: float, description: str):
        """
        初始化训练阶段

        Args:
            name: 阶段名称
            start_episode: 起始回合数
            end_episode: 结束回合数 (None表示无限)
            human_data_weight: 人类数据采样权重
            epsilon_start: 阶段开始时的epsilon
            epsilon_end: 阶段结束时的epsilon
            description: 阶段描述
        """
        self.name = name
        self.start_episode = start_episode
        self.end_episode = end_episode
        self.human_data_weight = human_data_weight
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.description = description

    def get_epsilon(self, current_episode: int) -> float:
        """
        计算当前回合的epsilon值（线性插值）

        Args:
            current_episode: 当前回合数

        Returns:
            epsilon值
        """
        if self.end_episode is None:
            # 无限阶段，返回结束值
            return self.epsilon_end

        # 计算进度 (0.0 - 1.0)
        progress = (current_episode - self.start_episode) / \
                  (self.end_episode - self.start_episode)
        progress = max(0.0, min(1.0, progress))  # 限制在[0,1]

        # 线性插值
        epsilon = self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress

        return epsilon

    def is_active(self, episode: int) -> bool:
        """检查阶段是否激活"""
        if self.end_episode is None:
            return episode >= self.start_episode
        return self.start_episode <= episode < self.end_episode


class TrainingStageManager:
    """
    训练阶段管理器

    管理多个训练阶段的切换和参数调整
    参考AlphaGo的分阶段训练策略
    """

    def __init__(self, stages_config: Dict = None):
        """
        初始化阶段管理器

        Args:
            stages_config: 阶段配置字典，如果为None则使用默认配置
        """
        if stages_config is None:
            # 使用默认的三阶段配置
            self.stages = self._create_default_stages()
        else:
            self.stages = self._create_stages_from_config(stages_config)

        self.current_stage = None
        self.current_stage_index = 0

    def _create_default_stages(self):
        """创建默认的三阶段训练配置"""
        return [
            TrainingStage(
                name="阶段1: 行为克隆",
                start_episode=0,
                end_episode=1000,         # 延长到1000回合
                human_data_weight=100.0,  # 极高权重，几乎只用人类数据
                epsilon_start=0.05,       # 极少探索
                epsilon_end=0.1,
                description="从人类专家数据中学习基础策略"
            ),
            TrainingStage(
                name="阶段2: 混合训练",
                start_episode=1000,       # 从1000回合开始
                end_episode=4000,         # 延长到4000回合
                human_data_weight=10.0,   # 高权重，人类数据占主导
                epsilon_start=0.3,        # 开始探索
                epsilon_end=0.7,          # 逐渐增加探索
                description="平衡人类策略与自主探索"
            ),
            TrainingStage(
                name="阶段3: 强化学习主导",
                start_episode=4000,       # 从4000回合开始
                end_episode=20000,        # 延长到20000回合
                human_data_weight=2.0,    # 低权重，主要靠AI自己
                epsilon_start=0.7,
                epsilon_end=0.2,          # 逐渐减少探索
                description="自主优化策略，逐步收敛"
            ),
            TrainingStage(
                name="阶段4: 精调收敛",
                start_episode=20000,      # 从20000回合开始
                end_episode=None,         # 无限
                human_data_weight=1.0,    # 极低权重
                epsilon_start=0.2,
                epsilon_end=0.1,          # 维持低探索
                description="维持最优策略，持续微调"
            )
        ]

    def _create_stages_from_config(self, config):
        """从配置创建阶段列表"""
        stages = []
        for stage_config in config:
            stage = TrainingStage(
                name=stage_config['name'],
                start_episode=stage_config['start_episode'],
                end_episode=stage_config.get('end_episode'),
                human_data_weight=stage_config['human_data_weight'],
                epsilon_start=stage_config['epsilon_start'],
                epsilon_end=stage_config['epsilon_end'],
                description=stage_config.get('description', '')
            )
            stages.append(stage)
        return stages

    def update(self, current_episode: int) -> Tuple[bool, Dict]:
        """
        更新当前阶段

        Args:
            current_episode: 当前回合数

        Returns:
            (stage_changed, stage_info): 阶段是否变化，当前阶段信息
        """
        # 找到当前激活的阶段
        new_stage = None
        new_stage_index = 0

        for i, stage in enumerate(self.stages):
            if stage.is_active(current_episode):
                new_stage = stage
                new_stage_index = i
                break

        # 检查是否切换阶段
        stage_changed = (new_stage != self.current_stage)

        if stage_changed and new_stage is not None:
            self.current_stage = new_stage
            self.current_stage_index = new_stage_index

        # 返回阶段信息
        if self.current_stage is None:
            return False, {}

        stage_info = {
            'name': self.current_stage.name,
            'index': self.current_stage_index,
            'description': self.current_stage.description,
            'human_data_weight': self.current_stage.human_data_weight,
            'epsilon': self.current_stage.get_epsilon(current_episode),
            'progress': self._get_stage_progress(current_episode),
        }

        return stage_changed, stage_info

    def _get_stage_progress(self, current_episode: int) -> float:
        """计算当前阶段的进度 (0.0-1.0)"""
        if self.current_stage is None:
            return 0.0

        if self.current_stage.end_episode is None:
            return 0.0  # 无限阶段无法计算进度

        progress = (current_episode - self.current_stage.start_episode) / \
                  (self.current_stage.end_episode - self.current_stage.start_episode)

        return max(0.0, min(1.0, progress))

    def get_human_data_weight(self, current_episode: int) -> float:
        """获取当前回合的人类数据权重"""
        _, stage_info = self.update(current_episode)
        return stage_info.get('human_data_weight', 1.0)

    def get_epsilon_override(self, current_episode: int) -> float:
        """
        获取当前回合的epsilon值（覆盖默认衰减）

        注意：这个epsilon会覆盖默认的epsilon衰减机制
        只在行为克隆阶段使用固定的低epsilon
        """
        _, stage_info = self.update(current_episode)
        return stage_info.get('epsilon', None)

    def should_use_human_data(self, current_episode: int) -> bool:
        """判断当前是否应该使用人类数据"""
        weight = self.get_human_data_weight(current_episode)
        return weight > 0.5  # 权重>0.5才启用

    def print_stage_info(self, current_episode: int):
        """打印当前阶段信息"""
        _, stage_info = self.update(current_episode)

        if not stage_info:
            return

        print(f"\n{'='*80}")
        print(f"🎯 {stage_info['name']}")
        print(f"{'='*80}")
        print(f"   描述: {stage_info['description']}")
        print(f"   回合: {current_episode} (阶段进度: {stage_info['progress']*100:.1f}%)")
        print(f"   人类数据权重: {stage_info['human_data_weight']:.1f}")
        print(f"   探索率 (Epsilon): {stage_info['epsilon']:.3f}")
        print(f"{'='*80}\n")

    def get_all_stages_summary(self) -> str:
        """获取所有阶段的摘要"""
        summary = "\n训练阶段规划（AlphaGo风格）:\n"
        summary += "=" * 80 + "\n"

        for i, stage in enumerate(self.stages, 1):
            end_desc = f"{stage.end_episode}" if stage.end_episode else "∞"
            summary += f"\n{i}. {stage.name}\n"
            summary += f"   回合范围: {stage.start_episode} - {end_desc}\n"
            summary += f"   人类数据权重: {stage.human_data_weight}\n"
            summary += f"   Epsilon范围: {stage.epsilon_start} → {stage.epsilon_end}\n"
            summary += f"   说明: {stage.description}\n"

        summary += "\n" + "=" * 80

        return summary


# 快速测试
if __name__ == '__main__':
    # 创建管理器
    manager = TrainingStageManager()

    # 打印所有阶段
    print(manager.get_all_stages_summary())

    # 测试不同回合数的阶段切换
    test_episodes = [0, 100, 500, 1000, 2000, 5000]

    print("\n\n阶段切换测试:")
    print("=" * 80)

    for episode in test_episodes:
        stage_changed, info = manager.update(episode)

        print(f"\n回合 {episode:5d}:")
        print(f"  阶段: {info['name']}")
        print(f"  人类数据权重: {info['human_data_weight']:.1f}")
        print(f"  Epsilon: {info['epsilon']:.3f}")
        print(f"  进度: {info['progress']*100:.1f}%")

        if stage_changed:
            print(f"  🔄 阶段切换!")
