"""
early_stopping.py - 训练早停和收敛监控系统
================================================

功能：
1. 自动检测训练收敛
2. 防止过拟合
3. 找到最佳训练时机
4. 节省训练时间

原理：
- 监控验证指标（胜率、平均奖励）
- 如果N个评估周期内没有提升，自动停止
- 保存表现最好的模型

参考：
- Keras EarlyStopping
- PyTorch ReduceLROnPlateau
"""

import numpy as np
from typing import Optional, List, Dict


class EarlyStopping:
    """
    早停监控器

    用于判断训练是否应该提前停止，避免过拟合和浪费时间

    使用示例：
        early_stopping = EarlyStopping(
            patience=5,
            min_delta=0.01,
            metric='win_rate'
        )

        for episode in range(max_episodes):
            # 训练...
            win_rate = evaluate()

            if early_stopping.step(win_rate, episode):
                print("训练收敛，提前停止")
                break
    """

    def __init__(self,
                 patience: int = 10,
                 min_delta: float = 0.01,
                 metric: str = 'win_rate',
                 mode: str = 'max',
                 baseline: Optional[float] = None):
        """
        初始化早停监控器

        Args:
            patience: 容忍的评估周期数
                - 如果patience个周期内指标没有提升，停止训练
                - 推荐值: 5-20（坦克游戏建议10）
            min_delta: 最小改进阈值
                - 只有改进大于此值才算真正提升
                - 推荐值: 0.01-0.05（胜率提升1%-5%）
            metric: 监控的指标名称
                - 'win_rate': 胜率（推荐）
                - 'avg_reward': 平均奖励
                - 'avg_length': 平均回合长度
            mode: 优化模式
                - 'max': 指标越大越好（胜率、奖励）
                - 'min': 指标越小越好（损失）
            baseline: 基线值
                - 只有超过基线才开始计数
                - 例如: 胜率必须>50%才算收敛
        """
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.mode = mode
        self.baseline = baseline

        # 内部状态
        self.best_value = None          # 历史最佳值
        self.best_episode = 0            # 最佳值对应的回合数
        self.counter = 0                 # 未改进的计数器
        self.stopped = False             # 是否已停止
        self.history = []                # 历史记录

        # 用于计算是否改进
        if mode == 'max':
            self.is_better = lambda new, best: new > best + min_delta
        else:
            self.is_better = lambda new, best: new < best - min_delta

    def step(self, value: float, episode: int) -> bool:
        """
        更新监控器状态

        Args:
            value: 当前指标值
            episode: 当前回合数

        Returns:
            bool: 是否应该停止训练
                - True: 应该停止
                - False: 继续训练
        """
        # 记录历史
        self.history.append({
            'episode': episode,
            'value': value,
            'is_best': False
        })

        # 检查是否超过基线
        if self.baseline is not None:
            if self.mode == 'max' and value < self.baseline:
                # 未达到基线，不开始计数
                return False
            elif self.mode == 'min' and value > self.baseline:
                return False

        # 初始化最佳值
        if self.best_value is None:
            self.best_value = value
            self.best_episode = episode
            self.history[-1]['is_best'] = True
            return False

        # 检查是否有改进
        if self.is_better(value, self.best_value):
            # 有改进，更新最佳值
            self.best_value = value
            self.best_episode = episode
            self.counter = 0
            self.history[-1]['is_best'] = True
            return False
        else:
            # 没有改进，计数器+1
            self.counter += 1

            if self.counter >= self.patience:
                # 超过容忍次数，停止训练
                self.stopped = True
                return True

        return False

    def get_status(self) -> Dict:
        """获取当前状态信息"""
        return {
            'best_value': self.best_value,
            'best_episode': self.best_episode,
            'counter': self.counter,
            'patience': self.patience,
            'stopped': self.stopped,
            'progress': f"{self.counter}/{self.patience}"
        }

    def print_status(self):
        """打印状态信息"""
        status = self.get_status()
        print(f"\n{'='*80}")
        print(f"📊 早停监控状态")
        print(f"{'='*80}")
        print(f"   监控指标: {self.metric}")

        # 处理 best_value 为 None 的情况（训练初期）
        if status['best_value'] is not None:
            print(f"   最佳值: {status['best_value']:.3f} (回合 {status['best_episode']})")
        else:
            print(f"   最佳值: 暂无 (等待首次评估)")

        print(f"   未改进次数: {status['progress']}")
        if status['stopped']:
            print(f"   状态: ⛔ 已停止（训练收敛）")
        else:
            print(f"   状态: ✅ 继续训练")
        print(f"{'='*80}\n")


class ConvergenceDetector:
    """
    训练收敛检测器

    比早停更复杂，综合多个指标判断是否收敛

    检测标准：
    1. 胜率稳定（标准差小）
    2. 奖励稳定
    3. 策略熵稳定
    4. 持续一定回合数
    """

    def __init__(self,
                 window_size: int = 100,
                 win_rate_threshold: float = 0.7,
                 stability_threshold: float = 0.05):
        """
        初始化收敛检测器

        Args:
            window_size: 滑动窗口大小
                - 在最近N个回合中检查稳定性
            win_rate_threshold: 胜率阈值
                - 只有胜率超过此值才可能收敛
            stability_threshold: 稳定性阈值
                - 标准差低于此值认为稳定
        """
        self.window_size = window_size
        self.win_rate_threshold = win_rate_threshold
        self.stability_threshold = stability_threshold

        # 历史数据
        self.win_rates = []
        self.rewards = []
        self.episode_lengths = []

    def update(self, win_rate: float, avg_reward: float, avg_length: float):
        """更新数据"""
        self.win_rates.append(win_rate)
        self.rewards.append(avg_reward)
        self.episode_lengths.append(avg_length)

        # 保持窗口大小
        if len(self.win_rates) > self.window_size * 2:
            self.win_rates = self.win_rates[-self.window_size:]
            self.rewards = self.rewards[-self.window_size:]
            self.episode_lengths = self.episode_lengths[-self.window_size:]

    def is_converged(self) -> bool:
        """判断是否收敛"""
        if len(self.win_rates) < self.window_size:
            return False

        # 获取最近window_size个数据
        recent_win_rates = self.win_rates[-self.window_size:]
        recent_rewards = self.rewards[-self.window_size:]

        # 检查1: 平均胜率是否足够高
        avg_win_rate = np.mean(recent_win_rates)
        if avg_win_rate < self.win_rate_threshold:
            return False

        # 检查2: 胜率是否稳定（标准差小）
        win_rate_std = np.std(recent_win_rates)
        if win_rate_std > self.stability_threshold:
            return False

        # 检查3: 奖励是否稳定
        reward_std = np.std(recent_rewards)
        reward_mean = np.mean(recent_rewards)
        if reward_mean != 0:
            reward_cv = reward_std / abs(reward_mean)  # 变异系数
            if reward_cv > self.stability_threshold * 2:
                return False

        return True

    def get_metrics(self) -> Dict:
        """获取当前指标"""
        if len(self.win_rates) < self.window_size:
            return {
                'converged': False,
                'reason': f'数据不足 ({len(self.win_rates)}/{self.window_size})'
            }

        recent_win_rates = self.win_rates[-self.window_size:]
        recent_rewards = self.rewards[-self.window_size:]

        avg_win_rate = np.mean(recent_win_rates)
        win_rate_std = np.std(recent_win_rates)
        avg_reward = np.mean(recent_rewards)
        reward_std = np.std(recent_rewards)

        converged = self.is_converged()

        return {
            'converged': converged,
            'avg_win_rate': avg_win_rate,
            'win_rate_std': win_rate_std,
            'avg_reward': avg_reward,
            'reward_std': reward_std,
            'data_points': len(recent_win_rates),
            'meets_win_rate': avg_win_rate >= self.win_rate_threshold,
            'is_stable': win_rate_std <= self.stability_threshold,
        }

    def print_metrics(self):
        """打印收敛指标"""
        metrics = self.get_metrics()

        print(f"\n{'='*80}")
        print(f"🔍 收敛检测报告")
        print(f"{'='*80}")

        if 'reason' in metrics:
            print(f"   状态: ⏳ {metrics['reason']}")
        else:
            print(f"   状态: {'✅ 已收敛' if metrics['converged'] else '⏳ 训练中'}")
            print(f"\n   指标分析:")
            print(f"   - 平均胜率: {metrics['avg_win_rate']*100:.1f}% "
                  f"({'✅' if metrics['meets_win_rate'] else '❌'} 目标: {self.win_rate_threshold*100:.0f}%)")
            print(f"   - 胜率标准差: {metrics['win_rate_std']*100:.2f}% "
                  f"({'✅' if metrics['is_stable'] else '❌'} 阈值: {self.stability_threshold*100:.0f}%)")
            print(f"   - 平均奖励: {metrics['avg_reward']:.2f} ± {metrics['reward_std']:.2f}")
            print(f"   - 统计样本: {metrics['data_points']} 回合")

        print(f"{'='*80}\n")


class TrainingRecommendation:
    """
    训练建议系统

    根据训练进度提供具体建议
    """

    @staticmethod
    def analyze(episode: int,
                win_rate: float,
                avg_reward: float,
                epsilon: float,
                stage_info: Optional[Dict] = None) -> Dict:
        """
        分析训练状态并提供建议

        Args:
            episode: 当前回合数
            win_rate: 当前胜率
            avg_reward: 平均奖励
            epsilon: 当前探索率
            stage_info: 当前训练阶段信息

        Returns:
            分析结果和建议
        """
        recommendations = []
        status = "正常"

        # 根据回合数和胜率判断
        if episode < 500:
            # 早期训练
            if win_rate < 0.2:
                status = "需要关注"
                recommendations.append("早期胜率偏低，建议检查：")
                recommendations.append("  1. 人类数据质量（运行 make diagnose-human-data）")
                recommendations.append("  2. 是否启用了分阶段训练")
                recommendations.append("  3. epsilon是否过高（应<0.3）")
            elif win_rate > 0.5:
                status = "表现优秀"
                recommendations.append("早期训练表现优秀！")
                recommendations.append("  - 可能是人类数据质量很高")
                recommendations.append("  - 建议继续训练到至少2000回合")

        elif episode < 2000:
            # 中期训练
            if win_rate < 0.4:
                status = "需要优化"
                recommendations.append("中期胜率不理想，建议：")
                recommendations.append("  1. 检查是否过早停止探索（epsilon是否过低）")
                recommendations.append("  2. 考虑增加网络容量")
                recommendations.append("  3. 调整奖励函数")
            elif 0.4 <= win_rate < 0.7:
                status = "正常进展"
                recommendations.append("训练正常，建议继续到至少5000回合")
            else:
                status = "快速收敛"
                recommendations.append("表现优秀！建议：")
                recommendations.append("  - 继续训练提升稳定性")
                recommendations.append("  - 可以考虑增加敌人数量")

        else:
            # 后期训练
            if win_rate < 0.5:
                status = "未充分收敛"
                recommendations.append("后期胜率仍然偏低，建议：")
                recommendations.append("  1. 检查是否存在训练崩溃")
                recommendations.append("  2. 降低学习率")
                recommendations.append("  3. 增加训练回合数")
            elif 0.5 <= win_rate < 0.75:
                status = "接近收敛"
                recommendations.append("训练进展良好，建议：")
                recommendations.append("  - 再训练2000-3000回合巩固")
                recommendations.append("  - 监控胜率稳定性")
            elif 0.75 <= win_rate < 0.9:
                status = "已收敛"
                recommendations.append("训练已基本收敛！")
                recommendations.append("  - 可以考虑停止训练")
                recommendations.append("  - 或继续训练追求更高胜率")
            else:
                status = "完全收敛"
                recommendations.append("训练已完全收敛！")
                recommendations.append("  - 建议停止训练")
                recommendations.append("  - 已达到当前配置下的最优性能")

        # 检查探索率
        if epsilon < 0.1 and episode < 5000:
            recommendations.append("\n⚠️  探索率过早降低！")
            recommendations.append("  - 建议调大 epsilon_decay")

        return {
            'status': status,
            'recommendations': recommendations,
            'should_continue': win_rate < 0.85,
            'estimated_final_win_rate': min(win_rate + 0.1, 0.95)
        }

    @staticmethod
    def estimate_remaining_episodes(current_episode: int,
                                   current_win_rate: float,
                                   target_win_rate: float = 0.8) -> int:
        """
        估计达到目标胜率还需要多少回合

        简单线性估计（实际可能非线性）
        """
        if current_win_rate >= target_win_rate:
            return 0

        # 假设前期提升快，后期慢
        # 使用对数模型
        if current_episode < 500:
            rate_per_1k = 0.15  # 前500回合，每1000回合提升15%
        elif current_episode < 2000:
            rate_per_1k = 0.10  # 500-2000回合，每1000回合提升10%
        else:
            rate_per_1k = 0.05  # 2000+回合，每1000回合提升5%

        remaining_improvement = target_win_rate - current_win_rate
        estimated_1k_episodes = remaining_improvement / rate_per_1k
        estimated_episodes = int(estimated_1k_episodes * 1000)

        return max(0, estimated_episodes)


# 快速测试
if __name__ == '__main__':
    # 测试早停
    early_stopping = EarlyStopping(patience=5, min_delta=0.01)

    print("测试早停系统:")
    print("=" * 80)

    # 模拟训练过程
    win_rates = [0.3, 0.4, 0.5, 0.6, 0.65, 0.67, 0.68, 0.68, 0.67, 0.68, 0.67, 0.68]

    for i, wr in enumerate(win_rates):
        episode = i * 100
        should_stop = early_stopping.step(wr, episode)

        print(f"回合 {episode:4d}: 胜率={wr:.2f}, 计数器={early_stopping.counter}/{early_stopping.patience}")

        if should_stop:
            print(f"\n✋ 训练应该停止！")
            if early_stopping.best_value is not None:
                print(f"   最佳胜率: {early_stopping.best_value:.2f} (回合 {early_stopping.best_episode})")
            break

    # 测试收敛检测
    print("\n\n测试收敛检测:")
    print("=" * 80)

    detector = ConvergenceDetector()

    # 模拟稳定的高胜率
    for i in range(150):
        wr = 0.75 + np.random.normal(0, 0.02)  # 胜率75% ± 2%
        reward = 150 + np.random.normal(0, 10)
        length = 500 + np.random.normal(0, 50)

        detector.update(wr, reward, length)

        if i == 50 or i == 100 or i == 149:
            detector.print_metrics()
