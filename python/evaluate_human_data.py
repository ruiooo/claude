#!/usr/bin/env python3
"""
人类数据质量评估工具

功能：
1. 分析所有人类游戏数据的质量
2. 识别好数据 vs 坏数据
3. 提供数据清理建议
4. 可选：自动移动/删除低质量数据
"""

import os
import sys
import struct
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import argparse

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import HUMAN_LEARNING_CONFIG


class HumanDataAnalyzer:
    """人类数据分析器"""

    def __init__(self, data_dir: str = 'human_data'):
        self.data_dir = data_dir
        self.files_info = []

    def load_experience_file(self, file_path: str) -> Dict:
        """
        加载单个经验文件并分析

        返回：
        {
            'file': 文件路径,
            'count': 经验数量,
            'avg_reward': 平均奖励,
            'total_reward': 总奖励,
            'min_reward': 最小奖励,
            'max_reward': 最大奖励,
            'win_rate': 胜利率（估算）,
            'quality_score': 质量评分 (0-100),
        }
        """
        try:
            with open(file_path, 'rb') as f:
                # 读取经验数量
                count_bytes = f.read(4)
                if len(count_bytes) < 4:
                    return None

                count = struct.unpack('i', count_bytes)[0]

                if count <= 0 or count > 10000:  # 合理性检查
                    return None

                # 读取所有数据
                state_dim = 43
                states = np.zeros((count, state_dim), dtype=np.float32)
                actions = np.zeros(count, dtype=np.int32)
                rewards = np.zeros(count, dtype=np.float32)
                next_states = np.zeros((count, state_dim), dtype=np.float32)
                dones = np.zeros(count, dtype=np.int32)

                # 读取状态
                for i in range(count):
                    state_bytes = f.read(state_dim * 4)
                    if len(state_bytes) < state_dim * 4:
                        return None
                    states[i] = np.frombuffer(state_bytes, dtype=np.float32)

                # 读取动作
                for i in range(count):
                    action_bytes = f.read(4)
                    if len(action_bytes) < 4:
                        return None
                    actions[i] = struct.unpack('i', action_bytes)[0]

                # 读取奖励
                for i in range(count):
                    reward_bytes = f.read(4)
                    if len(reward_bytes) < 4:
                        return None
                    rewards[i] = struct.unpack('f', reward_bytes)[0]

                # 读取下一状态
                for i in range(count):
                    next_state_bytes = f.read(state_dim * 4)
                    if len(next_state_bytes) < state_dim * 4:
                        return None
                    next_states[i] = np.frombuffer(next_state_bytes, dtype=np.float32)

                # 读取完成标志
                for i in range(count):
                    done_bytes = f.read(4)
                    if len(done_bytes) < 4:
                        return None
                    dones[i] = struct.unpack('i', done_bytes)[0]

                # 分析数据质量
                total_reward = rewards.sum()
                avg_reward = rewards.mean()
                min_reward = rewards.min()
                max_reward = rewards.max()

                # 估算击杀数（奖励>=40且<=60可能是击杀，简化奖励函数中击杀=+50）
                # 更准确的方法：检测奖励跳跃
                kill_rewards = rewards[(rewards >= 40) & (rewards <= 60)]
                estimated_kills = len(kill_rewards)

                # 估算受伤次数（-15到-5之间，简化奖励中受伤=-10）
                damage_count = len(rewards[(rewards >= -15) & (rewards <= -5)])

                # 估算死亡次数（奖励<-80，简化奖励中死亡=-100）
                death_count = len(rewards[rewards < -80])

                # 计算回合数（done标志的数量）
                episode_count = int(dones.sum())
                if episode_count == 0:  # 如果没有done标志，估计为1个回合
                    episode_count = 1

                # 估算胜率（最终奖励>80可能包含胜利奖励+100）
                final_rewards = rewards[dones == 1] if (dones == 1).any() else rewards[-10:]
                win_count = (final_rewards > 80).sum()
                win_rate = win_count / episode_count if episode_count > 0 else 0

                # 平均回合长度
                avg_episode_length = count / episode_count if episode_count > 0 else count

                # 计算动作多样性
                unique_actions = len(np.unique(actions))
                action_diversity = unique_actions / 9.0  # 总共9个动作

                # K/D比（击杀/死亡比）
                kd_ratio = estimated_kills / death_count if death_count > 0 else estimated_kills

                # 质量评分 (0-100)
                quality_score = 0

                # 1. 平均奖励评分（50分）
                if avg_reward > 20:
                    quality_score += 50
                elif avg_reward > 0:
                    quality_score += 25 + (avg_reward / 20) * 25
                elif avg_reward > -50:
                    quality_score += 10 + ((avg_reward + 50) / 50) * 15

                # 2. 胜率评分（30分）
                quality_score += win_rate * 30

                # 3. 动作多样性评分（20分）
                quality_score += action_diversity * 20

                return {
                    'file': os.path.basename(file_path),
                    'path': file_path,
                    'count': count,
                    'avg_reward': avg_reward,
                    'total_reward': total_reward,
                    'min_reward': min_reward,
                    'max_reward': max_reward,
                    'win_rate': win_rate,
                    'win_count': win_count,
                    'episode_count': episode_count,
                    'avg_episode_length': avg_episode_length,
                    'estimated_kills': estimated_kills,
                    'damage_count': damage_count,
                    'death_count': death_count,
                    'kd_ratio': kd_ratio,
                    'action_diversity': action_diversity,
                    'quality_score': quality_score,
                }

        except Exception as e:
            print(f"  ❌ 读取失败 {file_path}: {e}")
            return None

    def analyze_all_files(self) -> List[Dict]:
        """分析所有人类数据文件"""

        if not os.path.exists(self.data_dir):
            print(f"❌ 目录不存在: {self.data_dir}")
            return []

        files = sorted([
            os.path.join(self.data_dir, f)
            for f in os.listdir(self.data_dir)
            if f.endswith('.dat')
        ])

        if not files:
            print(f"❌ 未找到任何数据文件在 {self.data_dir}/")
            return []

        print(f"📊 正在分析 {len(files)} 个人类数据文件...\n")

        results = []
        for file_path in files:
            info = self.load_experience_file(file_path)
            if info:
                results.append(info)

        self.files_info = results
        return results

    def print_summary(self):
        """打印分析摘要"""

        if not self.files_info:
            print("❌ 没有可用的数据")
            return

        # 排序：质量评分从高到低
        sorted_files = sorted(self.files_info, key=lambda x: x['quality_score'], reverse=True)

        total_count = sum(f['count'] for f in self.files_info)
        total_reward = sum(f['total_reward'] for f in self.files_info)
        avg_quality = np.mean([f['quality_score'] for f in self.files_info])

        print("="*80)
        print("📊 人类数据质量分析报告")
        print("="*80)
        print(f"文件总数: {len(self.files_info)}")
        print(f"经验总数: {total_count:,}")
        print(f"总奖励: {total_reward:.1f}")
        print(f"平均质量评分: {avg_quality:.1f}/100")
        print()

        # 分类统计
        excellent = [f for f in self.files_info if f['quality_score'] >= 70]
        good = [f for f in self.files_info if 50 <= f['quality_score'] < 70]
        medium = [f for f in self.files_info if 30 <= f['quality_score'] < 50]
        poor = [f for f in self.files_info if f['quality_score'] < 30]

        print("质量分布:")
        print(f"  🌟 优秀 (≥70分): {len(excellent)} 个文件, {sum(f['count'] for f in excellent):,} 条经验")
        print(f"  ✅ 良好 (50-69分): {len(good)} 个文件, {sum(f['count'] for f in good):,} 条经验")
        print(f"  ⚠️  中等 (30-49分): {len(medium)} 个文件, {sum(f['count'] for f in medium):,} 条经验")
        print(f"  ❌ 差 (<30分): {len(poor)} 个文件, {sum(f['count'] for f in poor):,} 条经验")
        print()

        # 详细列表
        print("="*80)
        print("详细文件列表（按质量排序）:")
        print("="*80)
        print(f"{'文件名':<30} {'经验数':<8} {'平均奖励':<12} {'胜率':<8} {'质量':<8}")
        print("-"*80)

        for info in sorted_files:
            quality_icon = "🌟" if info['quality_score'] >= 70 else \
                          "✅" if info['quality_score'] >= 50 else \
                          "⚠️ " if info['quality_score'] >= 30 else "❌"

            print(f"{quality_icon} {info['file']:<28} {info['count']:<8} "
                  f"{info['avg_reward']:>+10.2f}  {info['win_rate']*100:>6.1f}%  "
                  f"{info['quality_score']:>6.1f}/100")

        print()

        # 清理建议
        if poor:
            print("="*80)
            print("🗑️  清理建议")
            print("="*80)
            poor_count = sum(f['count'] for f in poor)
            print(f"发现 {len(poor)} 个低质量文件（质量评分<30），共 {poor_count:,} 条经验")
            print()
            print("建议删除以下文件:")
            for info in poor:
                print(f"  ❌ {info['file']} - 评分: {info['quality_score']:.1f}, "
                      f"平均奖励: {info['avg_reward']:+.1f}, 胜率: {info['win_rate']*100:.1f}%")
            print()
            print("执行清理命令:")
            print(f"  python python/evaluate_human_data.py --clean-threshold 30")
            print()

    def print_tier_summary(self):
        """打印各档次汇总统计"""

        if not self.files_info:
            print("❌ 没有可用的数据")
            return

        # 按质量档次分组
        excellent = [f for f in self.files_info if f['quality_score'] >= 70]
        good = [f for f in self.files_info if 50 <= f['quality_score'] < 70]
        medium = [f for f in self.files_info if 30 <= f['quality_score'] < 50]
        poor = [f for f in self.files_info if f['quality_score'] < 30]

        tiers = [
            ("🌟 优秀 (≥70分)", excellent),
            ("✅ 良好 (50-69分)", good),
            ("⚠️  中等 (30-49分)", medium),
            ("❌ 差 (<30分)", poor),
        ]

        print("="*80)
        print("📈 各档次数据汇总统计")
        print("="*80)
        print()

        for tier_name, tier_files in tiers:
            if not tier_files:
                continue

            print(f"{tier_name}")
            print("-"*80)

            # 汇总统计
            total_files = len(tier_files)
            total_exp = sum(f['count'] for f in tier_files)
            total_episodes = sum(f['episode_count'] for f in tier_files)
            total_wins = sum(f['win_count'] for f in tier_files)
            total_kills = sum(f['estimated_kills'] for f in tier_files)
            total_deaths = sum(f['death_count'] for f in tier_files)
            total_damage = sum(f['damage_count'] for f in tier_files)

            avg_quality = np.mean([f['quality_score'] for f in tier_files])
            avg_reward = np.mean([f['avg_reward'] for f in tier_files])
            avg_episode_length = np.mean([f['avg_episode_length'] for f in tier_files])
            avg_kd = np.mean([f['kd_ratio'] for f in tier_files])

            # 计算档次胜率
            tier_win_rate = total_wins / total_episodes if total_episodes > 0 else 0

            # 计算档次K/D
            tier_kd = total_kills / total_deaths if total_deaths > 0 else total_kills

            print(f"  文件数量: {total_files:>4} 个")
            print(f"  经验总数: {total_exp:>6,} 条")
            print(f"  回合总数: {total_episodes:>4} 局")
            print(f"  平均质量: {avg_quality:>6.1f} 分")
            print()
            print(f"  📊 战绩统计:")
            print(f"     胜利局数: {total_wins:>4} 局  |  胜率: {tier_win_rate*100:>5.1f}%")
            print(f"     击杀总数: {total_kills:>4} 次")
            print(f"     死亡总数: {total_deaths:>4} 次")
            print(f"     受伤总数: {total_damage:>4} 次")
            print(f"     K/D比率: {tier_kd:>6.2f}  (档次平均: {avg_kd:>6.2f})")
            print()
            print(f"  📈 表现指标:")
            print(f"     平均奖励: {avg_reward:>+7.2f} 分/条")
            print(f"     平均回合长度: {avg_episode_length:>6.1f} 步/局")
            print(f"     平均击杀/局: {total_kills/total_episodes:>5.2f} 次" if total_episodes > 0 else "")
            print(f"     平均死亡/局: {total_deaths/total_episodes:>5.2f} 次" if total_episodes > 0 else "")
            print()

        # 总体对比
        print("="*80)
        print("📊 档次对比总览")
        print("="*80)
        print(f"{'档次':<20} {'文件':<8} {'经验':<10} {'回合':<8} {'胜率':<10} {'K/D':<10} {'平均质量':<10}")
        print("-"*80)

        for tier_name, tier_files in tiers:
            if not tier_files:
                continue

            total_exp = sum(f['count'] for f in tier_files)
            total_episodes = sum(f['episode_count'] for f in tier_files)
            total_wins = sum(f['win_count'] for f in tier_files)
            total_kills = sum(f['estimated_kills'] for f in tier_files)
            total_deaths = sum(f['death_count'] for f in tier_files)
            avg_quality = np.mean([f['quality_score'] for f in tier_files])

            tier_win_rate = total_wins / total_episodes if total_episodes > 0 else 0
            tier_kd = total_kills / total_deaths if total_deaths > 0 else total_kills

            print(f"{tier_name:<20} {len(tier_files):<8} {total_exp:<10,} {total_episodes:<8} "
                  f"{tier_win_rate*100:<9.1f}% {tier_kd:<9.2f} {avg_quality:<9.1f}")

        print()

    def clean_poor_data(self, threshold: float = 30, dry_run: bool = False):
        """
        清理低质量数据

        Args:
            threshold: 质量评分阈值，低于此值的文件将被删除
            dry_run: 是否为模拟运行（不实际删除）
        """

        if not self.files_info:
            print("❌ 没有可用的数据")
            return

        poor_files = [f for f in self.files_info if f['quality_score'] < threshold]

        if not poor_files:
            print(f"✅ 没有质量评分低于 {threshold} 的文件")
            return

        print("="*80)
        print(f"🗑️  清理低质量数据（阈值: {threshold}分）")
        print("="*80)

        if dry_run:
            print("⚠️  模拟运行模式（不会实际删除文件）\n")

        backup_dir = os.path.join(self.data_dir, 'low_quality_backup')

        for info in poor_files:
            file_path = info['path']

            if dry_run:
                print(f"  [模拟] 将删除: {info['file']} (评分: {info['quality_score']:.1f})")
            else:
                # 创建备份目录
                os.makedirs(backup_dir, exist_ok=True)

                # 移动到备份目录而不是直接删除
                backup_path = os.path.join(backup_dir, info['file'])
                os.rename(file_path, backup_path)
                print(f"  ✓ 已移动到备份: {info['file']} (评分: {info['quality_score']:.1f})")

        if not dry_run:
            print(f"\n✅ 已将 {len(poor_files)} 个低质量文件移动到: {backup_dir}/")
            print(f"   如需恢复，可从该目录复制回 {self.data_dir}/")
        else:
            print(f"\n✓ 模拟完成。实际清理请移除 --dry-run 参数")


def main():
    parser = argparse.ArgumentParser(
        description='人类数据质量评估工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 分析所有数据
  python evaluate_human_data.py

  # 清理质量评分<30的数据（模拟运行）
  python evaluate_human_data.py --clean-threshold 30 --dry-run

  # 实际清理质量评分<30的数据
  python evaluate_human_data.py --clean-threshold 30

  # 清理质量评分<20的数据
  python evaluate_human_data.py --clean-threshold 20
        """
    )

    parser.add_argument(
        '--data-dir',
        default='human_data',
        help='人类数据目录 (默认: human_data)'
    )

    parser.add_argument(
        '--clean-threshold',
        type=float,
        help='清理阈值：低于此质量评分的文件将被移动到备份目录'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='模拟运行：显示将要删除的文件但不实际删除'
    )

    args = parser.parse_args()

    # 创建分析器
    analyzer = HumanDataAnalyzer(data_dir=args.data_dir)

    # 分析所有文件
    analyzer.analyze_all_files()

    # 打印摘要
    analyzer.print_summary()

    # 打印各档次汇总统计
    analyzer.print_tier_summary()

    # 如果指定了清理阈值，执行清理
    if args.clean_threshold is not None:
        print()
        analyzer.clean_poor_data(
            threshold=args.clean_threshold,
            dry_run=args.dry_run
        )


if __name__ == '__main__':
    main()
