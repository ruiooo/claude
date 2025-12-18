#!/usr/bin/env python3
"""
diagnose_human_data.py - 人类数据质量诊断工具
====================================================

功能：
1. 分析人类经验数据的质量和分布
2. 提供详细的统计报告
3. 推荐最佳配置参数
4. 帮助优化模仿学习效果

使用方法：
    python python/diagnose_human_data.py
"""

import os
import numpy as np
from human_data_loader import HumanDataLoader
from config import HUMAN_LEARNING_CONFIG


def print_separator(title="", char="=", width=80):
    """打印分隔线"""
    if title:
        padding = (width - len(title) - 2) // 2
        print(f"\n{char * padding} {title} {char * padding}")
    else:
        print(f"\n{char * width}")


def analyze_data_distribution(data):
    """分析数据分布"""
    states, actions, rewards, next_states, dones = data
    total_count = len(states)

    print_separator("数据分布统计", "=")

    # 基础统计
    print(f"\n📊 基础统计：")
    print(f"   - 总经验数量: {total_count:,} 条")
    print(f"   - 完成的回合数: {int(np.sum(dones))} 个")
    print(f"   - 平均每回合步数: {total_count / max(np.sum(dones), 1):.1f} 步")

    # 奖励分布
    print(f"\n💰 奖励分布：")
    print(f"   - 平均奖励: {np.mean(rewards):.2f}")
    print(f"   - 中位数奖励: {np.median(rewards):.2f}")
    print(f"   - 最大奖励: {np.max(rewards):.2f}")
    print(f"   - 最小奖励: {np.min(rewards):.2f}")
    print(f"   - 标准差: {np.std(rewards):.2f}")

    # 奖励分段统计
    print(f"\n📈 奖励分段：")
    positive_rewards = np.sum(rewards > 0)
    small_negative = np.sum((rewards > -10) & (rewards <= 0))
    medium_negative = np.sum((rewards > -50) & (rewards <= -10))
    large_negative = np.sum(rewards <= -50)

    print(f"   - 正奖励 (>0):        {positive_rewards:6,} 条 ({positive_rewards/total_count*100:5.1f}%)")
    print(f"   - 小负奖励 (-10~0):   {small_negative:6,} 条 ({small_negative/total_count*100:5.1f}%)")
    print(f"   - 中负奖励 (-50~-10): {medium_negative:6,} 条 ({medium_negative/total_count*100:5.1f}%)")
    print(f"   - 大负奖励 (<-50):    {large_negative:6,} 条 ({large_negative/total_count*100:5.1f}%)")

    # ✅ 验证统计一致性：分段加和应等于总数
    segment_total = positive_rewards + small_negative + medium_negative + large_negative
    if segment_total == total_count:
        print(f"   ✓ 统计校验通过: {segment_total:,} 条 = 总数 {total_count:,} 条")
    else:
        print(f"   ⚠️ 统计校验失败: {segment_total:,} 条 ≠ 总数 {total_count:,} 条 (差异: {abs(segment_total - total_count):,})")

    # 动作分布
    print(f"\n🎮 动作分布：")
    action_names = ['静止', '上移', '下移', '左移', '右移',
                   '上射', '下射', '左射', '右射']
    action_total = 0
    for action_id in range(9):
        count = np.sum(actions == action_id)
        if count > 0:
            print(f"   - {action_names[action_id]:4s} (动作{action_id}): {count:6,} 次 ({count/total_count*100:5.1f}%)")
        action_total += count

    # ✅ 验证动作统计一致性
    if action_total == total_count:
        print(f"   ✓ 动作统计校验通过: {action_total:,} 次 = 总数 {total_count:,} 条")
    else:
        print(f"   ⚠️ 动作统计校验失败: {action_total:,} 次 ≠ 总数 {total_count:,} 条 (差异: {abs(action_total - total_count):,})")

    # 特殊奖励分析（与AI训练的奖励系统一致）
    print(f"\n🎯 关键事件：")
    # 击杀奖励: +100 + 奖励塑形(-1~+2) ≈ 99~102
    # 使用 >= 80 阈值以捕获所有击杀（包括负奖励塑形的极端情况）
    kill_rewards = np.sum(rewards >= 80)

    # 受伤惩罚: -15 + 基础存活(+0.02) + 奖励塑形(-1~+2) ≈ -16~-12
    # 使用 <= -10 阈值（保守，可能漏掉少量有高正奖励塑形的受伤）
    hit_rewards = np.sum(rewards <= -10)

    # 死亡惩罚: -100（直接返回，无其他奖励）
    # 使用 <= -100 阈值（准确）
    death_rewards = np.sum(rewards <= -100)

    print(f"   - 击杀敌人: {kill_rewards:6,} 次 ({kill_rewards/max(np.sum(dones), 1):.1f} 次/回合)")
    print(f"   - 受到伤害: {hit_rewards:6,} 次 ({hit_rewards/max(np.sum(dones), 1):.1f} 次/回合)")
    print(f"   - 玩家死亡: {death_rewards:6,} 次")

    return {
        'total_count': total_count,
        'episode_count': int(np.sum(dones)),
        'mean_reward': np.mean(rewards),
        'median_reward': np.median(rewards),
        'positive_ratio': positive_rewards / total_count,
        'kill_count': kill_rewards,
        'hit_count': hit_rewards,
        'death_count': death_rewards,
    }


def analyze_quality(data):
    """分析数据质量"""
    states, actions, rewards, next_states, dones = data

    print_separator("数据质量评估", "=")

    # 按回合聚合数据
    episode_rewards = []
    episode_lengths = []
    episode_outcomes = []

    current_reward = 0
    current_length = 0

    for i in range(len(rewards)):
        current_reward += rewards[i]
        current_length += 1

        if dones[i] == 1:
            episode_rewards.append(current_reward)
            episode_lengths.append(current_length)

            # 判断结果：死亡/胜利/平局
            # 根据最后一步的奖励判断（与AI训练的奖励系统一致）
            #
            # 奖励范围分析：
            # - 失败（死亡）: -100（直接返回，无其他奖励）
            # - 胜利: +300(胜利奖励) + 可能的击杀奖励(+100*n) + 奖励塑形(-1~+2) ≈ 50~500
            # - 平局（超时）: +0.02(存活) + 奖励塑形(-1~+2) ≈ -1~+3
            #
            # 判断阈值：
            # - <= -50: 失败（死亡惩罚-100，即使有少量奖励塑形也<-50）
            # - >= 50: 胜利（包含击杀+100或胜利+300奖励）
            # - 其他: 平局（小奖励，超时或无显著事件）
            if rewards[i] <= -50:
                episode_outcomes.append('失败')
            elif rewards[i] >= 50:
                episode_outcomes.append('胜利')
            else:
                episode_outcomes.append('平局')

            current_reward = 0
            current_length = 0

    if len(episode_rewards) == 0:
        print("\n⚠️  未找到完整回合数据")
        return {}

    episode_rewards = np.array(episode_rewards)
    episode_lengths = np.array(episode_lengths)

    # 回合质量统计
    print(f"\n🏆 回合质量：")
    print(f"   - 平均回合奖励: {np.mean(episode_rewards):.2f}")
    print(f"   - 平均回合长度: {np.mean(episode_lengths):.1f} 步")
    print(f"   - 最佳回合奖励: {np.max(episode_rewards):.2f}")
    print(f"   - 最差回合奖励: {np.min(episode_rewards):.2f}")

    # 胜率统计
    wins = episode_outcomes.count('胜利')
    losses = episode_outcomes.count('失败')
    draws = episode_outcomes.count('平局')
    total = len(episode_outcomes)

    print(f"\n📊 胜负统计：")
    print(f"   - 胜利: {wins:3d} 局 ({wins/total*100:5.1f}%)")
    print(f"   - 失败: {losses:3d} 局 ({losses/total*100:5.1f}%)")
    print(f"   - 平局: {draws:3d} 局 ({draws/total*100:5.1f}%)")
    print(f"   - 总计: {total:3d} 局")

    # ✅ 验证胜负统计一致性
    outcome_total = wins + losses + draws
    if outcome_total == total:
        print(f"   ✓ 胜负统计校验通过: {wins}胜 + {losses}负 + {draws}平 = {total} 局")
    else:
        print(f"   ⚠️ 胜负统计校验失败: {outcome_total} 局 ≠ 总计 {total} 局")

    # 质量分级
    excellent = np.sum(episode_rewards > 100)  # 优秀
    good = np.sum((episode_rewards > 0) & (episode_rewards <= 100))  # 良好
    fair = np.sum((episode_rewards > -50) & (episode_rewards <= 0))  # 一般
    poor = np.sum(episode_rewards <= -50)  # 较差

    print(f"\n⭐ 质量分级：")
    print(f"   - 优秀 (>100):   {excellent:3d} 局 ({excellent/total*100:5.1f}%)")
    print(f"   - 良好 (0~100):  {good:3d} 局 ({good/total*100:5.1f}%)")
    print(f"   - 一般 (-50~0):  {fair:3d} 局 ({fair/total*100:5.1f}%)")
    print(f"   - 较差 (<-50):   {poor:3d} 局 ({poor/total*100:5.1f}%)")

    # ✅ 验证质量分级一致性
    quality_total = excellent + good + fair + poor
    if quality_total == total:
        print(f"   ✓ 质量分级校验通过: {excellent} + {good} + {fair} + {poor} = {total} 局")
    else:
        print(f"   ⚠️ 质量分级校验失败: {quality_total} 局 ≠ 总计 {total} 局")

    return {
        'win_rate': wins / total,
        'avg_episode_reward': np.mean(episode_rewards),
        'avg_episode_length': np.mean(episode_lengths),
        'excellent_ratio': excellent / total,
        'good_ratio': good / total,
        'fair_ratio': fair / total,
        'poor_ratio': poor / total,
    }


def recommend_config(stats, quality):
    """推荐配置参数"""
    print_separator("配置推荐", "=")

    if not quality:
        print("\n⚠️  数据不足，无法提供推荐")
        return

    win_rate = quality.get('win_rate', 0)
    avg_reward = quality.get('avg_episode_reward', 0)
    excellent_ratio = quality.get('excellent_ratio', 0)

    # 评估数据整体质量
    print(f"\n📋 数据质量评级：")
    if win_rate > 0.7 and excellent_ratio > 0.3:
        grade = "A (优秀)"
        color = "🟢"
    elif win_rate > 0.5 and excellent_ratio > 0.2:
        grade = "B (良好)"
        color = "🟡"
    elif win_rate > 0.3:
        grade = "C (一般)"
        color = "🟠"
    else:
        grade = "D (较差)"
        color = "🔴"

    print(f"   {color} 整体评级: {grade}")
    print(f"   - 胜率: {win_rate*100:.1f}%")
    print(f"   - 平均回合奖励: {avg_reward:.1f}")

    # 推荐配置
    print(f"\n⚙️  推荐配置 (python/config.py)：")
    print(f"\n```python")
    print(f"HUMAN_LEARNING_CONFIG = {{")
    print(f"    'enabled': {win_rate > 0.3},  # {'建议启用' if win_rate > 0.3 else '建议禁用（质量不足）'}")
    print(f"    'preload': True,")
    print(f"    'filter_quality': True,")

    # 推荐min_reward
    if excellent_ratio > 0.3:
        min_reward = 50.0
        reason = "数据质量高，严格筛选"
    elif excellent_ratio > 0.1:
        min_reward = 0.0
        reason = "保留正奖励数据"
    elif win_rate > 0.3:
        min_reward = -30.0
        reason = "适度筛选，保留一般数据"
    else:
        min_reward = -50.0
        reason = "数据质量低，尽量保留"

    print(f"    'min_reward': {min_reward},  # {reason}")

    # 推荐sampling_weight
    if win_rate > 0.7:
        weight = 5.0
        reason = "高质量数据，适度权重"
    elif win_rate > 0.5:
        weight = 3.0
        reason = "中等质量，中等权重"
    elif win_rate > 0.3:
        weight = 2.0
        reason = "一般质量，低权重"
    else:
        weight = 1.0
        reason = "低质量数据，最小权重"

    print(f"    'sampling_weight': {weight},  # {reason}")

    print(f"    'periodic_usage': False,  # 分阶段训练模式下全程使用")
    print(f"}}")
    print("```")

    # 分阶段训练推荐
    print(f"\n🎯 分阶段训练建议：")

    if win_rate > 0.5:
        print(f"\n   阶段1 (0-500回合): 行为克隆")
        print(f"     - 纯人类数据训练")
        print(f"     - epsilon: 0.1 (少探索)")
        print(f"     - 学习率: 0.0003")
        print(f"\n   阶段2 (500-2000回合): 混合训练")
        print(f"     - 人类数据权重: {weight * 2}")
        print(f"     - epsilon: 0.3 → 0.7")
        print(f"\n   阶段3 (2000+回合): 强化学习主导")
        print(f"     - 人类数据权重: {weight}")
        print(f"     - epsilon: 0.7 → 0.99+")
    else:
        print(f"\n   ⚠️  数据质量不足以支持行为克隆预训练")
        print(f"   建议：")
        print(f"   1. 收集更多高质量数据（目标胜率>50%）")
        print(f"   2. 或直接使用强化学习，禁用人类数据")


def main():
    """主函数"""
    print_separator("人类数据质量诊断工具", "=")
    print(f"\n📁 数据目录: {HUMAN_LEARNING_CONFIG['human_data_dir']}")

    # 加载数据
    loader = HumanDataLoader(HUMAN_LEARNING_CONFIG['human_data_dir'])
    data = loader.load_all()

    if data is None:
        print(f"\n❌ 未找到人类数据文件")
        print(f"\n💡 提示：")
        print(f"   1. 运行 'make run-player' 开始玩家对战模式")
        print(f"   2. 启用经验记录功能 (输入 'y')")
        print(f"   3. 玩几局游戏，尽量取得胜利")
        print(f"   4. 数据会自动保存到 {HUMAN_LEARNING_CONFIG['human_data_dir']}/ 目录")
        print(f"   5. 再次运行本工具进行分析")
        return

    # 分析数据分布
    stats = analyze_data_distribution(data)

    # 分析数据质量
    quality = analyze_quality(data)

    # 推荐配置
    recommend_config(stats, quality)

    # 数据文件列表
    print_separator("数据文件列表", "=")
    files = sorted([f for f in os.listdir(HUMAN_LEARNING_CONFIG['human_data_dir'])
                   if f.endswith('.dat')])

    if files:
        print(f"\n📂 找到 {len(files)} 个数据文件：")
        for i, filename in enumerate(files[:10], 1):  # 最多显示10个
            filepath = os.path.join(HUMAN_LEARNING_CONFIG['human_data_dir'], filename)
            size = os.path.getsize(filepath) / 1024  # KB
            print(f"   {i:2d}. {filename:30s} ({size:8.1f} KB)")

        if len(files) > 10:
            print(f"   ... 还有 {len(files) - 10} 个文件")

    print_separator("", "=")
    print(f"\n✅ 诊断完成！\n")


if __name__ == '__main__':
    main()
